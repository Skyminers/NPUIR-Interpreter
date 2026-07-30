# NPUIR Interpreter

NPUIR Interpreter 是面向 **memref-form NPUIR（HIVM）** 的 CPU 解释器。它可以在
没有 Ascend 设备和 CANN Runtime 的环境中执行编译后的 kernel，并检查数值结果与
运行期同步语义。

核心能力：

- 通过 `inorder`、`lazy` 和 `fuzz` 三种调度模式执行 NPUIR；
- 模拟 S、V、M、MTE 等流水线的延迟提交与同步；
- 检测缺失同步、跨核数据竞争、死锁、越界访问和 layout 不一致；
- 使用 APInt/APFloat 覆盖 f16、bf16、f8 等低精度数值路径；
- 读取和输出 `.npy`，便于与 NumPy、PyTorch golden 结果比较。

解释器运行时不依赖 Ascend 硬件。源码构建依赖仓库固定版本的 AscendNPU-IR
submodule，以及该 submodule 固定的 LLVM/MLIR 版本。

## 整体架构

![Architecture](images/Architecture.png)

关键设计是**延迟提交**：带 pipe 的 op 不会在发射时立即修改内存，而是把 effect
放入对应的 `PipeEngine` 队列，只在 flag、barrier 或返回等硬件语义允许的位置提交。
因此缺失同步不会被主机程序顺序掩盖，而会表现为明确诊断、毒值、竞争或死锁。

## 项目结构

| 路径 | 内容 |
|---|---|
| `include/bishengir/Tools/Interp/` | 解释器公开接口：运行时值、内存、流水线与调度器 |
| `lib/` | 核心实现，以及按 op 家族拆分的 handler |
| `tools/` | `npuir-interp` 命令行入口 |
| `docs/` | 使用指南、架构设计与文档索引 |
| `unittests/` | `Value`、`ShadowMemory`、`PipeEngine` 等 gtest 单元测试 |
| `test/` | lit 功能、同步、竞争、死锁、越界、layout 和错误用例 |
| `test/Precision/` | 独立的低精度逐位扫描工具 |
| `third_party/AscendNPU-IR/` | 固定版本的 AscendNPU-IR submodule |
| `build.sh` | LLVM external project 的配置、构建与测试入口 |

## 获取源码

```bash
git clone --recursive https://github.com/Skyminers/NPUIR-Interpreter.git
cd NPUIR-Interpreter
```

已有普通 clone 时补齐 submodule：

```bash
git submodule update --init --recursive
```

## 构建与测试

需要 CMake、Ninja、Python 3，以及支持 C++17 的编译器。

```bash
./build.sh
```

默认目标会构建工具，并运行 lit 回归、gtest 单元测试和四组精度扫描。产物位于
`build/bin/npuir-interp`。

日常开发可以按需运行单个目标：

```bash
./build.sh --target npuir-interp
cmake --build build --target check-npuir-interpreter
cmake --build build --target check-npuir-interpreter-unit
cmake --build build --target check-npuir-interpreter-precision
```

## 快速使用

```bash
build/bin/npuir-interp kernel.mlir \
  --sched=lazy \
  --args=a.npy,b.npy,zeros \
  --out=out.
```

完整参数、调度模式和诊断示例见[中文使用指南](docs/Usage_zh.md)。实现原理见
[中文架构文档](docs/Architecture_zh.md)或[English architecture](docs/Architecture.md)；
全部文档入口见 [`docs/README.md`](docs/README.md)。

## 依赖版本

AscendNPU-IR 通过 `third_party/AscendNPU-IR` submodule 固定到来源分支
`feature/regbase` 的提交 `5a744afc17c0caa6833cf04f773898450041ff98`。不要通过
复制生成头文件或混用其他 LLVM 构建目录替换它。升级依赖时应提交新的 gitlink，
并运行完整回归与低精度逐位扫描。

## License

Apache License v2.0 with LLVM Exceptions，详见 [LICENSE.TXT](LICENSE.TXT)。
