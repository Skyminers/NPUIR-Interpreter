# NPUIR Interpreter

NPUIR Interpreter 是面向 memref-form NPUIR（HIVM）的 CPU 解释器，用于：

- 在没有 Ascend 设备和 CANN Runtime 的机器上验证 NPUIR 数值结果；
- 模拟 S/V/M/MTE 等流水线的延迟提交与同步；
- 检测缺失同步、数据竞争、死锁和越界访问。

解释器在运行时不依赖 Ascend 硬件；源码构建依赖仓库中固定版本的
AscendNPU-IR submodule，以及该 submodule 固定的 LLVM/MLIR 版本。

## 获取源码

```bash
git clone --recursive https://github.com/Skyminers/NPUIR-Interpreter.git
cd NPUIR-Interpreter
```

已有普通 clone 时：

```bash
git submodule update --init --recursive
```

## 构建与测试

需要 CMake、Ninja，以及支持 C++17 的编译器：

```bash
./build.sh
```

该命令构建解释器并运行回归测试和单元测试。产物位于：

```text
build/bin/npuir-interp
```

只构建工具：

```bash
./build.sh --target npuir-interp
```

## 快速使用

```bash
build/bin/npuir-interp kernel.mlir \
  --sched=lazy \
  --args=a.npy,b.npy,zeros \
  --out=out.
```

完整命令行说明见 [使用指南](使用指南.md)，设计与实现见
[中文架构文档](docs/BiShengIRInterp_zh.md)。

## 依赖版本

AscendNPU-IR 通过 `third_party/AscendNPU-IR` submodule 固定到 `feature/regbase`
分支的提交 `5a744afc17c0caa6833cf04f773898450041ff98`；
不要通过复制生成头文件或混用其他 LLVM 构建目录来替换它。升级依赖时应同时
执行完整回归测试与低精度逐位扫描。

依赖来源为官方 `Ascend/AscendNPU-IR` 仓库。submodule 记录的是固定提交，
`feature/regbase` 仅用于说明版本来源；升级依赖后应提交新的 gitlink。

## License

Apache License v2.0 with LLVM Exceptions，详见 [LICENSE.TXT](LICENSE.TXT)。
