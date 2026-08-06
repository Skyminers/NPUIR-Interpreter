# DSL E2E 环境

这组测试从仓库固定的 Triton Ascend 3.2.2 兼容源码构建 wheel，用它完成 DSL → TTAdapter。
不需要再从额外的 Python 包索引下载 `triton-ascend`。推荐环境是：

- Ubuntu Linux，x86_64 或 aarch64；
- C/C++ 编译器和 Python 3.12；
- CMake、Ninja、pybind11 等 Python 构建依赖由安装脚本准备。

TA submodule 指向项目兼容 fork，并固定到与本仓库当前 BiSheng 接口匹配的提交
`2c7d3bbf9ad5b3343db04701355cdce4370fe342`；安装脚本不会读取相邻目录或本机已有的
TA checkout。

## 从源码构建并安装

在仓库根目录运行安装脚本：

```bash
test/dsl_e2e/setup_venv.sh
source .venv/bin/activate
```

脚本会完成以下步骤：

1. 初始化 `third_party/triton-ascend` 及其嵌套 submodule；
2. 创建 `.venv` 并安装构建依赖；
3. 在 `build/ta-source/` 创建本地源码克隆，隔离 TA 构建时施加的临时补丁；
4. 调用 TA 的 `python/setup.py bdist_wheel` 编译源码；
5. 将 wheel 保存到 `build/ta-wheelhouse/<TA 提交>/`，再安装到 `.venv`；
6. 导入 `libtriton` 及 DSL E2E 使用的编译 API，并确认原始 submodule 干净。

需要指定其他 Python 3.10–3.13 解释器时设置 `NPUIR_PYTHON`，例如：

```bash
NPUIR_PYTHON=python3.11 test/dsl_e2e/setup_venv.sh
```

如需把 venv 放到其他位置，可设置 `NPUIR_VENV`。TA 构建默认下载其源码提交固定的
LLVM 工具链到仓库 `.cache/`；已有兼容 LLVM 安装时，可以通过 `LLVM_SYSPATH` 指定，
从而进行离线构建：

```bash
LLVM_SYSPATH=/opt/ta-llvm \
NPUIR_VENV=/opt/npuir-ta-venv \
test/dsl_e2e/setup_venv.sh
```

安装后检查实际发行包和关键编译接口：

```bash
python - <<'PY'
import importlib.metadata
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

print("triton-ascend", importlib.metadata.version("triton-ascend"))
print("triton", triton.__version__, triton.__file__)
PY
```

安装脚本通过 requirements 显式安装 TA 的 Python 依赖，并对本地 TA wheel 使用
`--no-deps`，因为该版本上游元数据还声明了社区版 `triton`，而两者使用相同的顶层
`triton` 目录。不要在同一 venv 中另外安装或升级社区版 `triton`，否则其文件可能
覆盖 Triton Ascend；`pip check` 因此会报告这一条已知的上游元数据冲突。

## 复用和分发构建结果

成功构建后，匹配当前 Python ABI 和平台的 wheel 位于：

```bash
find build/ta-wheelhouse -name 'triton_ascend-*.whl'
```

可以把该 wheel 复制到具有相同 Python ABI、CPU 架构和系统 ABI 的机器，然后安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps build/ta-wheelhouse/2c7d3bbf/triton_ascend-*.whl
```

离线从头构建时，需要预先提供 Python 构建依赖，并通过 `LLVM_SYSPATH` 指向兼容的
LLVM 安装；设置 `TRITON_OFFLINE_BUILD=1` 可禁止 TA 构建脚本下载工具链。

## 平台限制

源码构建产物与 Python ABI、操作系统和 CPU 架构绑定，不能跨平台复制。例如 macOS
构建的 wheel 不能用于 Linux，x86_64 wheel 也不能用于 aarch64。用于发布的 wheel
应在目标系统或与目标 ABI 一致的构建容器中生成。

本仓库的 DSL E2E 只使用 Triton Ascend 生成 TTAdapter，不会执行 NPU kernel；因此
没有 NPU 的 Linux 开发机也可以运行这条解释器验证链路。`npu-smi` 查询失败的提示
不代表 TTAdapter 编译失败。

## 运行验证

先构建本仓库工具，然后激活 venv：

```bash
./build.sh --target npuir-interp-debug
source .venv/bin/activate
```

运行全部用例：

```bash
python test/run_dsl_e2e.py --triton-python "$VIRTUAL_ENV/bin/python"
```

运行 Flash Attention 并打开调试界面：

```bash
python test/dsl_e2e/cases/flash_attention.py --debug --serve
```

该命令同时校验 lazy/inorder 输出、数学参考值和 fuzz 调度结果。
