# DSL E2E 环境

这组测试需要 Triton Ascend 的发行包来完成 DSL → TTAdapter。推荐环境是：

- Ubuntu Linux，x86_64 或 aarch64；
- Python 3.12；
- `triton-ascend==3.2.1`。

## 创建 venv 并安装发行包

在仓库根目录运行安装脚本：

```bash
test/dsl_e2e/setup_venv.sh
source .venv/bin/activate
```

脚本实际执行的手动安装步骤如下：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r test/dsl_e2e/requirements.txt
```

需要指定其他 Python 3.10–3.13 解释器时设置 `NPUIR_PYTHON`，例如：

```bash
NPUIR_PYTHON=python3.11 test/dsl_e2e/setup_venv.sh
```

`requirements.txt` 使用 Triton Ascend 官方包索引。pip 会根据当前 Python ABI 和
CPU 架构下载对应 wheel，例如 Python 3.12/aarch64 会选择 `cp312`、`aarch64` 的
manylinux wheel，不应手工安装其他 ABI 或架构的文件。

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

不要在同一 venv 中另外安装或升级社区版 `triton`。两个发行包使用相同的顶层
`triton` 目录，后安装的包可能覆盖 Triton Ascend 文件。

## 离线下载

在一台与目标机具有相同 Linux 架构和 Python 版本的联网机器上运行：

```bash
python3.12 -m pip download \
  --dest wheelhouse \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple \
  triton-ascend==3.2.1
```

将整个 `wheelhouse` 复制到目标机后安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse triton-ascend==3.2.1
```

必须复制整个目录，而不只是 `triton_ascend` wheel，因为其中还包含解析得到的依赖。

## 平台限制

官方 3.2.1 索引当前提供 Python 3.10–3.13 的 Linux x86_64/aarch64 wheel，另外提供
Python 3.9/x86_64 wheel；不提供 macOS wheel。macOS 上不能直接安装 manylinux
wheel，因为其中的 `libtriton` 是 Linux ELF 动态库。需要在 Linux 主机、Linux VM
或 Linux 容器中运行 DSL E2E。

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
