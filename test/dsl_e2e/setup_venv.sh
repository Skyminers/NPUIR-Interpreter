#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="${NPUIR_PYTHON:-python3.12}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: Triton Ascend 3.2.1 only publishes Linux wheels." >&2
  echo "Run DSL E2E in an x86_64/aarch64 Linux environment." >&2
  exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: $python_bin was not found; install Python 3.12 or set NPUIR_PYTHON." >&2
  exit 1
fi

"$python_bin" -m venv "$repo_root/.venv"
"$repo_root/.venv/bin/python" -m pip install --upgrade pip
"$repo_root/.venv/bin/python" -m pip install \
  -r "$repo_root/test/dsl_e2e/requirements.txt"
"$repo_root/.venv/bin/python" - <<'PY'
import importlib.metadata
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

version = importlib.metadata.version("triton-ascend")
if version != "3.2.1":
    raise RuntimeError(f"expected triton-ascend 3.2.1, got {version}")
print(f"OK: triton-ascend {version} ({triton.__file__})")
PY

echo "Activate with: source .venv/bin/activate"
