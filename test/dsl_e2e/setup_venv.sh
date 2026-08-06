#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="${NPUIR_PYTHON:-python3.12}"
triton_root="$repo_root/third_party/triton-ascend"
venv_dir="${NPUIR_VENV:-$repo_root/.venv}"
wheel_root="$repo_root/build/ta-wheelhouse"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: $python_bin was not found; install Python 3.12 or set NPUIR_PYTHON." >&2
  exit 1
fi

if [[ ! -f "$triton_root/python/setup.py" ]]; then
  git -C "$repo_root" submodule sync -- third_party/triton-ascend
  git -C "$repo_root" submodule update --init third_party/triton-ascend
fi
if [[ ! -f "$triton_root/python/setup.py" ]]; then
  echo "error: Triton Ascend submodule could not be initialized." >&2
  exit 1
fi
npuir_root="$triton_root/third_party/ascend/AscendNPU-IR"
if [[ ! -f "$npuir_root/CMakeLists.txt" ]]; then
  git -C "$triton_root" submodule update --init third_party/ascend/AscendNPU-IR
fi
triton_version="$(tr -d '[:space:]' < "$triton_root/version.txt")"
triton_commit="$(git -C "$triton_root" rev-parse HEAD)"
triton_commit_short="$(git -C "$triton_root" rev-parse --short=8 HEAD)"
npuir_commit="$(git -C "$npuir_root" rev-parse HEAD)"
wheel_dir="$wheel_root/$triton_commit_short"
ta_build_root="$repo_root/build/ta-source/$triton_commit_short"

if [[ -x "$venv_dir/bin/python" ]] && ! "$venv_dir/bin/python" -c 'import sys' >/dev/null 2>&1; then
  echo "repairing unusable Python environment: $venv_dir"
  "$python_bin" -m venv --clear "$venv_dir"
else
  "$python_bin" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" -m pip install --upgrade pip setuptools
"$venv_dir/bin/python" -m pip install \
  -r "$repo_root/test/dsl_e2e/requirements.txt"

mkdir -p "$wheel_dir" "$repo_root/.cache" "$(dirname "$ta_build_root")"

# TA's setup.py temporarily patches its AscendNPU-IR checkout. Build from local
# clones under build/ so source compilation never dirties either submodule in
# third_party/. The clones use the already initialized objects and need no
# network access.
"$venv_dir/bin/cmake" -E remove_directory "$ta_build_root"
git clone --quiet --shared --no-checkout "$triton_root" "$ta_build_root"
git -C "$ta_build_root" checkout --quiet --detach "$triton_commit"
mkdir -p "$ta_build_root/third_party/ascend"
git clone --quiet --shared --no-checkout "$npuir_root" \
  "$ta_build_root/third_party/ascend/AscendNPU-IR"
git -C "$ta_build_root/third_party/ascend/AscendNPU-IR" \
  checkout --quiet --detach "$npuir_commit"

cleanup_ta_build_root() {
  "$venv_dir/bin/cmake" -E remove_directory "$ta_build_root"
}
trap cleanup_ta_build_root EXIT

(
  cd "$ta_build_root/python"
  export CCACHE_DIR="${CCACHE_DIR:-$repo_root/.cache/ccache-ta}"
  export TRITON_HOME="${TRITON_HOME:-$repo_root/.cache}"
  if [[ -z "${TRITON_BUILD_WITH_CCACHE+x}" ]]; then
    if command -v ccache >/dev/null 2>&1; then
      export TRITON_BUILD_WITH_CCACHE=true
    else
      export TRITON_BUILD_WITH_CCACHE=false
    fi
  fi
  if [[ -z "${TRITON_BUILD_WITH_CLANG_LLD+x}" ]]; then
    if command -v clang >/dev/null 2>&1 && command -v lld >/dev/null 2>&1; then
      export TRITON_BUILD_WITH_CLANG_LLD=true
    else
      export TRITON_BUILD_WITH_CLANG_LLD=false
    fi
  fi
  export TRITON_BUILD_PROTON=OFF
  export TRITON_WHEEL_NAME=triton-ascend
  ta_cmake_args="-DTRITON_BUILD_UT=OFF"
  export TRITON_APPEND_CMAKE_ARGS="$ta_cmake_args ${TRITON_APPEND_CMAKE_ARGS:-}"
  "$venv_dir/bin/python" setup.py bdist_wheel --dist-dir "$wheel_dir"
)

wheel_path="$(find "$wheel_dir" -type f -name "triton_ascend-${triton_version}*.whl" -print | sort | tail -n 1)"
if [[ -z "$wheel_path" ]]; then
  echo "error: Triton Ascend wheel was not produced in $wheel_dir" >&2
  exit 1
fi

# Upstream declares the community `triton` distribution as a dependency even
# though this wheel provides the same top-level package. Installing --no-deps
# avoids letting pip overwrite the just-built Ascend implementation.
"$venv_dir/bin/python" -m pip install --force-reinstall --no-deps "$wheel_path"
"$venv_dir/bin/python" - "$triton_root/version.txt" "$wheel_path" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
import triton
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

expected = Path(sys.argv[1]).read_text().strip()
version = importlib.metadata.version("triton-ascend")
if version != expected and not version.startswith(f"{expected}+"):
    raise RuntimeError(f"expected triton-ascend {expected}, got {version}")
print(f"OK: built {sys.argv[2]}")
print(f"OK: installed triton-ascend {version} ({triton.__file__})")
PY

git -C "$triton_root" diff --quiet --exit-code
git -C "$npuir_root" diff --quiet --exit-code
echo "Activate with: source $venv_dir/bin/activate"
