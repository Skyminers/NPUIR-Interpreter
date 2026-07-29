#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_ROOT="$ROOT/third_party/AscendNPU-IR"
LLVM_ROOT="$ASCEND_ROOT/third-party/llvm-project"
BUILD_DIR="$ROOT/build"
BUILD_TYPE="Release"
JOBS=""
RECONFIGURE=0

# Keep compiler-cache state self-contained.  Callers may still override this.
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/.cache/ccache}"

usage() {
  echo "用法: ./build.sh [-r] [-j N] [--build-type TYPE] [--target TARGET]"
  echo "默认目标: 工具、lit、单元测试和精度扫描"
}

TARGETS=(npuir-interp check-npuir-interpreter
         check-npuir-interpreter-unit check-npuir-interpreter-precision)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--reconfigure) RECONFIGURE=1; shift ;;
    -j|--jobs) JOBS="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --target) TARGETS=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -f "$LLVM_ROOT/llvm/CMakeLists.txt" ]]; then
  echo "错误: submodule 尚未完整初始化，请执行:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

for tool in cmake ninja; do
  command -v "$tool" >/dev/null || {
    echo "错误: 找不到 $tool" >&2
    exit 1
  }
done

if [[ "$RECONFIGURE" == 1 ]]; then
  cmake -E remove_directory "$BUILD_DIR"
fi

if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  cmake -S "$LLVM_ROOT/llvm" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir;npuir_interpreter" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$ASCEND_ROOT/bishengir" \
    -DLLVM_EXTERNAL_NPUIR_INTERPRETER_SOURCE_DIR="$ROOT" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_HIIPU=ON \
    -DBSPUB_DAVINCI=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
    -DBISHENGIR_ENABLE_TRITON_COMPILE=ON \
    -DMLIR_ENABLE_BISHENGIR_EXTENTION=ON \
    -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON \
    -DBISHENGIR_PUBLISH=OFF \
    -DBISHENGIR_DISABLE_CANN=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DNPUIR_INTERPRETER_INCLUDE_TESTS=ON \
    -DBUILD_SHARED_LIBS=OFF
fi

if [[ -z "$JOBS" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 2)"
  elif command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS=2
  fi
fi

cmake --build "$BUILD_DIR" --parallel "$JOBS" --target "${TARGETS[@]}"

if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
  cmake -E create_symlink build/compile_commands.json "$ROOT/compile_commands.json"
fi

echo "完成: $BUILD_DIR/bin/npuir-interp"
