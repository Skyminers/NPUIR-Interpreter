#!/usr/bin/env python3
"""Compile a small Triton DSL kernel through the TTAdapter stage."""

import argparse
import inspect
import os
import pathlib

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.debug_barrier()
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def affine_abs_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.abs(x - 4.0) * 2.0
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def select_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x > 7.0, x, 7.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def cast_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = (x * 0.5 + 0.75).to(tl.int32).to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


CASES = {
    "add": (
        add_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"},
        1024,
    ),
    "affine_abs": (
        affine_abs_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32"},
        1024,
    ),
    "select": (
        select_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32"},
        32,
    ),
    "cast": (
        cast_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32"},
        1024,
    ),
}


def make_source(case: str) -> ASTSource:
    kernel, pointer_args, block_size = CASES[case]
    signature = {
        **pointer_args,
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }
    constants = {"BLOCK_SIZE": block_size}

    # Triton renamed this argument in 3.3. Keep this test usable with both the
    # 3.2 Ascend release and newer source builds.
    keyword = (
        "constexprs"
        if "constexprs" in inspect.signature(ASTSource).parameters
        else "constants"
    )
    return ASTSource(kernel, signature=signature, **{keyword: constants})


def compile_ttadapter(arch: str, case: str = "add") -> str:
    """Run Triton's in-process stages and stop before the external NPU compiler."""
    from triton._C.libtriton import ir

    target = GPUTarget("npu", arch, 0)
    backend = make_backend(target)
    options = backend.parse_options({"arch": arch})
    source = make_source(case)

    stages = {}
    if "language" in inspect.signature(backend.add_stages).parameters:
        backend.add_stages(stages, options, source.language)
    else:
        backend.add_stages(stages, options)
    if "ttadapter" not in stages:
        raise RuntimeError("the selected Triton backend has no TTAdapter stage")

    context = ir.context()
    ir.load_dialects(context)
    new_compiler_api = "target" in inspect.signature(source.make_ir).parameters
    if not new_compiler_api:
        from triton._C.libtriton import buffer_ir
        from triton._C.libtriton.ascend import ir as ascend_ir

        buffer_ir.load_dialects(context)
        ascend_ir.load_dialects(context)
    backend.load_dialects(context)

    if inspect.signature(backend.get_codegen_implementation).parameters:
        codegen_fns = backend.get_codegen_implementation(options)
    else:
        codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()
    if new_compiler_api:
        module = source.make_ir(target, options, codegen_fns, module_map, context)
    else:
        module = source.make_ir(options, codegen_fns, module_map, context)

    metadata = {
        "hash": "dump-ttadapter",
        "target": target,
        "triton_version": triton.__version__,
        **options.__dict__,
    }
    first_stage = list(stages).index(source.ext)
    for name, lower in list(stages.items())[first_stage:]:
        module = lower(module, metadata)
        if name == "ttadapter":
            text = str(module)
            if not text.strip():
                raise RuntimeError("Triton produced an empty TTAdapter module")
            return text

    raise RuntimeError("TTAdapter stage was not executed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a built-in Triton DSL test case to TTAdapter IR."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path("kernel.ttadapter.mlir"),
        help="output MLIR path (default: kernel.ttadapter.mlir)",
    )
    parser.add_argument(
        "--arch",
        default=os.environ.get("TRITON_ASCEND_ARCH", "Ascend910B4"),
        help="Ascend target architecture (default: TRITON_ASCEND_ARCH or Ascend910B4)",
    )
    parser.add_argument(
        "--case",
        choices=CASES,
        default="add",
        help="DSL test case to compile (default: add)",
    )
    parser.add_argument(
        "--print-ir", action="store_true", help="also print the generated IR"
    )
    args = parser.parse_args()

    ttadapter = compile_ttadapter(args.arch, args.case)
    args.output.write_text(ttadapter, encoding="utf-8")
    print(f"TTAdapter IR written to {args.output.resolve()}")
    if args.print_ir:
        print(ttadapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
