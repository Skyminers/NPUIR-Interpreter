#!/usr/bin/env python3
"""Compile one modular Triton DSL case through the TTAdapter stage."""

import argparse
import importlib
import inspect
import json
import os
import pathlib
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
CASES_DIR = TEST_ROOT / "dsl_e2e" / "cases"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


def case_names() -> list[str]:
    return sorted(path.stem for path in CASES_DIR.glob("[!_]*.py"))


def load_case(name: str):
    if name not in case_names():
        raise ValueError(f"unknown DSL E2E case: {name}")
    module = importlib.import_module(f"dsl_e2e.cases.{name}")
    case = module.CASE
    if case.name != name:
        raise RuntimeError(f"case module {name} declares name {case.name}")
    return case


def make_source(case_name: str) -> ASTSource:
    case = load_case(case_name)
    # Triton renamed this argument in 3.3. Keep this test usable with both the
    # 3.2 Ascend release and newer source builds.
    keyword = (
        "constexprs"
        if "constexprs" in inspect.signature(ASTSource).parameters
        else "constants"
    )
    return ASTSource(
        case.kernel,
        signature=case.signature,
        **{keyword: case.constants},
    )


def compile_ttadapter(arch: str, case_name: str = "add") -> str:
    """Run Triton's in-process stages and stop before the external NPU compiler."""
    from triton._C.libtriton import ir

    target = GPUTarget("npu", arch, 0)
    backend = make_backend(target)
    options = backend.parse_options({"arch": arch})
    source = make_source(case_name)

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
        description="Compile a modular Triton DSL test case to TTAdapter IR."
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
        choices=case_names(),
        default="add",
        help="DSL test case to compile (default: add)",
    )
    parser.add_argument(
        "--print-ir", action="store_true", help="also print the generated IR"
    )
    parser.add_argument(
        "--metadata", type=pathlib.Path, help="also write runtime metadata as JSON"
    )
    args = parser.parse_args()

    ttadapter = compile_ttadapter(args.arch, args.case)
    args.output.write_text(ttadapter, encoding="utf-8")
    if args.metadata:
        args.metadata.write_text(
            json.dumps(load_case(args.case).metadata()), encoding="utf-8"
        )
    print(f"TTAdapter IR written to {args.output.resolve()}")
    if args.print_ir:
        print(ttadapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
