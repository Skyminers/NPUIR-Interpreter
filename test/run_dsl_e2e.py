#!/usr/bin/env python3
"""Run modular Triton DSL cases through lowering and the interpreter."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

from dsl_e2e.common import check_output


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
CASES_DIR = TEST_ROOT / "dsl_e2e" / "cases"


def case_names() -> list[str]:
    return sorted(path.stem for path in CASES_DIR.glob("[!_]*.py"))


def run_case(args, case: str, work_dir: pathlib.Path) -> None:
    dump_script = TEST_ROOT / "dump_ttadapter.py"
    lower_script = TEST_ROOT / "lower_ttadapter_for_interp.py"
    ttadapter = work_dir / f"{case}.ttadapter.mlir"
    hivm = work_dir / f"{case}.hivm.mlir"
    metadata_path = work_dir / f"{case}.json"

    print(f"[{case}] DSL -> TTAdapter", flush=True)
    subprocess.run(
        [
            args.triton_python,
            dump_script,
            "--case",
            case,
            "--arch",
            args.arch,
            "--metadata",
            metadata_path,
            ttadapter,
        ],
        check=True,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["name"] != case:
        raise RuntimeError(f"case module {case} declares name {metadata['name']}")
    print(f"[{case}] TTAdapter -> post-GraphSyncSolver HIVM", flush=True)
    subprocess.run(
        [
            sys.executable,
            lower_script,
            args.compiler,
            ttadapter,
            hivm,
            "--",
            f"--target={args.arch}",
            "--enable-triton-kernel-compile=true",
            "--enable-hfusion-compile=true",
            "--enable-hivm-compile=true",
            "--enable-lir-compile=false",
            "--enable-hivm-graph-sync-solver=true",
        ],
        check=True,
    )

    lowered = hivm.read_text(encoding="utf-8")
    for op in metadata["ops"]:
        if op not in lowered:
            raise RuntimeError(f"{case}: lowered IR does not contain {op}")

    output_files = {}
    for schedule in ("lazy", "inorder"):
        prefix = work_dir / f"{case}.{schedule}."
        print(f"[{case}] interpreter sched={schedule}", flush=True)
        subprocess.run(
            [
                args.interpreter,
                hivm,
                f"--sched={schedule}",
                f"--block-dim={metadata['block_dim']}",
                f"--dyn-gm-elems={metadata['gm_elements']}",
                f"--args={metadata['args']}",
                f"--out={prefix}",
            ],
            check=True,
        )
        output_files[schedule] = work_dir / (
            f"{case}.{schedule}.arg{metadata['output_arg']}.npy"
        )

    if output_files["lazy"].read_bytes() != output_files["inorder"].read_bytes():
        raise RuntimeError(f"{case}: lazy and inorder outputs differ")
    check_output(metadata, output_files["lazy"])

    print(f"[{case}] interpreter sched=fuzz", flush=True)
    subprocess.run(
        [
            args.interpreter,
            hivm,
            "--sched=fuzz",
            "--seed=17",
            f"--block-dim={metadata['block_dim']}",
            f"--dyn-gm-elems={metadata['gm_elements']}",
            f"--args={metadata['args']}",
        ],
        check=True,
    )
    print(f"[{case}] PASS", flush=True)


def main() -> int:
    cases = case_names()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triton-python",
        default=os.environ.get("TRITON_PYTHON"),
        help="Python from a Triton Ascend environment (or set TRITON_PYTHON)",
    )
    parser.add_argument(
        "--compiler", default=str(ROOT / "build" / "bin" / "bishengir-compile")
    )
    parser.add_argument(
        "--interpreter", default=str(ROOT / "build" / "bin" / "npuir-interp")
    )
    parser.add_argument("--arch", default="Ascend910B4")
    parser.add_argument("--case", action="append", choices=cases)
    args = parser.parse_args()
    if not args.triton_python:
        parser.error("--triton-python or TRITON_PYTHON is required")

    selected = args.case or cases
    with tempfile.TemporaryDirectory(prefix="npuir-dsl-e2e-") as temp_dir:
        for case in selected:
            run_case(args, case, pathlib.Path(temp_dir))
    print(f"all {len(selected)} DSL end-to-end case(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
