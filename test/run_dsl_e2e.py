#!/usr/bin/env python3
"""Run the built-in Triton DSL cases through lowering and the interpreter."""

import argparse
import ast
import os
import pathlib
import struct
import subprocess
import sys
import tempfile


ELEMENTS = 1500
GM_ELEMENTS = 2048
CASES = {
    "add": {
        "args": "0,zeros,zeros,arange,1,zeros,1500,2,1,1",
        "block_dim": 2,
        "output_arg": 5,
        "ops": ("hivm.hir.vadd",),
        "expected": lambda i: float(i + 1),
    },
    "affine_abs": {
        "args": "0,zeros,zeros,arange,zeros,1500,2,1,1",
        "block_dim": 2,
        "output_arg": 4,
        "ops": ("hivm.hir.vmul", "hivm.hir.vabs"),
        "expected": lambda i: float(abs(i - 4) * 2),
    },
    "select": {
        "args": "0,zeros,zeros,arange,zeros,1500,47,1,1",
        "block_dim": 47,
        "output_arg": 4,
        "ops": ("hivm.hir.vcmp", "hivm.hir.vsel"),
        "expected": lambda i: float(max(i, 7)),
    },
    "cast": {
        "args": "0,zeros,zeros,arange,zeros,1500,2,1,1",
        "block_dim": 2,
        "output_arg": 4,
        "ops": ("hivm.hir.vcast",),
        "expected": lambda i: float(int(i * 0.5 + 0.75)),
    },
}


def read_f32_npy(path: pathlib.Path) -> list[float]:
    with path.open("rb") as file:
        if file.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"{path} is not an NPY file")
        major, _minor = file.read(2)
        size_bytes = file.read(2 if major == 1 else 4)
        header_size = struct.unpack("<H" if major == 1 else "<I", size_bytes)[0]
        header = ast.literal_eval(file.read(header_size).decode("latin1"))
        payload = file.read()

    if header["descr"] not in ("<f4", "=f4") or len(header["shape"]) != 1:
        raise RuntimeError(f"expected a one-dimensional f32 NPY file, got {header}")
    count = header["shape"][0]
    if len(payload) != count * 4:
        raise RuntimeError(f"invalid payload size in {path}")
    return [value for (value,) in struct.iter_unpack("<f", payload)]


def check_values(case: str, path: pathlib.Path) -> None:
    values = read_f32_npy(path)
    expected_at = CASES[case]["expected"]
    expected = [expected_at(i) if i < ELEMENTS else 0.0 for i in range(GM_ELEMENTS)]
    for index, (actual, wanted) in enumerate(zip(values, expected)):
        if actual != wanted:
            raise RuntimeError(
                f"{case}: output[{index}] is {actual}, expected {wanted}"
            )
    if len(values) != len(expected):
        raise RuntimeError(
            f"{case}: output has {len(values)} elements, expected {len(expected)}"
        )


def run_case(args, case: str, work_dir: pathlib.Path) -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    dump_script = root / "test" / "dump_ttadapter.py"
    lower_script = root / "test" / "lower_ttadapter_for_interp.py"
    ttadapter = work_dir / f"{case}.ttadapter.mlir"
    hivm = work_dir / f"{case}.hivm.mlir"

    print(f"[{case}] DSL -> TTAdapter", flush=True)
    subprocess.run(
        [
            args.triton_python,
            dump_script,
            "--case",
            case,
            "--arch",
            args.arch,
            ttadapter,
        ],
        check=True,
    )
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
    for op in CASES[case]["ops"]:
        if op not in lowered:
            raise RuntimeError(f"{case}: lowered IR does not contain {op}")

    output_arg = CASES[case]["output_arg"]
    output_files = {}
    for schedule in ("lazy", "inorder"):
        prefix = work_dir / f"{case}.{schedule}."
        print(f"[{case}] interpreter sched={schedule}", flush=True)
        subprocess.run(
            [
                args.interpreter,
                hivm,
                f"--sched={schedule}",
                f"--block-dim={CASES[case]['block_dim']}",
                f"--dyn-gm-elems={GM_ELEMENTS}",
                f"--args={CASES[case]['args']}",
                f"--out={prefix}",
            ],
            check=True,
        )
        output_files[schedule] = work_dir / f"{case}.{schedule}.arg{output_arg}.npy"

    if output_files["lazy"].read_bytes() != output_files["inorder"].read_bytes():
        raise RuntimeError(f"{case}: lazy and inorder outputs differ")
    check_values(case, output_files["lazy"])

    print(f"[{case}] interpreter sched=fuzz", flush=True)
    subprocess.run(
        [
            args.interpreter,
            hivm,
            "--sched=fuzz",
            "--seed=17",
            f"--block-dim={CASES[case]['block_dim']}",
            f"--dyn-gm-elems={GM_ELEMENTS}",
            f"--args={CASES[case]['args']}",
        ],
        check=True,
    )
    print(f"[{case}] PASS", flush=True)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triton-python",
        default=os.environ.get("TRITON_PYTHON"),
        help="Python from a Triton Ascend environment (or set TRITON_PYTHON)",
    )
    parser.add_argument(
        "--compiler", default=str(root / "build" / "bin" / "bishengir-compile")
    )
    parser.add_argument(
        "--interpreter", default=str(root / "build" / "bin" / "npuir-interp")
    )
    parser.add_argument("--arch", default="Ascend910B4")
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    if not args.triton_python:
        parser.error("--triton-python or TRITON_PYTHON is required")

    with tempfile.TemporaryDirectory(prefix="npuir-dsl-e2e-") as temp_dir:
        for case in args.case or CASES:
            run_case(args, case, pathlib.Path(temp_dir))
    print(f"all {len(args.case or CASES)} DSL end-to-end case(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
