#!/usr/bin/env python3
"""Lower TTAdapter IR to the post-GraphSyncSolver module used by tests."""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lower TTAdapter IR to the post-GraphSyncSolver HIVM module."
    )
    parser.add_argument("compiler")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("compile_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    compile_args = args.compile_args
    if compile_args[:1] == ["--"]:
        compile_args = compile_args[1:]

    with tempfile.TemporaryDirectory(prefix="npuir-ttadapter-") as temp_dir:
        tree_dir = pathlib.Path(temp_dir) / "ir"
        tree_dir.mkdir()
        command = [
            args.compiler,
            args.input,
            *compile_args,
            "--mlir-disable-threading",
            "--mlir-print-ir-after=hivm-graph-sync-solver",
            "--mlir-print-ir-module-scope",
            f"--mlir-print-ir-tree-dir={tree_dir}",
            "-o",
            str(pathlib.Path(temp_dir) / "device.bin"),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        dumps = sorted(tree_dir.rglob("*hivm-graph-sync-solver.mlir"))
        # The downstream hivmc step may be unavailable on interpreter-only
        # hosts. A unique pass dump proves that the stage we need completed.
        if len(dumps) != 1:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            sys.stderr.write(
                f"expected one post-GraphSyncSolver dump, found {len(dumps)}\n"
            )
            return result.returncode or 1
        shutil.copyfile(dumps[0], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
