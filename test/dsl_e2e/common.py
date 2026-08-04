"""Shared case metadata and output validation for Triton DSL E2E tests."""

import argparse
import ast
from dataclasses import dataclass
import json
import math
import pathlib
import struct
from typing import Any, Callable


@dataclass(frozen=True)
class E2ECase:
    name: str
    kernel: Any
    signature: dict[str, str]
    constants: dict[str, int]
    args: str
    block_dim: int
    output_arg: int
    ops: tuple[str, ...]
    elements: int
    expected: Callable[[int], float]
    atol: float = 0.0
    gm_elements: int = 2048

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "block_dim": self.block_dim,
            "output_arg": self.output_arg,
            "ops": self.ops,
            "gm_elements": self.gm_elements,
            "atol": self.atol,
            "expected": [
                self.expected(i) if i < self.elements else 0.0
                for i in range(self.gm_elements)
            ],
        }

    def check_output(self, path: pathlib.Path) -> None:
        check_output(self.metadata(), path)


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


def check_output(metadata: dict[str, Any], path: pathlib.Path) -> None:
    values = read_f32_npy(path)
    expected = metadata["expected"]
    if len(values) != len(expected):
        raise RuntimeError(
            f"{metadata['name']}: output has {len(values)} elements, "
            f"expected {len(expected)}"
        )
    atol = metadata["atol"]
    for index, (actual, wanted) in enumerate(zip(values, expected)):
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=atol):
            raise RuntimeError(
                f"{metadata['name']}: output[{index}] is {actual}, expected "
                f"{wanted} (atol={atol})"
            )


def case_main(case: E2ECase) -> int:
    parser = argparse.ArgumentParser(description=f"E2E case: {case.name}")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata", type=pathlib.Path)
    group.add_argument("--check", type=pathlib.Path)
    args = parser.parse_args()
    if args.metadata:
        args.metadata.write_text(json.dumps(case.metadata()), encoding="utf-8")
    else:
        case.check_output(args.check)
    return 0
