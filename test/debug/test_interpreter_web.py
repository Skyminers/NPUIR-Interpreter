#!/usr/bin/env python3

import importlib.util
import base64
import pathlib
import struct
import sys
import tempfile


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("interpreter_web", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_f32_npy(path: pathlib.Path, values: list[float]) -> None:
    header = repr({
        "descr": "<f4",
        "fortran_order": False,
        "shape": (len(values),),
    })
    padding = 16 - ((10 + len(header) + 1) % 16)
    encoded_header = (header + " " * padding + "\n").encode("latin1")
    payload = b"".join(struct.pack("<f", value) for value in values)
    path.write_bytes(
        b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded_header))
        + encoded_header + payload
    )


def main() -> int:
    web = load_module(pathlib.Path(sys.argv[1]))
    request = web.validate_request({
        "source": "module {}",
        "args": "zeros,zeros",
        "block_dim": 2,
        "sub_block_num": 2,
        "dyn_gm_elems": 16,
        "sched": "fuzz",
        "seed": 17,
        "debug_core": "AIV#0.0",
        "expected": {"arg": 1, "values": [43, 0], "atol": 0, "rtol": 0},
    })
    assert request["block_dim"] == 2
    assert request["expected"]["values"] == [43.0, 0.0]
    assert request["arch"] == "Ascend910B4"
    assert web.detect_ir_stage("module attributes {hivm.module_core_type} hacc.entry hivm.func_core_type") == "hivm"
    assert web.detect_ir_stage("module { %0 = bufferization.to_tensor %arg0 : memref<4xf32> }") == "ttadapter"

    for invalid in (
        {"source": ""},
        {"source": "module {}", "sched": "unknown"},
        {"source": "module {}", "sub_block_num": 3},
        {"source": "module {}", "debug_core": "AIV0"},
    ):
        try:
            web.validate_request(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid request accepted: {invalid}")

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "output.arg1.npy"
        write_f32_npy(output, [43.0, 0.0, 0.0, 0.0])
        summary = web.read_npy(output)
        assert summary["dtype"] == "<f4"
        assert summary["shape"] == [4]
        assert summary["preview"] == [43.0, 0.0, 0.0, 0.0]
        assert web.compare_expected(output, {
            "values": [43.0, 0.0], "atol": 0.0, "rtol": 0.0,
        })["status"] == "passed"
        mismatch = web.compare_expected(output, {
            "values": [42.0], "atol": 0.0, "rtol": 0.0,
        })
        assert mismatch["status"] == "failed" and mismatch["index"] == 0
        uploaded = web.validate_request({
            "source": "module {}",
            "args": "@input0,zeros",
            "input_files": [{
                "name": "input.npy",
                "data": base64.b64encode(output.read_bytes()).decode("ascii"),
            }],
        })
        assert uploaded["input_files"][0]["token"] == "@input0"
        assert uploaded["input_files"][0]["data"].startswith(b"\x93NUMPY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
