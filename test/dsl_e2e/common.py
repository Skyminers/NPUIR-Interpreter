"""Shared case metadata and output validation for Triton DSL E2E tests."""

import argparse
import ast
from dataclasses import dataclass
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
from typing import Any, Callable


DEBUG_TARGET_MODEL = "ascend-single-core/v2"


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
    ttadapter_counts: tuple[tuple[str, int], ...] = ()
    arch: str = "Ascend910B4"
    sub_block_num: int = 1
    compile_args: tuple[str, ...] = ()

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


def run_debug_case(case: E2ECase, args: argparse.Namespace) -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[2]
    test_root = root / "test"
    artifacts = (
        args.artifacts or root / "build" / "debug" / case.name
    ).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    ttadapter = artifacts / f"{case.name}.ttadapter.mlir"
    hivm = artifacts / f"{case.name}.hivm.mlir"
    metadata_path = artifacts / f"{case.name}.json"
    session = artifacts / f"{case.name}.debug.jsonl"
    lazy_prefix = artifacts / f"{case.name}.lazy."
    inorder_prefix = artifacts / f"{case.name}.inorder."

    print(f"[{case.name}] DSL -> TTAdapter", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(test_root / "dump_ttadapter.py"),
            "--case",
            case.name,
            "--arch",
            args.arch,
            "--metadata",
            str(metadata_path),
            str(ttadapter),
        ],
        check=True,
    )
    ttadapter_text = ttadapter.read_text(encoding="utf-8")
    for needle, expected_count in case.ttadapter_counts:
        actual_count = ttadapter_text.count(needle)
        if actual_count != expected_count:
            raise RuntimeError(
                f"{case.name}: TTAdapter contains {actual_count} occurrences "
                f"of {needle!r}, expected {expected_count}"
            )

    print(f"[{case.name}] TTAdapter -> post-GraphSyncSolver HIVM", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(test_root / "lower_ttadapter_for_interp.py"),
            str(args.compiler),
            str(ttadapter),
            str(hivm),
            "--",
            f"--target={args.arch}",
            "--enable-triton-kernel-compile=true",
            "--enable-hfusion-compile=true",
            "--enable-hivm-compile=true",
            "--enable-lir-compile=false",
            "--enable-hivm-graph-sync-solver=true",
            *case.compile_args,
        ],
        check=True,
    )
    lowered = hivm.read_text(encoding="utf-8")
    for op in case.ops:
        if op not in lowered:
            raise RuntimeError(f"{case.name}: lowered IR does not contain {op}")

    common_args = [
        str(hivm),
        f"--block-dim={case.block_dim}",
        f"--sub-block-num={case.sub_block_num}",
        f"--dyn-gm-elems={case.gm_elements}",
        f"--args={case.args}",
    ]
    debug_args = [
        str(args.debugger),
        *common_args,
        "--sched=lazy",
        f"--debug-output={session}",
        f"--out={lazy_prefix}",
    ]
    if args.debug_core.lower() != "all":
        debug_args.append(f"--debug-core={args.debug_core}")
    print(
        f"[{case.name}] debug sched=lazy core={args.debug_core}", flush=True
    )
    subprocess.run(debug_args, check=True)
    try:
        with session.open(encoding="utf-8") as file:
            session_meta = json.loads(file.readline())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{case.name}: invalid debug session: {session}") from error
    if session_meta.get("target_model") != DEBUG_TARGET_MODEL:
        raise RuntimeError(
            f"{case.name}: {args.debugger} is an outdated debugger binary; "
            "rebuild npuir-interp-debug before using --serve"
        )

    print(f"[{case.name}] reference sched=inorder", flush=True)
    subprocess.run(
        [
            str(args.interpreter),
            *common_args,
            "--sched=inorder",
            f"--out={inorder_prefix}",
        ],
        check=True,
    )
    lazy_output = artifacts / f"{case.name}.lazy.arg{case.output_arg}.npy"
    inorder_output = artifacts / f"{case.name}.inorder.arg{case.output_arg}.npy"
    if lazy_output.read_bytes() != inorder_output.read_bytes():
        raise RuntimeError(f"{case.name}: lazy and inorder outputs differ")
    case.check_output(lazy_output)

    print(f"[{case.name}] sched=fuzz seed=17", flush=True)
    subprocess.run(
        [
            str(args.interpreter),
            *common_args,
            "--sched=fuzz",
            "--seed=17",
        ],
        check=True,
    )
    print(f"[{case.name}] PASS: {session}", flush=True)
    return session


def serve_debug_session(session: pathlib.Path, args: argparse.Namespace) -> None:
    import importlib.util
    import urllib.parse
    import webbrowser

    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        relative_session = session.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            "--serve requires --artifacts to be inside the repository"
        ) from error
    query = urllib.parse.quote("/" + relative_session.as_posix())
    url = (
        f"http://127.0.0.1:{args.port}/tools/debug-ui/index.html"
        f"?session={query}"
    )
    web_module_path = root / "tools" / "interpreter_web.py"
    spec = importlib.util.spec_from_file_location("interpreter_web", web_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Interpreter Web server: {web_module_path}")
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    web_artifacts = root / "build" / "web-runs"
    web_artifacts.mkdir(parents=True, exist_ok=True)
    server = web.InterpreterWebServer(
        ("127.0.0.1", args.port), root, args.debugger.resolve(),
        args.compiler.resolve(),
        web_artifacts, 120.0,
    )
    print(f"Interpreter Web UI: {url}", flush=True)
    print("Press Ctrl-C to stop the local server.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDebugger server stopped.")
    finally:
        server.server_close()


def case_main(case: E2ECase) -> int:
    parser = argparse.ArgumentParser(description=f"E2E case: {case.name}")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata", type=pathlib.Path)
    group.add_argument("--check", type=pathlib.Path)
    group.add_argument(
        "--debug",
        action="store_true",
        help="run the complete lowering/validation pipeline and record a debug session",
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    parser.add_argument("--arch", default=case.arch)
    parser.add_argument("--artifacts", type=pathlib.Path)
    parser.add_argument(
        "--compiler",
        type=pathlib.Path,
        default=root / "build/bin/bishengir-compile",
    )
    parser.add_argument(
        "--interpreter",
        type=pathlib.Path,
        default=root / "build/bin/npuir-interp",
    )
    parser.add_argument(
        "--debugger",
        type=pathlib.Path,
        default=root / "build/bin/npuir-interp-debug",
    )
    parser.add_argument(
        "--debug-core",
        default=os.environ.get("NPUIR_DEBUG_CORE", "AIV#0.0"),
        help=(
            "lane selecting the recorded single-program AIC/AIV slice, "
            "or 'all' (default: AIV#0.0)"
        ),
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if args.metadata:
        args.metadata.write_text(json.dumps(case.metadata()), encoding="utf-8")
    elif args.check:
        case.check_output(args.check)
    else:
        session = run_debug_case(case, args)
        if args.serve:
            serve_debug_session(session, args)
        else:
            print("Open tools/debug-ui/index.html and load:")
            print(session)
    return 0
