#!/usr/bin/env python3
"""Local web entry point for running and replaying NPUIR kernels."""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import pathlib
import re
import struct
import subprocess
import sys
import time
import uuid
import webbrowser


MAX_REQUEST_BYTES = 96 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_EXPECTED_VALUES = 1_000_000
RUN_ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")
DEBUG_CORE_RE = re.compile(r"^(?:all|AIC#\d+|AIV#\d+\.\d+)$", re.IGNORECASE)
ARCH_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是整数") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return result


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是数字") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} 必须是非负有限数字")
    return result


def validate_request(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("请提供非空的 MLIR 源码")
    if len(source.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("MLIR 源码不能超过 4 MiB")

    sched = str(payload.get("sched", "lazy"))
    if sched not in {"lazy", "inorder", "fuzz"}:
        raise ValueError("sched 必须是 lazy、inorder 或 fuzz")
    debug_core = str(payload.get("debug_core", "AIV#0.0")).strip()
    if not DEBUG_CORE_RE.fullmatch(debug_core):
        raise ValueError("debug_core 格式应为 AIC#0、AIV#0.0 或 all")

    args = payload.get("args", "")
    if not isinstance(args, str) or "\n" in args or "\r" in args:
        raise ValueError("args 必须是单行字符串")

    arch = str(payload.get("arch", "Ascend910B4")).strip()
    if not ARCH_RE.fullmatch(arch):
        raise ValueError("arch 只能包含字母、数字、下划线和连字符")

    input_files = payload.get("input_files", [])
    if not isinstance(input_files, list):
        raise ValueError("input_files 必须是数组")
    normalized_inputs = []
    total_input_bytes = 0
    for index, item in enumerate(input_files):
        if not isinstance(item, dict):
            raise ValueError(f"input_files[{index}] 必须是对象")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", pathlib.Path(str(item.get("name", "input.npy"))).name)
        if not name.lower().endswith(".npy"):
            raise ValueError(f"input_files[{index}] 必须是 .npy 文件")
        try:
            data = base64.b64decode(str(item.get("data", "")), validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError(f"input_files[{index}] 不是合法的 base64") from error
        if not data.startswith(b"\x93NUMPY"):
            raise ValueError(f"input_files[{index}] 不是 NPY 文件")
        total_input_bytes += len(data)
        if total_input_bytes > MAX_INPUT_BYTES:
            raise ValueError("参数 NPY 文件总大小不能超过 64 MiB")
        normalized_inputs.append({
            "token": f"@input{index}", "name": name, "data": data,
        })

    expected = payload.get("expected")
    normalized_expected = None
    if expected is not None:
        if not isinstance(expected, dict) or not isinstance(expected.get("values"), list):
            raise ValueError("expected.values 必须是数字数组")
        if not expected["values"]:
            raise ValueError("expected.values 不能为空")
        if len(expected["values"]) > MAX_EXPECTED_VALUES:
            raise ValueError("期望值数量过多")
        values = []
        for index, value in enumerate(expected["values"]):
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"expected.values[{index}] 不是数字") from error
            if not math.isfinite(number):
                raise ValueError(f"expected.values[{index}] 必须是有限数字")
            values.append(number)
        normalized_expected = {
            "arg": _bounded_int(expected.get("arg", 0), "expected.arg", 0, 255),
            "values": values,
            "atol": _finite_float(expected.get("atol", 0), "expected.atol"),
            "rtol": _finite_float(expected.get("rtol", 0), "expected.rtol"),
        }

    return {
        "source": source,
        "args": args.strip(),
        "arch": arch,
        "input_files": normalized_inputs,
        "block_dim": _bounded_int(payload.get("block_dim", 1), "block_dim", 1, 65535),
        "sub_block_num": _bounded_int(payload.get("sub_block_num", 1), "sub_block_num", 1, 2),
        "dyn_gm_elems": _bounded_int(payload.get("dyn_gm_elems", 4096), "dyn_gm_elems", 1, 100_000_000),
        "sched": sched,
        "seed": _bounded_int(payload.get("seed", 0), "seed", 0, 2**63 - 1),
        "debug_core": debug_core,
        "expected": normalized_expected,
    }


def detect_ir_stage(source: str) -> str:
    if any(marker in source for marker in (
        "global_kernel", "tensor<", "tensor.", "bufferization.", "linalg.",
        "gpu.", "hfusion.",
    )):
        return "ttadapter"
    required = ("hacc.entry", "hivm.func_core_type", "hivm.module_core_type")
    if all(marker in source for marker in required):
        return "hivm"
    return "unlowered"


def read_npy(path: pathlib.Path, value_limit: int | None = 64) -> dict[str, object]:
    with path.open("rb") as file:
        if file.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path.name} 不是 NPY 文件")
        major, _minor = file.read(2)
        if major not in (1, 2, 3):
            raise ValueError(f"不支持 NPY v{major}")
        size_format = "<H" if major == 1 else "<I"
        size = struct.unpack(size_format, file.read(struct.calcsize(size_format)))[0]
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(file.read(size).decode(encoding))
        payload_offset = file.tell()
        file.seek(0, 2)
        payload_size = file.tell() - payload_offset

    shape = list(header.get("shape", ()))
    count = math.prod(shape) if shape else 1
    descr = str(header.get("descr", ""))
    match = re.fullmatch(r"([<>=|])([biuf])(\d+)", descr)
    if not match:
        return {
            "dtype": descr,
            "shape": shape,
            "count": count,
            "preview": [],
            "preview_error": "当前 Web UI 尚不能解码该 dtype",
        }
    byte_order, kind, width_text = match.groups()
    width = int(width_text)
    formats = {
        ("b", 1): "?",
        ("i", 1): "b", ("i", 2): "h", ("i", 4): "i", ("i", 8): "q",
        ("u", 1): "B", ("u", 2): "H", ("u", 4): "I", ("u", 8): "Q",
        ("f", 2): "e", ("f", 4): "f", ("f", 8): "d",
    }
    format_char = formats.get((kind, width))
    if not format_char or payload_size != count * width:
        return {
            "dtype": descr,
            "shape": shape,
            "count": count,
            "preview": [],
            "preview_error": "NPY payload 或 dtype 不受支持",
        }
    endian = ">" if byte_order == ">" else "<"
    limit = count if value_limit is None else min(count, value_limit)
    with path.open("rb") as file:
        file.seek(payload_offset)
        payload = file.read(limit * width)
    values = [item[0] for item in struct.iter_unpack(endian + format_char, payload[:limit * width])]
    preview = [value if not isinstance(value, float) or math.isfinite(value) else str(value) for value in values]
    return {
        "dtype": descr,
        "shape": shape,
        "count": count,
        "fortran_order": bool(header.get("fortran_order", False)),
        "preview": preview,
        "values": values,
    }


def compare_expected(output: pathlib.Path, expected: dict[str, object]) -> dict[str, object]:
    wanted = expected["values"]
    parsed = read_npy(output, value_limit=len(wanted))
    actual = parsed.get("values")
    if actual is None:
        return {"status": "failed", "message": parsed.get("preview_error", "无法解码输出")}
    if len(wanted) > len(actual):
        return {
            "status": "failed",
            "message": f"期望前缀有 {len(wanted)} 项，但输出只有 {len(actual)} 项",
        }
    atol = float(expected["atol"])
    rtol = float(expected["rtol"])
    for index, (got, want) in enumerate(zip(actual, wanted)):
        if not math.isclose(float(got), float(want), rel_tol=rtol, abs_tol=atol):
            return {
                "status": "failed",
                "message": f"output[{index}] = {got}，期望 {want}",
                "index": index,
                "actual": got,
                "expected": want,
            }
    return {
        "status": "passed",
        "message": f"前 {len(wanted)} 个值符合期望（atol={atol:g}, rtol={rtol:g}）",
    }


def output_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


class InterpreterWebHandler(SimpleHTTPRequestHandler):
    server_version = "NPUIRInterpreterWeb/1.0"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status: HTTPStatus, body: dict[str, object]) -> None:
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            binary = self.server.interpreter
            self._json(HTTPStatus.OK, {
                "available": binary.is_file(),
                "interpreter": str(binary),
                "compiler_available": self.server.compiler.is_file(),
                "compiler": str(self.server.compiler),
                "artifacts": str(self.server.artifacts),
            })
            return
        if self.path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/tools/debug-ui/index.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求必须使用 application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求体大小无效")
            payload = json.loads(self.rfile.read(length))
            request = validate_request(payload)
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        try:
            response = self._run(request)
        except (OSError, subprocess.SubprocessError) as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            return
        self._json(HTTPStatus.OK, response)

    def _run(self, request: dict[str, object]) -> dict[str, object]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        if not RUN_ID_RE.fullmatch(run_id):
            raise RuntimeError("内部 run id 无效")
        run_dir = self.server.artifacts / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        source = run_dir / "kernel.input.mlir"
        session = run_dir / "debug.jsonl"
        output_prefix = run_dir / "output."
        source.write_text(str(request["source"]), encoding="utf-8")

        input_dir = run_dir / "inputs"
        input_dir.mkdir()
        token_paths = {}
        input_summaries = []
        for index, item in enumerate(request["input_files"]):
            path = input_dir / f"{index}-{item['name']}"
            path.write_bytes(item["data"])
            token_paths[item["token"]] = str(path)
            input_summaries.append({
                "token": item["token"], "name": item["name"], "bytes": len(item["data"]),
            })
        arg_specs = str(request["args"]).split(",") if request["args"] else []
        resolved_args = ",".join(token_paths.get(spec.strip(), spec.strip()) for spec in arg_specs)

        input_stage = detect_ir_stage(str(request["source"]))
        execution_source = source
        lowering = {
            "performed": False,
            "input_stage": input_stage,
            "status": "not_needed",
            "message": "输入已经是可执行的 post-GraphSyncSolver HIVM",
            "duration_ms": 0,
            "source_url": "/" + source.relative_to(self.server.repo_root).as_posix(),
            "command": [], "stdout": "", "stderr": "",
        }
        if input_stage != "hivm":
            lowered = run_dir / "kernel.lowered.mlir"
            lower_command = [
                sys.executable,
                str(self.server.lowerer),
                str(self.server.compiler),
                str(source),
                str(lowered),
                "--",
                f"--target={request['arch']}",
                "--enable-triton-kernel-compile=true",
                "--enable-hfusion-compile=true",
                "--enable-hivm-compile=true",
                "--enable-lir-compile=false",
                "--enable-hivm-graph-sync-solver=true",
            ]
            lower_started = time.monotonic()
            if not self.server.compiler.is_file():
                lower_completed = None
                lower_error = f"找不到编译器：{self.server.compiler}"
            else:
                try:
                    lower_completed = subprocess.run(
                        lower_command,
                        cwd=self.server.repo_root,
                        capture_output=True,
                        text=True,
                        timeout=self.server.timeout,
                        check=False,
                    )
                    lower_error = ""
                except subprocess.TimeoutExpired as error:
                    lower_completed = error
                    lower_error = f"自动下降超过 {self.server.timeout:g} 秒"
            lower_duration = round((time.monotonic() - lower_started) * 1000)
            lower_ok = (
                lower_completed is not None
                and not isinstance(lower_completed, subprocess.TimeoutExpired)
                and lower_completed.returncode == 0
                and lowered.is_file()
            )
            if not lower_ok and not lower_error:
                lower_error = f"自动下降失败，退出码 {lower_completed.returncode}"
            lowering = {
                "performed": True,
                "input_stage": input_stage,
                "status": "passed" if lower_ok else "failed",
                "message": (
                    "检测到 IR 尚未完全下降，已自动生成 post-GraphSyncSolver HIVM"
                    if lower_ok else lower_error
                ),
                "duration_ms": lower_duration,
                "source_url": (
                    "/" + lowered.relative_to(self.server.repo_root).as_posix()
                    if lowered.is_file() else None
                ),
                "command": lower_command,
                "stdout": output_text(getattr(lower_completed, "stdout", ""))[-200_000:],
                "stderr": output_text(getattr(lower_completed, "stderr", ""))[-200_000:],
            }
            if not lower_ok:
                return {
                    "run_id": run_id,
                    "status": "failed",
                    "process": {
                        "status": "not_started", "return_code": None,
                        "duration_ms": 0, "stdout": "", "stderr": "", "command": [],
                    },
                    "lowering": lowering,
                    "verification": {"status": "not_requested", "message": "自动下降失败，未执行数值验证"},
                    "outputs": [], "inputs": input_summaries, "session_url": None,
                }
            execution_source = lowered

        command = [
            str(self.server.interpreter), str(execution_source),
            f"--block-dim={request['block_dim']}",
            f"--sub-block-num={request['sub_block_num']}",
            f"--dyn-gm-elems={request['dyn_gm_elems']}",
            f"--sched={request['sched']}",
            f"--seed={request['seed']}",
            f"--debug-output={session}",
            f"--out={output_prefix}",
        ]
        if resolved_args:
            command.append(f"--args={resolved_args}")
        if str(request["debug_core"]).lower() != "all":
            command.append(f"--debug-core={request['debug_core']}")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=self.server.repo_root,
                capture_output=True,
                text=True,
                timeout=self.server.timeout,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as error:
            completed = error
            timed_out = True
        elapsed_ms = round((time.monotonic() - started) * 1000)

        outputs = []
        for path in sorted(run_dir.glob("output.arg*.npy")):
            match = re.search(r"arg(\d+)\.npy$", path.name)
            summary = read_npy(path)
            summary.update({
                "arg": int(match.group(1)) if match else -1,
                "name": path.name,
                "url": "/" + path.relative_to(self.server.repo_root).as_posix(),
            })
            summary.pop("values", None)
            outputs.append(summary)

        verification = {"status": "not_requested", "message": "未提供数学期望值"}
        expected = request["expected"]
        if expected is not None:
            output = run_dir / f"output.arg{expected['arg']}.npy"
            verification = (
                compare_expected(output, expected)
                if output.is_file()
                else {"status": "failed", "message": f"没有生成 arg{expected['arg']} 输出"}
            )

        return_code = None if timed_out else completed.returncode
        process_passed = return_code == 0
        verified = verification["status"] != "failed"
        return {
            "run_id": run_id,
            "status": "passed" if process_passed and verified else "failed",
            "process": {
                "status": "timeout" if timed_out else ("passed" if process_passed else "failed"),
                "return_code": return_code,
                "duration_ms": elapsed_ms,
                "stdout": output_text(completed.stdout)[-200_000:],
                "stderr": output_text(completed.stderr)[-200_000:],
                "command": command,
            },
            "lowering": lowering,
            "verification": verification,
            "outputs": outputs,
            "inputs": input_summaries,
            "session_url": (
                "/" + session.relative_to(self.server.repo_root).as_posix()
                if session.is_file() else None
            ),
        }


class InterpreterWebServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        repo_root: pathlib.Path,
        interpreter: pathlib.Path,
        compiler: pathlib.Path,
        artifacts: pathlib.Path,
        timeout: float,
    ) -> None:
        handler = lambda *args, **kwargs: InterpreterWebHandler(  # noqa: E731
            *args, directory=str(repo_root), **kwargs
        )
        super().__init__(address, handler)
        self.repo_root = repo_root
        self.interpreter = interpreter
        self.compiler = compiler
        self.lowerer = repo_root / "test/lower_ttadapter_for_interp.py"
        self.artifacts = artifacts
        self.timeout = timeout


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the NPUIR Interpreter Web UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--interpreter", type=pathlib.Path,
        default=repo_root / "build/bin/npuir-interp-debug",
    )
    parser.add_argument(
        "--compiler", type=pathlib.Path,
        default=repo_root / "build/bin/bishengir-compile",
    )
    parser.add_argument("--artifacts", type=pathlib.Path, default=repo_root / "build/web-runs")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    interpreter = args.interpreter.resolve()
    compiler = args.compiler.resolve()
    artifacts = args.artifacts.resolve()
    try:
        artifacts.relative_to(repo_root)
    except ValueError:
        parser.error("--artifacts 必须位于仓库目录内，才能由页面读取")
    if not interpreter.is_file():
        parser.error(f"interpreter 不存在：{interpreter}")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    artifacts.mkdir(parents=True, exist_ok=True)

    server = InterpreterWebServer(
        ("127.0.0.1", args.port), repo_root, interpreter, compiler, artifacts, args.timeout
    )
    url = f"http://127.0.0.1:{server.server_port}/tools/debug-ui/index.html"
    print(f"Interpreter Web UI: {url}", flush=True)
    print("Press Ctrl-C to stop the local server.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterpreter Web server stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
