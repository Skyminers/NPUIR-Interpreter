#!/usr/bin/env python3

import json
import sys
from pathlib import Path


events = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line
]
states = [event for event in events if event["event"] == "state"]
by_op = {
    state["executed"]["name"]: state
    for state in states
    if state.get("executed")
}

assert events[0]["target_model"] == "ascend-single-core/v2"
assert [core["id"] for core in states[0]["cores"]] == ["AIC#0", "AIV#0.0"]

blocked = next(state for state in states if state["result"] == "block")
aic, aiv = blocked["cores"]
assert aic["status"] == "runnable"
assert aiv["status"] == "blocked_on_flag"
assert "flag=1" in aiv["blocked"]["what"]
assert aiv["blocked"]["operation"]["name"] == "hivm.hir.sync_block_wait"
assert aic["current"]["ir_line"] != aiv["current"]["ir_line"]

store_42 = next(
    state
    for state in states
    if state.get("executed")
    and state["executed"]["name"] == "memref.store"
    and state["executed"]["inputs"][0]["value"] == "42"
)
aic_pipe_s = next(
    pipe for pipe in store_42["cores"][0]["pipes"] if pipe["pipe"] == "PIPE_S"
)
assert [task["operation"]["name"] for task in aic_pipe_s["tasks"]] == [
    "memref.store"
]
assert any(
    patch["arena"] == 0 and patch["offset"] == 0 and patch["bytes"] == "2a"
    for patch in store_42["memory_patches"]
)

publish = by_op["hivm.hir.sync_block_set"]
assert publish["executed"]["sync"] == {
    "kind": "sync_block_set",
    "set_pipe": "PIPE_S",
    "wait_pipe": "PIPE_MTE2",
    "event_id": 1,
    "dynamic_event": False,
}
assert publish["cores"][1]["status"] == "runnable"
assert publish["cross_flags"][0]["aic_generation"] == 1

addi = by_op["arith.addi"]
assert [value["value"] for value in addi["executed"]["inputs"]] == ["42", "1"]
assert [value["value"] for value in addi["executed"]["outputs"]] == ["43"]

store_43 = next(
    state
    for state in states
    if state.get("executed")
    and state["executed"]["name"] == "memref.store"
    and state["executed"]["inputs"][0]["value"] == "43"
)
assert any(
    patch["arena"] == 0 and patch["offset"] == 64 and patch["bytes"] == "2b"
    for patch in store_43["memory_patches"]
)
assert all(core["status"] == "done" for core in states[-1]["cores"])
assert events[-1]["event"] == "finish" and events[-1]["result"] == "success"
