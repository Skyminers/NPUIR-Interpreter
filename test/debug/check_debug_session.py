#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_common(events: list[dict]) -> tuple[list[dict], dict]:
    assert events[0]["event"] == "meta"
    assert events[0]["schema"] == "npuir-interp-debug/v1"
    assert events[0]["target_model"] == "ascend-single-core/v2"
    assert "module" in events[0]["ir"]
    states = [event for event in events if event["event"] == "state"]
    assert states and states[0]["reason"] == "initial"
    assert states[0]["cores"]
    assert all(
        {"program_id", "kind", "sub_core_id"} <= core.keys()
        for state in states
        for core in state["cores"]
    )
    assert all(
        {pipe["pipe"] for pipe in core["pipes"]}
        == (
            {"PIPE_S", "PIPE_V", "PIPE_MTE2", "PIPE_MTE3"}
            if core["kind"] == "AIV"
            else {
                "PIPE_S", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2",
                "PIPE_MTE3", "PIPE_FIX",
            }
        )
        for state in states
        for core in state["cores"]
    )
    assert all(
        len(core["clock"].removeprefix("<").removesuffix(">").split(","))
        == len(state["cores"])
        for state in states
        for core in state["cores"]
    ), "vector clocks must only contain the recorded single-program slice"
    assert any(
        allocation.get("element_type")
        for arena in states[0]["arenas"]
        for allocation in arena["allocations"]
    )
    ub_lanes = [
        arena.get("sub_core_id")
        for arena in states[0]["arenas"]
        if arena["space"] == "ub"
    ]
    assert ub_lanes == list(range(events[0]["sub_block_num"]))
    finish = next(event for event in reversed(events) if event["event"] == "finish")
    return states, finish


def check_pipeline(states: list[dict], finish: dict) -> None:
    assert finish["result"] == "success"
    multiply_state = next(
        state for state in states
        if state["executed"] and state["executed"]["name"] == "arith.muli"
    )
    multiply = multiply_state["executed"]
    assert "arith.muli" in multiply["text"]
    assert multiply["ir_line"] > 0
    assert [value["value"] for value in multiply["inputs"]] == ["2", "3"]
    assert [value["value"] for value in multiply["outputs"]] == ["6"]
    assert all(value["ssa"].startswith("%") for value in multiply["inputs"])
    assert multiply["outputs"][0]["ssa"].startswith("%")
    load_state = next(
        state for state in states
        if state["executed"] and state["executed"]["name"] == "hivm.hir.load"
    )
    mte2 = next(pipe for pipe in load_state["cores"][0]["pipes"] if pipe["pipe"] == "PIPE_MTE2")
    assert any(task["operation"]["name"] == "hivm.hir.load" for task in mte2["tasks"])
    wait_state = next(
        state for state in states
        if state["executed"] and state["executed"]["name"] == "hivm.hir.wait_flag"
    )
    assert wait_state["executed"]["sync"] == {
        "kind": "wait_flag",
        "set_pipe": "PIPE_MTE2",
        "wait_pipe": "PIPE_MTE3",
        "event_id": 0,
        "dynamic_event": False,
    }
    assert wait_state["memory_patches"], "wait_flag must expose the committed load bytes"
    assert all(core["status"] == "done" for core in states[-1]["cores"])


def check_blocked(states: list[dict], finish: dict) -> None:
    assert finish["result"] == "failure"
    blocked = [
        core
        for state in states
        for core in state["cores"]
        if core["status"] == "blocked_on_flag"
    ]
    assert blocked
    assert any("flag=7" in core["blocked"]["what"] for core in blocked)
    assert any(core["blocked"]["operation"]["name"] == "hivm.hir.sync_block_wait" for core in blocked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=("pipeline", "blocked"), required=True)
    parser.add_argument("--core-filter")
    args = parser.parse_args()
    events = load_events(args.path)
    states, finish = check_common(events)
    if args.core_filter:
        assert events[0]["core_filter"] == args.core_filter
        assert events[0]["program_filter"] == 0
        assert {
            core["program_id"] for state in states for core in state["cores"]
        } == {0}
        assert {
            state["selected_core"]
            for state in states
            if state["selected_core"] is not None
        } == {args.core_filter}
        assert states[-1]["reason"] == "final"
    (check_pipeline if args.mode == "pipeline" else check_blocked)(states, finish)


if __name__ == "__main__":
    main()
