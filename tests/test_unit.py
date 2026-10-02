#!/usr/bin/env python3
"""Unit tests for the findings gate, the schema rules, and the retrieval tiers.

Run: pytest tests/test_unit.py

The gate is the one component whose failure mode is SILENT: everything else exits non-zero
and says why, while a validator that wrongly passes reports a clean review of a change
nobody reviewed. It has been wrong that way three times. Each regression is pinned here by
name, and each test is written so that reverting the fix it guards makes it fail — the
suite's own mutation harness checks that claim, because guards that read as coverage while
guarding nothing are how this package got into trouble twice.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# APPEND, never insert(0): the flake's unit check points PYTHONPATH at the BUILT package so
# the modules that ship are the modules exercised, and putting the source root ahead of it
# silently tests the source tree instead.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from persona_review import (  # noqa: E402
    assets,
    cli,
    config,
    errors,
    findings,
    flags,
    providers,
    runner,
    validate,
    verdicts,
)

# Which copy did we import? Only this process knows, so the flake check asserts it here
# rather than trusting the environment it set up.
_expect = os.environ.get("PERSONA_REVIEW_EXPECT_LIB")
if _expect and not validate.__file__.startswith(_expect):
    raise SystemExit(
        f"imported {validate.__file__}, not the packaged library under {_expect} — "
        "this suite is not testing what ships"
    )

# Mirrors the plugin schema's SHAPE, including the rules the real one actually uses: a
# ["string","null"] union, `minimum` on line, `maxLength` on title, and typed array items.
# A weaker fixture let type-invalid findings pass the suite while the real schema rejected
# them.
SCHEMA: dict[str, Any] = {
    "required": ["reviewer", "findings", "residual_risks", "testing_gaps"],
    "properties": {
        "reviewer": {"type": "string"},
        "residual_risks": {"type": "array"},
        "testing_gaps": {"type": "array"},
        "findings": {
            "items": {
                "required": ["title", "severity", "file", "line", "evidence"],
                "properties": {
                    "title": {"type": "string", "maxLength": 100},
                    "file": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "pre_existing": {"type": "boolean"},
                    "suggested_fix": {"type": ["string", "null"]},
                    "severity": {"enum": ["P0", "P1", "P2", "P3"]},
                    "confidence": {"enum": [0, 25, 50, 75, 100]},
                },
            }
        },
    },
}


def finding(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "t",
        "severity": "P1",
        "file": "f.py",
        "line": 1,
        "evidence": ["f.py:1 -- x"],
    }
    base.update(over)
    return base


def artifact(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer": "adversarial-reviewer",
        "findings": list(items),
        "residual_risks": [],
        "testing_gaps": [],
    }


def schema_with(field: str, spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(SCHEMA))
    out["properties"]["findings"]["items"]["properties"][field] = spec
    return out


def findings_of(art: validate.Artifact) -> list[Any]:
    """Narrow the parsed artifact for assertions. `len()` needs a real list."""
    items = art["findings"]
    assert isinstance(items, list)
    return items


def _app_error_classes() -> list[type[errors.AppError]]:
    """Every AppError subclass defined in errors.py, found by walking the module.

    Enumerated by reflection rather than by a hand-written list: a list is exactly the second
    copy these tests exist to make impossible, and a new class added without a status would
    simply be absent from it.
    """
    found = [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and issubclass(value, errors.AppError)
        if value is not errors.AppError
    ]
    assert found, "reflection found no error classes, so every assertion below is vacuous"
    return found


def _help_text() -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        cli.main(providers.GROK, ["--help"])
    return out.getvalue()


EMPTY_EXAMPLE = json.dumps(artifact())

# A run that DID inspect something. Every provenance test that is not about the tool-call
# refusal has to carry one, because a run with zero tool calls is refused before the summary
# line — so a zero here would quietly turn those tests into assertions about the refusal.
STATS = validate.RunStats(
    tool_calls=7,
    local_tool_calls=7,
    turns=4,
    output_tokens=4096,
    duration_s=61.5,
    local_tool_attempts=7,
)


# What a grok `tool_result` carries in its `content`, one per shape seen in real grok-4.7
# runs: every key and value type as recorded, with local paths, commit ids and long payloads
# replaced by neutral ones, because the recorded ones name a home directory and another
# repository's code.
def _octets(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _without(report: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != key}


READ_FILE = {
    "type": "ReadFile",
    "FileContent": {
        "content": "1→[project]\n",
        "content_concise": "1→[project]\n",
        "absolute_path": "/work/repo/pyproject.toml",
        "offset": None,
        "limit": 40,
        "raw_output": "[project]\n",
        "total_lines": 78,
    },
}


def bash_report(exit_code: int | None, output: str) -> dict[str, Any]:
    return {
        "type": "Bash",
        "output": _octets(output),
        "output_for_prompt": f"exit: {exit_code}\n{output}",
        "exit_code": exit_code,
        "command": "git diff --stat 0a1b2c3..HEAD",
        "truncated": False,
        "signal": None,
        "timed_out": False,
        "description": "Measure diff size and commit range",
        "current_dir": "/work/repo",
        "output_file": "/work/.grok/sessions/s/terminal/call-1.log",
        "total_bytes": len(output),
    }


BASH_OK = bash_report(0, " f.py | 2 +-\n 1 file changed\n")
# A command that ran and exited 2, returned with `is_error` false.
BASH_FAILED = bash_report(
    2, "error: Failed to initialize cache at `/work/.cache/uv`\n  Caused by: failed to open file\n"
)
# A search that matched nothing exits 1, like `rg`.
GREP_NO_MATCH = {
    "type": "GrepSearch",
    "stdout": _octets(
        '<workspace_result workspace_path="/work/repo">\nNo matches found\n</workspace_result>'
    ),
    "stderr": [],
    "exit_code": 1,
    "match_count": 0,
    "file_matches": [],
}
# A command moved to the background before anything came back.
BACKGROUND_STARTED = {
    "type": "BackgroundTaskStarted",
    "task_id": "call-1",
    "task_type": "bash",
    "output_file": "/work/.grok/sessions/s/terminal/call-1.log",
    "status": "running",
    "command": "uv run --no-project python probe.py",
    "summary": "Command has been automatically moved to background because it exceeded "
    "auto-background timeout limit of 15s. Process is still running.",
    "retrieval_hint": 'Use get_command_or_subagent_output with task_ids=["call-1"]',
    "pid": 42844,
}


# A bookkeeping result: the plan the model wrote itself. It has no exit code and no error,
# so it passes every result rule and inspected nothing.
TODO_UPDATED: dict[str, Any] = {
    "type": "Todo",
    "TodosUpdated": {
        "summary_for_prompt": "1 todo",
        "todos": [{"content": "Read the diff", "status": "in_progress"}],
        "state": {},
    },
}
KILL_TASK = {
    "type": "KillTask",
    "Result": {"task_id": "call-1", "outcome": "killed", "message": "Task was terminated"},
}
SEARCH_REPLACE = {"type": "SearchReplace", "path": "/work/repo/f.py", "replacements": 1}


def task_output(status: str, exit_code: int | None, output: str) -> dict[str, Any]:
    """The background command's own report, under `Result`."""
    return {
        "type": "TaskOutput",
        "Result": {
            "task_id": "call-1",
            "command": "uv run --no-project python probe.py",
            "status": status,
            "exit_code": exit_code,
            "started": "2026-09-26T19:49:03Z",
            "ended": None,
            "duration_secs": 78.890947,
            "output": output,
            "output_file": "/work/.grok/sessions/s/terminal/call-1.log",
            "truncated": False,
            "truncation_hint": "[truncated - use read_file on output_file for full content]",
            "raw_output_bytes": 0,
        },
    }


def task_outputs(*children: tuple[str, int | None]) -> dict[str, Any]:
    """A poll of several background commands at once: each one's report in a list under
    `MultiResult`. Keys and value types as a real grok-4.7 session recorded them."""
    results = [
        {
            "task_id": f"call-{n}",
            "command": "uv run --no-project python probe.py",
            "status": status,
            "exit_code": exit_code,
            "started": "2026-09-26T19:49:03Z",
            "ended": None if status == "running" else "2026-09-26T19:49:44Z",
            "duration_secs": 40.780223,
            "output": "",
            "output_file": f"/work/.grok/sessions/s/terminal/call-{n}.log",
            "truncated": False,
            "truncation_hint": "[truncated - use read_file on output_file for full content]",
            "raw_output_bytes": 0,
        }
        for n, (status, exit_code) in enumerate(children, 1)
    ]
    done = sum(status == "completed" for status, _ in children)
    return {
        "type": "TaskOutput",
        "MultiResult": {
            "mode": "wait_all",
            "results": results,
            "summary": f"{done}/{len(children)} tasks completed (wait_all)",
        },
    }


# Event fixtures in each provider's own vocabulary, at module scope because both the counter's
# tests and the gate's need them. Shapes copied from real runs: grok 1.0.13 and grok-4.7, and
# codex-cli 0.150.1, 0.152.1 and 0.156.1.
def grok_tool_result(
    ident: str, report: dict[str, Any] | str = READ_FILE, *, is_error: bool = False
) -> str:
    """A call's outcome, as grok returns it: a `tool_result` block in a `user` event."""
    content = report if isinstance(report, str) else json.dumps(report)
    block = {"type": "tool_result", "tool_use_id": ident, "content": content, "is_error": is_error}
    return json.dumps({"type": "user", "message": {"role": "user", "content": [block]}})


def grok_tool_call(
    name: str = "read_file", report: dict[str, Any] | str | None = READ_FILE, **result: Any
) -> str:
    """A call and, unless `report` is None, the result that answers it.

    Answered by default, because a call with no successful result does not count and the run
    behind it is refused: a fixture standing in for a real review has to carry one.
    """
    call = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": {}},
                ]
            },
        }
    )
    if report is None:
        return call
    return call + "\n" + grok_tool_result(f"toolu_{name}", report, **result)


def grok_result(**over: Any) -> str:
    event: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "stop_reason": "end_turn",
        "num_turns": 3,
        "usage": {"output_tokens": 4096},
    }
    event.update(over)
    return json.dumps(event)


# Tool kinds that change the working tree. A write does not show the tree was read.
WRITE_KINDS = ["file_change", "patch_apply"]


def codex_item(kind: str, item_type: str, ident: str | None = "item_1", **fields: Any) -> str:
    """One codex item event. A completed command SUCCEEDS unless `fields` say otherwise,
    because every completed command in a real stream carries an exit code: 0 when it worked.
    A completed write carries status "completed", as a real one does.
    """
    item: dict[str, Any] = {"type": item_type}
    if ident is not None:
        item["id"] = ident
    if kind == "item.completed" and item_type in validate.CODEX_COMMAND_ITEMS:
        item.update(aggregated_output="", exit_code=0, status="completed")
    elif kind == "item.completed" and item_type in WRITE_KINDS:
        item.update(status="completed")
    item.update(fields)
    return json.dumps({"type": kind, "item": item})


def codex_stream(*items: str) -> str:
    """A codex run: a thread, a turn, whatever items are given, and a terminal turn."""
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "th_1"}),
        json.dumps({"type": "turn.started"}),
        *items,
        json.dumps({"type": "turn.completed", "usage": {"output_tokens": 151}}),
    ]
    return "\n".join(lines) + "\n"


CODEX_ONE_CALL = codex_stream(
    codex_item("item.started", "command_execution"),
    codex_item("item.completed", "command_execution"),
)
CODEX_NO_CALLS = codex_stream(codex_item("item.completed", "agent_message", "item_0"))
# Tool kinds that are calls but do not prove the tree was read: the internet, or a tool that
# does not say where it runs.
NOT_LOCAL_KINDS = ["custom_tool_call", "function_call", "mcp_tool_call", "web_search"]
# Calls, and none of them local: the run read the internet and never opened the repository.
CODEX_ONLY_SEARCHED = codex_stream(
    codex_item("item.started", "web_search", "ws_1"),
    codex_item("item.completed", "web_search", "ws_1"),
    codex_item("item.completed", "web_search", "ws_2"),
)

# REAL STREAMS: every line below is copied from a kept event stream, keeping a subset of its
# lines, with the reviewed repository's commit ids and file names and the thread ids replaced
# by neutral values. Every key, value type, status, exit code and the runner's error text is
# as recorded. They are what the counting rule is proven on.
#
# codex-cli 0.156.1 on a runner where no command could start: five commands complete
# `failed` with exit code 1 beside two MCP calls, and the run answered with an empty
# findings array that 0.3.4 passed at exit 0.
C156_STREAM = r"""
{"type":"thread.started","thread_id":"00000000-0000-7000-8000-000000000001"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"I’ll measure the diff, inspect each changed area and its surrounding code, then return the review as JSON."}}
{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"/bin/bash -lc \"rg --files -g AGENTS.md -g '\"'!vendor'\"' -g '\"'!node_modules'\"'\"","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"/bin/bash -lc \"rg --files -g AGENTS.md -g '\"'!vendor'\"' -g '\"'!node_modules'\"'\"","aggregated_output":"error building bubblewrap command: cannot establish app-server socket mount isolation\n","exit_code":1,"status":"failed"}}
{"type":"item.started","item":{"id":"item_2","type":"command_execution","command":"/bin/bash -lc 'git diff --numstat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"/bin/bash -lc 'git diff --numstat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"error building bubblewrap command: cannot establish app-server socket mount isolation\n","exit_code":1,"status":"failed"}}
{"type":"item.started","item":{"id":"item_3","type":"command_execution","command":"/bin/bash -lc 'git diff --stat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_3","type":"command_execution","command":"/bin/bash -lc 'git diff --stat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"error building bubblewrap command: cannot establish app-server socket mount isolation\n","exit_code":1,"status":"failed"}}
{"type":"item.started","item":{"id":"item_4","type":"command_execution","command":"/bin/bash -lc 'git status --short'","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_4","type":"command_execution","command":"/bin/bash -lc 'git status --short'","aggregated_output":"error building bubblewrap command: cannot establish app-server socket mount isolation\n","exit_code":1,"status":"failed"}}
{"type":"item.completed","item":{"id":"item_5","type":"agent_message","text":"The local command runner is failing before Git starts, so I haven’t been able to read the diff. I’m checking whether another available tool can expose the repository before I return a result."}}
{"type":"item.started","item":{"id":"item_6","type":"mcp_tool_call","server":"codex","tool":"list_mcp_resources","arguments":{},"result":null,"error":null,"status":"in_progress"}}
{"type":"item.started","item":{"id":"item_7","type":"mcp_tool_call","server":"codex","tool":"list_mcp_resource_templates","arguments":{},"result":null,"error":null,"status":"in_progress"}}
{"type":"item.started","item":{"id":"item_8","type":"command_execution","command":"/bin/bash -c pwd","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_8","type":"command_execution","command":"/bin/bash -c pwd","aggregated_output":"error building bubblewrap command: cannot establish app-server socket mount isolation\n","exit_code":1,"status":"failed"}}
{"type":"item.completed","item":{"id":"item_7","type":"mcp_tool_call","server":"codex","tool":"list_mcp_resource_templates","arguments":{},"result":{"content":[{"type":"text","text":"{\"resourceTemplates\":[]}"}],"structured_content":null},"error":null,"status":"completed"}}
{"type":"item.completed","item":{"id":"item_9","type":"agent_message","text":"{\"reviewer\":\"adversarial-reviewer\",\"findings\":[],\"residual_risks\":[\"Review incomplete: the command runner failed before Git started, so I could not read git diff a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD or assess the changed code. An empty findings array does not indicate the change is safe.\"],\"testing_gaps\":[\"Could not inspect or run tests because the local command runner failed.\"]}"}}
{"type":"turn.completed","usage":{"input_tokens":110820,"cached_input_tokens":92288,"cache_write_input_tokens":0,"output_tokens":734,"reasoning_output_tokens":175}}
"""  # noqa: E501

# codex-cli 0.152.1 on the same diff, working.
C152M_STREAM = r"""
{"type":"thread.started","thread_id":"00000000-0000-7000-8000-000000000002"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"I’ll size the diff, identify whether it changes any verification mechanism, then trace the changed components and their call sites before emitting the required JSON."}}
{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"/bin/bash -lc 'git diff --numstat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD && git diff --stat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD && git diff --name-status a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"/bin/bash -lc 'git diff --numstat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD && git diff --stat a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD && git diff --name-status a1b2c3d4e5f60718293a4b5c6d7e8f9012345678..HEAD'","aggregated_output":"485\t0\tsrc/service.py\n32\t0\ttests/test_service.py\n src/service.py        | 485 ++++++++++++++++++++++++++++++++++\n tests/test_service.py |  32 +++\n 2 files changed, 517 insertions(+)\nA\tsrc/service.py\nA\ttests/test_service.py\n","exit_code":0,"status":"completed"}}
{"type":"turn.completed","usage":{"input_tokens":134325,"cached_input_tokens":107008,"cache_write_input_tokens":0,"output_tokens":3356,"reasoning_output_tokens":1128}}
"""  # noqa: E501

# A working review: two `rg` calls that ran and exited 1 and 2, both `failed` like the
# commands above that never started, and one command that exited 0.
Q2_MIXED_STREAM = r"""
{"type":"thread.started","thread_id":"00000000-0000-7000-8000-000000000003"}
{"type":"turn.started"}
{"type":"item.started","item":{"id":"item_5","type":"command_execution","command":"/bin/zsh -lc \"rg --files -g AGENTS.md -g '\"'!node_modules'\"' -g '\"'!vendor'\"'\"","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_5","type":"command_execution","command":"/bin/zsh -lc \"rg --files -g AGENTS.md -g '\"'!node_modules'\"' -g '\"'!vendor'\"'\"","aggregated_output":"","exit_code":1,"status":"failed"}}
{"type":"item.started","item":{"id":"item_71","type":"command_execution","command":"/bin/zsh -lc \"rg -n 'def locate|def anchors|def _find|def _stream|def _search|span|candidates' persona_review/anchors.py\"","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_71","type":"command_execution","command":"/bin/zsh -lc \"rg -n 'def locate|def anchors|def _find|def _stream|def _search|span|candidates' persona_review/anchors.py\"","aggregated_output":"rg: persona_review/anchors.py: IO error for operation on persona_review/anchors.py: No such file or directory (os error 2)\n","exit_code":2,"status":"failed"}}
{"type":"item.started","item":{"id":"item_98","type":"command_execution","command":"/bin/zsh -lc 'git diff --check 0a1b2c3..HEAD > /dev/null 2>&1; echo EXIT=$?'","aggregated_output":"","exit_code":null,"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_98","type":"command_execution","command":"/bin/zsh -lc 'git diff --check 0a1b2c3..HEAD > /dev/null 2>&1; echo EXIT=$?'","aggregated_output":"EXIT=0\n","exit_code":0,"status":"completed"}}
{"type":"turn.completed","usage":{"input_tokens":4559120,"cached_input_tokens":4356224,"cache_write_input_tokens":0,"output_tokens":18866,"reasoning_output_tokens":9988}}
"""  # noqa: E501

# A placeholder for a per-test fixture path inside a parametrize table, which is evaluated at
# import time and so cannot see instance state. Compared with `is`, never `==`.
ARTIFACT = "<the artifact under test>"


class TestObjectMode:
    """`--mode object` is strict, and the strictness is the feature.

    Regression: loosening this to tolerate prose-then-object imported a hole object mode was
    immune to. Two independent reviewers demonstrated both halves of it.
    """

    def test_a_bare_object_passes(self):
        assert validate.from_object_file(EMPTY_EXAMPLE)["findings"] == []

    def test_a_give_up_message_quoting_the_example_is_refused(self):
        # The exact shape a reviewer reproduced: an explicit refusal, then the brief's own
        # schema-valid empty example. Under the loosened gate this exited 0, "0 findings".
        text = (
            "I could not inspect the repository. The requested output shape is:\n\n" + EMPTY_EXAMPLE
        )
        with pytest.raises(validate.GateError) as caught:
            validate.from_object_file(text)
        assert "exactly one JSON object" in str(caught.value)

    def test_a_fenced_object_is_refused(self):
        text = "```json\n" + EMPTY_EXAMPLE + "\n```\n"
        with pytest.raises(validate.GateError):
            validate.from_object_file(text)

    def test_real_findings_followed_by_the_example_are_refused(self):
        # The worse half: a completed review that also pastes the example last. Accepting
        # this discards real defects and reports clean.
        text = json.dumps(artifact(finding(severity="P0"))) + "\n\n" + EMPTY_EXAMPLE
        with pytest.raises(validate.GateError):
            validate.from_object_file(text)

    def test_prose_only_is_refused(self):
        with pytest.raises(validate.GateError):
            validate.from_object_file("I could not complete the review.")

    def test_json_without_a_findings_key_is_refused(self):
        with pytest.raises(validate.GateError) as caught:
            validate.from_object_file('{"summary": "all good"}')
        assert "no findings key" in str(caught.value)


class TestSchemaRules:
    """The rules the installed plugin schema actually uses."""

    def test_union_type_accepts_both_members(self):
        schema = schema_with("suggested_fix", {"type": ["string", "null"]})
        for value in ("do the thing", None):
            validate.validate(artifact(finding(suggested_fix=value)), schema)

    def test_union_type_rejects_a_non_member(self):
        schema = schema_with("suggested_fix", {"type": ["string", "null"]})
        with pytest.raises(validate.GateError):
            validate.validate(artifact(finding(suggested_fix=42)), schema)

    def test_a_numeric_union_still_rejects_a_boolean(self):
        # `True` satisfies isinstance(x, int), so a union pairing a numeric type with a
        # non-numeric one must not lose the bool carve-out.
        schema = schema_with("line", {"type": ["integer", "null"]})
        validate.validate(artifact(finding(line=None)), schema)
        with pytest.raises(validate.GateError):
            validate.validate(artifact(finding(line=True)), schema)

    def test_a_union_that_admits_booleans_takes_one(self):
        schema = schema_with("pre_existing", {"type": ["boolean", "null"]})
        validate.validate(artifact(finding(pre_existing=True)), schema)

    def test_minimum_and_maximum(self):
        validate.validate(artifact(finding(line=1)), SCHEMA)
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(finding(line=0)), SCHEMA)
        assert ">= 1" in str(caught.value)

    def test_max_length(self):
        validate.validate(artifact(finding(title="x" * 100)), SCHEMA)
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(finding(title="x" * 101)), SCHEMA)
        assert "maxLength" in str(caught.value)

    def test_array_item_types(self):
        validate.validate(artifact(finding(evidence=["f.py:1 -- x"])), SCHEMA)
        for bad in ([{"a": 1}], [None], ["ok", 3], []):
            with pytest.raises(validate.GateError):
                validate.validate(artifact(finding(evidence=bad)), SCHEMA)

    def test_enums_are_enforced(self):
        for bad in ({"severity": "critical"}, {"confidence": 72}):
            with pytest.raises(validate.GateError):
                validate.validate(artifact(finding(**bad)), SCHEMA)

    def test_a_type_name_the_gate_cannot_check_fails(self):
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(finding()), schema_with("title", {"type": "sTrInG"}))
        assert "cannot check" in str(caught.value)

    def test_an_unimplemented_schema_keyword_fails_loudly(self):
        # Silently ignoring anyOf/oneOf/$ref/const certifies against rules nobody checked.
        for keyword in ("anyOf", "oneOf", "allOf", "$ref", "const", "pattern", "not"):
            with pytest.raises(validate.GateError) as caught:
                validate.validate(artifact(finding()), schema_with("title", {keyword: "whatever"}))
            assert keyword in str(caught.value)

    def test_annotations_are_tolerated(self):
        schema = schema_with("title", {"type": "string", "description": "the title", "default": ""})
        validate.validate(artifact(finding()), schema)

    def test_a_non_object_property_spec_is_refused(self):
        # It used to be ACCEPTED, on the grounds that it did not crash. Silently skipping a
        # property spec the gate cannot read is the same silent certification the schema
        # keyword check exists to prevent.
        schema = schema_with("title", "not-a-spec")  # type: ignore[arg-type]
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(finding()), schema)
        assert "not a schema object" in str(caught.value)

    # The accept-list is global but enforcement is positional, so these passed the keyword
    # check and were then never applied: `evidence: [123, {"a": 1}]` validated under a
    # tuple-form items, and `{}` validated under a nested required.
    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "array", "items": [{"type": "string"}]},
            {"type": "object", "required": ["file", "line"]},
            {"type": "object", "properties": {"file": {"type": "string"}}},
        ],
    )
    def test_rules_this_gate_only_enforces_shallowly_are_refused_when_nested(
        self, spec: dict[str, Any]
    ):
        with pytest.raises(validate.GateError):
            validate.validate(artifact(finding()), schema_with("loc", spec))

    def test_additional_properties_is_not_treated_as_an_annotation(self):
        # It is a constraint, and the likeliest keyword for the plugin to add. Filing it as
        # metadata would keep this gate returning 0 while enforcing nothing about extra keys.
        schema: dict[str, Any] = json.loads(json.dumps(SCHEMA))
        schema["additionalProperties"] = False
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(finding()), schema)
        assert "additionalProperties" in str(caught.value)

    def test_missing_required_keys(self):
        with pytest.raises(validate.GateError) as caught:
            validate.validate({"findings": []}, SCHEMA)
        assert "missing required keys" in str(caught.value)

    # Distinct from the top-level check above: deleting the per-finding required loop would
    # leave that one green, because it only ever sees an empty findings array.
    @pytest.mark.parametrize("absent", ["title", "severity", "file", "line", "evidence"])
    def test_a_finding_missing_required_fields_is_refused(self, absent: str):
        item = {k: v for k, v in finding().items() if k != absent}
        with pytest.raises(validate.GateError) as caught:
            validate.validate(artifact(item), SCHEMA)
        assert absent in str(caught.value)


class TestGrokRunGating:
    """grok's terminal event, not just its payload.

    Schema-constrained decoding means a truncated or refused run still returns a well-formed
    `{"findings": []}`, indistinguishable from a clean review by the payload alone.
    """

    def _events(self, **result: Any) -> str:
        # The production shape: a real success carries subtype AND stop_reason, so the happy
        # path must exercise both new checks rather than skipping them for absence.
        event = {
            "type": "result",
            "is_error": False,
            "subtype": "success",
            "stop_reason": "end_turn",
            **result,
        }
        return '{"type":"system"}\n' + json.dumps(event) + "\n"

    def test_a_healthy_run_passes(self):
        got = validate.from_grok_events(self._events(structured_output=artifact()))
        assert got["findings"] == []

    @pytest.mark.parametrize("empty_first", [False, True])
    def test_two_HEALTHY_result_events_are_refused(self, empty_first: bool):
        # The case the old "last wins" rule got wrong, and the one no test covered: a real
        # review followed by an empty one. Both events pass every terminal-status check, so
        # nothing else can catch it — the second silently becomes the verdict and a P0 review
        # reports clean.
        real = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "stop_reason": "end_turn",
                "structured_output": artifact(finding(severity="P0")),
            }
        )
        empty = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "stop_reason": "end_turn",
                "structured_output": artifact(),
            }
        )
        order = (empty, real) if empty_first else (real, empty)
        with pytest.raises(validate.GateError) as caught:
            validate.from_grok_events("\n".join(order) + "\n")
        assert "2 `result` events" in str(caught.value)

    def test_an_early_success_cannot_certify_a_run_that_later_failed(self):
        # The intent the old "last wins" rule served, now satisfied by refusing both. A
        # stream with a healthy event followed by a failure has no single verdict, and
        # picking either one is a guess the gate is not entitled to make.
        early = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "stop_reason": "end_turn",
                "structured_output": artifact(finding(title="early")),
            }
        )
        late = json.dumps({"type": "result", "is_error": True, "stop_reason": "max_tokens"})
        with pytest.raises(validate.GateError) as caught:
            validate.from_grok_events(early + "\n" + late + "\n")
        assert "2 `result` events" in str(caught.value)

    def test_a_non_success_subtype_fails(self):
        with pytest.raises(validate.GateError):
            validate.from_grok_events(
                self._events(subtype="error_max_turns", structured_output=artifact())
            )

    def test_early_stop_reasons_fail_even_with_a_well_formed_answer(self):
        for stop in ("max_tokens", "length", "content_filter", "refusal", "timeout"):
            with pytest.raises(validate.GateError) as caught:
                validate.from_grok_events(
                    self._events(stop_reason=stop, structured_output=artifact())
                )
            assert stop in str(caught.value)

    def test_a_mistyped_terminal_field_fails_closed(self):
        for bad in ({"subtype": 3}, {"stop_reason": ["end_turn"]}, {"is_error": "false"}):
            with pytest.raises(validate.GateError):
                validate.from_grok_events(self._events(structured_output=artifact(), **bad))

    @pytest.mark.parametrize("missing", ["is_error", "subtype", "stop_reason"])
    def test_an_absent_terminal_field_fails_closed(self, missing: str):
        # Absence is not evidence of success. A terminal event carrying none of these plus
        # the schema-generated empty findings object was accepted as a clean review — the
        # exact shape this gate exists to reject, since constrained decoding produces that
        # payload whether or not the run completed.
        full = {
            "type": "result",
            "is_error": False,
            "subtype": "success",
            "stop_reason": "end_turn",
            "structured_output": artifact(),
        }
        validate.from_grok_events(json.dumps(full) + "\n")  # control: the full shape passes
        event = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(validate.GateError):
            validate.from_grok_events(json.dumps(event) + "\n")

    def test_a_bare_result_event_with_an_empty_answer_is_refused(self):
        event = {"type": "result", "structured_output": artifact()}
        with pytest.raises(validate.GateError):
            validate.from_grok_events(json.dumps(event) + "\n")

    def test_an_unfamiliar_stop_reason_does_not_fail_a_good_run(self):
        got = validate.from_grok_events(
            self._events(stop_reason="finished_normally", structured_output=artifact())
        )
        assert got["findings"] == []

    # PRESENT but wrong is a malformed answer, not a reason to read a different channel.
    # Falling back to the raw text when --json-schema was in force means the gate quietly
    # answered from a channel nobody asked for.
    @pytest.mark.parametrize("bad", [{"oops": 1}, [], "a string", 7])
    def test_a_structured_output_that_is_not_findings_fails_rather_than_falling_through(
        self, bad: Any
    ):
        real = json.dumps(artifact(finding(title="from the raw text")))
        with pytest.raises(validate.GateError) as caught:
            validate.from_grok_events(self._events(structured_output=bad, result=real))
        assert "not a findings object" in str(caught.value)

    def test_no_structured_output_at_all_still_reads_the_raw_text_strictly(self):
        # The unconstrained case: no schema was in force, so the final text is all there is.
        got = validate.from_grok_events(self._events(result=EMPTY_EXAMPLE))
        assert got["findings"] == []
        with pytest.raises(validate.GateError):
            validate.from_grok_events(self._events(result="I gave up. " + EMPTY_EXAMPLE))

    def test_an_unknown_extraction_mode_is_refused(self):
        # Mutating this guard to `return from_object_file(text)` left the whole unit suite
        # green — an unguarded guard found by asking which ones had no mutation entry.
        with pytest.raises(validate.GateError) as caught:
            validate.extract("transcript", EMPTY_EXAMPLE)
        assert "unknown extraction mode" in str(caught.value)
        for mode in ("", "grok", "OBJECT"):
            with pytest.raises(validate.GateError):
                validate.extract(mode, EMPTY_EXAMPLE)

    def test_the_two_real_modes_dispatch_correctly(self):
        assert validate.extract("object", EMPTY_EXAMPLE)["findings"] == []
        assert (
            validate.extract("grok-events", self._events(structured_output=artifact()))["findings"]
            == []
        )

    def test_no_result_event_fails(self):
        with pytest.raises(validate.GateError):
            validate.from_grok_events('{"type":"system"}\n')

    def test_a_raw_text_result_goes_through_the_strict_object_reader(self):
        got = validate.from_grok_events(self._events(result=EMPTY_EXAMPLE))
        assert got["findings"] == []
        with pytest.raises(validate.GateError):
            validate.from_grok_events(self._events(result="I gave up. " + EMPTY_EXAMPLE))


class TestRunEvidence:
    """What the run DID, counted from its own event stream.

    The incident this closes: grok returned a schema-valid EMPTY findings artifact from one
    turn, zero tool calls, 151 output tokens and four and a half seconds, and exited 0.
    Nothing about the ANSWER separated that from a clean review — only the transcript did,
    and the exit status a gating caller branches on said CLEAN.
    """

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _file(self, *lines: str) -> Path:
        path = self.dir / "events.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _grok(self, *lines: str, duration: float | None = None) -> validate.RunStats:
        return validate.run_stats(
            "grok-messages", validate.file_objects(self._file(*lines)), duration
        )

    def _codex(self, *items: str) -> validate.RunStats:
        return validate.run_stats(
            "codex-items", validate.objects(codex_stream(*items).splitlines()), None
        )

    def test_grok_counts_tool_use_blocks_and_reads_the_run_s_own_numbers(self):
        stats = self._grok(
            '{"type":"system","subtype":"init"}',
            grok_tool_call("grep"),
            grok_tool_call("read_file"),
            grok_result(),
            duration=12.5,
        )
        assert stats.tool_calls == 2
        assert (stats.turns, stats.output_tokens, stats.duration_s) == (3, 4096, 12.5)

    def test_grok_does_not_count_the_tool_results_coming_back(self):
        # Every call is echoed as a `tool_result` block inside a USER message. Counting
        # content blocks without testing the block type doubles every total, which would
        # make one real call look like two and — worse — make a stream of nothing but
        # results look like work.
        echo = json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_x"}]},
            }
        )
        assert self._grok(grok_tool_call(), echo, grok_result()).tool_calls == 1

    def test_the_incident_stream_counts_zero(self):
        # The shape of the run that started this: one assistant turn carrying thinking and
        # text, no tool_use anywhere, a healthy terminal event, 151 output tokens. The real
        # stream produces the same counts — checked against the kept artifact, not inferred.
        answered = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "The user wants me to review a diff"},
                        {"type": "text", "text": EMPTY_EXAMPLE},
                    ]
                },
            }
        )
        stats = self._grok(
            '{"type":"system","subtype":"init"}',
            answered,
            grok_result(num_turns=1, usage={"output_tokens": 151}),
            duration=4.5,
        )
        assert stats.tool_calls == 0
        assert (stats.turns, stats.output_tokens) == (1, 151)

    def test_codex_counts_one_call_per_item_not_one_per_event(self):
        # codex emits `item.started` AND `item.completed` for the same call. Two of the three
        # calls below carry an id and are deduped by it; the third carries NONE, which used to
        # fall past the dedupe and be counted once per event — the fail-OPEN direction, and
        # the reason an id-less pair is in this fixture rather than a test of its own.
        stats = self._codex(
            codex_item("item.started", "command_execution", "item_1"),
            codex_item("item.completed", "command_execution", "item_1"),
            codex_item("item.started", "command_execution", "item_2"),
            codex_item("item.completed", "command_execution", "item_2"),
            codex_item("item.started", "command_execution", None),
            codex_item("item.completed", "command_execution", None),
        )
        assert stats.tool_calls == 3
        assert (stats.turns, stats.output_tokens) == (1, 151)

    def test_web_searches_are_calls_but_not_local_ones(self):
        # A run that only searched the web read the internet, not the repository: it made
        # calls, so `tool_calls` says so, and none of them acted on the tree it reviewed.
        stats = self._codex(
            codex_item("item.started", "web_search", "ws_1"),
            codex_item("item.completed", "web_search", "ws_1"),
            codex_item("item.completed", "command_execution", "item_2"),
            codex_item("item.completed", "web_search", None),
        )
        assert (stats.tool_calls, stats.local_tool_calls) == (3, 1)

    @pytest.mark.parametrize("kind", sorted(validate.CODEX_LOCAL_TOOL_ITEMS))
    def test_every_local_kind_is_counted_as_local(self, kind: str):
        # With an id and without, so both counting arms carry the local tallies.
        for ident in ("item_1", None):
            stats = self._codex(codex_item("item.completed", kind, ident))
            assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1), ident

    @pytest.mark.parametrize("kind", NOT_LOCAL_KINDS + WRITE_KINDS)
    def test_a_kind_that_does_not_prove_the_tree_was_read_is_a_call_but_not_local(self, kind: str):
        stats = self._codex(codex_item("item.completed", kind, "item_1"))
        assert (stats.tool_calls, stats.local_tool_calls) == (1, 0)

    def test_the_local_kinds_are_the_ones_that_read_the_working_directory(self):
        local = validate.CODEX_LOCAL_TOOL_ITEMS
        assert local == {"command_execution", "local_shell_call"}
        not_local = validate.CODEX_TOOL_ITEMS - local
        assert not_local == set(NOT_LOCAL_KINDS) | set(WRITE_KINDS)

    def test_grok_counts_every_inspecting_call_as_local(self):
        # The argv disables grok's web tools, so no inspecting call left is one known to leave
        # the machine.
        assert "--disable-web-search" in providers.GROK.argv(
            providers.Invocation(
                model="m",
                effort="e",
                repo=Path("."),
                prompt_file=Path("p"),
                schema_text="{}",
                last_file=None,
            )
        )
        stats = self._grok(grok_tool_call("grep"), grok_tool_call("read_file"), grok_result())
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (2, 2, 2)

    def test_an_id_less_call_counts_once_never_twice(self):
        # Stated on its own as well, because the rule is not "dedupe": without an id the two
        # events cannot be paired, so the terminal one is counted and the start is not.
        assert self._codex(codex_item("item.completed", "web_search", None)).tool_calls == 1
        assert self._codex(codex_item("item.started", "web_search", None)).tool_calls == 0

    def test_codex_does_not_mistake_the_model_talking_to_itself_for_a_tool_call(self):
        # `agent_message` and `reasoning` are items too. A count that took every item would
        # certify a run that only ever thought and answered — precisely the dud shape.
        stats = self._codex(
            codex_item("item.completed", "agent_message", "item_0"),
            codex_item("item.completed", "reasoning", "item_1"),
        )
        assert stats.tool_calls == 0

    def test_a_todo_list_is_not_evidence_that_anything_was_inspected(self):
        # It IS a tool invocation, and it reaches nothing: a run whose only tool call was
        # writing itself a plan inspected exactly as much as one that made none. Counting it
        # would let the dud shape through by one event.
        assert self._codex(codex_item("item.completed", "todo_list", "item_3")).tool_calls == 0

    def test_a_partial_last_line_is_skipped_rather_than_fatal(self):
        # The stream is append-only and a killed run leaves a half-written line. That is a
        # condition the watchdogs already judged; re-deciding it here would fail runs that
        # completed.
        assert self._grok(grok_tool_call(), grok_result(), '{"type":"assi').tool_calls == 1

    def test_an_unopenable_stream_is_an_environment_error_not_a_silent_zero(self):
        # It must not fall through to zero and refuse the run with the wrong reason: this
        # process wrote that file moments ago, so failing to open it is the machine's
        # problem, not the model's.
        with pytest.raises(errors.EnvError):
            validate.run_stats(
                "grok-messages", validate.file_objects(self.dir / "never-written.jsonl"), None
            )

    def test_an_unknown_event_mode_is_an_environment_error(self):
        # A mis-wired build, not a bad answer. Reporting it as a gate failure would blame the
        # model for a defect in the wrapper — the same misdiagnosis the drift check prevents.
        with pytest.raises(errors.EnvError) as caught:
            validate.run_stats("transcript", validate.objects(["{}"]), None)
        assert "unknown event mode" in str(caught.value)

    def test_every_provider_names_a_mode_this_module_understands(self):
        # The drift control. `events_mode` is declared in providers.py and dispatched in
        # validate.py, so the two are free to disagree — and the failure would be a provider
        # whose runs all refuse, or worse, one whose evidence is never counted.
        for provider in providers.PROVIDERS.values():
            source = CODEX_ONE_CALL if provider.name == "codex" else grok_tool_call()
            stats = validate.run_stats(
                provider.events_mode, validate.objects(source.splitlines()), None
            )
            assert stats.tool_calls == 1, provider.name

    def test_the_answer_modes_and_the_events_modes_share_no_value(self):
        # They once shared "grok-events", so handing an ANSWER mode where an events mode
        # belongs was caught for codex and silently accepted for grok. Distinct values make
        # that mis-wiring fail for both providers rather than one.
        answer_modes = {providers.MODE_GROK_EVENTS, providers.MODE_OBJECT}
        event_modes = {providers.EVENTS_GROK, providers.EVENTS_CODEX}
        assert not (answer_modes & event_modes), sorted(answer_modes & event_modes)
        for mode in answer_modes:
            with pytest.raises(errors.EnvError):
                validate.run_stats(mode, validate.objects(["{}"]), None)


class TestACallCountsOnlyIfItSucceeded:
    """A local call inspected the tree only if it ran and worked.

    The incident this closes: codex-cli 0.156.1 could not start a single command, every one
    completed `failed` with exit code 1, and 0.3.4 counted the five of them as inspection and
    passed an empty findings array at exit 0 — a clean review of nothing.
    """

    def _codex(self, text: str) -> validate.RunStats:
        return validate.run_stats("codex-items", validate.objects(text.splitlines()), None)

    def _grok(self, *lines: str) -> validate.RunStats:
        text = "\n".join(lines) + "\n"
        return validate.run_stats("grok-messages", validate.objects(text.splitlines()), None)

    def test_the_c156_stream_attempted_five_local_calls_and_none_succeeded(self):
        stats = self._codex(C156_STREAM)
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (7, 5, 0)
        # The first COMPLETION's output, not the empty output its `item.started` carries.
        assert stats.first_failure == (
            "error building bubblewrap command: cannot establish app-server socket mount isolation"
        )

    def test_a_working_stream_counts_its_command(self):
        stats = self._codex(C152M_STREAM)
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (1, 1, 1)

    def test_commands_that_ran_and_exited_non_zero_do_not_count_and_do_not_sink_the_run(self):
        # `rg` exiting 1 and 2 reads `failed`, exactly like a command that never started; the
        # one command that exited 0 is what makes this a review.
        stats = self._codex(Q2_MIXED_STREAM)
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (3, 3, 1)
        assert stats.first_failure == "", "the first failure, `rg` exiting 1, printed nothing"

    @pytest.mark.parametrize("kind", sorted(validate.CODEX_COMMAND_ITEMS))
    @pytest.mark.parametrize("exit_code", [1, 2, None, True, False, "0", 0.0])
    def test_a_command_counts_only_on_exit_code_0(self, kind: str, exit_code: Any):
        stats = self._codex(codex_stream(codex_item("item.completed", kind, exit_code=exit_code)))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)
        stats = self._codex(codex_stream(codex_item("item.completed", kind, exit_code=0)))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1)

    @pytest.mark.parametrize("kind", WRITE_KINDS)
    def test_a_completed_write_is_a_tool_call_and_never_a_local_one(self, kind: str):
        # Grok's edits are skipped for the same reason: changing the tree shows no read of it.
        stats = self._codex(
            codex_stream(codex_item("item.started", kind), codex_item("item.completed", kind))
        )
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (1, 0, 0)

    def test_every_local_kind_succeeds_by_exit_code(self):
        # A local kind is judged by its exit code alone, so one that carries none would never
        # succeed: adding such a kind means giving it a rule in `_codex_succeeded` first.
        assert validate.CODEX_COMMAND_ITEMS == validate.CODEX_LOCAL_TOOL_ITEMS

    def test_a_call_reported_complete_twice_succeeds_once(self):
        # So the succeeded count never exceeds the attempts it is a part of.
        done = codex_item("item.completed", "command_execution", "item_1")
        stats = self._codex(codex_stream(done, done))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1)

    def test_a_codex_id_carried_by_two_kinds_pairs_with_neither(self):
        # A command's start and a write's completion are two calls, and neither is shown to
        # have worked.
        wrote = self._codex(
            codex_stream(
                codex_item("item.started", "command_execution", "i"),
                codex_item("item.completed", "file_change", "i"),
            )
        )
        assert (wrote.tool_calls, wrote.local_tool_attempts, wrote.local_tool_calls) == (2, 1, 0)
        ran = self._codex(
            codex_stream(
                codex_item("item.started", "local_shell_call", "i"),
                codex_item("item.completed", "command_execution", "i"),
            )
        )
        assert (ran.tool_calls, ran.local_tool_attempts, ran.local_tool_calls) == (2, 2, 0)
        searched = self._codex(
            codex_stream(
                codex_item("item.completed", "web_search", "i"),
                codex_item("item.completed", "command_execution", "i"),
            )
        )
        assert (searched.tool_calls, searched.local_tool_calls) == (2, 0)

    @pytest.mark.parametrize("failed_first", [True, False])
    def test_a_codex_id_completed_both_ways_counts_for_neither(self, failed_first: bool):
        failed = codex_item(
            "item.completed", "command_execution", "i", exit_code=1, status="failed"
        )
        worked = codex_item("item.completed", "command_execution", "i")
        order = (failed, worked) if failed_first else (worked, failed)
        stats = self._codex(codex_stream(*order))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)

    def test_a_long_first_line_is_cut(self):
        long = codex_item(
            "item.completed", "command_execution", exit_code=1, aggregated_output="x" * 500
        )
        assert self._codex(codex_stream(long)).first_failure == "x" * 200 + "..."

    @pytest.mark.parametrize(
        ("report", "is_error", "succeeded"),
        [
            (READ_FILE, False, 1),
            (BASH_OK, False, 1),
            (task_output("completed", 0, "Python 3.14.6\n"), False, 1),
            # Each of these is refused by exactly one rule, so each rule is proven alone.
            (READ_FILE, True, 0),
            (BASH_FAILED, False, 0),
            (GREP_NO_MATCH, False, 0),
            (bash_report(None, ""), False, 0),
            (BACKGROUND_STARTED, False, 0),
            (task_output("completed", 2, "Traceback\n"), False, 0),
            (task_output("running", None, "\n\nWaited the requested 60s.\n"), False, 0),
        ],
    )
    def test_a_grok_call_counts_only_on_a_result_reporting_success(
        self, report: dict[str, Any], is_error: bool, succeeded: int
    ):
        stats = self._grok(grok_tool_call("run_terminal_command", report, is_error=is_error))
        assert (stats.tool_calls, stats.local_tool_attempts) == (1, 1)
        assert stats.local_tool_calls == succeeded

    @pytest.mark.parametrize(
        ("report", "succeeded"),
        [
            (task_outputs(("completed", 0), ("completed", 0)), 1),
            (task_outputs(("completed", 0), ("running", None)), 1),
            (task_outputs(("failed", 1), ("running", None)), 0),
            (task_outputs(("failed", 1), ("completed", 2)), 0),
            (task_outputs(), 0),
            ({"type": "TaskOutput", "MultiResult": {"mode": "wait_all"}}, 0),
            ({"type": "TaskOutput", "MultiResult": {"results": {"exit_code": 0}}}, 0),
            ({"type": "TaskOutput", "MultiResult": {"results": [{"exit_code": "0"}]}}, 0),
            ({"type": "TaskOutput", "MultiResult": [{"exit_code": 0}]}, 0),
            (task_outputs(("failed", 0)), 0),
            (task_outputs(("failed", 0), ("completed", 0)), 1),
            ({"type": "TaskOutput", "MultiResult": {"results": [{"exit_code": 0}]}}, 1),
            ({"type": "TaskOutput", "MultiResult": {"results": [{"status": "completed"}]}}, 0),
        ],
        ids=[
            "all-exited-0",
            "one-exited-0-one-running",
            "failed-and-running",
            "all-failed",
            "no-results",
            "results-missing",
            "results-not-a-list",
            "exit-code-not-a-number",
            "batch-not-an-object",
            "exited-0-but-failed",
            "exited-0-but-failed-beside-one-completed",
            "exited-0-without-a-status",
            "completed-without-an-exit-code",
        ],
    )
    def test_a_batch_poll_counts_only_when_one_of_its_commands_completed_with_exit_0(
        self, report: dict[str, Any], succeeded: int
    ):
        # No status or exit code on the outer object: reading only that counted a poll of
        # commands that all failed as a successful read. A child needs both halves, an
        # `exit_code` of 0 and no status other than "completed".
        stats = self._grok(grok_tool_call("get_command_or_subagent_output", report))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, succeeded)

    @pytest.mark.parametrize(
        "content",
        [
            "command not found",
            json.dumps([{"type": "text", "text": "x"}]),
            "",
            json.dumps("oops"),
            [{"type": "text", "text": "x"}],
        ],
        ids=["plain-string", "json-array", "empty-string", "json-string", "raw-list"],
    )
    def test_a_grok_result_counts_only_when_its_content_is_a_json_object(self, content: Any):
        # `is_error` false beside content that reports nothing: no object, so no status and
        # no exit code to read, and an empty list of reports would pass every rule on it.
        call = grok_tool_call("run_terminal_command", None)
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": "toolu_run_terminal_command"}
        block.update(content=content, is_error=False)
        answer = json.dumps({"type": "user", "message": {"role": "user", "content": [block]}})
        stats = self._grok(call, answer)
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)
        # The control: the same call, answered with an object reporting exit code 0.
        stats = self._grok(grok_tool_call("run_terminal_command", BASH_OK))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1)

    @pytest.mark.parametrize(
        ("name", "report", "succeeded"),
        [
            # A status, and no exit code to fail on: at the top, and in a background
            # command's own report.
            ("run_terminal_command", {**_without(BASH_OK, "exit_code"), "status": "failed"}, 0),
            (
                "get_command_or_subagent_output",
                {"type": "TaskOutput", "Result": {"task_id": "call-1", "status": "failed"}},
                0,
            ),
            # A status grok has never reported is not taken for success.
            ("run_terminal_command", {**BASH_OK, "status": "succeeded"}, 0),
            ("read_file", {**READ_FILE, "status": None}, 0),
            # The two a report may carry and still count.
            ("run_terminal_command", {**BASH_OK, "status": "completed"}, 1),
            ("read_file", READ_FILE, 1),
            (
                "get_command_or_subagent_output",
                {"type": "TaskOutput", "Result": {"task_id": "call-1", "status": "completed"}},
                1,
            ),
        ],
        ids=[
            "failed-no-exit-code",
            "nested-failed-no-exit-code",
            "unknown-status",
            "null-status",
            "completed",
            "no-status",
            "nested-completed-no-exit-code",
        ],
    )
    def test_a_grok_report_counts_only_with_no_status_or_status_completed(
        self, name: str, report: dict[str, Any], succeeded: int
    ):
        stats = self._grok(grok_tool_call(name, report))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, succeeded)

    def test_a_bookkeeping_call_beside_failed_commands_is_not_inspection(self):
        # The bypass reproduced on the poster's stopgap: every command failed, and one Todo
        # result, which carries no exit code, read as "1 succeeded".
        stats = self._grok(
            grok_tool_call("run_terminal_command", BASH_FAILED),
            grok_tool_call("todo_write", TODO_UPDATED),
            grok_result(),
        )
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (2, 1, 0)

    @pytest.mark.parametrize(
        ("name", "report"),
        [
            ("todo_write", TODO_UPDATED),
            ("kill_command_or_subagent", KILL_TASK),
            ("search_replace", SEARCH_REPLACE),
            ("write", SEARCH_REPLACE),
        ],
    )
    def test_a_bookkeeping_call_alone_attempted_nothing(self, name: str, report: dict[str, Any]):
        # And is not drift either: these names are known, so a run of nothing else stays the
        # model's doing, exit 6.
        stats = self._grok(grok_tool_call(name, report), grok_result())
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (1, 0, 0)

    def test_an_unknown_grok_tool_with_no_inspecting_call_is_drift(self):
        with pytest.raises(errors.EnvError) as caught:
            self._grok(grok_tool_call("mcp_repo_search", READ_FILE), grok_result())
        message = str(caught.value)
        assert "mcp_repo_search" in message and "drift" in message, message
        named = set(re.findall(r"validate\.([A-Z_]+)", message))
        assert named == {"GROK_INSPECTING_TOOLS", "GROK_QUIET_TOOLS"}, message

    def test_an_unknown_grok_tool_beside_an_inspecting_call_is_not_drift(self):
        stats = self._grok(
            grok_tool_call("mcp_repo_search", READ_FILE), grok_tool_call("read_file"), grok_result()
        )
        assert (stats.tool_calls, stats.local_tool_attempts, stats.local_tool_calls) == (2, 1, 1)

    def test_the_grok_tool_lists_do_not_overlap(self):
        assert not validate.GROK_INSPECTING_TOOLS & validate.GROK_QUIET_TOOLS

    def test_a_grok_result_without_is_error_false_does_not_count(self):
        call = grok_tool_call("read_file", None)
        result = json.loads(grok_tool_result("toolu_read_file"))
        del result["message"]["content"][0]["is_error"]
        assert self._grok(call, json.dumps(result)).local_tool_calls == 0

    def test_a_grok_call_with_no_result_is_an_attempt_that_did_not_succeed(self):
        stats = self._grok(grok_tool_call("read_file", None), grok_result())
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)
        assert stats.first_failure is None

    def test_a_grok_result_counts_once_and_only_for_a_call_this_run_made(self):
        stranger = grok_tool_result("toolu_never_called")
        assert self._grok(stranger, grok_result()).local_tool_calls == 0
        twice = grok_tool_result("toolu_read_file")
        stats = self._grok(grok_tool_call("read_file"), twice, grok_result())
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1)

    @staticmethod
    def _called(ident: str, name: str) -> str:
        use: dict[str, Any] = {"type": "tool_use", "id": ident, "name": name, "input": {}}
        return json.dumps({"type": "assistant", "message": {"content": [use]}})

    def test_a_grok_id_two_calls_share_pairs_a_result_with_neither(self):
        # A todo reusing a command's id would otherwise lend the command its result.
        shared = self._grok(
            self._called("a", "run_terminal_command"),
            self._called("a", "todo_write"),
            grok_tool_result("a", TODO_UPDATED),
        )
        assert (shared.tool_calls, shared.local_tool_attempts, shared.local_tool_calls) == (2, 1, 0)
        both = self._grok(
            self._called("a", "read_file"), self._called("a", "read_file"), grok_tool_result("a")
        )
        assert (both.local_tool_attempts, both.local_tool_calls) == (2, 0)
        # Reuse after the result is reuse too: the stream never says which call it answered.
        later = self._grok(
            self._called("a", "read_file"), grok_tool_result("a"), self._called("a", "todo_write")
        )
        assert later.local_tool_calls == 0
        alone = self._grok(self._called("a", "read_file"), grok_tool_result("a"))
        assert alone.local_tool_calls == 1

    @pytest.mark.parametrize("first_is_error", [False, True])
    def test_a_grok_id_answered_both_ways_counts_for_neither(self, first_is_error: bool):
        stats = self._grok(
            self._called("a", "read_file"),
            grok_tool_result("a", is_error=first_is_error),
            grok_tool_result("a", is_error=not first_is_error),
        )
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)

    def test_a_grok_result_before_its_call_answers_nothing(self):
        stats = self._grok(grok_tool_result("a"), self._called("a", "read_file"))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0)

    def test_a_failed_grok_call_quotes_the_first_line_it_printed(self):
        stats = self._grok(grok_tool_call("run_terminal_command", BASH_FAILED))
        assert stats.first_failure == "error: Failed to initialize cache at `/work/.cache/uv`"
        refused = self._grok(
            grok_tool_call("read_file", "\nError: permission denied\nmore", is_error=True)
        )
        assert refused.first_failure == "Error: permission denied"
        # A failed todo is not a failed local call, so it is not the one quoted.
        after_todo = self._grok(
            grok_tool_call("todo_write", "todo store unavailable", is_error=True),
            grok_tool_call("run_terminal_command", BASH_FAILED),
        )
        assert after_todo.first_failure == "error: Failed to initialize cache at `/work/.cache/uv`"

    def test_the_first_failed_grok_call_is_the_one_quoted(self):
        # Each failure prints a different first line, so quoting a later one shows here.
        stats = self._grok(
            self._called("a", "run_terminal_command"),
            grok_tool_result("a", bash_report(2, "first failure line\n")),
            self._called("b", "run_terminal_command"),
            grok_tool_result("b", bash_report(1, "second failure line\n")),
        )
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (2, 0)
        assert stats.first_failure == "first failure line"

    def test_the_run_description_names_the_attempts_beside_none_succeeded(self):
        # "(0 local)" alone would describe a run that never tried.
        described = validate.describe_run(self._codex(C156_STREAM))
        assert "7 tool calls (5 local, 0 succeeded)" in described, described
        # And the web-search-only wording is unchanged, because that run attempted nothing.
        assert "(0 local)" in validate.describe_run(self._codex(CODEX_ONLY_SEARCHED))


class TestDriftIsNotBlamedOnTheModel:
    """A renamed event vocabulary is a broken wrapper, and must not be told as a bad model.

    After a provider-CLI upgrade renames its item kinds, every run counts zero. Refusing
    those with "the model never opened the diff" would be a falsehood repeated identically on
    every run, about the one component that was working — a permanent outage wearing the
    costume of a bad model, and a README that says retry-once-then-blame-the-model.
    """

    def _codex(self, text: str) -> validate.RunStats:
        return validate.run_stats("codex-items", validate.objects(text.splitlines()), None)

    def test_unrecognised_item_kinds_with_no_tool_calls_are_an_environment_error(self):
        with pytest.raises(errors.EnvError) as caught:
            self._codex(
                codex_stream(
                    codex_item("item.completed", "shell_call_v2", "item_1"),
                    codex_item("item.completed", "file_patch_v2", "item_2"),
                )
            )
        message = str(caught.value)
        assert "shell_call_v2" in message and "file_patch_v2" in message, message
        assert "drift" in message

    def test_the_drift_recovery_names_both_kind_lists(self):
        # Following a recovery that names only the tool list silences this check for a
        # renamed tree-reading kind, and every run then exits 6. Names are read back out of
        # the message and resolved, so renaming either constant fails here.
        with pytest.raises(errors.EnvError) as caught:
            self._codex(codex_stream(codex_item("item.completed", "shell_call_v2", "item_1")))
        message = str(caught.value)
        named = set(re.findall(r"validate\.([A-Z_]+)", message))
        assert named == {"CODEX_TOOL_ITEMS", "CODEX_LOCAL_TOOL_ITEMS"}, message
        assert validate.CODEX_LOCAL_TOOL_ITEMS <= validate.CODEX_TOOL_ITEMS
        assert "exits 6" in message, message

    def test_a_web_search_beside_a_renamed_kind_is_still_drift(self):
        # The refusal reads the LOCAL count, so the drift check must too: otherwise a search
        # beside a renamed local kind counts one call, skips this check, and the rename is
        # reported as a model that never opened the diff.
        with pytest.raises(errors.EnvError) as caught:
            self._codex(
                codex_stream(
                    codex_item("item.completed", "web_search", "ws_1"),
                    codex_item("item.completed", "shell_call_v2", "item_1"),
                )
            )
        assert "shell_call_v2" in str(caught.value)

    @pytest.mark.parametrize("kind", WRITE_KINDS)
    def test_a_completed_write_beside_a_renamed_kind_is_still_drift(self, kind: str):
        # A write is no local attempt, so it cannot vouch for the vocabulary either.
        with pytest.raises(errors.EnvError) as caught:
            self._codex(
                codex_stream(
                    codex_item("item.completed", kind, "item_1"),
                    codex_item("item.completed", "shell_call_v2", "item_2"),
                )
            )
        assert "shell_call_v2" in str(caught.value) and "drift" in str(caught.value)

    def test_a_recognised_kind_alongside_them_is_still_a_review(self):
        # The control that keeps the check from firing on every mixed stream: one kind we do
        # understand is evidence the vocabulary still overlaps ours, so this is not drift.
        stats = self._codex(
            codex_stream(
                codex_item("item.completed", "command_execution", "item_1"),
                codex_item("item.completed", "shell_call_v2", "item_2"),
            )
        )
        assert stats.tool_calls == 1

    def test_the_kinds_we_deliberately_skip_are_not_mistaken_for_drift(self):
        # THE OTHER CONTROL, and the one that decides whether exit 6 still exists: a genuine
        # dud emits agent_message and reasoning and nothing else. If those counted as
        # unrecognized, every vacuous run would report drift and the refusal would be dead.
        stats = self._codex(CODEX_NO_CALLS)
        assert stats.tool_calls == 0

    def test_a_codex_stream_with_no_events_at_all_is_an_environment_error(self):
        # `codex exec --json` opens every run with a thread and a turn, so an empty stream is
        # a runner that did not run — not a model that did nothing.
        with pytest.raises(errors.EnvError) as caught:
            self._codex("")
        assert "no events at all" in str(caught.value)

    def test_a_grok_run_that_called_nothing_is_not_drift(self):
        # The grok side's control: with no tool name to be unfamiliar, a run that called
        # nothing is still the model's doing, so exit 6 stays reachable.
        stats = validate.run_stats(
            "grok-messages", validate.objects(grok_result().splitlines()), None
        )
        assert (stats.tool_calls, stats.local_tool_attempts) == (0, 0)


class TestTheGateRefusesARunThatInspectedNothing:
    """Zero tool calls is not a small number of tool calls; it is no review at all."""

    def _gate(
        self,
        tmp: Path,
        answer: str,
        events: str,
        mode: str = "object",
        evidence_mode: str = "codex-items",
    ) -> tuple[int, str, str]:
        (tmp / "answer.txt").write_text(answer, encoding="utf-8")
        (tmp / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
        (tmp / "events.jsonl").write_text(events, encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = validate.gate(
                    answer_file=tmp / "answer.txt",
                    schema_path=tmp / "schema.json",
                    mode=mode,
                    findings_out=tmp / "out.json",
                    provenance_out=tmp / "out-provenance.json",
                    prov_pairs=[],
                    prov_files={},
                    evidence=validate.Evidence(
                        events_file=tmp / "events.jsonl", mode=evidence_mode, duration_s=4.5
                    ),
                    label="ce-persona",
                )
            except errors.AppError as exc:
                return exc.exit_code, out.getvalue(), str(exc)
        return code, out.getvalue(), err.getvalue()

    @pytest.mark.parametrize(
        "answer", [EMPTY_EXAMPLE, json.dumps(artifact(finding(severity="P0")))]
    )
    def test_no_tool_calls_is_refused_whether_or_not_it_reported_findings(self, answer: str):
        # Both arms, because refusing only the EMPTY one would read a populated array as
        # evidence the model worked. A model that read nothing and reported a P0 invented it.
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), answer, CODEX_NO_CALLS)
            assert code == 6, message
            assert out == "", "the summary line must not be printed for a refused run"
            assert "no local tool calls" in message
            # The artifacts survive: the dud IS the evidence of what was refused.
            record = json.loads((Path(tmp) / "out-provenance.json").read_text(encoding="utf-8"))
            # And the refusal points at the SIDECAR, never at the findings file: naming the
            # artifact invites the caller into the very listing that was just refused.
            assert str(Path(tmp) / "out-provenance.json") in message
            assert str(Path(tmp) / "out.json") not in message
        assert record["run_stats"]["tool_calls"] == 0

    def test_a_run_that_only_searched_the_web_is_refused(self):
        # It made calls, so a total count passed it. None acted on the repository, so it is
        # the same unfounded answer, and the refusal says what the run did instead.
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), EMPTY_EXAMPLE, CODEX_ONLY_SEARCHED)
            record = json.loads((Path(tmp) / "out-provenance.json").read_text(encoding="utf-8"))
        assert code == 6, message
        assert out == ""
        assert "no local tool calls" in message
        assert "2 tool calls (0 local)" in message, message
        assert record["run_stats"]["tool_calls"] == 2
        assert record["run_stats"]["local_tool_calls"] == 0

    @pytest.mark.parametrize("kind", NOT_LOCAL_KINDS + WRITE_KINDS)
    def test_a_run_whose_only_calls_are_not_local_is_refused(self, kind: str):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(
                Path(tmp), EMPTY_EXAMPLE, codex_stream(codex_item("item.completed", kind, "i1"))
            )
        assert code == 6, message
        assert out == ""
        assert "1 tool call (0 local)" in message, message

    def test_one_tool_call_is_enough(self):
        # The control. Without it every assertion above holds for a gate that refuses
        # everything, which is the same cannot-fail defect in the other direction.
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(Path(tmp), EMPTY_EXAMPLE, CODEX_ONE_CALL)
        assert code == 0, err
        assert "0 findings" in out

    def test_a_run_whose_every_local_call_failed_exits_3_and_keeps_its_evidence(self):
        # The c156 run, with the answer it actually gave: 0.3.4 passed it at exit 0.
        answer = json.loads(C156_STREAM.strip().splitlines()[-2])["item"]["text"]
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), answer, C156_STREAM)
            record = json.loads((Path(tmp) / "out-provenance.json").read_text(encoding="utf-8"))
            assert (Path(tmp) / "out.json").is_file(), "the refused artifact is the evidence"
            assert str(Path(tmp) / "out-provenance.json") in message
            assert str(Path(tmp) / "out.json") not in message
        assert code == 3, message
        assert out == "", "the summary line must not be printed for a refused run"
        assert "none of whose local tool calls succeeded" in message, message
        assert "7 tool calls (5 local, 0 succeeded)" in message, message
        assert (
            "'error building bubblewrap command: cannot establish app-server socket mount "
            "isolation'"
        ) in message, message
        # Neither the model's fault nor one diagnosed cause: a command that ran and exited
        # non-zero lands here too.
        assert "never opened the diff" not in message
        assert "sandbox" not in message
        assert "\n" not in message, "stderr carries one line"
        stats = record["run_stats"]
        assert (stats["tool_calls"], stats["local_tool_attempts"], stats["local_tool_calls"]) == (
            7,
            5,
            0,
        )

    @pytest.mark.parametrize("stream", [C152M_STREAM, Q2_MIXED_STREAM])
    def test_a_run_with_one_command_that_worked_is_a_review(self, stream: str):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(Path(tmp), EMPTY_EXAMPLE, stream)
        assert code == 0, err
        assert "0 findings" in out

    def test_failed_commands_beside_an_unrecognized_kind_are_not_drift(self):
        # The drift check reads ATTEMPTS: these commands are a kind this build knows, and
        # calling their failure a renamed vocabulary would send someone to the wrong fix.
        failed = codex_item(
            "item.completed",
            "command_execution",
            exit_code=1,
            status="failed",
            aggregated_output="rg: f.py: No such file or directory (os error 2)\n",
        )
        renamed = codex_item("item.completed", "shell_call_v2", "item_2")
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), EMPTY_EXAMPLE, codex_stream(failed, renamed))
        assert code == 3, message
        assert out == ""
        assert "none of whose local tool calls succeeded" in message, message
        assert "drift" not in message, message

    @pytest.mark.parametrize("kind", WRITE_KINDS)
    def test_failed_commands_beside_a_completed_write_exit_3(self, kind: str):
        # The write is the one call that completed, and it read nothing.
        failed = codex_item(
            "item.completed",
            "command_execution",
            exit_code=1,
            status="failed",
            aggregated_output="rg: f.py: No such file or directory (os error 2)\n",
        )
        wrote = codex_item("item.completed", kind, "item_2")
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), EMPTY_EXAMPLE, codex_stream(failed, wrote))
        assert code == 3, message
        assert out == ""
        assert "2 tool calls (1 local, 0 succeeded)" in message, message

    def test_a_run_whose_local_calls_never_finished_says_so(self):
        started = codex_item("item.started", "command_execution", exit_code=None)
        with tempfile.TemporaryDirectory() as tmp:
            code, _, message = self._gate(Path(tmp), EMPTY_EXAMPLE, codex_stream(started))
        assert code == 3, message
        assert "(1 local, 0 succeeded)" in message and "none of them finished" in message

    def test_a_run_whose_only_inspecting_call_polled_failed_commands_exits_3(self):
        poll = task_outputs(("failed", 1), ("running", None))
        stream = "\n".join(
            [
                grok_tool_call("get_command_or_subagent_output", poll),
                grok_result(structured_output=artifact()),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(
                Path(tmp), stream, stream + "\n", "grok-events", "grok-messages"
            )
        assert code == 3, message
        assert out == ""
        assert "(1 local, 0 succeeded)" in message, message

    @pytest.mark.parametrize(
        ("mode", "evidence_mode", "stream"),
        [
            # `rg` that ran, searched the tree and matched nothing: exit 1, nothing printed.
            (
                "object",
                "codex-items",
                codex_stream(
                    codex_item("item.completed", "command_execution", exit_code=1, status="failed")
                ),
            ),
            # grok's own search, reporting no matches the same way.
            (
                "grok-events",
                "grok-messages",
                grok_tool_call("grep", GREP_NO_MATCH)
                + "\n"
                + grok_result(structured_output=artifact())
                + "\n",
            ),
        ],
        ids=["codex", "grok"],
    )
    def test_a_search_that_matched_nothing_is_not_told_as_nothing_read(
        self, mode: str, evidence_mode: str, stream: str
    ):
        # The search did read the tree. It fails the rule like a command that never started,
        # because the two cannot be told apart, so the refusal says what is known -- no local
        # call succeeded, and how many were made -- and nothing about what was read.
        answer = EMPTY_EXAMPLE if mode == "object" else stream
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), answer, stream, mode, evidence_mode)
        assert code == 3, message
        assert out == ""
        assert "none of whose local tool calls succeeded" in message, message
        assert "(1 local, 0 succeeded)" in message, message
        assert re.search(r"\bread\b", message) is None, message

    def test_a_stream_this_build_cannot_count_leaves_no_artifact_behind(self):
        # Drift is decided BEFORE anything is written, unlike the vacuous refusal: there is no
        # verdict to keep evidence of, and an artifact on disk beside an environment failure
        # is exactly the stale answer `_clear_run_dir` exists to prevent.
        with tempfile.TemporaryDirectory() as tmp:
            code, _, message = self._gate(
                Path(tmp),
                EMPTY_EXAMPLE,
                codex_stream(codex_item("item.completed", "shell_call_v2", "item_1")),
            )
            assert code == 3, message
            assert not (Path(tmp) / "out.json").exists()
            assert not (Path(tmp) / "out-provenance.json").exists()

    def test_the_gate_counts_grok_from_the_text_it_already_read(self, monkeypatch: Any):
        # grok's answer arrives INSIDE its event stream, so the answer file and the events
        # file are one path and the ~1.4 MB is read once.
        #
        # Proved by making the file source UNUSABLE rather than by asserting the counts: a
        # second read would produce exactly the same numbers, so an outcome assertion here
        # would hold whether or not the reuse existed — which is the shape of guard this
        # repository keeps finding.
        def opened_again(path: Path) -> object:
            raise AssertionError(f"the gate opened {path} a second time to count it")

        monkeypatch.setattr(validate, "file_objects", opened_again)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = grok_tool_call() + "\n" + grok_result(structured_output=artifact()) + "\n"
            (root / "events.jsonl").write_text(stream, encoding="utf-8")
            (root / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = validate.gate(
                    answer_file=root / "events.jsonl",
                    schema_path=root / "schema.json",
                    mode="grok-events",
                    findings_out=root / "out.json",
                    provenance_out=root / "out-provenance.json",
                    prov_pairs=[],
                    prov_files={},
                    evidence=validate.Evidence(
                        events_file=root / "events.jsonl",
                        mode="grok-messages",
                        duration_s=1.0,
                    ),
                    label="ce-persona",
                )
            record = json.loads((root / "out-provenance.json").read_text(encoding="utf-8"))
        assert code == 0, out.getvalue()
        assert record["run_stats"]["tool_calls"] == 1


# Characters JSON carries raw inside a string and `str.splitlines` also breaks a line at.
# Escaped here, never literal, so the source says which character it means.
LINE_BREAKS_INSIDE_JSON = pytest.mark.parametrize(
    "separator", ["\u2028", "\u2029", "\x85"], ids=["U+2028", "U+2029", "U+0085"]
)


def raw_json(event: dict[str, Any]) -> str:
    """One event as grok writes it: non-ASCII characters raw rather than escaped."""
    return json.dumps(event, ensure_ascii=False)


class TestAStreamIsSplitAtNewlinesOnly:
    """An NDJSON record ends at "\\n" and nowhere else.

    `str.splitlines` also breaks at U+2028, U+2029 and U+0085. A successful read of a file
    holding one broke into two halves that do not parse, and the run was refused as one whose
    calls never finished; an answer holding one lost its `result` event.
    """

    def _gate(self, stream: str) -> tuple[int, str, dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.jsonl").write_text(stream, encoding="utf-8")
            (root / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = validate.gate(
                        answer_file=root / "events.jsonl",
                        schema_path=root / "schema.json",
                        mode="grok-events",
                        findings_out=root / "out.json",
                        provenance_out=root / "out-provenance.json",
                        prov_pairs=[],
                        prov_files={},
                        evidence=validate.Evidence(
                            events_file=root / "events.jsonl",
                            mode="grok-messages",
                            duration_s=1.0,
                        ),
                        label="ce-persona",
                    )
                except errors.AppError as exc:
                    return exc.exit_code, str(exc), {}
            record = json.loads((root / "out-provenance.json").read_text(encoding="utf-8"))
        return code, err.getvalue(), record["run_stats"]

    @LINE_BREAKS_INSIDE_JSON
    def test_a_successful_read_of_a_file_holding_a_line_break_is_counted(self, separator: str):
        report = json.loads(json.dumps(READ_FILE))
        report["FileContent"]["content"] = f"1→x = 1  # a{separator}b\n"
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_read_file",
            "content": raw_json(report),
            "is_error": False,
        }
        result = raw_json({"type": "user", "message": {"role": "user", "content": [block]}})
        assert separator in result, "the character must be raw in the line, as grok writes it"
        stream = "\n".join(
            [grok_tool_call("read_file", None), result, grok_result(structured_output=artifact())]
        )
        code, message, stats = self._gate(stream + "\n")
        assert code == 0, message
        assert (stats["local_tool_attempts"], stats["local_tool_calls"]) == (1, 1)

    @LINE_BREAKS_INSIDE_JSON
    def test_an_answer_holding_a_line_break_is_read(self, separator: str):
        title = f"a{separator}b"
        answer = raw_json(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "end_turn",
                "structured_output": artifact(finding(title=title)),
            }
        )
        assert separator in answer
        found = validate.from_grok_events(grok_tool_call() + "\n" + answer + "\n")
        assert findings_of(found)[0]["title"] == title

    @LINE_BREAKS_INSIDE_JSON
    def test_a_codex_stream_read_from_its_file_splits_at_newlines_too(self, separator: str):
        # codex's stream is read a line at a time from the file, which already splits at
        # newlines only; this holds it there.
        done = raw_json(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "cat f.py",
                    "aggregated_output": f"a{separator}b\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
        assert separator in done
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text(codex_stream(done), encoding="utf-8")
            stats = validate.run_stats("codex-items", validate.file_objects(events), None)
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1)


class TestARefusalSurvivesBeingHandedOn:
    """`validate.refused_run`: the reader's half of the vacuous-run refusal.

    The review command refuses with exit 6, or 3 when every local call failed, and keeps the
    artifact as evidence. Without this,
    `ce-persona-findings <artifact>` rendered that same dud as an ordinary listing at exit 0 —
    the package laundering its own refusal, one command later, through its own reader.
    """

    def _artifact(self, tmp: Path, stats: dict[str, Any] | None) -> Path:
        art = tmp / "adversarial-reviewer-grok.json"
        art.write_text(EMPTY_EXAMPLE, encoding="utf-8")
        if stats is not None:
            (tmp / "adversarial-reviewer-grok-provenance.json").write_text(
                json.dumps({"provider": "grok", "run_stats": stats}), encoding="utf-8"
            )
        return art

    def test_a_sidecar_recording_no_tool_calls_is_a_refusal(self):
        # Written before `local_tool_calls` existed: read by the rule it was written under.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(
                Path(tmp),
                {"tool_calls": 0, "turns": 1, "output_tokens": 151, "duration_s": 4.5},
            )
            stats = validate.refused_run(art)
        assert stats is not None
        assert (stats.tool_calls, stats.local_tool_calls) == (0, 0)
        assert (stats.turns, stats.output_tokens) == (1, 151)

    @pytest.mark.parametrize("calls", [0, 3])
    def test_a_sidecar_recording_no_local_tool_calls_is_a_refusal(self, calls: int):
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(
                Path(tmp), {"tool_calls": calls, "local_tool_calls": 0, "turns": 2}
            )
            stats = validate.refused_run(art)
        assert stats is not None
        assert (stats.tool_calls, stats.local_tool_calls, stats.turns) == (calls, 0, 2)

    @pytest.mark.parametrize("local", [3, "0", True, False, -1, 0.0, None])
    def test_a_zero_total_refuses_whatever_the_local_count_says(self, local: Any):
        # Every sidecar the reader refused before the field existed is still refused.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), {"tool_calls": 0, "local_tool_calls": local})
            stats = validate.refused_run(art)
        assert stats is not None
        assert (stats.tool_calls, stats.local_tool_calls) == (0, 0)

    @pytest.mark.parametrize(
        "stats",
        [
            {"tool_calls": 1},
            {"tool_calls": 99, "turns": 31},
            # A malformed count is not a positive reading of zero.
            {"tool_calls": "0"},
            {"tool_calls": True},
            {"tool_calls": -1},
            {"tool_calls": 0.0},
            {},
            # Present beside a nonzero total, so it is what the reading relies on, and it is
            # not a whole number.
            {"tool_calls": 3, "local_tool_calls": "0"},
            {"tool_calls": 3, "local_tool_calls": True},
            {"tool_calls": 3, "local_tool_calls": False},
            {"tool_calls": 3, "local_tool_calls": -1},
            {"tool_calls": 3, "local_tool_calls": 0.0},
            {"tool_calls": 3, "local_tool_calls": None},
            {"tool_calls": 3, "local_tool_calls": 1},
            # A zero local count beside a total that is not a count is a malformed record.
            {"tool_calls": "3", "local_tool_calls": 0},
            {"tool_calls": -1, "local_tool_calls": 0},
            {"local_tool_calls": 0},
        ],
    )
    def test_anything_short_of_a_positive_zero_renders_normally(self, stats: dict[str, Any]):
        with tempfile.TemporaryDirectory() as tmp:
            assert validate.refused_run(self._artifact(Path(tmp), stats)) is None

    def test_an_artifact_with_no_sidecar_still_renders(self):
        # The sidecar is a record, not a gate. An artifact written before this field existed,
        # or one a person assembled by hand, must not become unreadable.
        with tempfile.TemporaryDirectory() as tmp:
            assert validate.refused_run(self._artifact(Path(tmp), None)) is None

    def test_a_malformed_sidecar_still_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), {"tool_calls": 0})
            (Path(tmp) / "adversarial-reviewer-grok-provenance.json").write_text(
                "{not json", encoding="utf-8"
            )
            assert validate.refused_run(art) is None

    @pytest.mark.parametrize("sidecar", ["[]", '[{"run_stats": {"tool_calls": 0}}]', "0"])
    def test_a_sidecar_that_is_not_an_object_is_no_record(self, sidecar: str):
        # Valid JSON with no fields to read: read as a record, it raises out of every mode.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), {"tool_calls": 0})
            (Path(tmp) / "adversarial-reviewer-grok-provenance.json").write_text(
                sidecar, encoding="utf-8"
            )
            assert validate.refused_run(art) is None
            assert validate.reviewed_head(art) == "unresolved: no provenance"

    def test_the_command_refuses_every_output_mode_including_json(self):
        # --json especially. A programmatic caller is the one most likely to act on these
        # findings without a person ever reading them.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(
                Path(tmp),
                {"tool_calls": 0, "turns": 1, "output_tokens": 151, "duration_s": 4.5},
            )
            for args in (
                [str(art)],
                [str(art), "--all"],
                [str(art), "--json"],
                [str(art), "--show", "all"],
                [str(art), "--anchors", "-C", tmp],
            ):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = findings.main(args)
                assert code == 6, args
                assert out.getvalue() == "", args
                assert "no local tool calls" in err.getvalue(), args
                assert "adversarial-reviewer-grok-provenance.json" in err.getvalue(), args

    def test_the_command_refuses_a_run_that_only_searched_the_web(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), {"tool_calls": 4, "local_tool_calls": 0})
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = findings.main([str(art)])
        assert code == 6
        assert "4 tool calls (0 local)" in err.getvalue(), err.getvalue()

    def _banner(self, stats: dict[str, Any]) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), stats)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = findings.main([str(art)])
        assert out.getvalue() == ""
        return code, err.getvalue()

    # The c156 run's sidecar as 0.3.5 writes it.
    C156_SIDECAR: ClassVar[dict[str, Any]] = {
        "tool_calls": 7,
        "local_tool_calls": 0,
        "local_tool_attempts": 5,
        "turns": 1,
        "output_tokens": 734,
        "duration_s": 12.0,
    }

    def test_a_sidecar_whose_local_calls_all_failed_is_refused_as_the_exit_3_it_was(self):
        code, banner = self._banner(self.C156_SIDECAR)
        # 6 from this command either way: it has no environment status, and to its caller the
        # two refusals mean the same thing. The banner says which one the review made.
        assert code == 6, banner
        assert "none of whose local tool calls succeeded" in banner, banner
        assert "(5 local, 0 succeeded)" in banner, banner
        assert "exit 3;" in banner and "exit 6" not in banner, banner
        # A search that ran and matched nothing is refused the same way, so the banner makes
        # no claim about what was read.
        assert "was read" not in banner and "read nothing" not in banner, banner

    @pytest.mark.parametrize(
        "stats",
        [
            # 0.3.4 and 0.3.3: no `local_tool_attempts`, and a zero `local_tool_calls` meant
            # nothing local was attempted. Not reinterpreted as "attempted, none succeeded".
            {"tool_calls": 4, "local_tool_calls": 0},
            {"tool_calls": 0, "local_tool_calls": 0},
            {"tool_calls": 0},
            # 0.3.5 with nothing local attempted, and attempts that are not a count.
            {"tool_calls": 4, "local_tool_calls": 0, "local_tool_attempts": 0},
            {"tool_calls": 4, "local_tool_calls": 0, "local_tool_attempts": -1},
            {"tool_calls": 4, "local_tool_calls": 0, "local_tool_attempts": "5"},
            {"tool_calls": 4, "local_tool_calls": 0, "local_tool_attempts": True},
        ],
    )
    def test_a_sidecar_that_attempted_nothing_local_keeps_the_exit_6_banner(
        self, stats: dict[str, Any]
    ):
        code, banner = self._banner(stats)
        assert code == 6, banner
        assert "no local tool calls" in banner, banner
        assert "exit 6;" in banner, banner
        assert "succeeded" not in banner, banner

    def test_a_0_3_4_sidecar_that_counted_failed_attempts_still_renders(self):
        # THE KNOWN GAP, asserted so closing it fails loudly: 0.3.4 counted the c156 run's
        # five failed commands as local calls, and a sidecar is read by the rule it was
        # written under.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), {"tool_calls": 7, "local_tool_calls": 5})
            assert validate.refused_run(art) is None
            with contextlib.redirect_stdout(io.StringIO()):
                assert findings.main([str(art)]) == 0

    @pytest.mark.parametrize(
        "stats", [{"tool_calls": 12}, {"tool_calls": 12, "local_tool_calls": 12}]
    )
    def test_the_same_artifact_with_a_real_run_behind_it_renders(self, stats: dict[str, Any]):
        # The control for the whole class: without it every assertion above is satisfied by a
        # command that refuses everything.
        with tempfile.TemporaryDirectory() as tmp:
            art = self._artifact(Path(tmp), stats)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = findings.main([str(art)])
        assert code == 0
        assert "no findings" in out.getvalue()


class TestAssets:
    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.assets = Path(self.tmp.name)
        (self.assets / "personas").mkdir()
        (self.assets / "personas" / "adversarial-reviewer.md").write_text(
            "# Adversarial\n\nReturn findings matching the findings schema.\n", encoding="utf-8"
        )
        (self.assets / "personas" / "agent-native-reviewer.md").write_text(
            "# Agent Native\n\n## Output Format\n\nA markdown table.\n", encoding="utf-8"
        )
        # A findings-capable brief OUTSIDE personas/, so the traversal case below is not
        # vacuous: without it `../outside` fails as "unknown persona" whether or not the
        # name is validated.
        (self.assets / "outside.md").write_text(
            "# Outside\n\nReturn findings matching the findings schema.\n", encoding="utf-8"
        )
        (self.assets / "findings-schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def test_a_capable_persona_resolves(self):
        name, brief = assets.resolve_persona(self.assets, "adversarial-reviewer")
        assert name == "adversarial-reviewer"
        assert brief.is_file()

    def test_the_ce_prefix_and_md_suffix_are_accepted(self):
        for spelling in ("ce-adversarial-reviewer", "adversarial-reviewer.md"):
            name, _ = assets.resolve_persona(self.assets, spelling)
            assert name == "adversarial-reviewer"

    def test_a_markdown_only_persona_is_refused(self):
        with pytest.raises(assets.UsageError) as caught:
            assets.resolve_persona(self.assets, "agent-native-reviewer")
        assert "markdown output format" in str(caught.value)

    def test_a_persona_name_must_be_a_bare_brief_name(self):
        # The name reaches every artifact path; a separator would resolve a brief from
        # outside personas/ and place artifacts outside the run directory.
        for hostile in ("../outside", "/etc/passwd", ".hidden", "a/b", "", "."):
            with pytest.raises(assets.UsageError) as caught:
                assets.resolve_persona(self.assets, hostile)
            assert "bare brief name" in str(caught.value), hostile

    def test_the_traversal_fixture_would_otherwise_resolve(self):
        # Control for the test above: assets/outside.md really is a findings-capable brief,
        # so `../outside` is refused by the NAME check and not merely by being absent.
        assert assets.emits_findings(self.assets / "outside.md")
        assert (self.assets / "personas" / ".." / "outside.md").is_file()

    def _roots(self, *versions: str) -> list[Path]:
        made: list[Path] = []
        for v in versions:
            path = (
                Path(self.tmp.name) / f"compound-engineering/{v}/skills/ce-code-review/references"
            )
            path.mkdir(parents=True, exist_ok=True)
            made.append(path)
        return made

    def test_plugin_roots_sort_numerically_not_lexically(self):
        # Lexical order puts 3.9 after 3.13, silently pinning an old brief set.
        made = self._roots("3.9.0", "3.13.1", "3.10.0")
        assert sorted(made, key=assets.version_key)[-1] == made[1]

    def test_a_release_outranks_its_own_prerelease_regardless_of_input_order(self):
        # Dropping non-numeric components makes these tie, and `sorted` is stable — so the
        # winner would be whichever order the filesystem happened to yield. That decides
        # which briefs EVERY review runs against.
        made = self._roots("3.22.0", "3.22.0-rc1")
        release, prerelease = made[0], made[1]
        for order in ([release, prerelease], [prerelease, release]):
            assert sorted(order, key=assets.version_key)[-1] == release, order

    def test_a_digit_like_non_integer_version_does_not_crash(self):
        # `'²'.isdigit()` is True while `int('²')` raises, so the obvious parse escapes as a
        # traceback instead of an exit code.
        (made,) = self._roots("3.²")
        assets.version_key(made)

    def test_the_prompt_carries_brief_rubric_boundary_and_the_strict_output_clause(self):
        prompt = assets.build_prompt(
            provider=providers.GROK,
            persona="adversarial-reviewer",
            brief=self.assets / "personas" / "adversarial-reviewer.md",
            assets=self.assets,
            schema_text=json.dumps(SCHEMA),
            base="HEAD~1",
            context="extra context here",
        )
        assert "Return findings matching the findings schema." in prompt
        assert "Anchors 0 and 25 mean SUPPRESS" in prompt  # the fallback rubric
        assert "extra context here" in prompt
        assert "git diff HEAD~1..HEAD" in prompt
        # The strict output clause is what lets the gate stay strict.
        assert "exactly one JSON object" in prompt
        assert prompt.rstrip().endswith(assets.BOUNDARY)

    def test_the_plugin_rubric_replaces_the_fallback_when_present(self):
        (self.assets / "subagent-template.md").write_text(
            "**Schema conformance** — every finding carries file, line, evidence.\n"
            "RUBRIC-MARKER-FROM-TEMPLATE\n"
            "Example of a schema-valid finding:\n",
            encoding="utf-8",
        )
        text = assets.rubric(self.assets)
        assert "RUBRIC-MARKER-FROM-TEMPLATE" in text
        assert "Anchors 0 and 25" not in text


class TestProviders:
    def _inv(self) -> providers.Invocation:
        return providers.Invocation(
            model="m",
            effort="xhigh",
            repo=Path("/repo"),
            prompt_file=Path("/tmp/p.md"),
            schema_text="{}",
            last_file=Path("/tmp/last.json"),
        )

    def test_grok_runs_read_only_and_asks_for_a_streamed_schema(self):
        argv = providers.GROK.argv(self._inv())
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv[argv.index("--output-format") + 1] == "streaming-messages-json"
        assert "--json-schema" in argv

    def test_codex_runs_read_only_and_writes_only_the_final_message(self):
        argv = providers.CODEX.argv(self._inv())
        assert argv[argv.index("-s") + 1] == "read-only"
        assert argv[argv.index("-o") + 1] == "/tmp/last.json"
        assert "--output-schema" not in argv
        assert argv[-1] == "-"

    def test_codex_omits_the_effort_flag_when_it_is_unset(self):
        inv = providers.Invocation(
            model="m",
            effort="",
            repo=Path("/r"),
            prompt_file=Path("/p"),
            schema_text="{}",
            last_file=None,
        )
        assert "model_reasoning_effort" not in " ".join(providers.CODEX.argv(inv))

    def test_every_provider_is_covered_by_the_flag_probe_shape(self):
        argv = flags.reference_argv()
        assert flags.valueless_flags(argv), "an empty probe set would cover nothing"
        assert "--verbatim" in flags.valueless_flags(argv)
        assert "--json-schema" in flags.valued_flags(argv)


class TestBudget:
    def test_over_budget_counts_prompt_plus_diff(self):
        budget = runner.Budget(prompt_bytes=1200, diff_bytes=92_000, limit_tokens=2_000)
        assert budget.tokens == (1200 + 92_000) // 4
        assert budget.over

    def test_the_prompt_alone_is_not_enough_to_fire(self):
        assert not runner.Budget(prompt_bytes=1200, diff_bytes=0, limit_tokens=2_000).over


class TestFindingsRetrieval:
    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "artifact.json")
        # The fixture carries the tier-2 fields. Without them `render_detail` could collapse
        # into `render_row` — no why_it_matters, no evidence array, no suggested fix, no
        # routing — and every test here stayed green, which is to say the two tiers that are
        # this module's entire product were unasserted.
        Path(self.path).write_text(
            json.dumps(
                artifact(
                    finding(title="a p2", severity="P2", file="b.py", line=2),
                    finding(
                        title="a p0",
                        severity="P0",
                        file="a.py",
                        line=1,
                        confidence=100,
                        first_evidence="a.py:1 -- the motivating line",
                        evidence=["a.py:1 -- the motivating line", "a.py:9 -- corroboration"],
                        why_it_matters="Callers read a stale value and bill the wrong account.",
                        suggested_fix="Guard the lookup the way b.py:2 already does.",
                        autofix_class="gated_auto",
                        owner="downstream-resolver",
                    ),
                )
            ),
            encoding="utf-8",
        )

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_default_tier_shows_p0_p1_and_hides_the_rest(self):
        code, out, _ = self._run(self.path)
        assert code == 0
        assert "a p0" in out
        assert "a p2" not in out
        assert "hidden" in out

    def test_all_shows_every_severity(self):
        _, out, _ = self._run(self.path, "--all")
        assert "a p0" in out
        assert "a p2" in out

    def test_list_is_accepted(self):
        code, out, _ = self._run(self.path, "--list")
        assert code == 0
        assert "a p0" in out

    def test_show_n_is_the_finding_numbered_n_in_the_listing(self):
        _, listing, _ = self._run(self.path, "--all")
        rows = [ln for ln in listing.splitlines() if ln.startswith("#")]
        assert len(rows) == 2
        for row in rows:
            number = int(row.split()[0].lstrip("#"))
            title = row.split(" — ", 1)[1].split(" (confidence")[0]
            _, detail, _ = self._run(self.path, "--show", str(number))
            assert title in detail

    def test_tier_1_carries_the_quoted_line_and_nothing_heavier(self):
        # The quoted line is what makes a title trustworthy without the evidence array; the
        # tier is worthless if it renders a bare title, and over-costed if it renders detail.
        _, out, _ = self._run(self.path, "--all")
        assert "a.py:1 -- the motivating line" in out
        assert "(confidence 100)" in out
        assert "why:" not in out
        assert "Guard the lookup" not in out
        assert "a.py:9 -- corroboration" not in out

    def test_tier_2_carries_why_evidence_fix_and_routing(self):
        _, out, _ = self._run(self.path, "--show", "2")
        assert "why: Callers read a stale value" in out
        assert "fix: Guard the lookup" in out
        assert "a.py:9 -- corroboration" in out  # the FULL evidence array, not just [0]
        assert "autofix_class=gated_auto" in out
        assert "owner=downstream-resolver" in out

    def test_the_listing_is_ordered_most_severe_first(self):
        _, out, _ = self._run(self.path, "--all")
        rows = [ln for ln in out.splitlines() if ln.startswith("#")]
        severities = [ln.split()[1] for ln in rows]
        assert severities == ["P0", "P2"], out

    def test_rendered_output_is_fenced_as_untrusted(self):
        # Everything rendered here was written by a model and is being handed to another
        # agent as its input.
        _, out, _ = self._run(self.path, "--all")
        assert "BEGIN UNTRUSTED MODEL OUTPUT" in out
        assert "END UNTRUSTED MODEL OUTPUT" in out
        _, detail, _ = self._run(self.path, "--show", "1")
        assert "BEGIN UNTRUSTED MODEL OUTPUT" in detail

    def test_the_fence_nonce_differs_between_runs(self):
        # A fixed delimiter could be closed by the text inside it.
        _, first, _ = self._run(self.path, "--all")
        _, second, _ = self._run(self.path, "--all")
        assert first != second

    def test_json_is_not_fenced_and_round_trips(self):
        code, out, _ = self._run(self.path, "--json")
        assert code == 0
        assert "UNTRUSTED" not in out
        assert json.loads(out) == json.loads(Path(self.path).read_text(encoding="utf-8"))

    def test_a_bad_artifact_is_a_data_error_not_a_usage_error(self):
        # The same vocabulary the review commands use: a caller must be able to tell "I
        # asked for this wrongly" from "the artifact is not usable" without reading stderr.
        #
        # LITERAL 1, not findings.EXIT_DATA. These numbers are a published contract, and an
        # assertion written against the module's own constant moves with it -- mutation
        # testing caught exactly that here: redefining EXIT_USAGE to 1 left the suite green.
        other = str(Path(self.tmp.name) / "other.json")
        Path(other).write_text('{"hello": "world"}', encoding="utf-8")
        code, _, err = self._run(other)
        assert code == 1
        assert "not a findings or verdicts artifact" in err

        code, _, _ = self._run(str(Path(self.tmp.name) / "nope.json"))
        assert code == 1

    # ARTIFACT stands in for `self.path`, which does not exist until setup_method runs and so
    # cannot appear in a decorator evaluated at import time.
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ((), "artifact path is required"),
            (("--nope",), "unknown option"),
            # A trailing --show once fell through to the unknown-option arm and reported a
            # documented flag as unknown.
            ((ARTIFACT, "--show"), "wants a finding number"),
            ((ARTIFACT, "--show", "x"), "wants a finding number"),
            # Only the lowercase word is the sentinel; anything else is still a number.
            ((ARTIFACT, "--show", "ALL"), "wants a finding number"),
            ((ARTIFACT, "--show", "99"), "no finding #99"),
        ],
    )
    def test_usage_mistakes_are_usage_errors(self, args: tuple[str, ...], expected: str):
        code, _, err = self._run(*(self.path if a is ARTIFACT else a for a in args))
        assert code == 2, err
        assert expected in err
        assert "usage:" in err

    def test_help_exits_zero_and_documents_every_tier(self):
        # A documented flag that exits non-zero with "unknown option" is how an agent
        # concludes a tool is broken; --list already had that problem once.
        for flag in ("-h", "--help"):
            code, out, _ = self._run(flag)
            assert code == 0
            for token in ("--list", "--all", "--show", "--json", "UNTRUSTED MODEL OUTPUT"):
                assert token in out, flag

    def test_the_fence_stays_cheap(self):
        # It wraps every tier-1 and tier-2 read, and tier 1 budgets ~20 tokens a finding.
        _, out, _ = self._run(self.path, "--show", "1")
        overhead = [ln for ln in out.splitlines() if "UNTRUSTED MODEL OUTPUT" in ln]
        assert len(overhead) == 2, "the fence should cost two lines, not a paragraph"

    def test_show_all_is_every_tier_2_render_inside_one_fence(self):
        # One fence, not one per finding: the point of the mode is that a caller relaying
        # every finding pays one invocation and one fence rather than one of each per finding.
        code, out, err = self._run(self.path, "--show", "all")
        assert code == 0, err
        lines = out.splitlines()
        assert len([ln for ln in lines if "UNTRUSTED MODEL OUTPUT" in ln]) == 2, out
        assert lines[0].startswith("--- BEGIN UNTRUSTED MODEL OUTPUT"), out
        assert lines[-1].startswith("--- END UNTRUSTED MODEL OUTPUT"), out
        body = "\n".join(lines[2:-1])
        # Byte-for-byte the renders `--show N` produces, joined by the separator line, so
        # the two modes cannot come to render a finding differently.
        singles: list[str] = []
        for n in (1, 2):
            _, one, _ = self._run(self.path, "--show", str(n))
            singles.append("\n".join(one.splitlines()[2:-1]))
        assert body == f"\n{findings.SHOW_ALL_SEPARATOR}\n".join(singles), out
        assert "a p2" in body and "why: Callers read a stale value" in body

    def test_show_all_is_in_number_order_not_severity_order(self):
        # #1 is the P2 on disk. `#` order is what makes the entries line up with the numbers
        # a caller already holds; severity order would reshuffle them.
        _, out, _ = self._run(self.path, "--show", "all")
        heads = [ln.split()[0] for ln in out.splitlines() if ln.startswith("#")]
        assert heads == ["#1", "#2"], out
        assert out.count(f"\n{findings.SHOW_ALL_SEPARATOR}\n") == 1, out

    def test_show_all_on_an_empty_artifact_says_so_as_all_does(self):
        Path(self.path).write_text(json.dumps(artifact()), encoding="utf-8")
        code, out, _ = self._run(self.path, "--show", "all")
        _, listed, _ = self._run(self.path, "--all")
        assert code == 0
        assert out == listed == "no findings\n"

    def test_json_outranks_show_all(self):
        code, out, _ = self._run(self.path, "--show", "all", "--json")
        assert code == 0
        assert json.loads(out) == json.loads(Path(self.path).read_text(encoding="utf-8"))

    def test_show_all_wins_over_the_listing_modes(self):
        _, out, _ = self._run(self.path, "--list", "--show", "all")
        assert "why: Callers read a stale value" in out
        assert "hidden" not in out


class TestProvenance:
    def test_paths_with_quotes_and_backslashes_round_trip(self):
        # Structured data, written by json.dump rather than interpolated into a heredoc,
        # which produced invalid JSON the moment a path held a quote or a backslash.
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / 'we"ird\\name.md'
            odd.write_text("brief", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validate.write_provenance(
                out, ["provider=grok", "base_ref="], {"persona": str(odd)}, STATS
            )
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["provider"] == "grok"
        assert record["base_ref"] == ""
        assert record["persona_file"] == str(odd)
        assert len(record["persona_sha256"]) == 64

    def test_an_unreadable_asset_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, [], {"schema": str(Path(tmp) / "gone.json")}, STATS)
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["schema_sha256"].startswith("unreadable:")

    def test_a_precomputed_digest_is_recorded_instead_of_re_reading_the_path(self):
        # Provenance attests what the run USED. Re-reading the path at write time attests
        # whatever is there afterwards, so a file replaced mid-run is recorded as the one
        # the model was given.
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "validator-input.json"
            batch.write_text("replaced after the run", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validate.write_provenance(
                out, [], {"batch": str(batch)}, STATS, {"batch": "the-bytes-that-were-read"}
            )
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["batch_sha256"] == "the-bytes-that-were-read"
        assert record["batch_file"] == str(batch)

    def test_a_file_with_no_precomputed_digest_is_still_hashed_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "persona.md"
            asset.write_text("brief", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, [], {"persona": str(asset)}, STATS, {"batch": "x"})
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["persona_sha256"] == hashlib.sha256(b"brief").hexdigest()

    def test_what_the_run_did_is_recorded_beside_what_produced_it(self):
        # The local counts decide the exit status, so they have to be auditable after the
        # fact: a refusal a caller cannot check is one it has to take on trust.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            stats = validate.RunStats(2, 0, 1, 151, 4.5, local_tool_attempts=2)
            validate.write_provenance(out, [], {}, stats)
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["run_stats"] == {
            "tool_calls": 2,
            "local_tool_calls": 0,
            "local_tool_attempts": 2,
            "turns": 1,
            "output_tokens": 151,
            "duration_s": 4.5,
        }


class TestSettings:
    """The environment boundary. Parsed once, validated wholly, frozen."""

    def test_the_defaults_stand_when_nothing_is_set(self):
        s = config.Settings.from_env({})
        assert s.idle_secs == config.DEFAULT_IDLE_SECS
        assert s.hard_secs == config.DEFAULT_HARD_SECS
        assert s.max_prompt_tokens == config.DEFAULT_MAX_TOKENS
        assert s.run_dir is None
        assert s.assets_override is None

    @pytest.mark.parametrize("name", ["CE_PERSONA_IDLE_SECS", "CE_PERSONA_HARD_SECS"])
    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "abc", ""])
    def test_a_timeout_is_strictly_positive_and_finite(self, name: str, value: str):
        # "" is in the table as a CONTROL that is expected to pass: an unset variable takes
        # the default, and a parser that refused everything would satisfy every other row
        # here while breaking every real invocation.
        if value == "":
            assert config.Settings.from_env({name: value}).idle_secs > 0
            return
        with pytest.raises(errors.UsageError) as caught:
            config.Settings.from_env({name: value})
        assert name in str(caught.value), "the message must name the variable to be actionable"

    def test_zero_is_refused_with_the_reason_spelled_out(self):
        # Singled out because it is the one value a person types on purpose, meaning "do not
        # wait", and it used to mean "do not watch".
        with pytest.raises(errors.UsageError) as caught:
            config.Settings.from_env({"CE_PERSONA_IDLE_SECS": "0"})
        assert "greater than zero" in str(caught.value)

    def test_an_unrecognised_setting_under_the_prefix_is_refused(self):
        with pytest.raises(errors.UsageError) as caught:
            config.Settings.from_env({"CE_PERSONA_IDEL_SECS": "30"})
        assert "CE_PERSONA_IDEL_SECS" in str(caught.value)
        assert "CE_PERSONA_IDLE_SECS" in str(caught.value)

    def test_a_variable_outside_the_prefix_is_none_of_this_module_s_business(self):
        # The control for the check above. Forbidding everything unknown would break every
        # real environment, which carries PATH, HOME and hundreds of others.
        config.Settings.from_env({"PATH": "/usr/bin", "EDITOR": "vi", "CE_REVIEW_ASSETS": "/a"})

    def test_an_unknown_home_directory_is_a_usage_error(self):
        with pytest.raises(errors.UsageError):
            config.Settings.from_env({"CE_PERSONA_RUN_DIR": "~nosuchuser0987/run"})

    def test_settings_are_frozen(self):
        s = config.Settings.from_env({})
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is not public
            s.idle_secs = 1.0  # type: ignore[misc]


class TestErrorVocabulary:
    def test_every_error_class_carries_an_exit_code(self):
        # The point of the hierarchy: a class cannot be added without choosing a status, so
        # `main`'s mapping is total by construction rather than by six except-blocks.
        for cls in _app_error_classes():
            assert isinstance(getattr(cls, "exit_code", None), int), f"{cls.__name__} has none"

    def test_no_two_error_kinds_share_a_status(self):
        # MissingTool deliberately shares EnvError's, so compare the classes that DEFINE one.
        defined = [c for c in _app_error_classes() if "exit_code" in c.__dict__]
        codes = [c.exit_code for c in defined]
        assert len(set(codes)) == len(codes), sorted((c.__name__, c.exit_code) for c in defined)

    def test_the_base_class_has_no_code_of_its_own(self):
        # So a subclass that forgets to set one fails where it is used rather than silently
        # reporting whatever the base happened to say.
        assert "exit_code" not in errors.AppError.__dict__

    def test_the_help_exit_table_is_rendered_from_the_error_classes(self):
        # Not "the numbers appear somewhere in --help", which a hand-written table also
        # satisfies. The rendered block must be present VERBATIM, so the help text cannot
        # carry a second copy that drifts.
        rendered = errors.render_exit_table(providers.GROK.binary)
        assert rendered in _help_text(), rendered

    def test_the_rendered_table_names_every_status_the_cli_can_return(self):
        rendered = errors.render_exit_table("grok")
        for code in (0, *(c.exit_code for c in _app_error_classes())):
            assert f"  {code} " in rendered or f"  {code}  " in rendered, f"exit {code} missing"

    def test_the_table_substitutes_the_provider_binary(self):
        # The control for the {runner} placeholder: an unsubstituted table would still
        # contain every number and pass the two checks above.
        assert "{runner}" not in errors.render_exit_table("codex")
        assert "codex itself exited non-zero" in errors.render_exit_table("codex")

    def test_the_cli_constants_are_the_class_attributes(self):
        assert errors.GateError.exit_code == cli.EXIT_GATE
        assert errors.UsageError.exit_code == cli.EXIT_USAGE
        assert errors.EnvError.exit_code == cli.EXIT_ENV
        assert errors.RunnerError.exit_code == cli.EXIT_RUNNER
        assert errors.RunTimeout.exit_code == cli.EXIT_TIMEOUT
        assert errors.VacuousRun.exit_code == cli.EXIT_VACUOUS
        assert errors.BudgetError.exit_code == cli.EXIT_BUDGET
        # LITERALS, because the codes are the published contract. An assertion written only
        # against the module's own constants moves with them: mutation testing caught exactly
        # that elsewhere, where redefining EXIT_USAGE to 1 left the suite green.
        assert (cli.EXIT_GATE, cli.EXIT_USAGE, cli.EXIT_ENV) == (1, 2, 3)
        assert (cli.EXIT_RUNNER, cli.EXIT_TIMEOUT, cli.EXIT_VACUOUS) == (4, 5, 6)
        assert cli.EXIT_BUDGET == 78


class TestTheEnvironmentIsReadInOnePlace:
    """A rule you can check with one command beats a rule you have to remember.

    The same structural move that keeps `subprocess.PIPE` greppably absent from runner.py.
    Scattered `os.environ.get` calls meant a malformed timeout was discovered three quarters
    of the way through `main`, and that a variable read twice could be validated once.
    """

    ALLOWED = {
        "config.py",  # the boundary itself
        # Runs before argument parsing and therefore before Settings exists — if the gate is
        # not this package's gate, nothing it goes on to report means anything.
        "cli.py",
        # expanduser("~") for the plugin cache location, which is not a setting.
        "assets.py",
        # Builds the child environment for the provider CLI; reads no setting of its own.
        "runner.py",
    }

    def test_no_module_outside_the_boundary_reads_the_environment(self):
        package = Path(validate.__file__).parent
        offenders: dict[str, list[str]] = {}
        for path in sorted(package.glob("*.py")):
            if path.name in self.ALLOWED:
                continue
            hits = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if "os.environ" in line or "getenv" in line
            ]
            if hits:
                offenders[path.name] = hits
        assert offenders == {}, f"read the environment outside config.py: {offenders}"

    @staticmethod
    def _setting_reads(text: str) -> list[str]:
        """Lines that both name a CE_PERSONA_* setting and read the environment.

        Naming one is fine and often required — runner.py puts CE_PERSONA_IDLE_SECS in its
        timeout message precisely so the error is actionable, and the process suite asserts
        that it does. READING one outside the boundary is the regression.
        """
        return [
            line.strip()
            for line in text.splitlines()
            if ("os.environ" in line or "getenv" in line)
            and any(var in line for var in config.KNOWN_VARS)
        ]

    def test_the_detector_matches_the_boundary_itself(self):
        # The control, and it is not ceremony: a grep-shaped test whose pattern matches
        # nothing passes over any codebase at all, which is the exact defect this repo keeps
        # finding. config.py is where settings ARE read, so the detector must fire on it —
        # otherwise the test below is green for every file for the wrong reason.
        boundary = (Path(validate.__file__).parent / "config.py").read_text(encoding="utf-8")
        hits = self._setting_reads(boundary + '\nos.environ.get("CE_PERSONA_IDLE_SECS")\n')
        assert hits, "the detector finds no setting read even in an explicit one"

    def test_no_module_outside_the_boundary_reads_a_CE_PERSONA_SETTING(self):
        package = Path(validate.__file__).parent
        offenders = {
            path.name: hits
            for path in sorted(package.glob("*.py"))
            if path.name != "config.py"
            and (hits := self._setting_reads(path.read_text(encoding="utf-8")))
        }
        assert offenders == {}, f"read a CE_PERSONA_* setting outside config.py: {offenders}"


class TestGateReadsItsSchemaDefensively:
    """findings-schema.json belongs to the compound-engineering plugin, not to this package.

    It is read fresh every run from a directory this package does not own, so its shape is an
    input, not an invariant. `gate()` runs it through `_as_object` for exactly that reason —
    and that guard had no test until a property test called `validate()` directly and turned
    up the four tracebacks it prevents. The crash is not reachable through `gate()`, so the
    property was wrong and was narrowed; the missing guard test is the real finding.
    """

    def _gate(self, tmp: Path, schema_text: str) -> tuple[int, str]:
        (tmp / "answer.txt").write_text(EMPTY_EXAMPLE, encoding="utf-8")
        (tmp / "schema.json").write_text(schema_text, encoding="utf-8")
        (tmp / "events.jsonl").write_text(CODEX_ONE_CALL, encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = validate.gate(
                answer_file=tmp / "answer.txt",
                schema_path=tmp / "schema.json",
                mode="object",
                findings_out=tmp / "out.json",
                provenance_out=tmp / "prov.json",
                prov_pairs=[],
                prov_files={},
                # A run that DID inspect something, so these stay tests of the SCHEMA path:
                # with no tool call the gate refuses before it reaches any of this.
                evidence=validate.Evidence(
                    events_file=tmp / "events.jsonl", mode="codex-items", duration_s=1.0
                ),
                label="ce-persona",
            )
        return code, err.getvalue()

    def test_the_control_a_well_formed_schema_passes(self):
        # Without this the two cases below would pass against a gate that refuses every
        # schema, which is the same "cannot fail" defect in the opposite direction.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, err = self._gate(root, json.dumps(SCHEMA))
            assert code == 0, err
            assert (root / "out.json").is_file(), "a passing gate must write its artifact"

    @pytest.mark.parametrize("body", ["[]", "null", '"a string"', "7", "[{}]"])
    def test_a_schema_that_is_not_an_object_is_refused_legibly(self, body: str):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self._gate(Path(tmp), body)
        assert code == 1, f"{body} was not refused"
        assert "findings schema" in err, err

    def test_an_unparseable_schema_is_refused_legibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self._gate(Path(tmp), "{not json")
        assert code == 1
        assert "cannot read findings schema" in err, err


# ---------------------------------------------------------------------------------------
# Property tests.
#
# Every regression above was an input shape nobody pictured: a ["integer","null"] union that
# admitted a boolean, a terminal event missing `stop_reason` entirely, two healthy `result`
# events, a non-empty object followed by the brief's empty example, tuple-form `items`.
# Reviewers found those by hand, one at a time, over several rounds. The generators below
# produce that family mechanically, so the next member does not need a reviewer.
#
# derandomize + database=None deliberately: this suite runs 20-odd times inside the mutation
# harness, and a property test that passes on some seeds and fails on others is
# indistinguishable there from a guard firing. A flaky PASS is worse still -- it lets a
# reverted fix be recorded as killed. Determinism is worth more here than the extra shapes a
# random seed would reach over time.
PROPERTY = settings(max_examples=150, deadline=None, derandomize=True, database=None)

json_values: st.SearchStrategy[Any] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**6), max_value=10**6)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=6,
)

# Text a runner could plausibly hand the gate. The last two arms are the shapes that
# produced both P0s: an answer with prose around it, and several objects in one message.
gate_text: st.SearchStrategy[str] = st.one_of(
    st.text(max_size=64),
    json_values.map(json.dumps),
    st.sampled_from([EMPTY_EXAMPLE, json.dumps(artifact(finding()))]),
    st.tuples(st.text(max_size=24), st.sampled_from([EMPTY_EXAMPLE, "{}"])).map("".join),
    st.lists(
        st.sampled_from([EMPTY_EXAMPLE, json.dumps(artifact(finding())), "{}"]),
        min_size=2,
        max_size=3,
    ).map("\n\n".join),
)


# Keys drawn from the real vocabulary as well as arbitrary text, so the generator reaches
# the enforcement code rather than bouncing off the unknown-keyword check every time.
schema_keys = st.sampled_from(
    sorted(validate.ENFORCED_KEYWORDS | validate.IGNORED_KEYWORDS)
) | st.text(max_size=6)
schema_shapes: st.SearchStrategy[dict[str, Any]] = st.dictionaries(
    schema_keys, json_values, max_size=4
) | st.builds(
    # The real schema with ONE property spec replaced by an arbitrary keyword dict. Without
    # this arm the generator almost always trips the "no properties object" check and returns
    # before reaching any per-finding rule — measured: reverting the tuple-form-items guard
    # and the unknown-keyword guard both survived a property-only run until this was added.
    schema_with,
    st.sampled_from(["title", "line", "evidence", "severity", "loc"]),
    st.dictionaries(schema_keys, json_values, max_size=3),
)


def _artifact_of(items: list[dict[str, Any]]) -> dict[str, Any]:
    """A named function rather than a lambda: strict mode cannot infer a lambda's parameter."""
    return artifact(*items)


artifact_shapes: st.SearchStrategy[dict[str, Any]] = st.dictionaries(
    st.sampled_from(["reviewer", "findings", "residual_risks", "testing_gaps"])
    | st.text(max_size=6),
    json_values,
    max_size=4,
) | st.builds(_artifact_of, st.lists(st.just(finding()), max_size=2))


def _healthy_event() -> dict[str, Any]:
    """The shape a completed, schema-constrained grok run really emits."""
    return {
        "type": "result",
        "is_error": False,
        "subtype": "success",
        "stop_reason": "end_turn",
        "structured_output": artifact(finding()),
    }


def _result_event() -> st.SearchStrategy[dict[str, Any]]:
    """A terminal event with each status field independently present, absent, or mistyped.

    Absence and mistyping are separate arms because the gate failed OPEN on each of them at
    different times, and a strategy that only ever omitted fields would have missed the
    `"is_error": "false"` case.
    """
    return st.fixed_dictionaries(
        {"type": st.just("result")},
        optional={
            "is_error": st.booleans() | st.sampled_from(["false", 0, None]),
            "subtype": st.sampled_from(["success", "error_max_turns", 3, None]),
            "stop_reason": st.sampled_from(["end_turn", "max_tokens", "refusal", ["x"], None]),
            "structured_output": st.sampled_from(
                [artifact(), artifact(finding()), {"oops": 1}, []]
            ),
            "result": st.sampled_from([EMPTY_EXAMPLE, "I gave up.", ""]),
        },
    )


class TestGateIsFailClosed:
    """The one property the whole package rests on: the gate never passes by accident.

    Each property here was checked against a reverted guard before being believed, the same
    way the mutation table is. Ten of the twelve reversions swept die on these tests alone.
    Two do NOT and are pinned only by the named tests above — tuple-form `items`, and the
    per-finding `required` loop. Both need a realistic artifact and a realistic schema to
    line up at once, which a joint generator reaches too rarely to rely on. Recorded so this
    class is not read as blanket coverage: `tests/test_mutations.py` is what holds those two.
    """

    @PROPERTY
    @given(text=gate_text)
    def test_object_mode_accepts_only_text_that_is_WHOLLY_one_json_object(self, text: str):
        # Two claims, and the second is the one with teeth. Refusing or returning findings
        # (never a traceback) is the weaker half: it holds even for a gate that hunts for an
        # object inside prose, which is precisely the loosening that produced both P0s. So
        # acceptance must also mean the whole message WAS the answer -- if json.loads cannot
        # read the same text the gate accepted, the gate validated something it found rather
        # than something it was sent.
        try:
            art = validate.from_object_file(text)
        except validate.GateError:
            return
        assert isinstance(art.get("findings"), list)
        assert json.loads(text) == art, "the gate answered from a fragment of the message"

    @PROPERTY
    @given(
        # The healthy arm is drawn explicitly and often. Composing terminal fields
        # independently makes a fully-healthy event rare, and a stream is only ACCEPTED when
        # its first event is healthy -- so a strategy without this arm generates thousands of
        # streams the gate rejects for some other reason and never tests the count at all.
        # Verified: without it, reverting the multiple-events guard leaves this test green.
        events=st.lists(st.just(_healthy_event()) | _result_event(), max_size=3),
        noise=st.lists(st.text(max_size=16), max_size=2),
    )
    def test_grok_mode_accepts_only_a_stream_with_exactly_one_result_event(
        self, events: list[dict[str, Any]], noise: list[str]
    ):
        # The invariant, stated without re-implementing the status rules: whatever else the
        # gate checks, a stream carrying zero or several verdicts has no answer to give. Two
        # HEALTHY events pass every status check individually, so nothing but a count can
        # reject them -- and "last wins" silently let the second one overwrite a real review.
        lines = [json.dumps(e) for e in events] + [n for n in noise if "result" not in n]
        try:
            art = validate.from_grok_events("\n".join(lines) + "\n")
        except validate.GateError:
            return
        assert len(events) == 1, f"accepted a stream with {len(events)} result events"
        assert isinstance(art.get("findings"), list)

    @PROPERTY
    @given(found=artifact_shapes, schema=schema_shapes)
    def test_validate_refuses_rather_than_crashing_on_any_nested_shape(
        self, found: dict[str, Any], schema: dict[str, Any]
    ):
        # Both arguments are objects, because that is the contract: `gate()` runs the schema
        # file through `_as_object` and `found` through `extract`, so neither can be a scalar
        # here. Generating scalars instead just proves the type annotations -- the real risk
        # is NESTED, since the schema belongs to the compound-engineering plugin and is free
        # to grow keywords and nest them anywhere. Every hand-found bug in this validator was
        # one level down: tuple-form `items`, a property spec that is a string, `required`
        # inside an object spec.
        try:
            validate.validate(found, schema)
        except validate.GateError:
            return
        assert isinstance(found.get("findings"), list)

    @PROPERTY
    @given(mode=st.text(max_size=16), text=gate_text)
    def test_extract_never_dispatches_to_a_mode_it_does_not_have(self, mode: str, text: str):
        try:
            validate.extract(mode, text)
        except validate.GateError:
            return
        assert mode in ("object", "grok-events"), f"{mode!r} was dispatched somewhere"

    @PROPERTY
    @given(
        field=st.sampled_from(["title", "line", "evidence", "severity"]),
        spec=st.dictionaries(schema_keys, json_values, min_size=1, max_size=3),
        value=json_values,
    )
    def test_a_schema_keyword_is_either_enforced_or_refused_never_skipped(
        self, field: str, spec: dict[str, Any], value: Any
    ):
        # Generalizes the hand-written anyOf/oneOf/$ref/const case over the whole keyword
        # vocabulary, including keywords the plugin has not invented yet. Silently ignoring
        # one certifies a review against rules nobody checked -- so acceptance has to mean
        # every keyword present was one of the two declared sets.
        try:
            validate.validate(artifact(finding(**{field: value})), schema_with(field, spec))
        except validate.GateError:
            return
        unknown = set(spec) - validate.ENFORCED_KEYWORDS - validate.IGNORED_KEYWORDS
        assert not unknown, f"accepted a schema using {sorted(unknown)} without enforcing it"

    @PROPERTY
    @given(
        names=st.lists(
            st.sampled_from(sorted(validate.JSON_TYPES)), min_size=1, max_size=3, unique=True
        )
    )
    def test_a_boolean_passes_only_where_the_schema_actually_says_boolean(self, names: list[str]):
        # `True` satisfies isinstance(x, int), so every union pairing a numeric type with
        # anything else has to keep the bool carve-out. Stated over all unions rather than
        # the one ["integer","null"] pair a reviewer happened to try.
        try:
            validate.validate(artifact(finding(line=True)), schema_with("line", {"type": names}))
        except validate.GateError:
            return
        assert "boolean" in names, f"True passed a field typed {names}"


# The keys a grok report is judged by. Payloads never carry them, so only an arm below or a
# spoiler decides them. `MultiResult` is among them: a batch poll counts when any one child
# worked, a different rule, held by its own table in `TestACallCountsOnlyIfItSucceeded`.
_GROK_JUDGED_KEYS = frozenset({"status", "exit_code", "Result", "MultiResult", "type"})
_grok_payload: st.SearchStrategy[dict[str, Any]] = st.dictionaries(
    st.text(max_size=8).filter(lambda key: key not in _GROK_JUDGED_KEYS), json_values, max_size=3
)
_grok_task_report: st.SearchStrategy[dict[str, Any]] = st.fixed_dictionaries(
    {"status": st.just("completed"), "exit_code": st.just(0), "output": json_values}
)

# What an inspecting grok call returns when it worked, one arm per shape real grok-4.7
# streams carry: a file read, a directory listing, a search and a command that exited 0, and
# a background command's own report completing with 0.
grok_successes: st.SearchStrategy[tuple[str, dict[str, Any]]] = st.one_of(
    st.tuples(
        st.just("read_file"),
        st.fixed_dictionaries({"type": st.just("ReadFile"), "FileContent": _grok_payload}),
    ),
    st.tuples(
        st.just("list_dir"),
        st.fixed_dictionaries({"type": st.just("ListDir"), "Content": json_values}),
    ),
    st.tuples(
        st.just("grep"),
        st.fixed_dictionaries(
            {"type": st.just("GrepSearch"), "exit_code": st.just(0), "stdout": json_values}
        ),
    ),
    st.tuples(
        st.just("run_terminal_command"),
        st.fixed_dictionaries(
            {"type": st.just("Bash"), "exit_code": st.just(0), "output": json_values}
        ),
    ),
    st.tuples(
        st.just("get_command_or_subagent_output"),
        st.fixed_dictionaries({"type": st.just("TaskOutput"), "Result": _grok_task_report}),
    ),
)


def _holds_no_object(text: str) -> bool:
    try:
        return not isinstance(json.loads(text), dict)
    except ValueError:
        return True


# A result block without an `is_error` key at all.
_ABSENT = object()

# Each way a result fails to report success, built directly rather than asked of the code
# under test: content holding no JSON object, a status other than "completed", an exit code
# other than the integer 0, and an `is_error` other than false.
grok_spoilers: st.SearchStrategy[tuple[str, str, Any]] = st.one_of(
    st.tuples(
        st.just("content"),
        st.just("top"),
        st.one_of(
            json_values.filter(lambda value: not isinstance(value, dict)).map(json.dumps),
            st.text(max_size=24).filter(_holds_no_object),
            json_values.filter(lambda value: not isinstance(value, (dict, str))),
        ),
    ),
    st.tuples(
        st.just("status"),
        st.sampled_from(["top", "nested"]),
        st.one_of(
            st.text(max_size=12).filter(lambda status: status != "completed"),
            st.sampled_from(["failed", "running", "succeeded", "Completed", None, 0, True, []]),
        ),
    ),
    st.tuples(
        st.just("exit_code"),
        st.sampled_from(["top", "nested"]),
        st.one_of(
            st.integers().filter(lambda code: code != 0),
            st.sampled_from([None, False, True, 0.0, "0", [0]]),
        ),
    ),
    st.tuples(
        st.just("is_error"), st.just("top"), st.sampled_from([True, None, 0, "false", _ABSENT])
    ),
)


class TestAGrokResultCountsOnlyWhenItReportsSuccess:
    """The success rule over generated results rather than the few shapes named above.

    Both directions, because either alone is satisfied by a rule that is wrong the other way:
    every real success shape counts, and every one of them stops counting when any single
    rule is broken in it.
    """

    def _answered(self, name: str, content: Any, is_error: Any = False) -> validate.RunStats:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": f"toolu_{name}",
            "content": content,
        }
        if is_error is not _ABSENT:
            block["is_error"] = is_error
        lines = [
            grok_tool_call(name, None),
            json.dumps({"type": "user", "message": {"role": "user", "content": [block]}}),
        ]
        return validate.run_stats("grok-messages", validate.objects(lines), None)

    @PROPERTY
    @given(success=grok_successes, extra=_grok_payload, completed=st.booleans())
    def test_every_real_success_shape_counts(
        self, success: tuple[str, dict[str, Any]], extra: dict[str, Any], completed: bool
    ):
        name, report = success
        report = {**extra, **report}
        if completed:
            report["status"] = "completed"
        stats = self._answered(name, json.dumps(report))
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 1), report

    @PROPERTY
    @given(success=grok_successes, extra=_grok_payload, spoiler=grok_spoilers)
    def test_a_success_with_any_one_rule_broken_does_not_count(
        self,
        success: tuple[str, dict[str, Any]],
        extra: dict[str, Any],
        spoiler: tuple[str, str, Any],
    ):
        name, report = success
        report = {**extra, **report}
        rule, where, value = spoiler
        nested = report.get("Result")
        nested_report = cast(dict[str, Any], nested) if isinstance(nested, dict) else None
        target = nested_report if where == "nested" and nested_report is not None else report
        content: Any = json.dumps(report)
        is_error: Any = False
        if rule == "content":
            content = value
        elif rule == "is_error":
            is_error = value
        else:
            target[rule] = value
            content = json.dumps(report)
        stats = self._answered(name, content, is_error)
        assert (stats.local_tool_attempts, stats.local_tool_calls) == (1, 0), (spoiler, content)


version_names = st.text(alphabet="0123456789.-abz²", min_size=1, max_size=8)


def _plugin_root(version: str) -> Path:
    """version_key reads `path.parent.parent.parent.name`, so the shape has to be real.

    An earlier draft passed `Path("/plugins") / version`, whose third parent is `/` — the key
    was computed from the empty string for every input and the property held vacuously.
    """
    return Path("/plugins") / version / "skills" / "ce-code-review" / "references"


dotted = st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=4).map(
    lambda xs: ".".join(str(x) for x in xs)
)


class TestVersionOrderIsTotal:
    """Three separate claims, because no one of them pins this key on its own.

    Injectivity without ordering is satisfied by `((), 0, name)` — a purely lexical key,
    which is injective and puts 3.9 above 3.13, the very bug the key exists to prevent.
    Ordering without injectivity is satisfied by dropping the tiebreakers, which lets
    `3.22.0` and `3.22.0-rc1` tie and hands the choice of brief set to filesystem order.
    """

    @PROPERTY
    @given(versions=st.lists(version_names, min_size=2, max_size=6, unique=True))
    def test_distinct_plugin_versions_never_tie(self, versions: list[str]):
        # `sorted` is stable, so any tie hands the decision to whatever order `glob` yielded.
        keys = [assets.version_key(_plugin_root(v)) for v in versions]
        assert len(set(keys)) == len(versions), f"distinct versions collided: {versions}"

    @PROPERTY
    @given(a=dotted, b=dotted)
    def test_numeric_order_beats_lexical_order(self, a: str, b: str):
        # The headline claim: 3.9 sorts BELOW 3.13. Compared against the component tuples
        # rather than against a second copy of the implementation, so a key that reverts to
        # string comparison disagrees here on the first pair whose digit counts differ.
        ka, kb = assets.version_key(_plugin_root(a)), assets.version_key(_plugin_root(b))
        ta = tuple(int(p) for p in a.split("."))
        tb = tuple(int(p) for p in b.split("."))
        assert (ka < kb) == (ta < tb), f"{a} vs {b}"

    @PROPERTY
    @given(version=dotted, tag=st.text(alphabet="abcr0123456789", min_size=1, max_size=4))
    def test_a_release_outranks_its_own_prerelease(self, version: str, tag: str):
        # 3.22.0 beats 3.22.0-rc1. They share a numeric tuple, so only the purity flag
        # separates them — and dropping it makes the winner filesystem order.
        assert assets.version_key(_plugin_root(version)) > assets.version_key(
            _plugin_root(f"{version}-{tag}")
        )

    @PROPERTY
    @given(version=version_names)
    def test_an_exotic_version_component_never_escapes_as_a_traceback(self, version: str):
        # `'²'.isdigit()` is True while `int('²')` raises.
        assets.version_key(_plugin_root(version))


class TestPersonaNamesAreBare:
    @PROPERTY
    @given(name=st.text(max_size=24))
    def test_a_name_that_survives_normalisation_can_never_leave_its_directory(self, name: str):
        # The name reaches every artifact path. Anything accepted here must be a single
        # filename, so `Path(run_dir) / f"{name}-grok.json"` cannot escape the run directory.
        try:
            bare = assets.normalise_persona(name)
        except assets.UsageError:
            return
        assert bare == Path(bare).name
        assert not bare.startswith(".")
        assert (Path("/run") / f"{bare}-grok.json").parent == Path("/run")


# The merge-tier projection lives at the end of this file because its property test reuses
# PROPERTY and `json_values` above.


def _return_finding(**over: Any) -> dict[str, Any]:
    """A finding carrying every key the merge helper reads, plus the ones it must not get."""
    base: dict[str, Any] = {
        "title": "a stale account id is billed",
        "severity": "P1",
        "file": "src/f.py",
        "line": 2,
        "confidence": 75,
        "autofix_class": "manual",
        "owner": "human",
        "requires_verification": True,
        "pre_existing": False,
        "suggested_fix": "read the id from the request",
        "settled_conflict": "kept as P1 over the peer's P2",
        "reviewers": ["correctness", "api-contract"],
        "independent_reviewers": ["api-contract"],
        "evidence": ["src/f.py:2 -- return  bill(account)", "src/f.py:9 -- corroboration"],
        "why_it_matters": "Callers bill the wrong account.",
        "an_unknown_key": {"the helper": "never reads this"},
    }
    base.update(over)
    return base


class TestTheMergeTierProjection:
    """`--return`: the artifact as the compact RETURN the plugin's merge helper consumes.

    The helper merges reviewer returns, not artifacts, and demotes a 75/100 finding whose
    `first_evidence` is missing to 50 — where its own confidence gate suppresses it. A lens
    that filled only `evidence` therefore reads as having found nothing, so the projection's
    one judgment is the `evidence[0]` fallback tier 1 already applies for display.
    """

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = str(self.dir / "correctness.json")

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _write(self, art: dict[str, Any]) -> None:
        Path(self.path).write_text(json.dumps(art), encoding="utf-8")

    def _run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def _project(self, art: dict[str, Any], *args: str) -> tuple[dict[str, Any], str]:
        self._write(art)
        code, out, err = self._run(self.path, "--return", *args)
        assert code == 0, err
        obj: dict[str, Any] = json.loads(out)
        return obj, err

    def _one(self, art: dict[str, Any], *args: str) -> dict[str, Any]:
        obj, _ = self._project(art, *args)
        first: dict[str, Any] = obj["findings"][0]
        return first

    def test_only_the_keys_the_helper_reads_survive(self):
        row = self._one(artifact(_return_finding()))
        assert set(row) == {
            "title",
            "severity",
            "file",
            "line",
            "confidence",
            "autofix_class",
            "owner",
            "requires_verification",
            "pre_existing",
            "suggested_fix",
            "first_evidence",
        }, row

    def test_the_merge_state_keys_are_never_copied_from_an_artifact(self):
        # Merge state is the orchestrator's to stamp on its own reconciled returns. A truthy
        # `settled_conflict` exempts a finding from the helper's confidence gate, so a lens
        # artifact carrying one would walk a finding whose quote --verify-quotes had just
        # dropped straight past the gate this projection exists to feed.
        row = self._one(artifact(_return_finding()))
        for key in ("settled_conflict", "reviewers", "independent_reviewers"):
            assert key not in row, key

    def test_the_artifact_only_keys_are_dropped(self):
        # A return is a different shape from an artifact, not a subset of one: the helper's
        # REQUIRED_FINDING has no `why_it_matters` and no `evidence`, and the tokens they
        # cost buy the merge nothing.
        row = self._one(artifact(_return_finding()))
        assert "why_it_matters" not in row
        assert "evidence" not in row
        assert "an_unknown_key" not in row

    def test_a_lens_written_first_evidence_is_kept_verbatim(self):
        row = self._one(artifact(_return_finding(first_evidence="src/f.py:2 --  bill(account)  ")))
        assert row["first_evidence"] == "src/f.py:2 --  bill(account)  "

    def test_a_missing_first_evidence_is_backfilled_from_evidence_zero(self):
        # THE POINT OF THE COMMAND. Without this the helper demotes the finding to 50 and
        # the gate suppresses it, so a real P1 reads as a clean review.
        row = self._one(artifact(_return_finding()))
        assert row["first_evidence"] == "src/f.py:2 -- return  bill(account)"

    def test_a_blank_first_evidence_is_treated_as_absent(self):
        # The helper marks a finding whose `first_evidence` is present but blank MALFORMED
        # and drops it, which is worse than the demotion the backfill exists to avoid.
        row = self._one(artifact(_return_finding(first_evidence="   \n ")))
        assert row["first_evidence"] == "src/f.py:2 -- return  bill(account)"

    @pytest.mark.parametrize(
        "evidence", [[], "src/f.py:2 -- not a list", [None], [""], [{"quote": "x"}]]
    )
    def test_no_usable_quote_leaves_the_key_absent_rather_than_empty(self, evidence: Any):
        row = self._one(artifact(_return_finding(first_evidence=" ", evidence=evidence)))
        assert "first_evidence" not in row

    def test_a_finding_that_is_not_an_object_passes_through_and_order_is_kept(self):
        # The helper counts a non-object finding malformed. Projecting it away would hide a
        # defect in the artifact behind a return that reads as clean.
        art = artifact(_return_finding(title="first"), _return_finding(title="third"))
        art["findings"].insert(1, "not an object")
        obj, _ = self._project(art)
        rows = obj["findings"]
        assert rows[1] == "not an object"
        assert [rows[0]["title"], rows[2]["title"]] == ["first", "third"]

    def test_every_other_top_level_key_is_copied_verbatim(self):
        # `independence_verified` decides cross-model promotion for an `adversarial-*`
        # reviewer, so a projection that dropped unknown metadata would change the merge.
        art = artifact(_return_finding())
        art["independence_verified"] = True
        art["residual_risks"] = ["the cache path is untested"]
        obj, _ = self._project(art)
        assert obj["independence_verified"] is True
        assert obj["residual_risks"] == ["the cache path is untested"]
        assert obj["reviewer"] == "adversarial-reviewer"

    def test_absent_list_fields_are_emitted_empty(self):
        art = artifact(_return_finding())
        del art["residual_risks"]
        del art["testing_gaps"]
        obj, _ = self._project(art)
        assert obj["residual_risks"] == []
        assert obj["testing_gaps"] == []

    @pytest.mark.parametrize(
        ("art", "expected"),
        [
            ({"findings": []}, "no reviewer name"),
            ({"reviewer": None, "findings": []}, "no reviewer name"),
            ({"reviewer": "  ", "findings": []}, "no reviewer name"),
            ({"reviewer": "r", "findings": [], "residual_risks": {}}, "residual_risks"),
            ({"reviewer": "r", "findings": [], "testing_gaps": "none"}, "testing_gaps"),
        ],
    )
    def test_a_return_the_helper_would_drop_whole_is_refused(
        self, art: dict[str, Any], expected: str
    ):
        # LITERAL 1: these numbers are a published contract, and an assertion written
        # against the module's own constant moves with it. The helper drops a malformed
        # return WITH every finding in it and says nothing, so emitting one would turn a
        # reviewer's whole pass into silence.
        self._write(art)
        code, out, err = self._run(self.path, "--return")
        assert code == 1, out
        assert expected in err
        assert out == ""

    def test_the_summary_line_counts_the_findings_and_the_backfills(self):
        # stdout is the object and nothing else; the counts a caller needs to audit the
        # projection go to stderr.
        art = artifact(
            _return_finding(),
            _return_finding(first_evidence="src/f.py:2 -- kept"),
            _return_finding(first_evidence=" ", evidence=[]),
        )
        _, err = self._project(art)
        assert (
            err.strip() == "ce-persona-findings: adversarial-reviewer: 3 findings, "
            "1 first_evidence backfilled from evidence[0]"
        )

    def test_the_object_is_unfenced_so_a_caller_can_parse_it(self):
        self._write(artifact(_return_finding()))
        _, out, _ = self._run(self.path, "--return")
        assert "UNTRUSTED" not in out
        assert json.loads(out)["reviewer"] == "adversarial-reviewer"

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ((ARTIFACT, "--return", "--json"), "cannot be combined"),
            ((ARTIFACT, "--json", "--return"), "cannot be combined"),
            ((ARTIFACT, "--return", "--show", "1"), "cannot be combined"),
            ((ARTIFACT, "--return", "--show", "all"), "cannot be combined"),
            ((ARTIFACT, "--verify-quotes"), "only applies to --return"),
            ((ARTIFACT, "--return", "--verify-quotes"), "wants -C"),
            ((ARTIFACT, "--return", "--verify-quotes", "-C"), "-C wants a directory"),
            # A -C nobody asked to use is not discarded: the output would be byte-identical
            # to an unverified --return at exit 0, and a machine caller has no channel on
            # which to notice it got no verification.
            ((ARTIFACT, "--return", "-C", "."), "-C only applies to --verify-quotes or --anchors"),
            ((ARTIFACT, "-C", "."), "-C only applies to --verify-quotes or --anchors"),
        ],
    )
    def test_a_mode_that_does_not_exist_is_a_usage_error(
        self, args: tuple[str, ...], expected: str
    ):
        self._write(artifact(_return_finding()))
        code, out, err = self._run(*(self.path if a is ARTIFACT else a for a in args))
        assert code == 2, err
        assert expected in err
        assert out == ""

    def test_a_c_that_is_not_a_usable_directory_is_a_usage_error(self):
        self._write(artifact(_return_finding()))
        not_a_dir = str(Path(self.tmp.name) / "correctness.json")
        for spec, expected in (
            (not_a_dir, "is not a directory"),
            (str(Path(self.tmp.name) / "nope"), "is not a directory"),
            # expanduser raises RuntimeError for an unknown user -- not OSError, and not a
            # type a caller would think to catch, so it needs its own arm to reach exit 2.
            ("~nosuchuser0123/x", "names a home directory that does not exist"),
        ):
            code, out, err = self._run(self.path, "--return", "--verify-quotes", "-C", spec)
            assert code == 2, err
            assert expected in err, err
            assert out == ""
            assert "Traceback" not in err

    def test_the_vacuous_run_refusal_precedes_the_projection(self):
        # The laundering route, in the mode a machine consumes: exit 6 keeps the artifact as
        # evidence, and a return built from it would feed a review nobody performed straight
        # into a merge.
        art = self.dir / "adversarial-reviewer-grok.json"
        art.write_text(json.dumps(artifact(_return_finding())), encoding="utf-8")
        (self.dir / ("adversarial-reviewer-grok" + validate.PROVENANCE_SUFFIX)).write_text(
            json.dumps({"provider": "grok", "run_stats": {"tool_calls": 0, "turns": 1}}),
            encoding="utf-8",
        )
        code, out, err = self._run(str(art), "--return")
        assert code == 6, out
        assert out == ""
        assert "no local tool calls" in err

    def test_help_documents_the_new_modes(self):
        code, out, _ = self._run("--help")
        assert code == 0
        for token in ("--return", "--verify-quotes", "-C <dir>", "--anchors -C <dir>"):
            assert token in out
        # Wrapped in HELP, one sentence in README: the words are the contract, the line
        # breaks are layout, so the comparison is against the collapsed text.
        assert (
            "the file is unreadable, is neither a findings nor a verdicts artifact, is"
            " both at once, or could not be projected into a usable return"
        ) in " ".join(out.split())

    @PROPERTY
    @given(
        raw=st.dictionaries(
            st.sampled_from([*findings.RETURN_KEYS, "why_it_matters", "evidence", "junk"]),
            json_values,
            max_size=8,
        )
    )
    def test_a_projected_finding_never_carries_a_key_the_helper_cannot_read(
        self, raw: dict[str, Any]
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"reviewer": "r", "findings": [raw]}), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = findings.main([str(path), "--return"])
        assert code == 0
        row = json.loads(out.getvalue())["findings"][0]
        assert set(row) <= set(findings.RETURN_KEYS)
        if "first_evidence" in row:
            # An empty one is worse than none: the helper marks that finding malformed.
            assert isinstance(row["first_evidence"], str) and row["first_evidence"].strip()


class TestQuotesAreCheckedAgainstTheTree:
    """`--verify-quotes -C <dir>`: a quote the reviewed tree does not carry is dropped.

    Dropped, never rewritten. Rewriting a quote to whatever the file holds would manufacture
    evidence the lens did not give; removing it lets the merge helper demote the finding on
    the same rule it applies to a lens that quoted nothing at all.
    """

    LINE = "    return bill(account)"

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = str(self.dir / "correctness.json")
        self.tree = self.dir / "tree"
        (self.tree / "src").mkdir(parents=True)
        (self.tree / "src" / "f.py").write_text(f"import billing\n{self.LINE}\n", encoding="utf-8")

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _run(self, quote: str, **over: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """The finding as `--return` alone gives it, the stderr, and as verified."""
        art = artifact(_return_finding(first_evidence=quote, **over))
        Path(self.path).write_text(json.dumps(art), encoding="utf-8")
        plain, verified, err = io.StringIO(), io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(plain), contextlib.redirect_stderr(io.StringIO()):
            assert findings.main([self.path, "--return"]) == 0
        with contextlib.redirect_stdout(verified), contextlib.redirect_stderr(err):
            code = findings.main([self.path, "--return", "--verify-quotes", "-C", str(self.tree)])
        assert code == 0, err.getvalue()
        before: dict[str, Any] = json.loads(plain.getvalue())
        after: dict[str, Any] = json.loads(verified.getvalue())
        # THE INVARIANT, asserted on every case rather than once: removing a first_evidence
        # is the ONLY difference this mode may make to the object.
        assert [k for k in after] == [k for k in before]
        assert all(before[k] == after[k] for k in before if k != "findings")
        plain_rows: list[Any] = before["findings"]
        checked_rows: list[Any] = after["findings"]
        for plain_row, checked in zip(plain_rows, checked_rows, strict=True):
            if not isinstance(plain_row, dict):
                assert checked == plain_row
                continue
            row = cast(dict[str, Any], plain_row)
            kept = {
                key: value
                for key, value in row.items()
                if key != "first_evidence" or key in checked
            }
            assert checked == kept
        return before["findings"][0], err.getvalue(), after["findings"][0]

    @pytest.mark.parametrize(
        "quote",
        [
            "src/f.py:2 -- return bill(account)",
            "src/f.py:2: return bill(account)",
            "`return bill(account)` -- src/f.py:2",
            # Whitespace is collapsed on both sides, so a re-indented quote still matches.
            "src/f.py:2 --     return   bill(account)",
            # A substring of the line is enough, down to the floor: lenses quote the
            # fragment that matters, and `bill(account)` is 13 characters.
            "src/f.py:2 -- bill(account)",
            # Decoration around the citation, a `:col` suffix, and a backticked remainder:
            # shapes lenses write every day, each of which used to drop a verbatim quote.
            "**src/f.py:2** -- return bill(account)",
            "return bill(account) (src/f.py:2)",
            "`src/f.py:2` -- return bill(account)",
            "<src/f.py:2> -- return bill(account)",
            "src/f.py:2:5 -- return bill(account)",
            "src/f.py:2 -- `return bill(account)`",
            # The motivating LINES, which is what the evidence contract asks for: the
            # comparison is sized to the quote rather than to one line.
            "src/f.py:1 -- import billing\n    return bill(account)",
        ],
    )
    def test_a_quote_the_tree_carries_is_kept(self, quote: str):
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote
        assert err.strip().endswith("0 dropped by --verify-quotes")

    @pytest.mark.parametrize(
        "quote",
        [
            "src/f.py:1-2: import billing\n    return bill(account)",
            "src/f.py:1–2: import billing\n    return bill(account)",
            "src/f.py:2-2 -- return bill(account)",
            "`return bill(account)` -- src/f.py:2-3",
        ],
    )
    def test_a_range_cited_quote_the_tree_carries_is_kept(self, quote: str):
        # Half of the lenses' citations are ranges. Read as `path:first`, the rest of the
        # range stayed in the compared text, and every one of them was dropped.
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err
        assert err.strip().endswith("0 dropped by --verify-quotes")

    def test_a_basename_citation_is_resolved_through_the_findings_own_file(self):
        # Lenses routinely cite `f.py:2` while `file` carries the repo-relative path.
        _, _, after = self._run("f.py:2 -- return bill(account)", file="src/f.py")
        assert after["first_evidence"] == "f.py:2 -- return bill(account)"

    @pytest.mark.parametrize(
        ("quote", "over", "reason"),
        [
            (
                "src/f.py:2 -- return charge(account)",
                {},
                "quoted text is not on src/f.py:2",
            ),
            ("src/f.py:1 -- return bill(account)", {}, "quoted text is not on src/f.py:1"),
            ("src/f.py:99 -- return bill(account)", {}, "line 99 is out of range for src/f.py"),
            ("src/gone.py:2 -- return bill(account)", {}, "no file:line reference resolves"),
            # A citation that resolves to a DIRECTORY inside the tree: readable, contained,
            # and still not a file with a line 1.
            ("src:1 -- return bill(account)", {}, "no file:line reference resolves"),
            ("a finding with no citation at all", {}, "no file:line reference resolves"),
            ("src/f.py:2 --", {}, "quoted text is empty"),
            ("`` -- src/f.py:2", {}, "quoted text is empty"),
            # A backticked span is read only when the remainder IS one. Reading it wherever a
            # backtick appeared checked `bill` -- four characters of an aside -- and
            # certified the first two of these, one of which is pure prose.
            (
                "src/f.py:2 -- return charge(account)  (the `bill` path is the correct one)",
                {},
                "quoted text is not on src/f.py:2",
            ),
            (
                "src/f.py:2 -- the account is never re-read before `bill` is called",
                {},
                "quoted text is not on src/f.py:2",
            ),
            (
                "src/f.py:2 -- return bill(account), not the `import billing` on line 1",
                {},
                "quoted text is not on src/f.py:2",
            ),
            # The floor. Both are ON the cited line as substrings, and neither says anything
            # about the finding: verification that cannot fail is worse than none.
            ("src/f.py:2 -- r", {}, "quoted text is too short to check (1 chars, floor 12)"),
            (
                "src/f.py:5 -- account",
                {},
                "quoted text is too short to check (7 chars, floor 12)",
            ),
            # A two-line quote needs two lines under it; checked before the comparison so the
            # reason says which lines were wanted.
            (
                "src/f.py:2 -- import billing\n    return bill(account)",
                {},
                "lines 2-3 are out of range for src/f.py (2 lines)",
            ),
            # A trailing cross-reference: the compared text is the quote minus the citation
            # being checked, so the aside is part of it and the remainder is not verbatim.
            (
                "src/f.py:2 -- return bill(account)  (see also src/g.py:9)",
                {},
                "quoted text is not on src/f.py:2",
            ),
            # Two citations, only the second resolving: the reason names src/f.py:2, so the
            # reader did not stop at the first citation it could not resolve.
            (
                "src/gone.py:1 and src/f.py:2 -- return charge(account)",
                {},
                "quoted text is not on src/f.py:2",
            ),
        ],
    )
    def test_a_quote_the_tree_does_not_carry_is_dropped_with_its_reason(
        self, quote: str, over: dict[str, Any], reason: str
    ):
        before, err, after = self._run(quote, **over)
        assert before["first_evidence"] == quote, "the plain projection must still carry it"
        assert "first_evidence" not in after
        assert reason in err, err
        assert "verify-quotes: finding #1 (src/f.py:2)" in err
        assert err.strip().endswith("1 dropped by --verify-quotes")

    def test_an_unreadable_file_drops_the_quote_rather_than_crashing(self):
        if os.geteuid() == 0:
            pytest.skip("root reads a mode-000 file, so the case cannot be produced")
        secret = self.tree / "src" / "secret.py"
        secret.write_text("x = 1\n", encoding="utf-8")
        secret.chmod(0o000)
        try:
            _, err, after = self._run("src/secret.py:1 -- x = 1")
        finally:
            secret.chmod(0o600)
        assert "first_evidence" not in after
        assert "no file:line reference resolves" in err

    def test_the_artifact_on_disk_is_never_modified(self):
        original = Path(self.path)
        self._run("src/f.py:2 -- return charge(account)")
        art = json.loads(original.read_text(encoding="utf-8"))
        assert art["findings"][0]["first_evidence"] == "src/f.py:2 -- return charge(account)"

    def test_a_backfilled_quote_is_verified_too(self):
        # The backfill is where most quotes come from, so verifying only lens-written ones
        # would leave the common case unchecked.
        art = artifact(
            _return_finding(evidence=["src/f.py:2 -- return nothing_like_this(x)"]),
        )
        Path(self.path).write_text(json.dumps(art), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main([self.path, "--return", "--verify-quotes", "-C", str(self.tree)])
        assert code == 0
        assert "first_evidence" not in json.loads(out.getvalue())["findings"][0]
        assert "quoted text is not on src/f.py:2" in err.getvalue()

    def test_findings_with_no_quote_to_check_are_left_alone_and_numbering_holds(self):
        # #N must stay the artifact's own numbering, so a stderr line names the same finding
        # the listing and `--show N` do.
        art = artifact(
            _return_finding(first_evidence=" ", evidence=[]),
            _return_finding(first_evidence="src/f.py:2 -- return charge(account)"),
        )
        art["findings"].insert(1, "not an object")
        Path(self.path).write_text(json.dumps(art), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main([self.path, "--return", "--verify-quotes", "-C", str(self.tree)])
        assert code == 0
        rows = json.loads(out.getvalue())["findings"]
        assert "first_evidence" not in rows[0]
        assert rows[1] == "not an object"
        assert "first_evidence" not in rows[2]
        assert "finding #3 (src/f.py:2)" in err.getvalue(), err.getvalue()
        assert err.getvalue().strip().endswith("1 dropped by --verify-quotes")

    def test_a_citation_cannot_steer_the_reader_out_of_the_tree(self):
        # The quote is model-written text: an input, not a destination. Each of these names a
        # real file whose cited line carries the quote verbatim, and each must still drop.
        secret = "the secret line nobody pointed this reader at"
        outside = self.dir / "outside.py"
        outside.write_text(f"{secret}\n", encoding="utf-8")
        # A link INSIDE the tree, cited by its own path, whose target is outside: the check
        # on the cited STRING refuses the other two and admits this one, which is how the
        # verifier became a read oracle over every file its caller can open.
        (self.tree / "src" / "link.py").symlink_to(outside)
        for citation in (
            f"{outside}:1 -- {secret}",
            f"../outside.py:1 -- {secret}",
            f"src/link.py:1 -- {secret}",
        ):
            _, err, after = self._run(citation)
            assert "first_evidence" not in after, citation
            assert "cites a path outside the reviewed tree" in err, citation

    def test_an_unreadable_directory_drops_the_quote_rather_than_crashing(self):
        # The other half of the unreadable case: this one fails in the stat, before the read,
        # and _resolve's contract is that an unresolvable citation is dropped, never fatal.
        if os.geteuid() == 0:
            pytest.skip("root traverses a mode-000 directory, so the case cannot be produced")
        locked = self.tree / "src" / "locked"
        locked.mkdir()
        (locked / "x.py").write_text("x = 1  # a line long enough to check\n", encoding="utf-8")
        locked.chmod(0o000)
        try:
            _, err, after = self._run("src/locked/x.py:1 -- x = 1  # a line long enough to check")
        finally:
            locked.chmod(0o700)
        assert "first_evidence" not in after
        assert "no file:line reference resolves" in err

    def test_a_line_number_too_long_to_parse_drops_the_quote_rather_than_crashing(self):
        # CPython refuses to int() a 5000-digit string, and the digits come straight out of
        # model-written text.
        _, err, after = self._run("src/f.py:" + "9" * 5000 + " -- return bill(account)")
        assert "first_evidence" not in after
        assert "no file:line reference resolves" in err

    def test_any_citation_of_the_findings_own_file_may_corroborate_it(self):
        # The first citation resolves and contradicts; the second resolves and carries the
        # text. Committing to the first resolving citation dropped a quote the tree does
        # hold -- and line 3 is a line shaped like a citation, which is what a test fixture
        # or a log line in a reviewed tree looks like.
        (self.tree / "src" / "f.py").write_text(
            f"import billing\n{self.LINE}\n# src/f.py:99 -- return bill(account)\n",
            encoding="utf-8",
        )
        quote = "src/f.py:99 -- return bill(account) (src/f.py:3)"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err
        assert err.strip().endswith("0 dropped by --verify-quotes")

    def test_a_citation_of_another_file_cannot_found_this_finding(self):
        # A quote citing any real line in the tree used to keep first_evidence, so evidence
        # for a location the finding is not at founded it anyway.
        (self.tree / "README.md").write_text(
            "persona-review working notes here\n", encoding="utf-8"
        )
        (self.tree / "src" / "pkg").mkdir()
        (self.tree / "src" / "pkg" / "__init__.py").write_text(
            "from .billing import bill\n", encoding="utf-8"
        )
        quote = "README.md:1 -- persona-review working notes here"
        _, err, after = self._run(quote, file="src/pkg/__init__.py")
        assert "first_evidence" not in after
        assert "cites README.md:1 but the finding is at src/pkg/__init__.py" in err, err

    def test_a_contradicted_same_file_citation_keeps_its_own_reason(self):
        # The both-locations reason is for a quote that founds somewhere else, not for one
        # that cites the right file and gets the line wrong.
        _, err, after = self._run("src/f.py:2 -- return charge(account)")
        assert "first_evidence" not in after
        assert "quoted text is not on src/f.py:2" in err, err
        assert "but the finding is at" not in err, err

    def _package_tree(self, root_line: str, own_line: str) -> None:
        """A bare `__init__.py` at the root and the finding's own one under `src/pkg`."""
        (self.tree / "__init__.py").write_text(f"{root_line}\n", encoding="utf-8")
        (self.tree / "src" / "pkg").mkdir()
        (self.tree / "src" / "pkg" / "__init__.py").write_text(f"{own_line}\n", encoding="utf-8")

    def test_a_basename_citation_tries_the_findings_own_path_first(self):
        # `__init__.py` names one file per package, and the bare one at the root resolves
        # first. Checking it instead dropped a quote verbatim from the finding's own file.
        self._package_tree("# root package", "from .billing import bill")
        quote = "__init__.py:1 -- from .billing import bill"
        _, err, after = self._run(quote, file="src/pkg/__init__.py")
        assert after["first_evidence"] == quote, err

    def test_a_basename_citation_is_not_corroborated_by_the_file_at_the_root(self):
        # The inverse: the root file carries the text and the finding's own file does not,
        # so the quote founds a location this finding is not at.
        self._package_tree("from .billing import bill", "# the package this finding is at")
        _, err, after = self._run(
            "__init__.py:1 -- from .billing import bill", file="src/pkg/__init__.py"
        )
        assert "first_evidence" not in after
        assert "quoted text is not on src/pkg/__init__.py:1" in err, err

    def _five_line_file(self) -> None:
        """A file long enough for a quote to cite two separate lines of it."""
        (self.tree / "src" / "f.py").write_text(
            "def bill(account):\n    return bill(account)\n\n    return refund(account)\n# end\n",
            encoding="utf-8",
        )

    def test_each_citation_is_checked_against_its_own_line(self):
        # Two snippets, each cited at the line that carries it. Compared as one remainder
        # against each citation in turn, neither can match, and a true quote was dropped.
        self._five_line_file()
        quote = "src/f.py:2 -- return bill(account)\nsrc/f.py:4 -- return refund(account)"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err
        assert err.strip().endswith("0 dropped by --verify-quotes")

    def test_a_multi_citation_quote_is_dropped_when_one_segment_is_wrong(self):
        self._five_line_file()
        quote = "src/f.py:2 -- return bill(account)\nsrc/f.py:4 -- return refund(customer)"
        _, err, after = self._run(quote)
        assert "first_evidence" not in after
        assert "quoted text is not on src/f.py:4" in err, err

    def test_a_quote_first_multi_citation_quote_is_checked_per_citation(self):
        # The other shape lenses write: the text precedes the citation it belongs to.
        self._five_line_file()
        quote = "`return bill(account)` -- src/f.py:2; `return refund(account)` -- src/f.py:4"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err
        assert err.strip().endswith("0 dropped by --verify-quotes")

    def test_a_multi_citation_quote_still_founds_the_findings_own_file(self):
        # Every segment true is not enough: none of them is at the finding's location.
        self._five_line_file()
        (self.tree / "README.md").write_text(
            "persona-review working notes here\nthe second note in this file\n",
            encoding="utf-8",
        )
        quote = "README.md:1 -- persona-review working notes here\n"
        quote += "README.md:2 -- the second note in this file"
        _, err, after = self._run(quote, file="src/f.py")
        assert "first_evidence" not in after
        assert "cites README.md:1 but the finding is at src/f.py" in err, err

    def test_a_backticked_path_before_the_colon_resolves(self):
        # `src/f.py`:2 -- the closing backtick sits between the path and the line number,
        # and the whole citation used to resolve to nothing.
        quote = "`src/f.py`:2 -- return bill(account)"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err

    def test_a_newline_padded_quote_does_not_widen_the_window(self):
        # The leading newline inside the backticks used to count as a line of the quote, so
        # the window reached line 2 and the text there certified a citation of line 1.
        _, err, after = self._run("src/f.py:1 -- `\n    return bill(account)`")
        assert "first_evidence" not in after
        assert "quoted text is not on src/f.py:1" in err, err

    def test_a_padded_quote_is_kept_at_the_line_it_is_actually_on(self):
        quote = "src/f.py:2 -- `\n    return bill(account)`"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err

    def test_a_fabricated_citation_beside_true_ones_drops_the_quote(self):
        # A citation that resolves to nothing carries text of its own here, so it is a claim
        # like any other segment: unchecked, a lens prefixes an invented line to two true
        # ones and the quote survives on their strength.
        self._five_line_file()
        quote = "src/nope.py:1 -- authorize_everything()\n"
        quote += "src/f.py:2 -- return bill(account)\nsrc/f.py:4 -- return refund(account)"
        _, err, after = self._run(quote)
        assert "first_evidence" not in after
        assert "cites src/nope.py:1, which does not resolve" in err, err

    def test_a_segmented_quote_whose_citations_all_resolve_is_kept(self):
        # The control for the case above: the same quote without the invented first line.
        self._five_line_file()
        quote = "src/f.py:2 -- return bill(account)\nsrc/f.py:4 -- return refund(account)"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err

    def test_a_citation_inside_the_quoted_source_is_not_a_claim_of_its_own(self):
        # The quoted LINE contains a citation, which is what a test fixture or a log line in
        # a reviewed tree looks like. Read as a second claim it splits the quote, and the
        # text left to the finding's own citation is too short to check.
        (self.tree / "tests").mkdir()
        line = '    quote = "README.md:1 -- persona-review working notes here"'
        (self.tree / "tests" / "t.py").write_text(f"x = 1\ny = 2\n{line}\n", encoding="utf-8")
        (self.tree / "README.md").write_text(
            "persona-review working notes here\n", encoding="utf-8"
        )
        quote = f"tests/t.py:3 -- {line.strip()}"
        _, err, after = self._run(quote, file="tests/t.py", line=3)
        assert after["first_evidence"] == quote, err

    @pytest.mark.parametrize("separator", ["--", ":"])
    def test_a_doubled_citation_of_one_location_is_one_claim(self, separator: str):
        # The same location cited twice is one claim about it, so both citations leave the
        # compared text. Removing only the one being checked leaves the other in the
        # remainder, where it is not on the line.
        quote = f"src/f.py:2 {separator} return bill(account) (src/f.py:2)"
        _, err, after = self._run(quote)
        assert after["first_evidence"] == quote, err

    def test_a_version_token_on_the_quoted_line_does_not_split_the_quote(self):
        # `python:3` is citation-shaped, and a file named `python` at the root makes it
        # resolve. Split on it, the finding's own citation keeps `FROM` and the quote dies
        # on the floor -- on a line the tree carries verbatim.
        (self.tree / "src" / "f.py").write_text("FROM python:3.12\n", encoding="utf-8")
        (self.tree / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        quote = "src/f.py:1 -- FROM python:3.12"
        _, err, after = self._run(quote, line=1)
        assert after["first_evidence"] == quote, err

    def test_a_citation_of_the_findings_own_file_by_another_path_founds_it(self):
        # An in-tree absolute path and a path through `..` both name the finding's own file.
        # Compared lexically they name some other location, and a verbatim quote is dropped.
        for cited in (str(self.tree / "src" / "f.py"), "src/../src/f.py"):
            quote = f"{cited}:2 -- return bill(account)"
            _, err, after = self._run(quote, file="src/f.py")
            assert after["first_evidence"] == quote, err


def _decorated(path: str, where: str, shape: str) -> str:
    """A named function rather than a lambda: strict mode cannot infer a lambda's parameter."""
    return f"`{path}`:{where}" if shape == "`path`" else shape.format(f"{path}:{where}")


# A leading citation in every shape `claim_v1` takes off, with what joins it to the code.
_leading_citations = st.tuples(
    st.builds(
        _decorated,
        st.sampled_from(
            ["src/a.py", "a.py", "lib/other.py", "docs/guide.md", "src\\win.py", "Makefile.am"]
        ),
        st.tuples(
            st.integers(min_value=1, max_value=99999).map(str),
            st.sampled_from(["", ":7"]),
            st.sampled_from(["", "-40", "–40", "—40"]),
        ).map("".join),
        st.sampled_from(["{}", "**{}**", "({})", "`{}`", "<{}>", "[{}]", "`path`"]),
    ),
    st.sampled_from(["", " (verbatim)", " (Verbatim)"]),
    st.sampled_from([": ", " : ", " -- ", "-- ", " — ", " – ", " - ", " "]),
).map("".join)

# Code a claim is made of. No colon, so it holds no citation of its own, and a letter first,
# so it opens with neither a separator nor `(verbatim)`.
_claimed_code = st.tuples(
    st.sampled_from("abcxyz"), st.text(alphabet="abcxyz_019 =+.,()[]{}/\"'#", max_size=30)
).map("".join)


class TestTheCitationScanner:
    """One reading of a quote's citations, which the claim is cut from."""

    def test_a_bracketed_own_path_is_one_citation(self):
        # `_REFERENCE` alone reads `id]/page.tsx:12` from offset 4. Kept beside the literal,
        # the two readings would cut the quote in two different places.
        found = findings.citations("app/[id]/page.tsx:12: foo(bar, baz)", "app/[id]/page.tsx")
        assert found == [findings.Citation("app/[id]/page.tsx", "12", 0, 20, True)]

    def test_a_trailing_bracketed_own_path_is_one_citation(self):
        found = findings.citations("foo(bar, baz) -- app/[id]/page.tsx:12", "app/[id]/page.tsx")
        assert found == [findings.Citation("app/[id]/page.tsx", "12", 17, 37, True)]

    def test_a_path_ending_in_the_basename_is_another_file(self):
        # `lib/a.py` ends in `a.py`, and a finding at `src/a.py` tries that basename. Read as
        # its own citation, a quote of another file would found this one.
        found = findings.citations("lib/a.py:5 -- total = compute(a, b)", "src/a.py")
        assert found == [findings.Citation("lib/a.py", "5", 0, 10, False)]

    def test_a_decorated_basename_is_the_findings_own_file(self):
        found = findings.citations("(a.py:12) total = compute(a, b)", "src/a.py")
        assert found == [findings.Citation("a.py", "12", 0, 9, True)]

    def test_an_empty_own_path_names_no_citation(self):
        # An empty literal would match every `:N` that follows whitespace.
        for own in ("", "src/"):
            found = findings.citations("retry(limit) :12 and (:40)", own)
            assert not any(c.own for c in found), (own, found)


class TestKeysV1:
    """The claim and the two keys a poster writes into hidden markers. Frozen.

    A marker outlives every release, so these vectors are never updated to follow the code:
    a failure here means a definition moved, which orphans every comment already posted. A
    different definition is a `v2` beside these, with a one-time re-post.
    """

    CODE = "total = compute(a, b)"
    OWN = "src/a.py"

    # (quote, the finding's file, the claim). Each claim is read off the grammar in
    # `claim_v1`'s docstring, not off its output, or a vector would pin whatever the code does.
    CLAIMS: tuple[tuple[str, str | None, str], ...] = (
        # A leading citation, with each separator a lens writes.
        ("src/a.py:12: total = compute(a, b)", OWN, CODE),
        ("src/a.py:12 -- total = compute(a, b)", OWN, CODE),
        ("src/a.py:12 — total = compute(a, b)", OWN, CODE),
        ("src/a.py:12 – total = compute(a, b)", OWN, CODE),
        ("src/a.py:12 - total = compute(a, b)", OWN, CODE),
        # No separator: the rest is re-stripped before it is unwrapped.
        ("src/a.py:12 `total = compute(a, b)`", OWN, CODE),
        ("src/a.py:12 (verbatim): `total = compute(a, b)`", OWN, CODE),
        ("src/a.py:12–14: total = compute(a, b)", OWN, CODE),
        ("src/a.py:12—14: total = compute(a, b)", OWN, CODE),
        ("src/a.py:12-14 -- total = compute(a, b)", OWN, CODE),
        ("src/a.py:12:5 -- total = compute(a, b)", OWN, CODE),
        ("src/a.py:12 --     total   =  compute(a, b)", OWN, CODE),
        ("src/a.py:12 -- `  total = compute(a, b)  `", OWN, CODE),
        # A trailing citation: joined by a separator, or parenthesized.
        ("`total = compute(a, b)` -- src/a.py:12", OWN, CODE),
        ("total = compute(a, b) (src/a.py:12)", OWN, CODE),
        ("total = compute(a, b) -- (verbatim) src/a.py:12", OWN, CODE),
        ("total = compute(a, b) – src/a.py:12", OWN, CODE),
        ("total = compute(a, b) -- (Verbatim) src/a.py:12", OWN, CODE),
        # An opening parenthesis alone does not set a citation apart.
        (
            "total = compute(a, b) (src/a.py:12",
            OWN,
            "total = compute(a, b) (src/a.py:12",
        ),
        # A citation of another file at column 0 is the line's locator, though the finding's
        # own path, read first, is cited later in the line.
        ("lib/b.py:3 -- x … src/a.py:12", OWN, "x … src/a.py:12"),
        # Bare and unjoined, it may be what the line says.
        ("total = compute(a, b) src/a.py:12", OWN, "total = compute(a, b) src/a.py:12"),
        ("total = a- src/a.py:12", OWN, "total = a- src/a.py:12"),
        # Prose after the citation: it is not at the edge, so nothing is taken off.
        (
            "`total = compute(a, b)` -- src/a.py:12, called twice",
            OWN,
            "`total = compute(a, b)` -- src/a.py:12, called twice",
        ),
        # The finding's own path, read as a literal where `_REFERENCE` would stop at `[`/`(`.
        ("app/[id]/page.tsx:12: foo(bar, baz)", "app/[id]/page.tsx", "foo(bar, baz)"),
        ("(app/(auth)/page.tsx:12) -- foo(bar, baz)", "app/(auth)/page.tsx", "foo(bar, baz)"),
        ("foo(bar, baz) -- app/[id]/page.tsx:12", "app/[id]/page.tsx", "foo(bar, baz)"),
        # Spelled with a `./` on either side, which `norm_path` reads as the same file.
        (
            "app/[id]/page.tsx:2 -- return renderPage(props);",
            "app/[id]/page.tsx",
            "return renderPage(props);",
        ),
        (
            "app/[id]/page.tsx:2 -- return renderPage(props);",
            "./app/[id]/page.tsx",
            "return renderPage(props);",
        ),
        (
            "./app/[id]/page.tsx:2 -- return renderPage(props);",
            "app/[id]/page.tsx",
            "return renderPage(props);",
        ),
        # Its basename.
        ("a.py:12 -- total = compute(a, b)", OWN, CODE),
        ("(a.py:12) total = compute(a, b)", OWN, CODE),
        # Another file, stripped because its path looks like one (step 2's path rule).
        ("**src/f.py:2** -- return bill(account)", "src/g.py", "return bill(account)"),
        ("`src/f.py`:2 -- return bill(account)", "src/g.py", "return bill(account)"),
        ("lib/a.py:5 -- total = compute(a, b)", OWN, CODE),
        ("src\\a.py:12 -- total = compute(a, b)", OWN, CODE),
        ("f.py:2 -- return bill(account)", "src/g.py", "return bill(account)"),
        ("bin/deploy:3 -- set -euo pipefail", OWN, "set -euo pipefail"),
        ("C:\\bin\\Makefile:42 -- $(CC) -o app main.c", OWN, "$(CC) -o app main.c"),
        # Citation-shaped, not a path: what the line says.
        ("timeout:30 -- seconds", OWN, "timeout:30 -- seconds"),
        ("`timeout`:30 -- seconds", OWN, "`timeout`:30 -- seconds"),
        (
            "https://example.com:443 - the port the proxy listens on",
            OWN,
            "https://example.com:443 - the port the proxy listens on",
        ),
        (":12 -- retry(limit)", "", ":12 -- retry(limit)"),
        # Only a citation: no code is claimed.
        ("a.py:12", OWN, ""),
        ("(src/a.py:12)", OWN, ""),
        # A citation mid-quote is what the line says.
        (
            "retry(3) # see src/a.py:12 for the limit",
            OWN,
            "retry(3) # see src/a.py:12 for the limit",
        ),
        # Never split, never elided.
        ("src/a.py:3 -- a = 1 / b = 2", OWN, "a = 1 / b = 2"),
        ("src/a.py:3 -- first() ... last()", OWN, "first() ... last()"),
        # A literal backslash-n stays; a real newline is whitespace.
        ('src/a.py:3 -- log("one\\ntwo")', OWN, 'log("one\\ntwo")'),
        ("src/a.py:3 -- a = 1\n    b = 2", OWN, "a = 1 b = 2"),
        # A citation starting any line is that line's locator, in each shape a lens writes.
        ("src/x.py:81: a = 1\nsrc/x.py:82: b = 2", "src/x.py", "a = 1 b = 2"),
        ("src/x.py:81: `a = 1`\nsrc/x.py:82: `b = 2`", "src/x.py", "a = 1 b = 2"),
        (
            "src/x.py:81 (verbatim) -- a = 1\nsrc/x.py:82 (verbatim) -- b = 2",
            "src/x.py",
            "a = 1 b = 2",
        ),
        (
            "src/x.py:81 (verbatim) -- `a = 1`\nsrc/x.py:82 (verbatim) -- `b = 2`",
            "src/x.py",
            "a = 1 b = 2",
        ),
        ("a = 1\nsrc/x.py:82: b = 2", "src/x.py", "a = 1 b = 2"),
        ("src/x.py:81: a = 1\n    src/x.py:82: b = 2", "src/x.py", "a = 1 b = 2"),
        # A line starting with a token that is not a path, or citing mid-line, is code.
        ("src/x.py:81: a = 1\nretries:3 -- b = 2", "src/x.py", "a = 1 retries:3 -- b = 2"),
        (
            "src/x.py:81: a = 1\nb = 2 # see src/x.py:82 for c",
            "src/x.py",
            "a = 1 b = 2 # see src/x.py:82 for c",
        ),
        # Backticks on a line no citation came off are code; around the whole rest they are not.
        ("src/x.py:81: a = 1\n`b = 2`", "src/x.py", "a = 1 `b = 2`"),
        ("src/x.py:81:\n`a = 1`", "src/x.py", "a = 1"),
        # A separator or `(verbatim)` not beside a removed citation stays.
        ("-- SELECT id FROM users", OWN, "-- SELECT id FROM users"),
        (
            "src/a.py:12 -- `total = compute(a, b)` (verbatim)",
            OWN,
            "`total = compute(a, b)` (verbatim)",
        ),
        ("src/a.py:12 -1 if index is None else index", OWN, "-1 if index is None else index"),
        ("config.yml:42-column-limit", "config.yml", "-column-limit"),
        # Backticks are taken off only when they wrap the whole rest.
        (
            "src/a.py:12 -- total = compute(a, b)  (the `compute` call)",
            OWN,
            "total = compute(a, b) (the `compute` call)",
        ),
        # The shapes the poster's own tests carry.
        (
            "tools/interval_stats.py:162 (verbatim): `merged[-1] = (merged[-1][0], end)`",
            "tools/interval_stats.py",
            "merged[-1] = (merged[-1][0], end)",
        ),
        (
            "tools/interval_stats.py:15–16: merged[-1] = (merged[-1][0], end)",
            "tools/interval_stats.py",
            "merged[-1] = (merged[-1][0], end)",
        ),
        (
            "`merged[-1] = (merged[-1][0], end)` -- tools/interval_stats.py:15",
            "tools/interval_stats.py",
            "merged[-1] = (merged[-1][0], end)",
        ),
    )

    # sha256("persona-review/quote-key/1\0" + claim), each reproducible outside Python with
    # `printf 'persona-review/quote-key/1\0<claim>' | shasum -a 256`.
    QUOTE_KEYS: dict[str, str] = {
        CODE: "f6d53bad0a55cc1481554a6bbc8409829d9663465c7a3069569a784f951ad97c",
        "foo(bar, baz)": "a1eac5fd51a84970a6897ba2f848f1f07001d0c4ec7b8eb97a3ac979df0d4147",
        "return renderPage(props);": (
            "9abc73d462d4cb3277a9269340fe56936e230c6ee64799115ee2fb0014af75c3"
        ),
        "return bill(account)": "606eafd0681fda2ab232ea5812d0d13a2c8ab0a015af3899622c3718330aaa1e",
        "merged[-1] = (merged[-1][0], end)": (
            "2843e8f84c816e9b966ee53801fe30ce07a5c5c24f0f7a4b2d20768d0af46fda"
        ),
        "total = compute(a, b) src/a.py:12": (
            "cf559ec1534669c7e302ee9faed89278afec47fa45d701962ecafd10edb447a5"
        ),
        "total = a- src/a.py:12": (
            "7347abb80e57b21f2e1c2ea794aff5929ea9c7c04a10bc1a571a1a836432125d"
        ),
        "total = compute(a, b) (src/a.py:12": (
            "fad40a85775f1f0d2ba14d679d2a699089ab905901c6d6c4d8d1e3695c43c9f6"
        ),
        "x … src/a.py:12": "8b021871d435fcd08fcbbaed9791b9a9691da7c81966078c57792dc82feb3904",
        "`total = compute(a, b)` -- src/a.py:12, called twice": (
            "e5cfafacc298867b95c3c3217fbd05201af58aa0ed8eb64f263c43ec1d49c7f0"
        ),
        "timeout:30 -- seconds": "569f99f20b19f5f460c0d938dbcb47da31bc75dc418e7fa76f35317ee8916252",
        "`timeout`:30 -- seconds": (
            "ac1c5725838a7d1bf30c4e9bc1dd1921d892a653b3797f6a84526cac500346d6"
        ),
        "https://example.com:443 - the port the proxy listens on": (
            "19b2efec6d600516a24c6d00974c66aa97f5d021ee50b88be20ebad7b3b01c38"
        ),
        ":12 -- retry(limit)": "ce87b1e8d3a814e6ae270c81fd60823fe7cd6744f29a8ee673d1449e0bb23c03",
        "retry(3) # see src/a.py:12 for the limit": (
            "078393f24ca9e0b88c89527719aa02d3a8b2c76dfeb0580dda2147deafe1e2ba"
        ),
        "a = 1 / b = 2": "c45fbc9af3195b5caccba8fc12914bf79201155a8bac8b8ac59039cc52f68902",
        "first() ... last()": "efe561ceaff7295351e0c532c42072ed6505880928a46ec38b1baba9de8a75c2",
        'log("one\\ntwo")': "80b17a3b6e0d4e79905127a98b633e57754e45ad1cda77540d9db562ba24c630",
        "a = 1 b = 2": "5e0b3835e0d80a4fd2b78939e7144a12db1d693d5c9e42f7869064945089534d",
        "a = 1 retries:3 -- b = 2": (
            "97f50348864a25d7c44d668b6ddf99339bc0d15ead98dd144aa2740600ebc52a"
        ),
        "a = 1 b = 2 # see src/x.py:82 for c": (
            "2a39b5e50e42a47f90e0a0c23744e1f34eb81b4c9a9d9cc2ebe302d04b904166"
        ),
        "a = 1 `b = 2`": "57e371fa25a3711fc6659982926ae672be720e1f054a0178363b648cfa777019",
        "a = 1": "679e59eae4622f19a810bac3bf4344383ff50238de3f265128a981eaf790cbb6",
        "-- SELECT id FROM users": (
            "d3a523425cdc9275fbfd8f1e0846bdb10836ba3b96fb6aad64b7a4d33b3b8d0b"
        ),
        "`total = compute(a, b)` (verbatim)": (
            "db26194a59084ae1b2aa57d164cddfc9c0bd4d68183ac6402b0ed94805caf864"
        ),
        "-1 if index is None else index": (
            "e91ec774b0b88bfdff4be7946efd62a84b4aa09c675cec86a8b885ea2c7f233b"
        ),
        "-column-limit": "ab755a113e1b4169b84ab122624213501e4ec7401df4d2f25465cf0f1cc7ac0d",
        "set -euo pipefail": "d32999f83f5146ad89b77f328f7959bfd5eba509f1c0634bee75549c83c77574",
        "$(CC) -o app main.c": "da72122c02dfcb681a06d5ffdce9875947f84f1308e2f1b3fb6db136ec686bff",
        "total = compute(a, b) (the `compute` call)": (
            "15253257dd06b90ce021219c42fff8a2922a4a53b22b6591ce8fd77ebd0b187d"
        ),
    }

    @pytest.mark.parametrize(("quote", "own", "claim"), CLAIMS)
    def test_the_claim(self, quote: str, own: str | None, claim: str):
        assert findings.claim_v1(quote, own) == claim

    @pytest.mark.parametrize(("quote", "own", "claim"), CLAIMS)
    def test_the_quote_key(self, quote: str, own: str | None, claim: str):
        # None for an empty claim: a key shared by every citation-only quote would let one
        # finding adopt another's thread.
        assert findings.quote_key_v1(findings.claim_v1(quote, own)) == self.QUOTE_KEYS.get(claim)

    def test_every_vector_is_pinned(self):
        claims = {claim for _, _, claim in self.CLAIMS}
        assert claims - {""} == set(self.QUOTE_KEYS)
        assert findings.quote_key_v1("") is None

    @pytest.mark.parametrize(("claim", "key"), sorted(QUOTE_KEYS.items()))
    def test_the_quote_key_is_the_documented_hash(self, claim: str, key: str):
        spelled = hashlib.sha256(f"persona-review/quote-key/1\0{claim}".encode()).hexdigest()
        assert spelled == key

    def test_a_claim_holding_a_lone_surrogate_still_has_a_key(self):
        # A lone surrogate is valid JSON (`"\ud800"`), and the key must not raise on it.
        claim = "x = '\ud800'"
        assert json.loads("\"x = '\\ud800'\"") == claim
        assert findings.quote_key_v1(claim) == (
            "38995cc362aa6d34b0e67804b18d3edb31b1fec8207b6770e3c1c394a85098af"
        )

    # sha256("persona-review/evidence-key/1\0" + path + "\0" + the span's non-blank lines,
    # each whitespace-collapsed, joined by "\n"), reproducible the same way.
    EVIDENCE_ONE_LINE = "b945015ae72d102ebf4ffc5c5cba50ec5e4abdf965bb3dee64925369f425b464"
    EVIDENCE_TWO_LINES = "0c5b6db6414216e304f8f52adf0eaf5447614658bc961e90650e88a2503a399a"

    def test_the_evidence_key(self):
        assert (
            findings.evidence_key_v1("src/a.py", ["    total = compute(a, b)"])
            == self.EVIDENCE_ONE_LINE
        )
        assert (
            findings.evidence_key_v1("src/a.py", ["  def   f():", "    return 1"])
            == self.EVIDENCE_TWO_LINES
        )

    def test_a_blank_line_inside_the_span_does_not_move_the_evidence_key(self):
        spans = (["def f():", "", "    return 1"], ["def f():", "   \t", "    return 1"])
        for lines in spans:
            assert findings.evidence_key_v1("src/a.py", lines) == self.EVIDENCE_TWO_LINES

    def test_the_evidence_key_carries_the_path(self):
        assert findings.evidence_key_v1("lib/a.py", ["    total = compute(a, b)"]) == (
            "cc65c9d107387d4392abbac4bd1a0bf61e6c62846315906df5118c1da8b559f8"
        )

    @PROPERTY
    @given(
        cite=st.one_of(st.just(""), _leading_citations),
        code=_claimed_code,
        ticks=st.sampled_from(["{}", "`{}`", "` {} `"]),
    )
    def test_a_leading_citation_or_backticks_never_move_the_quote_key(
        self, cite: str, code: str, ticks: str
    ):
        quote = cite + ticks.format(code)
        assert findings.quote_key_v1(findings.claim_v1(quote, self.OWN)) == findings.quote_key_v1(
            findings.claim_v1(code, self.OWN)
        ), quote

    @PROPERTY
    @given(
        cite=st.one_of(st.just(""), _leading_citations),
        code=_claimed_code,
        ticks=st.sampled_from(["{}", "`{}`"]),
    )
    def test_the_claim_is_its_own_claim(self, cite: str, code: str, ticks: str):
        # Over the shapes lenses write. Not over any text: a quote opening with two
        # citations loses one per pass, because only the edge citation is the lens's.
        claim = findings.claim_v1(cite + ticks.format(code), self.OWN)
        assert findings.claim_v1(claim, self.OWN) == claim

    @PROPERTY
    @given(
        path=st.sampled_from(["app/[id]/page.tsx", "app/(auth)/page.tsx", "src/a.py"]),
        form=st.sampled_from(["{}", "./{}", "/{}", "{}/"]),
        separator=st.sampled_from(["/", "//", "/./", "\\"]),
        written=st.sampled_from(["{}", "./{}"]),
        decoration=st.sampled_from(["{}", "**{}**", "({})", "`{}`", "[{}]", "`path`"]),
        shape=st.sampled_from(
            ["{cite} -- {code}", "{code} -- {cite}", "{cite}: {code}\n{cite}: {code}"]
        ),
        code=_claimed_code,
    )
    def test_every_spelling_of_the_findings_file_makes_one_claim(
        self,
        path: str,
        form: str,
        separator: str,
        written: str,
        decoration: str,
        shape: str,
        code: str,
    ):
        # `norm_path` is how a poster keys the file, so every spelling it reads as one names
        # the same file, in the finding's `file` and in the quote alike.
        own = form.format(path.replace("/", separator))
        assert findings.norm_path(own) == path
        quote = shape.format(cite=_decorated(written.format(path), "2", decoration), code=code)
        plain = shape.format(cite=_decorated(path, "2", decoration), code=code)
        assert findings.claim_v1(quote, own) == findings.claim_v1(plain, path), (quote, own)


# The file every `TestAnchors` case reads, one entry per line so a number is easy to check.
_ANCHORED = (
    "import billing",
    "",
    "def charge(account):",
    "    total = compute_total(account)",
    "    return bill(account, total)",
    "",
    "def refund(account):",
    "    total = compute_total(account)",
    "    return credit(account, total)",
    "",
    "def audit(account):",
    '    log_event("audit", account)',
    "    result = summarize(",
    "        account, verbose=True)",
    "    first_value = compute(a)",
    "",
    "    second_value = compute(b)",
)


def _citing(path: str, line: int, code: str) -> str:
    return f"{path}:{line} -- {code}"


def _as_finding(file: str | None, line: int | bool | None, quote: str | None) -> dict[str, Any]:
    return {"file": file, "line": line, "first_evidence": quote}


_ANCHORED_CODE = [line.strip() for line in _ANCHORED if line.strip()]

# Quotes that reach every state against `_ANCHORED`: whole lines, lines cited from the file,
# its basename or another file, two lines at once, and text it does not hold.
_anchored_quotes = st.one_of(
    st.sampled_from(_ANCHORED_CODE),
    st.builds(
        _citing,
        st.sampled_from(["src/f.py", "f.py", "./src/f.py", "src/g.py"]),
        st.integers(min_value=1, max_value=20),
        st.sampled_from(_ANCHORED_CODE),
    ),
    st.sampled_from(["\n".join(_ANCHORED[n : n + 2]) for n in range(len(_ANCHORED) - 1)]),
    st.text(max_size=20),
)

# The poster's own `norm_path` cases, shared so the two ends agree on which path an entry
# names. (path, what it normalizes to).
NORMALIZED_PATHS = (
    ("../../../../evil-org/payload/x.py", "evil-org/payload/x.py"),
    ("..%2fevil/x.py", "..%2fevil/x.py"),
    ("%2e%2e/%2e%2e/evil/x.py", "%2e%2e/%2e%2e/evil/x.py"),
    ("\\..\\..\\evil\\x.py", "evil/x.py"),
    ("/etc/passwd", "etc/passwd"),
    ("....//....//evil/x.py", "..../..../evil/x.py"),
    ("a/./../../evil.py", "a/evil.py"),
    ("../" * 40 + "evil.py", "evil.py"),
    ("tools/a.py", "tools/a.py"),
    ("tools/a b(c).py", "tools/a b(c).py"),
    (".. /../evil.py", "evil.py"),
    ("..\u2028/../evil.py", "evil.py"),
    ("a /", "a"),
    (".. /", ""),
    ("a /..", "a"),
)

_STATES = ("verified", "relocated", "ambiguous", "not_found", "unverifiable", "no_evidence")


class TestAnchors:
    """`locate` and `anchors`: where each finding's quote is, and the keys a poster writes.

    Every state is reached from a small tree, and each case asserts the span, `via` and
    both keys, since those are what a poster places and deduplicates by.
    """

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tree = self.dir / "tree"
        src = self.tree / "src"
        src.mkdir(parents=True)
        (src / "f.py").write_text("\n".join(_ANCHORED) + "\n", encoding="utf-8")
        (src / "alias.py").symlink_to(src / "f.py")
        outside = self.dir / "outside.py"
        outside.write_text("\n".join(_ANCHORED) + "\n", encoding="utf-8")
        (src / "out.py").symlink_to(outside)

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _locate(self, quote: str | None, line: Any = 5, file: Any = "src/f.py") -> dict[str, Any]:
        finding: dict[str, Any] = {"line": line}
        if file is not None:
            finding["file"] = file
        if quote is not None:
            finding["first_evidence"] = quote
        return cast(dict[str, Any], findings.locate(finding, self.tree))

    @staticmethod
    def _evidence(path: str, *numbers: int) -> str:
        return findings.evidence_key_v1(path, [_ANCHORED[n - 1] for n in numbers])

    def test_a_quote_on_the_findings_line_is_verified(self):
        entry = self._locate("src/f.py:5 -- return bill(account, total)", line=5)
        assert entry == {
            "file": "src/f.py",
            "path": "src/f.py",
            "line": 5,
            "state": "verified",
            "via": "line",
            "start": 5,
            "end": 5,
            "occurrences": 1,
            "candidates": [],
            "reason": "on the finding's line",
            "quote_key": findings.quote_key_v1("return bill(account, total)"),
            "evidence_key": self._evidence("src/f.py", 5),
        }

    def test_the_findings_line_wins_over_the_line_the_quote_cites(self):
        # The code is on lines 4 and 8. The finding says 8, so 8 is corroborated, whatever
        # line the lens wrote beside the quote.
        entry = self._locate("src/f.py:4 -- total = compute_total(account)", line=8)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "verified",
            "line",
            8,
            8,
        )
        assert entry["evidence_key"] == self._evidence("src/f.py", 8)

    def test_a_quote_off_its_line_moves_to_the_line_it_cites(self):
        # On lines 4 and 8, so searching alone is ambiguous: the citation says which.
        entry = self._locate("src/f.py:8 -- total = compute_total(account)", line=30)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            8,
            8,
        )
        assert (entry["occurrences"], entry["candidates"]) == (2, [])
        assert entry["quote_key"] == findings.quote_key_v1("total = compute_total(account)")
        assert entry["evidence_key"] == self._evidence("src/f.py", 8)

    def test_a_quote_whose_citations_both_hold_it_moves_to_the_first_it_cites(self):
        # Each line cites one of the two places the code is, and the finding names neither.
        code = "total = compute_total(account)"
        entry = self._locate(f"src/f.py:8: {code}\nsrc/f.py:4: {code}", line=30)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            8,
            8,
        )
        assert entry["evidence_key"] == self._evidence("src/f.py", 8)

    @pytest.mark.parametrize(
        "quote",
        [
            # Spelled as the finding's own path, and resolving to it.
            "./src/f.py:5 -- return bill(account, total)",
            # Its basename, which resolves to nothing at the root.
            "f.py:5 -- return bill(account, total)",
            # Spelled so that no literal of its path matches, and naming it only once resolved.
            "src//f.py:5 -- return bill(account, total)",
            "src/../src/f.py:5 -- return bill(account, total)",
            "src/alias.py:5 -- return bill(account, total)",
        ],
    )
    def test_a_citation_of_the_findings_own_file_places_it(self, quote: str):
        # One occurrence, so a search would find the same line: `via` says the citation did.
        entry = self._locate(quote, line=40)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            5,
            5,
        )
        assert entry["evidence_key"] == self._evidence("src/f.py", 5)

    def test_a_quote_occurring_once_moves_to_that_line(self):
        entry = self._locate("return credit(account, total)", line=2)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "search",
            9,
            9,
        )
        assert (entry["occurrences"], entry["candidates"]) == (1, [])
        assert entry["evidence_key"] == self._evidence("src/f.py", 9)

    def test_a_quote_occurring_twice_off_its_line_is_ambiguous(self):
        entry = self._locate("total = compute_total(account)", line=1)
        assert entry["state"] == "ambiguous"
        assert (entry["via"], entry["start"], entry["end"], entry["evidence_key"]) == (
            None,
            None,
            None,
            None,
        )
        assert (entry["occurrences"], entry["candidates"]) == (2, [[4, 4], [8, 8]])
        assert entry["quote_key"] == findings.quote_key_v1("total = compute_total(account)")
        assert entry["reason"] == "occurs 2 times in the file"

    def test_an_ambiguous_quote_lists_at_most_twenty_places(self):
        (self.tree / "src" / "many.py").write_text(
            "    retry_request(session)\n" * 25, encoding="utf-8"
        )
        entry = self._locate("retry_request(session)", line=100, file="src/many.py")
        assert (entry["state"], entry["occurrences"]) == ("ambiguous", 25)
        assert entry["candidates"] == [[n, n] for n in range(1, 21)]

    def test_a_quote_the_file_does_not_carry_is_not_found(self):
        entry = self._locate("src/f.py:5 -- return charge(account)", line=5)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "not_found",
            None,
            None,
            None,
        )
        assert (entry["occurrences"], entry["candidates"], entry["evidence_key"]) == (0, [], None)
        assert entry["quote_key"] == findings.quote_key_v1("return charge(account)")
        assert entry["path"] == "src/f.py"

    def test_the_span_is_the_lines_the_match_covers(self):
        # One line of quote, two lines of file: a window sized by the quote would stop at 13.
        entry = self._locate("src/f.py:13 -- result = summarize( account, verbose=True)", line=13)
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 13, 14)
        assert entry["evidence_key"] == self._evidence("src/f.py", 13, 14)

    def test_a_blank_line_inside_the_match_is_inside_the_span(self):
        entry = self._locate("first_value = compute(a)\n\n    second_value = compute(b)", line=15)
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 15, 17)
        assert entry["evidence_key"] == self._evidence("src/f.py", 15, 16, 17)

    def test_a_locator_on_every_line_matches_the_lines_it_quotes(self):
        quote = (
            "src/f.py:4: total = compute_total(account)\nsrc/f.py:5: return bill(account, total)"
        )
        entry = self._locate(quote, line=4)
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 4, 5)
        assert entry["quote_key"] == findings.quote_key_v1(
            "total = compute_total(account) return bill(account, total)"
        )

    def test_a_snippet_is_checked_at_the_line_its_citation_names(self):
        # Two citations, each owning a snippet. Neither the whole remainder nor the primary
        # claim is on any line; the second snippet is on the line it cites and the finding's.
        quote = (
            "src/f.py:4 -- total = compute_total(account); "
            'src/f.py:12 -- log_event("audit", account)'
        )
        entry = self._locate(quote, line=12)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "verified",
            "line",
            12,
            12,
        )
        assert entry["occurrences"] == 0
        assert entry["evidence_key"] == self._evidence("src/f.py", 12)

    def test_a_snippet_is_never_searched_for_off_its_cited_line(self):
        # The second snippet is on lines 4 and 8, and the finding says 8, but it cites 9. Found
        # wherever it occurs, a fragment would verify, or be ambiguous, on text it never cited.
        quote = (
            'src/f.py:12 -- log_event("nothing", account); '
            "src/f.py:9 -- total = compute_total(account)"
        )
        entry = self._locate(quote, line=8)
        assert (entry["state"], entry["occurrences"], entry["candidates"]) == ("not_found", 0, [])

    @pytest.mark.parametrize(
        ("second", "occurrences"),
        [
            # On no line of the file.
            ("src/f.py:5: return charge(account, fee)", 0),
            # In the file, but not on the line it cites. The whole claim is on 4-5, so a search
            # finds it once.
            ("src/f.py:12: return bill(account, total)", 1),
            # Past the end of the file.
            ("src/f.py:40: return bill(account, total)", 1),
            ("src/f.py:" + "9" * 5000 + ": return bill(account, total)", 1),
            # Joined by `; ` rather than a newline.
            ("; src/f.py:5 -- return charge(account, fee)", 0),
        ],
        ids=["absent", "elsewhere", "past-the-end", "unparsable-line", "semicolon-joined"],
    )
    def test_a_true_snippet_does_not_place_a_false_one_beside_it(
        self, second: str, occurrences: int
    ):
        # The first snippet is on line 4, the finding's line. Placed, the comment would publish
        # the second as code the file holds.
        joiner = "" if second.startswith(";") else "\n"
        quote = f"src/f.py:4: total = compute_total(account){joiner}{second}"
        entry = self._locate(quote, line=4)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "not_found",
            None,
            None,
            None,
        )
        assert (entry["occurrences"], entry["candidates"]) == (occurrences, [])
        assert entry["reason"] == "a snippet is not on the lines its citation names"
        assert entry["evidence_key"] is None
        assert entry["quote_key"] == findings.quote_key_v1(findings.claim_v1(quote, "src/f.py"))

    def _other(self) -> None:
        (self.tree / "src" / "g.py").write_text(
            "import credit\n    return credit(account, total)\n", encoding="utf-8"
        )

    @pytest.mark.parametrize(
        "second",
        [
            # On no line of that file.
            "src/g.py:2: this is not in g.py at all",
            # In that file, but not on the line it cites.
            "src/g.py:1: return credit(account, total)",
            # In a file the tree does not have.
            "src/h.py:2: return credit(account, total)",
            # Through a link to a file outside the tree that holds it on line 9: never read.
            "src/out.py:9: return credit(account, total)",
        ],
        ids=["absent", "elsewhere", "no-such-file", "outside-the-tree"],
    )
    def test_a_snippet_of_another_file_is_checked_in_that_file(self, second: str):
        self._other()
        entry = self._locate(f"src/f.py:5: return bill(account, total)\n{second}", line=5)
        assert (entry["state"], entry["start"], entry["evidence_key"]) == ("not_found", None, None)
        assert entry["reason"] == "a snippet is not on the lines its citation names"

    def test_a_true_snippet_of_another_file_leaves_the_quote_placed(self):
        # The second snippet is also on line 9 of this file, but it cites line 2 of g.py.
        self._other()
        quote = "src/f.py:5: return bill(account, total)\nsrc/g.py:2: return credit(account, total)"
        entry = self._locate(quote, line=5)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "verified",
            "line",
            5,
            5,
        )
        assert entry["evidence_key"] == self._evidence("src/f.py", 5)

    def _ranged(self) -> None:
        body = [f"# line {n}" for n in range(1, 41)]
        body[9] = body[20] = "    total = compute_total(items)"
        (self.tree / "src" / "range.py").write_text("\n".join(body) + "\n", encoding="utf-8")

    @pytest.mark.parametrize(
        "quote",
        [
            "src/range.py:20-22 -- total = compute_total(items)",
            "src/range.py:20–22: total = compute_total(items)",
            "**src/range.py:20:5—22** total = compute_total(items)",
            "`total = compute_total(items)` -- src/range.py:20-22",
        ],
    )
    def test_a_quote_inside_the_range_it_cites_moves_there(self, quote: str):
        # On lines 10 and 21, so searching alone is ambiguous: the range covers 21 without
        # starting on it.
        self._ranged()
        entry = self._locate(quote, line=5, file="src/range.py")
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            21,
            21,
        )
        assert (entry["occurrences"], entry["candidates"]) == (2, [])

    def test_a_range_written_backwards_cites_its_first_line(self):
        self._ranged()
        quote = "src/range.py:21-20 -- total = compute_total(items)"
        entry = self._locate(quote, line=5, file="src/range.py")
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            21,
            21,
        )

    def test_a_range_end_too_long_to_parse_cites_its_first_line(self):
        entry = self._locate("src/f.py:5-" + "9" * 5000 + " -- return bill(account, total)", 30)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            5,
            5,
        )

    def test_a_snippet_is_checked_across_the_range_its_citation_names(self):
        # The first snippet is on line 5, inside the 3-5 it cites, and the second on the 9 it
        # cites. Neither the whole remainder nor the primary claim is on any line.
        first, second = "return bill(account, total)", "return credit(account, total)"
        entry = self._locate(f"src/f.py:3-5 -- {first}\nsrc/f.py:9 -- {second}", line=30)
        assert (entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "relocated",
            "citation",
            5,
            5,
        )
        assert entry["evidence_key"] == self._evidence("src/f.py", 5)

    @pytest.mark.parametrize("end", ["", "\n"])
    def test_a_doubly_escaped_quote_matches_as_the_lines_it_escaped(self, end: str):
        # A newline ending the quote is not one inside it.
        quote = "src/f.py:3 -- def charge(account):\\n\\ttotal = compute_total(account)" + end
        entry = self._locate(quote, line=3)
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 3, 4)
        # Matching only: the key is the claim as written.
        assert entry["quote_key"] == findings.quote_key_v1(
            "def charge(account):\\n\\ttotal = compute_total(account)"
        )

    def test_an_escape_beside_a_real_newline_is_the_codes_own(self):
        quote = (
            "def charge(account):\\n    total = compute_total(account)\nreturn bill(account, total)"
        )
        entry = self._locate(quote, line=3)
        assert (entry["state"], entry["occurrences"]) == ("not_found", 0)

    def test_an_escaped_quote_is_held_to_the_floor_once_unescaped(self):
        # Unescaped, it is empty, and an empty string occurs everywhere.
        entry = self._locate("src/f.py:2 -- \\n\\n\\n\\n\\n\\n", line=2)
        assert (entry["state"], entry["occurrences"], entry["candidates"]) == ("not_found", 0, [])

    def test_a_citation_line_too_long_to_parse_is_a_state_not_a_crash(self):
        entry = self._locate("src/f.py:" + "9" * 5000 + " -- return bill(account, total)", line=5)
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 5, 5)

    def test_a_line_that_is_a_boolean_is_no_line(self):
        entry = self._locate("import billing", line=True)
        assert entry["line"] is None
        assert (entry["state"], entry["via"], entry["start"]) == ("relocated", "search", 1)

    def test_lines_are_numbered_as_a_diff_numbers_them(self):
        # A form feed is a line break to `splitlines` and not to a diff, which would put the
        # code on line 5.
        (self.tree / "src" / "ff.py").write_text(
            "import billing\n\x0c\ndef g(account):\n    return bill(account, total)\n",
            encoding="utf-8",
        )
        entry = self._locate("return bill(account, total)", line=4, file="src/ff.py")
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 4, 4)

    def test_a_crlf_file_anchors_like_its_lf_twin(self):
        (self.tree / "src" / "crlf.py").write_bytes(
            "\r\n".join(_ANCHORED).encode("utf-8") + b"\r\n"
        )
        entry = self._locate("src/crlf.py:5 -- return bill(account, total)", file="src/crlf.py")
        assert (entry["state"], entry["start"], entry["end"]) == ("verified", 5, 5)
        assert entry["evidence_key"] == self._evidence("src/crlf.py", 5)

    @pytest.mark.parametrize("file", ["./src/f.py", "src//f.py", "src/./f.py"])
    def test_a_path_spelled_another_way_still_names_its_file(self, file: str):
        entry = self._locate("return bill(account, total)", file=file)
        assert (entry["path"], entry["state"]) == ("src/f.py", "verified")

    @pytest.mark.parametrize(
        ("file", "quote"),
        [
            ("./app/[id]/page.tsx", "app/[id]/page.tsx:2 -- return renderPage(props);"),
            ("app/[id]/page.tsx", "./app/[id]/page.tsx:2 -- return renderPage(props);"),
        ],
    )
    def test_a_bracketed_path_spelled_another_way_still_cites_its_file(self, file: str, quote: str):
        # `_REFERENCE` reads either citation as one of `id]/page.tsx`, another file, so only
        # the finding's own path can say which file the quote cites.
        page = self.tree / "app" / "[id]"
        page.mkdir(parents=True)
        (page / "page.tsx").write_text(
            "export default function Page() {\n  return renderPage(props);\n}\n",
            encoding="utf-8",
        )
        entry = self._locate(quote, line=7, file=file)
        assert (entry["path"], entry["state"], entry["via"], entry["start"], entry["end"]) == (
            "app/[id]/page.tsx",
            "relocated",
            "citation",
            2,
            2,
        )
        assert entry["quote_key"] == findings.quote_key_v1("return renderPage(props);")

    def test_a_file_whose_name_is_padded_is_another_path(self):
        # The poster strips each segment, so it would look up `src/f.py`, not this file.
        (self.tree / " src ").mkdir()
        (self.tree / " src " / " f.py").write_text("\n".join(_ANCHORED), encoding="utf-8")
        entry = self._locate("return bill(account, total)", file=" src / f.py")
        assert (entry["state"], entry["reason"]) == ("unverifiable", "resolves to another path")

    @pytest.mark.parametrize(
        ("file", "reason"),
        [
            ("src/missing.py", "no such file"),
            ("src/f\x00.py", "no such file"),
            ("src", "not a file"),
            ("src/out.py", "outside the tree"),
            ("../outside.py", "outside the tree"),
            # Through a link or a `..` the file read is another path than the one a poster
            # would place the finding on.
            ("src/alias.py", "resolves to another path"),
            ("src/../src/f.py", "resolves to another path"),
        ],
    )
    def test_a_file_that_cannot_be_read_as_named_is_unverifiable(self, file: str, reason: str):
        entry = self._locate("src/f.py:5 -- return bill(account, total)", file=file)
        assert (entry["state"], entry["reason"], entry["path"]) == ("unverifiable", reason, None)
        assert (entry["start"], entry["end"], entry["evidence_key"]) == (None, None, None)
        assert entry["quote_key"] == findings.quote_key_v1("return bill(account, total)")

    def test_an_absolute_path_outside_the_tree_is_unverifiable(self):
        entry = self._locate("return bill(account, total)", file=str(self.dir / "outside.py"))
        assert (entry["state"], entry["reason"]) == ("unverifiable", "outside the tree")

    def test_an_unreadable_file_is_unverifiable(self):
        if os.geteuid() == 0:
            pytest.skip("root reads a mode-000 file, so the case cannot be produced")
        secret = self.tree / "src" / "secret.py"
        secret.write_text("return bill(account, total)\n", encoding="utf-8")
        secret.chmod(0o000)
        try:
            entry = self._locate("return bill(account, total)", file="src/secret.py")
        finally:
            secret.chmod(0o600)
        assert (entry["state"], entry["reason"], entry["path"]) == (
            "unverifiable",
            "unreadable",
            None,
        )

    def test_a_finding_with_no_file_is_unverifiable(self):
        entry = self._locate("return bill(account, total)", file=None)
        assert (entry["file"], entry["state"], entry["reason"]) == (None, "unverifiable", "no file")

    def test_a_quote_citing_only_another_file_is_unverifiable(self):
        # The code IS in src/f.py, and a search would find it: the quote says it is elsewhere.
        entry = self._locate("src/g.py:5 -- return bill(account, total)", line=5)
        assert (entry["state"], entry["reason"], entry["path"]) == (
            "unverifiable",
            "cites only other files",
            "src/f.py",
        )
        assert entry["quote_key"] == findings.quote_key_v1("return bill(account, total)")

    def test_a_quote_under_the_floor_is_unverifiable(self):
        # It is on line 5, and short enough to be on many lines of any tree.
        entry = self._locate("src/f.py:5 -- return bill", line=5)
        assert (entry["state"], entry["reason"]) == (
            "unverifiable",
            "too short (11 chars, floor 12)",
        )
        assert entry["quote_key"] == findings.quote_key_v1("return bill")

    @pytest.mark.parametrize(
        ("quote", "reason"),
        [
            (None, "no quote"),
            ("   ", "no quote"),
            ("src/f.py:5", "the quote is only a citation"),
            # Before the file it cites is looked at: an empty claim has no key, and a poster
            # refuses any other state without one.
            ("src/g.py:5", "the quote is only a citation"),
        ],
    )
    def test_a_finding_with_no_code_to_look_for_has_no_evidence(
        self, quote: str | None, reason: str
    ):
        entry = self._locate(quote, line=5)
        assert (entry["state"], entry["reason"]) == ("no_evidence", reason)
        assert (entry["path"], entry["quote_key"], entry["evidence_key"]) == (None, None, None)

    @pytest.mark.parametrize(("path", "expected"), NORMALIZED_PATHS)
    def test_paths_normalize_as_the_poster_normalizes_them(self, path: str, expected: str):
        assert findings.norm_path(path) == expected

    # ---- the document ----

    def _artifact(self, raw: bytes, head: str | None) -> Path:
        art = self.dir / "correctness-grok.json"
        art.write_bytes(raw)
        if head is not None:
            (self.dir / "correctness-grok-provenance.json").write_text(
                json.dumps({"head_sha": head, "run_stats": {"tool_calls": 3}}), encoding="utf-8"
            )
        return art

    def test_the_document(self):
        items: list[Any] = [
            finding(file="src/f.py", line=5, first_evidence="return bill(account, total)"),
            "not a finding",
            finding(file="src/f.py", line=1, first_evidence="src/f.py:9"),
        ]
        raw = json.dumps({"findings": items}).encode("utf-8")
        art = self._artifact(raw, "a" * 40)
        doc = findings.anchors(str(art), raw, self.tree)
        assert list(doc) == [
            "anchors_version",
            "artifact",
            "artifact_sha256",
            "tree",
            "head",
            "findings",
        ]
        assert doc["anchors_version"] == 1
        assert doc["artifact"] == str(art)
        assert doc["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
        assert doc["tree"] == str(self.tree.resolve())
        assert doc["head"] == "a" * 40
        rows = cast(list[dict[str, Any]], doc["findings"])
        # Keyed by raw position, the entry that is not a finding skipped rather than renumbered.
        assert [(row["#"], row["state"]) for row in rows] == [(1, "verified"), (3, "no_evidence")]
        assert list(rows[0]) == [
            "#",
            "file",
            "path",
            "line",
            "state",
            "via",
            "start",
            "end",
            "occurrences",
            "candidates",
            "reason",
            "quote_key",
            "evidence_key",
        ]
        json.dumps(doc, allow_nan=False)

    def test_the_bytes_given_are_the_bytes_located(self):
        # Hashed and located from one read: a file rewritten in between changes neither.
        raw = json.dumps({"findings": [finding(file="src/f.py", line=5)]}).encode("utf-8")
        art = self._artifact(b'{"findings": []}', "a" * 40)
        doc = findings.anchors(str(art), raw, self.tree)
        assert doc["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
        assert len(cast(list[Any], doc["findings"])) == 1

    def test_a_tree_reached_through_a_link_still_holds_its_files(self):
        # Containment compares resolved paths, so the root has to be resolved too.
        link = self.dir / "link"
        link.symlink_to(self.tree)
        items = [finding(file="src/f.py", line=5, first_evidence="return bill(account, total)")]
        raw = json.dumps({"findings": items}).encode("utf-8")
        doc = findings.anchors(str(self._artifact(raw, None)), raw, link)
        assert doc["tree"] == str(self.tree.resolve())
        rows = cast(list[dict[str, Any]], doc["findings"])
        assert (rows[0]["state"], rows[0]["path"]) == ("verified", "src/f.py")
        entry = findings.locate(items[0], link)
        assert (entry["state"], entry["path"]) == ("verified", "src/f.py")

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ("0123456789abcdef" * 2 + "01234567", "0123456789abcdef" * 2 + "01234567"),
            ("0123456789abcdef" * 4, "0123456789abcdef" * 4),
            (None, "unresolved: no provenance"),
            ("unresolved: HEAD", "unresolved: the provenance records no head commit"),
            (
                "0123456789ABCDEF" * 2 + "01234567",
                "unresolved: the provenance records no head commit",
            ),
            ("a" * 41, "unresolved: the provenance records no head commit"),
            ("", "unresolved: the provenance records no head commit"),
        ],
    )
    def test_the_head_is_the_reviewed_commit_or_unresolved(self, head: str | None, expected: str):
        raw = json.dumps({"findings": []}).encode("utf-8")
        art = self._artifact(raw, head)
        assert findings.anchors(str(art), raw, self.tree)["head"] == expected

    def test_a_head_recorded_as_no_string_is_unresolved(self):
        raw = json.dumps({"findings": []}).encode("utf-8")
        art = self._artifact(raw, None)
        (self.dir / "correctness-grok-provenance.json").write_text(
            json.dumps({"head_sha": 7}), encoding="utf-8"
        )
        assert findings.anchors(str(art), raw, self.tree)["head"] == (
            "unresolved: the provenance records no head commit"
        )

    def test_a_verdicts_artifact_is_refused(self):
        raw = json.dumps({"verdicts": [{"#": 1, "validated": True}]}).encode("utf-8")
        art = self._artifact(raw, "a" * 40)
        with pytest.raises(findings.FindingsError, match="not a findings artifact"):
            findings.anchors(str(art), raw, self.tree)

    def test_bytes_that_are_not_an_artifact_are_refused(self):
        for raw in (b"\xff\xfe", b"[]", b"{}"):
            with pytest.raises(findings.FindingsError):
                findings.anchors(str(self.dir / "x.json"), raw, self.tree)

    # ---- the command ----

    def _main(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_the_command_prints_the_document_and_counts_every_state(self):
        bill = "return bill(account, total)"
        items: list[Any] = [
            finding(file="src/f.py", line=5, first_evidence=bill),
            finding(file="src/f.py", line=1, first_evidence=bill),
            finding(file="src/f.py", line=1, first_evidence="total = compute_total(account)"),
            finding(file="src/f.py", line=1, first_evidence="this text is in no file at all"),
            finding(file="src/missing.py", line=1, first_evidence=bill),
            finding(file="src/f.py", line=1, first_evidence="src/f.py:9"),
            finding(file="src/f.py", line=5, first_evidence=f"src/f.py:5 -- {bill}"),
        ]
        raw = json.dumps({"findings": items}).encode("utf-8")
        art = self._artifact(raw, "a" * 40)
        code, out, err = self._main(str(art), "--anchors", "-C", str(self.tree))
        assert code == 0, err
        doc = json.loads(out)
        assert doc == findings.anchors(str(art), raw, self.tree)
        assert doc["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
        assert [row["state"] for row in doc["findings"]] == [
            "verified",
            "relocated",
            "ambiguous",
            "not_found",
            "unverifiable",
            "no_evidence",
            "verified",
        ]
        assert err == (
            "ce-persona-findings: anchors: 7 findings (2 verified, 1 relocated, 1 ambiguous,"
            " 1 not found, 1 unverifiable, 1 no evidence)\n"
        )

    def test_the_summary_of_an_artifact_with_no_findings_counts_zero_in_every_state(self):
        raw = json.dumps({"findings": ["not a finding"]}).encode("utf-8")
        art = self._artifact(raw, "a" * 40)
        code, out, err = self._main(str(art), "--anchors", "-C", str(self.tree))
        assert code == 0, err
        assert json.loads(out)["findings"] == []
        assert err == (
            "ce-persona-findings: anchors: 0 findings (0 verified, 0 relocated, 0 ambiguous,"
            " 0 not found, 0 unverifiable, 0 no evidence)\n"
        )

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (("--anchors",), "--anchors wants -C <dir>"),
            (("--anchors", "-C"), "-C wants a directory"),
            (("--anchors", "-C", "ARTIFACT"), "is not a directory"),
            (("--anchors", "--json", "-C", "TREE"), "--anchors cannot be combined with --json"),
            (("--json", "--anchors", "-C", "TREE"), "--anchors cannot be combined with --json"),
            (("--anchors", "--show", "1", "-C", "TREE"), "cannot be combined with --show"),
            (("--anchors", "--show", "all", "-C", "TREE"), "cannot be combined with --show"),
            (("--anchors", "--return", "-C", "TREE"), "cannot be combined with --return"),
            (("--anchors", "--verify-quotes", "-C", "TREE"), "only applies to --return"),
        ],
    )
    def test_a_mode_it_cannot_serve_is_a_usage_error(self, args: tuple[str, ...], expected: str):
        raw = json.dumps({"findings": [finding(file="src/f.py", line=5)]}).encode("utf-8")
        art = self._artifact(raw, "a" * 40)
        given = {"ARTIFACT": str(art), "TREE": str(self.tree)}
        code, out, err = self._main(str(art), *(given.get(a, a) for a in args))
        assert code == 2, err
        assert expected in err, err
        assert out == ""

    def test_a_verdicts_artifact_is_a_usage_error_before_anything_is_located(self):
        raw = json.dumps({"verdicts": [{"#": 1, "validated": True, "reason": "r"}]}).encode()
        art = self._artifact(raw, "a" * 40)
        code, out, err = self._main(str(art), "--anchors", "-C", str(self.tree))
        assert code == 2, err
        assert "a verdict has no quote to locate" in err
        assert out == ""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(None, "cannot read findings artifact"), (b"[]", "is not a findings or verdicts")],
    )
    def test_an_artifact_it_cannot_use_is_a_data_error(self, raw: bytes | None, expected: str):
        art = self.dir / "correctness-grok.json" if raw is None else self._artifact(raw, None)
        code, out, err = self._main(str(art), "--anchors", "-C", str(self.tree))
        assert code == 1, err
        assert expected in err, err
        assert out == ""

    # ---- properties, over findings drawn from the shapes that reach every state ----

    @PROPERTY
    @given(
        entries=st.lists(
            st.one_of(
                st.builds(
                    _as_finding,
                    st.sampled_from(
                        ["src/f.py", "./src/f.py", "src/alias.py", "src/missing.py", "src", None]
                    ),
                    st.one_of(st.none(), st.booleans(), st.integers(min_value=-1, max_value=20)),
                    st.one_of(st.none(), _anchored_quotes),
                ),
                st.sampled_from(["text", 3, None, ["list"]]),
            ),
            max_size=6,
        )
    )
    def test_every_entry_keeps_the_contract_a_poster_enforces(self, entries: list[Any]):
        raw = json.dumps({"findings": entries}).encode("utf-8")
        doc = findings.anchors(str(self.dir / "a.json"), raw, self.tree)
        rows = cast(list[dict[str, Any]], doc["findings"])
        # One entry per finding that is an object, at its raw position.
        assert [row["#"] for row in rows] == [
            n for n, item in enumerate(entries, 1) if isinstance(item, dict)
        ]
        for row in rows:
            state = row["state"]
            placed = state in ("verified", "relocated")
            assert state in _STATES
            assert (row["evidence_key"] is not None) == placed, row
            assert (row["start"] is not None) == placed == (row["end"] is not None), row
            assert (row["quote_key"] is None) == (state == "no_evidence"), row
            assert (row["candidates"] != []) == (state == "ambiguous"), row
            if placed:
                assert 1 <= row["start"] <= row["end"], row
            if state == "verified":
                assert row["start"] <= row["line"] <= row["end"], row
            if row["path"] is not None:
                assert row["path"] == findings.norm_path(row["file"]), row


# ---------------------------------------------------------------------------------------
# VALIDATOR MODE. A review asks what a model finds; a validation asks it to judge findings
# somebody else already wrote, and the answer is only usable if it addresses each of them
# exactly once. The tests below cover the library layer of that: the prompt, the batch, the
# verdicts schema, the coverage rule and the gate they run through.

# The plugin's template is prose ABOUT a prompt, wrapped around the prompt in one fenced
# block. The fixture keeps both halves, including a literal JSON example with braces in it:
# that example is why the fill is `str.replace` and not `str.format`.
VALIDATOR_TEMPLATE = """# Validator batch

Dispatch this to a second model when a review's findings need independent judgment.

```
You are validating findings that another reviewer reported.

Scope: {scope_mode_and_remote_refs}

Diff: {diff}

Findings to validate:

{findings_json}

Return one verdict per finding, in this shape:

{"verdicts": [{"#": 1, "validated": true, "reason": "confirmed at f.py:2"}]}
```

Notes for the dispatcher, which are not part of the prompt and must not reach the model.
"""


def batch_item(n: int, **over: Any) -> dict[str, Any]:
    """One element of the plugin's validator batch.

    The key set is the one a real batch carries (the u3 rounds' `validator-input.json`),
    reproduced here rather than read from that file: these tests also run against the built
    package in a sandbox where nothing outside the source tree exists.
    """
    item: dict[str, Any] = {
        "#": n,
        "title": "t",
        "severity": "P1",
        "file": "f.py",
        "line": 2,
        "confidence": 100,
        "why_it_matters": "w",
        "evidence": ["f.py:2 -- x"],
        "first_evidence": "f.py:2 -- x",
        "suggested_fix": "s",
        "reviewers": ["grok"],
    }
    item.update(over)
    return item


def batch_text(*numbers: int) -> str:
    return json.dumps([batch_item(n) for n in numbers])


def verdict(n: int, **over: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"#": n, "validated": True, "reason": "confirmed at f.py:2"}
    item.update(over)
    return item


def verdicts_of(*items: dict[str, Any]) -> dict[str, Any]:
    return {"verdicts": list(items)}


def verdicts_schema() -> dict[str, Any]:
    """The schema as it SHIPS, read through the accessor the gate uses."""
    return cast(dict[str, Any], json.loads(verdicts.schema_path().read_text(encoding="utf-8")))


class TestTheValidatorPrompt:
    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.assets = Path(self.tmp.name)
        (self.assets / assets.VALIDATOR_TEMPLATE).write_text(VALIDATOR_TEMPLATE, encoding="utf-8")

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _prompt(self, *, base: str = "HEAD~1", context: str = "") -> str:
        return assets.build_validator_prompt(
            batch_text=batch_text(1, 2),
            assets=self.assets,
            schema_text=json.dumps(verdicts_schema()),
            base=base,
            context=context,
        )

    def test_the_prompt_is_the_fence_body_and_not_the_prose_around_it(self):
        # The wrapper tells a dispatcher when to use this. Sending it would ask the model to
        # decide whether to validate rather than to validate.
        body = assets.validator_body(self.assets)
        assert "You are validating findings" in body
        assert "Dispatch this to a second model" not in body
        assert "Notes for the dispatcher" not in body
        assert "```" not in body

    def test_a_missing_template_is_an_environment_failure(self):
        with tempfile.TemporaryDirectory() as bare, pytest.raises(errors.EnvError) as caught:
            assets.validator_body(Path(bare))
        assert "validator batch template" in str(caught.value)
        assert caught.value.exit_code == 3

    def test_a_template_with_no_fence_says_the_plugin_restructured_it(self):
        # Not "no prompt found": the file is present and readable, so the actionable fact is
        # that its SHAPE changed. Substituting into the surrounding prose would send the
        # model something that is not the validator prompt at all.
        (self.assets / assets.VALIDATOR_TEMPLATE).write_text("# No fence here\n", encoding="utf-8")
        with pytest.raises(errors.EnvError) as caught:
            assets.validator_body(self.assets)
        assert str(self.assets / assets.VALIDATOR_TEMPLATE) in str(caught.value)
        assert "restructured" in str(caught.value)
        assert caught.value.exit_code == 3

    def test_the_batch_goes_in_verbatim(self):
        # Verbatim, not re-serialized: the batch is the document the caller assembled, and
        # the `#` values are the only part this package reads.
        assert batch_text(1, 2) in self._prompt()
        assert "{findings_json}" not in self._prompt()

    def test_a_base_ref_tells_the_model_to_diff_it_itself(self):
        prompt = self._prompt(base="HEAD~3")
        assert "git diff HEAD~3..HEAD" in prompt
        assert "{diff}" not in prompt

    def test_without_a_base_ref_the_working_tree_as_a_whole_is_the_change(self):
        # The alternative -- leaving the slot empty -- asks the model to validate against a
        # diff it was never given, and a validator with no scope validates by vibe.
        prompt = self._prompt(base="")
        assert "No base ref was given" in prompt
        assert "git diff" not in prompt

    def test_the_scope_block_is_filled_and_carries_the_context_when_given(self):
        assert "{scope_mode_and_remote_refs}" not in self._prompt()
        assert "local-aligned" in self._prompt()
        assert "Additional validation context" not in self._prompt()
        with_context = self._prompt(context="the reviewed tree is a worktree at HEAD")
        assert "Additional validation context:" in with_context
        assert "the reviewed tree is a worktree at HEAD" in with_context

    def test_the_answer_contract_is_appended_after_the_plugin_text(self):
        prompt = self._prompt()
        assert "Return the verdicts as a JSON object matching this schema:" in prompt
        # The schema itself, because this is how codex is told it: only grok's --json-schema
        # reads `schema_text` out of band.
        assert json.dumps(verdicts_schema()) in prompt
        assert "exactly one JSON object" in prompt
        assert prompt.rstrip().endswith(assets.BOUNDARY_VERDICTS)

    def test_placeholder_text_inside_the_batch_reaches_the_validator_verbatim(self):
        # The batch is another model's prose, so it may contain the template's own slot
        # names. Filling in one pass keeps them text: a second pass would rewrite them and
        # hand the validator a different batch from the one the caller assembled.
        batch = json.dumps(
            [batch_item(1, title="{diff}", suggested_fix="{scope_mode_and_remote_refs}")]
        )
        prompt = assets.build_validator_prompt(
            batch_text=batch,
            assets=self.assets,
            schema_text=json.dumps(verdicts_schema()),
            base="HEAD~3",
            context="",
        )
        assert batch in prompt
        # ...and the TEMPLATE's slots are still filled, so holding the batch intact did not
        # cost the substitution.
        assert "git diff HEAD~3..HEAD" in prompt
        assert "local-aligned" in prompt
        assert "{findings_json}" not in prompt

    def test_the_fill_is_replacement_so_the_templates_literal_braces_survive(self):
        # The example verdict is literal JSON. It must reach the model intact...
        assert '{"verdicts": [{"#": 1, "validated": true' in self._prompt()
        # ...and this is the control: `format` reads those braces as fields and raises
        # before substituting anything, which is why the fill cannot use it.
        with pytest.raises((KeyError, IndexError, ValueError)):
            assets.validator_body(self.assets).format(
                findings_json="x", diff="y", scope_mode_and_remote_refs="z"
            )


class TestTheValidatorBatch:
    def test_the_shape_a_real_batch_has_is_accepted_in_input_order(self):
        assert verdicts.parse_batch(batch_text(3, 1, 2), "batch.json") == [3, 1, 2]

    @pytest.mark.parametrize(
        ("text", "because"),
        [
            ("not json at all", "is not JSON"),
            ('{"findings": []}', "must be a JSON array"),
            ("[1]", "not a finding object"),
            ("[]", "empty array"),
        ],
    )
    def test_a_batch_that_is_not_a_list_of_finding_objects_is_refused(
        self, text: str, because: str
    ):
        with pytest.raises(errors.UsageError) as caught:
            verdicts.parse_batch(text, "batch.json")
        assert because in str(caught.value), str(caught.value)
        assert "batch.json" in str(caught.value)
        assert caught.value.exit_code == 2

    @pytest.mark.parametrize(
        "number",
        [None, "1", 1.5, True, False, 0, -1],
        ids=["missing", "str", "float", "true", "false", "zero", "negative"],
    )
    def test_every_finding_needs_an_integer_number_of_one_or_more(self, number: Any):
        # `isinstance(True, int)` is True, so `"#": true` would otherwise pass as 1 and
        # address another finding's verdict.
        item = batch_item(1)
        if number is None:
            del item["#"]
        else:
            item["#"] = number
        with pytest.raises(errors.UsageError) as caught:
            verdicts.parse_batch(json.dumps([item]), "batch.json")
        assert "element 1 has #=" in str(caught.value), str(caught.value)
        assert caught.value.exit_code == 2

    def test_a_repeated_number_is_refused_and_named(self):
        with pytest.raises(errors.UsageError) as caught:
            verdicts.parse_batch(batch_text(1, 2, 1), "batch.json")
        assert "element 3 repeats #1" in str(caught.value)
        assert caught.value.exit_code == 2

    # PROPERTY's settings: its lack of a deadline matters under a loaded build sandbox.
    @settings(PROPERTY, max_examples=60)
    @given(numbers=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=6))
    def test_a_batch_is_accepted_exactly_when_its_numbers_are_a_set(self, numbers: list[int]):
        # The property, over MULTISETS: uniqueness is the whole contract, because the
        # verdicts are matched back on these numbers and nothing else.
        text = json.dumps([batch_item(n) for n in numbers])
        if len(set(numbers)) == len(numbers):
            assert verdicts.parse_batch(text, "b.json") == numbers
        else:
            with pytest.raises(errors.UsageError) as caught:
                verdicts.parse_batch(text, "b.json")
            assert "repeats" in str(caught.value)


class TestTheVerdictsSchemaIsThisPackages:
    """`findings-schema.json` is the plugin's file; this one is ours and has to earn it."""

    def test_the_shipped_schema_is_where_the_gate_looks_and_uses_only_supported_keywords(self):
        # A keyword this gate does not implement would be silently ignored, certifying
        # against a rule nobody checked -- so the schema we OWN must pass the same support
        # check the plugin's does.
        assert verdicts.schema_path().name == verdicts.SCHEMA_FILE
        assert verdicts.schema_path().is_file(), verdicts.schema_path()
        validate.check_schema_supported(verdicts_schema())

    def test_a_complete_verdict_set_passes_and_is_counted(self):
        found = verdicts_of(verdict(1), verdict(2, validated=False, reason="not reproducible"))
        assert validate.check_object(found, verdicts_schema(), "verdicts") == 2

    @pytest.mark.parametrize(
        ("entry", "because"),
        [
            (verdict(1, validated="yes"), "must be boolean"),
            (verdict(1, validated=None), "must be boolean"),
            (verdict(1, reason=""), "under the schema's minLength"),
            (verdict(1, **{"#": 0}), "must be >= 1"),
            (verdict(1, **{"#": True}), "must be integer"),
            ({"#": 1, "validated": True}, "missing reason"),
            ({"validated": True, "reason": "r"}, "missing #"),
        ],
    )
    def test_a_verdict_that_does_not_meet_the_schema_is_refused(
        self, entry: dict[str, Any], because: str
    ):
        with pytest.raises(errors.GateError) as caught:
            validate.check_object(verdicts_of(entry), verdicts_schema(), "verdicts")
        assert because in str(caught.value), str(caught.value)
        # The messages name ONE verdict, so a caller can find it in a batch of forty.
        assert "verdict 1" in str(caught.value), str(caught.value)

    def test_an_element_that_is_not_an_object_is_refused(self):
        with pytest.raises(errors.GateError) as caught:
            validate.check_object({"verdicts": ["yes"]}, verdicts_schema(), "verdicts")
        assert "verdict 1 is not an object" in str(caught.value)

    def test_a_missing_or_non_array_verdicts_key_is_refused(self):
        cases: list[dict[str, Any]] = [{}, {"verdicts": {}}]
        for found in cases:
            with pytest.raises(errors.GateError):
                validate.check_object(found, verdicts_schema(), "verdicts")

    def test_the_findings_only_demands_are_not_made_of_this_schema(self):
        # The control for the split. The verdicts schema declares no enums at all, so a walk
        # that kept the findings gate's demands would fail closed on EVERY validation -- and
        # the failure would read as a bad answer rather than as a wrong gate.
        schema = verdicts_schema()
        found = verdicts_of(verdict(1))
        assert validate.check_object(found, schema, "verdicts") == 1
        with pytest.raises(errors.GateError) as caught:
            validate.check_object(found, schema, "verdicts", demand_item_rules=True)
        assert "enums" in str(caught.value)


class TestOneVerdictForEveryFindingExactlyOnce:
    """The rule a silently short answer breaks: a finding nobody judged reads as judged."""

    def test_a_complete_set_covers_the_batch(self):
        verdicts.check_coverage(verdicts_of(verdict(1), verdict(2)), [1, 2])

    @pytest.mark.parametrize(
        ("found", "expected", "because"),
        [
            (verdicts_of(verdict(1)), [1, 2], "no verdict for #2"),
            (verdicts_of(verdict(1), verdict(3)), [1, 2], "findings that were not sent: #3"),
            (verdicts_of(verdict(1), verdict(1)), [1], "more than one verdict for #1"),
        ],
    )
    def test_missing_extra_and_duplicated_numbers_are_named(
        self, found: dict[str, Any], expected: list[int], because: str
    ):
        with pytest.raises(errors.GateError) as caught:
            verdicts.check_coverage(cast(validate.Artifact, found), expected)
        assert because in str(caught.value), str(caught.value)
        assert caught.value.exit_code == 1

    def test_the_summary_counts_both_sides(self):
        found = verdicts_of(verdict(1), verdict(2, validated=False), verdict(3, validated=False))
        assert verdicts.summarize(found, 3) == "1 validated, 2 rejected"

    def test_the_summary_counts_from_the_flag_rather_than_from_the_total(self):
        # Control: with `rejected = count - validated`, a verdict whose flag is neither true
        # nor false would be counted as rejected. The schema refuses that shape first; the
        # arithmetic must not depend on it having done so.
        found = verdicts_of(verdict(1), verdict(2, validated="maybe"))
        assert verdicts.summarize(found, 2) == "1 validated, 0 rejected"


class TestAnAnswerCarryingBothShapesIsRefused:
    """The producer must not certify an answer its own documented reader cannot render.

    `ce-persona-findings` refuses a file carrying a `findings` list AND a `verdicts` list,
    so a validation that returned both would exit 0 while the only reader this package
    ships exits 1 on what it wrote.
    """

    def _both(self, findings_value: Any) -> validate.Artifact:
        found = verdicts_of(verdict(1))
        found["findings"] = findings_value
        return cast(validate.Artifact, found)

    def test_a_verdicts_only_answer_is_accepted(self):
        verdicts.check_single_shape(cast(validate.Artifact, verdicts_of(verdict(1))))

    @pytest.mark.parametrize("also", [[], [{"title": "t"}]])
    def test_an_answer_that_also_carries_a_findings_list_is_refused(self, also: Any):
        # Empty as well as populated: the reader decides on the KEY being a list, so an
        # empty one is the same unrenderable file and refusing only the populated case
        # would leave the producer certifying a file the reader rejects.
        with pytest.raises(errors.GateError) as caught:
            verdicts.check_single_shape(self._both(also))
        assert "findings" in str(caught.value), str(caught.value)
        assert caught.value.exit_code == 1

    def test_a_findings_key_that_is_not_a_list_is_not_a_second_shape(self):
        # The reader's rule exactly: a non-list `findings` is not a findings artifact, so
        # refusing it here would refuse answers `ce-persona-findings` renders happily.
        verdicts.check_single_shape(self._both(None))


class TestTheGateOnVerdicts:
    """The same gate frame a review runs through, with the four validator-mode arguments."""

    def _check(self, expected: list[int]) -> Any:
        def check(found: validate.Artifact, schema: validate.JSONObject) -> int:
            count = validate.check_object(found, schema, "verdicts")
            verdicts.check_coverage(found, expected)
            return count

        return check

    def _gate(
        self,
        tmp: Path,
        answer: str,
        events: str,
        *,
        expected: list[int],
        mode: str = "object",
        evidence_mode: str = "codex-items",
    ) -> tuple[int, str, str]:
        answer_file = tmp / ("events.jsonl" if mode == "grok-events" else "answer.txt")
        answer_file.write_text(answer, encoding="utf-8")
        events_file = tmp / "events.jsonl"
        if events_file != answer_file:
            events_file.write_text(events, encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = validate.gate(
                    answer_file=answer_file,
                    schema_path=verdicts.schema_path(),
                    mode=mode,
                    findings_out=tmp / "validator-grok.json",
                    provenance_out=tmp / "validator-grok-provenance.json",
                    prov_pairs=[],
                    prov_files={},
                    evidence=validate.Evidence(
                        events_file=events_file, mode=evidence_mode, duration_s=4.5
                    ),
                    label="ce-grok-validate",
                    key="verdicts",
                    check=self._check(expected),
                    summarize=verdicts.summarize,
                    noun="verdicts",
                )
            except errors.AppError as exc:
                return exc.exit_code, out.getvalue(), str(exc)
        return code, out.getvalue(), err.getvalue()

    def test_a_complete_answer_passes_and_reports_the_split_in_one_line(self):
        answer = json.dumps(verdicts_of(verdict(1), verdict(2, validated=False, reason="no")))
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(Path(tmp), answer, CODEX_ONE_CALL, expected=[1, 2])
            written = json.loads((Path(tmp) / "validator-grok.json").read_text(encoding="utf-8"))
        assert code == 0, err
        assert out.strip().endswith("validator-grok.json")
        assert "ce-grok-validate: 2 verdicts (1 validated, 1 rejected) ->" in out
        assert out.count("\n") == 1, out
        assert written == verdicts_of(verdict(1), verdict(2, validated=False, reason="no"))

    def test_the_same_answer_arrives_through_groks_event_stream(self):
        # Both extraction modes, because the top-level key check that a verdicts answer has
        # to pass lives in each of them -- and before this parameter existed, a perfectly
        # good verdicts object died at exit 1 on BOTH providers.
        stream = (
            grok_tool_call()
            + "\n"
            + grok_result(structured_output=verdicts_of(verdict(1), verdict(2)))
            + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(
                Path(tmp),
                stream,
                stream,
                expected=[1, 2],
                mode="grok-events",
                evidence_mode="grok-messages",
            )
        assert code == 0, err
        assert "2 verdicts (2 validated, 0 rejected)" in out

    def test_an_answer_that_misses_a_finding_is_a_gate_failure(self):
        # Exit 1 and no summary line: a batch that comes back a verdict short would
        # otherwise read as a completed validation, and the unjudged finding as judged.
        answer = json.dumps(verdicts_of(verdict(1)))
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(Path(tmp), answer, CODEX_ONE_CALL, expected=[1, 2])
        assert code == 1, err
        assert out == ""
        assert "no verdict for #2" in err

    def test_a_validation_that_made_no_tool_calls_is_refused_in_its_own_words(self):
        # The whole reason this mode exists: `validated: true` across the board from a run
        # that inspected nothing is the answer it must never certify.
        answer = json.dumps(verdicts_of(verdict(1), verdict(2)))
        with tempfile.TemporaryDirectory() as tmp:
            code, out, message = self._gate(Path(tmp), answer, CODEX_NO_CALLS, expected=[1, 2])
            record = json.loads(
                (Path(tmp) / "validator-grok-provenance.json").read_text(encoding="utf-8")
            )
        assert code == 6, message
        assert out == ""
        assert "refusing to report 2 verdicts" in message, message
        assert "findings" not in message, message
        assert record["run_stats"]["tool_calls"] == 0

    def test_a_findings_answer_does_not_pass_as_verdicts(self):
        # Control for the key parameter pointing the other way: without it the gate would
        # read whatever top-level key it was hardcoded to.
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self._gate(Path(tmp), EMPTY_EXAMPLE, CODEX_ONE_CALL, expected=[1])
        assert code == 1
        assert "no verdicts key" in err, err


class TestTheReaderRendersVerdicts:
    """`ce-persona-findings` on what a validation wrote, not on what a review wrote.

    One command reads both shapes because both are model output being handed to an agent,
    and one reader is one place to keep the fence and the vacuous-run refusal. What it must
    not do is blur them: a verdict has no severity to hide behind and no merge shape to be
    projected into, so the modes that mean nothing here say so rather than approximating.
    """

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = str(self.dir / "validator-grok.json")
        # Out of `#` order on disk, so the ordering assertion below is not satisfied by the
        # file's own layout.
        self._write(
            verdicts_of(
                verdict(2, validated=False, reason="the handler re-raises one line down"),
                verdict(1, reason="confirmed at f.py:2"),
            )
        )

    def teardown_method(self) -> None:
        self.tmp.cleanup()

    def _write(self, obj: dict[str, Any]) -> None:
        Path(self.path).write_text(json.dumps(obj), encoding="utf-8")

    def _run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_every_verdict_renders_in_number_order_with_its_call_and_reason(self):
        code, out, err = self._run(self.path)
        assert code == 0, err
        rows = [ln for ln in out.splitlines() if ln.startswith("#")]
        assert rows == [
            "#1 validated — confirmed at f.py:2",
            "#2 REJECTED — the handler re-raises one line down",
        ], out

    def test_the_listing_is_fenced_as_untrusted(self):
        # A verdict is model-written text about somebody else's model-written text, and it
        # is being handed to a third agent. If anything here needs the fence, this does.
        _, out, _ = self._run(self.path)
        assert "BEGIN UNTRUSTED MODEL OUTPUT" in out
        assert "END UNTRUSTED MODEL OUTPUT" in out

    def test_show_renders_the_verdict_addressed_to_that_finding(self):
        code, out, err = self._run(self.path, "--show", "2")
        assert code == 0, err
        assert "#2 REJECTED — the handler re-raises one line down" in out
        assert "#1" not in out

    def test_show_for_a_finding_nobody_judged_is_a_usage_error(self):
        # LITERAL 2: these numbers are a published contract, and an assertion written
        # against the module's own constant moves with it.
        code, _, err = self._run(self.path, "--show", "9")
        assert code == 2
        assert "no verdict #9" in err

    def test_show_all_renders_the_default_listing(self):
        # Accepted for symmetry with a findings artifact: a verdict row is already its full
        # detail, so there is no heavier render to widen into.
        _, plain, _ = self._run(self.path)
        code, shown, err = self._run(self.path, "--show", "all")
        assert code == 0, err
        rows = [ln for ln in shown.splitlines() if ln.startswith("#")]
        assert rows == [ln for ln in plain.splitlines() if ln.startswith("#")]
        assert len(rows) == 2
        assert "BEGIN UNTRUSTED MODEL OUTPUT" in shown

    def test_show_all_on_an_empty_verdicts_artifact_says_so(self):
        self._write(verdicts_of())
        code, out, _ = self._run(self.path, "--show", "all")
        assert code == 0
        assert out == "no verdicts\n"

    def test_all_changes_nothing_because_no_verdict_is_hidden(self):
        _, plain, _ = self._run(self.path)
        _, widened, _ = self._run(self.path, "--all")
        assert [ln for ln in plain.splitlines() if ln.startswith("#")] == [
            ln for ln in widened.splitlines() if ln.startswith("#")
        ]

    def test_json_is_the_raw_object_unfenced(self):
        code, out, _ = self._run(self.path, "--json")
        assert code == 0
        assert "UNTRUSTED" not in out
        assert json.loads(out) == json.loads(Path(self.path).read_text(encoding="utf-8"))

    @pytest.mark.parametrize("args", [("--return",), ("--return", "--verify-quotes", "-C", ".")])
    def test_the_merge_projection_is_refused_rather_than_approximated(self, args: tuple[str, ...]):
        # The merge helper reads findings. Projecting a verdict into that shape would put an
        # object with no title, file or line into a merge that would drop the whole return.
        code, out, err = self._run(self.path, *args)
        assert code == 2, err
        assert out == ""
        assert "--return" in err

    def test_an_artifact_carrying_both_shapes_is_a_data_error(self):
        # Neither reading is the file's answer, and rendering one half would report a
        # complete result for a file that is two half-written ones.
        self._write({"findings": [], "verdicts": [verdict(1)]})
        code, _, err = self._run(self.path)
        assert code == 1
        assert "both findings and verdicts" in err

    def test_an_object_with_neither_key_is_still_a_data_error(self):
        # The control for the widening: `load` must not have become "any JSON object".
        self._write({"hello": "world"})
        code, _, err = self._run(self.path)
        assert code == 1
        assert "not a findings or verdicts artifact" in err

    def test_a_validation_that_inspected_nothing_is_refused_through_its_own_sidecar(self):
        # Proved rather than assumed: the sidecar lookup is by artifact STEM, and the stem
        # of a validation is `validator-<provider>`, not `<persona>-<provider>`.
        (self.dir / "validator-grok-provenance.json").write_text(
            json.dumps(
                {
                    "provider": "grok",
                    "kind": "validator",
                    "run_stats": {"tool_calls": 0, "turns": 1, "output_tokens": 151},
                }
            ),
            encoding="utf-8",
        )
        for args in ((), ("--json",), ("--show", "1"), ("--show", "all"), ("--return",)):
            code, out, err = self._run(self.path, *args)
            assert code == 6, (args, err)
            assert out == "", args
            assert "no local tool calls" in err, args
            assert "validator-grok-provenance.json" in err, args

    def test_the_help_documents_the_verdicts_rows(self):
        code, out, _ = self._run("--help")
        assert code == 0
        for token in ("verdicts", "REJECTED", "ce-grok-validate"):
            assert token in out


class TestTheExitTableSpeaksTheFlowsWords:
    def test_the_default_rendering_is_the_review_wording_unchanged(self):
        # Byte-for-byte the sentences the review commands published before the table took
        # word sets at all: the flow parameter must not have edited the shipped contract.
        rendered = errors.render_exit_table("grok")
        assert "  0   schema-valid findings (an empty findings array is valid)" in rendered
        assert "  1   the answer was not schema-valid findings" in rendered
        assert "  2   usage error: bad arguments, unknown or markdown-only persona," in rendered
        assert "nothing, so its findings -- empty or not -- attest to nothing" in rendered

    def test_the_exit_3_row_claims_only_that_no_local_call_succeeded(self):
        # A search that ran and matched nothing reaches this row too, so "it read nothing"
        # would be false for it.
        row = " ".join(dict(errors.EXIT_TABLE)[errors.EnvError.exit_code])
        assert "none succeeded" in row, row
        assert re.search(r"\bread\b", row) is None, row

    def test_the_validate_rendering_swaps_the_nouns_and_nothing_else(self):
        rendered = errors.render_exit_table("grok", errors.VALIDATE_WORDS)
        assert "schema-valid verdicts (one verdict for every input #, exactly once)" in rendered
        assert (
            "the answer was not schema-valid verdicts, or also carries a findings list"
        ) in rendered
        assert "a batch that is not an array of findings carrying a `#` each" in rendered
        # The word the validate commands must never use for their OWN answer: they take no
        # persona, and what they gate is verdicts. The two places it does appear are the
        # batch they are handed and the second list an unrenderable answer carries.
        stripped = rendered.replace("array of findings", "").replace("a findings list", "")
        assert "findings" not in stripped
        assert "persona" not in rendered
        # The statuses are the same table: same numbers, same runner substitution.
        assert "grok itself exited non-zero" in rendered
        for code in (0, 1, 2, 3, 4, 5, 6, 78):
            assert f"  {code} " in rendered, f"exit {code} missing"

    def test_no_word_slot_survives_either_rendering(self):
        # The control for the substitution pass. An unfilled `{answer}` would still leave a
        # table that lists every status, which is what the assertions above mostly check.
        for words in (errors.REVIEW_WORDS, errors.VALIDATE_WORDS):
            rendered = errors.render_exit_table("grok", words)
            for slot in ("{answer}", "{ok}", "{bad_argument}", "{also}", "{runner}"):
                assert slot not in rendered, (slot, words)
