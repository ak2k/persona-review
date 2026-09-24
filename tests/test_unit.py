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
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

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
    tool_calls=7, local_tool_calls=7, turns=4, output_tokens=4096, duration_s=61.5
)


# Event fixtures in each provider's own vocabulary, at module scope because both the counter's
# tests and the gate's need them. Shapes copied from real runs: grok 1.0.13 and codex-cli
# 0.150.1.
def grok_tool_call(name: str = "read_file") -> str:
    return json.dumps(
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


def codex_item(kind: str, item_type: str, ident: str | None = "item_1") -> str:
    item: dict[str, Any] = {"type": item_type}
    if ident is not None:
        item["id"] = ident
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
# Calls, and none of them local: the run read the internet and never opened the repository.
CODEX_ONLY_SEARCHED = codex_stream(
    codex_item("item.started", "web_search", "ws_1"),
    codex_item("item.completed", "web_search", "ws_1"),
    codex_item("item.completed", "web_search", "ws_2"),
)

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
    def test_every_other_tool_kind_is_local(self, kind: str):
        # With an id and without, so both counting arms carry the local tally.
        assert self._codex(codex_item("item.completed", kind, "item_1")).local_tool_calls == 1
        assert self._codex(codex_item("item.completed", kind, None)).local_tool_calls == 1

    def test_the_local_kinds_are_the_tool_kinds_less_web_search(self):
        not_local = validate.CODEX_TOOL_ITEMS - validate.CODEX_LOCAL_TOOL_ITEMS
        assert not_local == {"web_search"}

    def test_grok_counts_every_call_as_local(self):
        # The argv disables grok's web tools, so no call left is one known to leave the machine.
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
        assert (stats.tool_calls, stats.local_tool_calls) == (2, 2)

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
        # unrecognised, every vacuous run would report drift and the refusal would be dead.
        stats = self._codex(CODEX_NO_CALLS)
        assert stats.tool_calls == 0

    def test_a_codex_stream_with_no_events_at_all_is_an_environment_error(self):
        # `codex exec --json` opens every run with a thread and a turn, so an empty stream is
        # a runner that did not run — not a model that did nothing.
        with pytest.raises(errors.EnvError) as caught:
            self._codex("")
        assert "no events at all" in str(caught.value)

    def test_grok_has_no_kind_list_to_drift(self):
        # Stated so the asymmetry is deliberate rather than an omission: grok names a tool
        # call structurally (`tool_use`), so there is no vocabulary to fall out of date and
        # nothing for a drift check to detect.
        stats = validate.run_stats(
            "grok-messages", validate.objects(grok_result().splitlines()), None
        )
        assert stats.tool_calls == 0


class TestTheGateRefusesARunThatInspectedNothing:
    """Zero tool calls is not a small number of tool calls; it is no review at all."""

    def _gate(self, tmp: Path, answer: str, events: str) -> tuple[int, str, str]:
        (tmp / "answer.txt").write_text(answer, encoding="utf-8")
        (tmp / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
        (tmp / "events.jsonl").write_text(events, encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = validate.gate(
                    answer_file=tmp / "answer.txt",
                    schema_path=tmp / "schema.json",
                    mode="object",
                    findings_out=tmp / "out.json",
                    provenance_out=tmp / "out-provenance.json",
                    prov_pairs=[],
                    prov_files={},
                    evidence=validate.Evidence(
                        events_file=tmp / "events.jsonl", mode="codex-items", duration_s=4.5
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

    def test_one_tool_call_is_enough(self):
        # The control. Without it every assertion above holds for a gate that refuses
        # everything, which is the same cannot-fail defect in the other direction.
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._gate(Path(tmp), EMPTY_EXAMPLE, CODEX_ONE_CALL)
        assert code == 0, err
        assert "0 findings" in out

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


class TestARefusalSurvivesBeingHandedOn:
    """`validate.refused_run`: the reader's half of the vacuous-run refusal.

    The review command refuses with exit 6 and keeps the artifact as evidence. Without this,
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
            # Present, so it is what the reading relies on, and it is not a whole number.
            {"tool_calls": 0, "local_tool_calls": "0"},
            {"tool_calls": 0, "local_tool_calls": True},
            {"tool_calls": 0, "local_tool_calls": False},
            {"tool_calls": 0, "local_tool_calls": -1},
            {"tool_calls": 0, "local_tool_calls": 0.0},
            {"tool_calls": 0, "local_tool_calls": None},
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
        # `tool_calls` decides the exit status, so it has to be auditable after the fact:
        # a refusal a caller cannot check is one it has to take on trust.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, [], {}, validate.RunStats(2, 0, 1, 151, 4.5))
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["run_stats"] == {
            "tool_calls": 2,
            "local_tool_calls": 0,
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
        # Generalises the hand-written anyOf/oneOf/$ref/const case over the whole keyword
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
            ((ARTIFACT, "--return", "-C", "."), "-C only applies to --verify-quotes"),
            ((ARTIFACT, "-C", "."), "-C only applies to --verify-quotes"),
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
        for token in ("--return", "--verify-quotes", "-C <dir>"):
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

    @settings(max_examples=60)
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
