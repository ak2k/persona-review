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
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# APPEND, never insert(0): the flake's unit check points PYTHONPATH at the BUILT package so
# the modules that ship are the modules exercised, and putting the source root ahead of it
# silently tests the source tree instead.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from persona_review import assets, findings, flags, providers, runner, validate  # noqa: E402

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


EMPTY_EXAMPLE = json.dumps(artifact())

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
        assert "not a findings artifact" in err

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


class TestProvenance:
    def test_paths_with_quotes_and_backslashes_round_trip(self):
        # Structured data, written by json.dump rather than interpolated into a heredoc,
        # which produced invalid JSON the moment a path held a quote or a backslash.
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / 'we"ird\\name.md'
            odd.write_text("brief", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, ["provider=grok", "base_ref="], {"persona": str(odd)})
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["provider"] == "grok"
        assert record["base_ref"] == ""
        assert record["persona_file"] == str(odd)
        assert len(record["persona_sha256"]) == 64

    def test_an_unreadable_asset_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, [], {"schema": str(Path(tmp) / "gone.json")})
            record = json.loads(out.read_text(encoding="utf-8"))
        assert record["schema_sha256"].startswith("unreadable:")


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
