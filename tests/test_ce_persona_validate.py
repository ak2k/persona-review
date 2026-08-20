#!/usr/bin/env python3
"""Unit tests for the persona wrappers' findings gate and CLI-drift parser.

Run: python3 tests/test_ce_persona_validate.py

The modules under test are a real package, so they are imported normally; importing only
defines things, since their main()s are __main__-guarded.

This gate is the one piece of the wrappers whose failure mode is SILENT: everything else
exits non-zero and says why, while a validator that wrongly passes reports a clean review
of a change nobody reviewed. It has been wrong that way twice -- see TestEchoedPrompt and
TestEndAnchored, which pin both regressions.

The wrapper assertions parse the invocation instead of searching the file text. The
earlier substring versions were satisfied by a comment and by a variable assignment, so
deleting the real `--sandbox read-only` flag and the real validator call left the suite
green -- a guard that cannot fail is worse than no guard, because it reads as coverage.

The schema fixture below mirrors the shape of the compound-engineering plugin's
findings-schema.json rather than reading it: the plugin is not installed in the build
sandbox, and this file tests the validator's logic, not the plugin's contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# The directory under test. Defaulting to ../bin runs the suite against the source tree;
# $PERSONA_REVIEW_BIN points it at a BUILT package instead, which is how the flake check
# exercises the thing that actually ships — wrapper substitution, permissions and all —
# rather than the files a developer happens to have edited.
BIN = Path(os.environ.get("PERSONA_REVIEW_BIN") or Path(__file__).parent.parent / "bin")


# Make the package importable when this file is run directly (python3 tests/…).
#
# APPEND, never insert(0): the flake's unit check points PYTHONPATH at the BUILT
# ${persona-review}/lib so the modules that ship are the modules exercised, and putting
# the source root ahead of it silently tested the source tree instead. That check passed
# with every packaged module replaced by `raise RuntimeError`, which means it could not
# fail on anything installPhase got wrong.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from persona_review import findings as project  # noqa: E402
from persona_review import flags  # noqa: E402
from persona_review import validate as validator  # noqa: E402

# Which copy did we actually import? Only this process knows, so the flake check sets
# PERSONA_REVIEW_EXPECT_LIB and the answer is asserted here rather than trusted.
_expect_lib = os.environ.get("PERSONA_REVIEW_EXPECT_LIB")
if _expect_lib and not validator.__file__.startswith(_expect_lib):
    raise SystemExit(
        f"imported {validator.__file__}, not the packaged library under {_expect_lib} — "
        "this suite is not testing what ships"
    )

# Mirrors the plugin schema's SHAPE, including the types and array minimums -- an
# enum-only fixture let type-invalid findings ("line": "nope", "evidence": []) pass the
# suite while the real schema rejected them.
SCHEMA = {
    "required": ["reviewer", "findings", "residual_risks", "testing_gaps"],
    "properties": {
        "reviewer": {"type": "string"},
        "residual_risks": {"type": "array"},
        "testing_gaps": {"type": "array"},
        "findings": {
            "items": {
                "required": ["title", "severity", "file", "line", "evidence"],
                "properties": {
                    "title": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "evidence": {"type": "array", "minItems": 1},
                    "pre_existing": {"type": "boolean"},
                    "severity": {"enum": ["P0", "P1", "P2", "P3"]},
                    "confidence": {"enum": [0, 25, 50, 75, 100]},
                    "autofix_class": {"enum": ["gated_auto", "manual", "advisory"]},
                    "owner": {"enum": ["downstream-resolver", "human", "release"]},
                },
            }
        },
    },
}

# The last line build_prompt() emits, i.e. the echo boundary transcript mode cuts on.
# TestWrapperInvariants asserts both wrappers still emit it, so the fixture cannot drift
# away from the code it stands in for. Both wrappers now use native extraction, so this
# only matters if plain output is ever used again — which is exactly when a silently
# drifted boundary would bite.
BOUNDARY = "testing_gaps when they apply."
PROMPT = f"# Adversarial Reviewer\n\n...persona body...\n\n{BOUNDARY}\n"

# 13 of the 16 persona briefs carry one of these. It is schema-valid.
PERSONA_EXAMPLE = json.dumps(
    {
        "reviewer": "adversarial",
        "findings": [],
        "residual_risks": [],
        "testing_gaps": [],
    }
)


def finding(**overrides: Any) -> dict[str, Any]:
    base = {
        "title": "t",
        "severity": "P1",
        "file": "f",
        "line": 1,
        "evidence": ["e"],
    }
    base.update(overrides)
    return base


def answer(findings: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "reviewer": "adversarial-reviewer",
            "findings": findings,
            "residual_risks": [],
            "testing_gaps": [],
        }
    )


class TestFindingsObject(unittest.TestCase):
    def test_prose_only_is_not_a_review(self):
        with self.assertRaises(SystemExit) as cm:
            validator.findings_object("I could not complete the review.", PROMPT)
        self.assertIn("no findings JSON object", str(cm.exception))

    def test_genuinely_empty_findings_is_valid(self):
        obj = validator.findings_object(f"prose\n{answer([])}", PROMPT)
        self.assertEqual(obj["findings"], [])

    def test_later_object_wins(self):
        text = f"{answer([finding(title='early')])}\n{answer([finding(title='late')])}"
        obj = validator.findings_object(text, PROMPT)
        self.assertEqual(obj["findings"][0]["title"], "late")

    def test_closing_code_fence_still_counts_as_the_end(self):
        obj = validator.findings_object(f"```json\n{answer([finding()])}\n```\n", PROMPT)
        self.assertEqual(len(obj["findings"]), 1)


class TestEchoedPrompt(unittest.TestCase):
    """codex replays its prompt; the persona example inside it must not count.

    Regression: scanning the whole output made a prose-only answer 'validate' against
    the persona's own example object and report a clean review.
    """

    def _echoed(self, tail: str) -> str:
        return f"# Adversarial Reviewer\n\n```json\n{PERSONA_EXAMPLE}\n```\n{BOUNDARY}\n{tail}"

    def test_echo_then_prose_fails(self):
        with self.assertRaises(SystemExit) as cm:
            validator.findings_object(self._echoed("I ran out of context."), PROMPT)
        self.assertIn("no findings JSON object", str(cm.exception))

    def test_echo_then_real_answer_passes(self):
        obj = validator.findings_object(self._echoed(f"here it is\n{answer([finding()])}"), PROMPT)
        self.assertEqual(len(obj["findings"]), 1)

    def test_no_echo_leaves_whole_output_in_scope(self):
        obj = validator.findings_object(answer([finding()]), "prompt with no shared tail\n")
        self.assertEqual(len(obj["findings"]), 1)


class TestEndAnchored(unittest.TestCase):
    """A quoted object is not an answer.

    Regression: the boundary cut removed the prompt echo, so a give-up answer that
    QUOTED a schema-shaped object after the boundary -- tool output, a replayed brief --
    was accepted as the model's own findings. The same false green, one paragraph later.
    """

    def test_object_quoted_in_tool_output_then_prose_fails(self):
        text = (
            f"{BOUNDARY}\n"
            "I ran a command. Its output was:\n"
            f"```\n{PERSONA_EXAMPLE}\n```\n"
            "I ran out of context and did not complete the review.\n"
        )
        with self.assertRaises(SystemExit) as cm:
            validator.findings_object(text, PROMPT)
        self.assertIn("not at the end", str(cm.exception))

    def test_quoted_object_then_a_real_answer_passes(self):
        text = (
            f"{BOUNDARY}\n"
            f"tool said:\n```\n{PERSONA_EXAMPLE}\n```\n"
            f"and here is my review:\n{answer([finding()])}\n"
        )
        obj = validator.findings_object(text, PROMPT)
        self.assertEqual(len(obj["findings"]), 1)


class TestValidate(unittest.TestCase):
    def test_valid_returns_count(self):
        obj = json.loads(answer([finding(), finding(severity="P0")]))
        self.assertEqual(validator.validate(obj, SCHEMA), 2)

    def test_missing_top_level_key(self):
        obj: dict[str, Any] = {"reviewer": "r", "findings": []}
        with self.assertRaises(SystemExit) as cm:
            validator.validate(obj, SCHEMA)
        self.assertIn("residual_risks", str(cm.exception))

    def test_findings_not_an_array(self):
        obj: dict[str, Any] = json.loads(answer([]))
        obj["findings"] = {"nope": True}
        with self.assertRaises(SystemExit) as cm:
            validator.validate(obj, SCHEMA)
        self.assertIn("must be an array", str(cm.exception))

    def test_missing_required_field(self):
        bad = finding()
        del bad["evidence"]
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([bad])), SCHEMA)
        self.assertIn("evidence", str(cm.exception))

    def test_finding_not_an_object(self):
        obj: dict[str, Any] = json.loads(answer([]))
        obj["findings"] = ["just a string"]
        with self.assertRaises(SystemExit) as cm:
            validator.validate(obj, SCHEMA)
        self.assertIn("not an object", str(cm.exception))


class TestEveryEnum(unittest.TestCase):
    """Every enum the schema declares, not just severity.

    The prompt shape this wrapper replaced taught `autofix_class: safe_auto` and
    `owner: review-fixer`; neither is valid under the current schema, and both passed
    when only severity was checked.
    """

    def test_severity(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(severity="CRITICAL")])), SCHEMA)
        self.assertIn("CRITICAL", str(cm.exception))

    def test_confidence(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(confidence=90)])), SCHEMA)
        self.assertIn("confidence", str(cm.exception))

    def test_autofix_class(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(autofix_class="safe_auto")])), SCHEMA)
        self.assertIn("autofix_class", str(cm.exception))

    def test_owner(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(owner="review-fixer")])), SCHEMA)
        self.assertIn("owner", str(cm.exception))

    def test_absent_optional_enum_key_is_not_an_error(self):
        self.assertEqual(validator.validate(json.loads(answer([finding()])), SCHEMA), 1)


class TestNativeExtraction(unittest.TestCase):
    """The two paths that replace transcript scanning.

    grok hands back `structured_output` already parsed in its terminal `result` event;
    `codex exec -o` writes only the final message. Neither needs the prompt-echo cut or
    the end anchor -- but both need their own failure modes to stay loud.
    """

    def test_object_mode_reads_a_bare_findings_object(self):
        obj = validator.from_object_file(answer([finding()]))
        self.assertEqual(len(obj["findings"]), 1)

    def test_object_mode_rejects_prose(self):
        # Assert the refusal, not its wording: a final message with no object in it at
        # all now falls through the end-anchored scan and reports that instead.
        with self.assertRaises(SystemExit) as cm:
            validator.from_object_file("I could not complete the review.")
        self.assertIn("no findings JSON object", str(cm.exception))

    def test_object_mode_rejects_json_without_findings(self):
        with self.assertRaises(SystemExit) as cm:
            validator.from_object_file('{"status": "ok"}')
        self.assertIn("no findings key", str(cm.exception))

    def _events(self, result_event: dict[str, Any]) -> str:
        return "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init"}),
                json.dumps({"type": "assistant", "message": {"content": []}}),
                json.dumps(result_event),
            ]
        )

    def test_grok_events_reads_structured_output(self):
        stream = self._events(
            {
                "type": "result",
                "is_error": False,
                "stop_reason": "end_turn",
                "structured_output": json.loads(answer([finding()])),
            }
        )
        self.assertEqual(len(validator.from_grok_events(stream)["findings"]), 1)

    def test_grok_events_falls_back_to_the_raw_result_text(self):
        stream = self._events({"type": "result", "is_error": False, "result": answer([finding()])})
        self.assertEqual(len(validator.from_grok_events(stream)["findings"]), 1)

    def test_grok_events_honours_is_error(self):
        stream = self._events({"type": "result", "is_error": True, "stop_reason": "max_tokens"})
        with self.assertRaises(SystemExit) as cm:
            validator.from_grok_events(stream)
        self.assertIn("max_tokens", str(cm.exception))

    def test_grok_events_without_a_result_event_fails(self):
        stream = json.dumps({"type": "system", "subtype": "init"})
        with self.assertRaises(SystemExit) as cm:
            validator.from_grok_events(stream)
        self.assertIn("no `result` event", str(cm.exception))

    def test_grok_events_ignores_unparsable_lines(self):
        stream = "not json\n" + self._events(
            {
                "type": "result",
                "is_error": False,
                "structured_output": json.loads(answer([])),
            }
        )
        self.assertEqual(validator.from_grok_events(stream)["findings"], [])


class TestDeclaredTypes(unittest.TestCase):
    """Presence and enums are not shape.

    Found by a live grok run against this branch: the gate printed 'findings JSON valid'
    for objects the plugin schema rejects, so downstream trusting the gate would parse a
    string where it expected a number.
    """

    def test_integer_field_given_a_string(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(line="nope")])), SCHEMA)
        self.assertIn("line", str(cm.exception))

    def test_integer_field_given_a_bool(self):
        # True is an int in Python; the gate must not inherit that.
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(line=True)])), SCHEMA)
        self.assertIn("line", str(cm.exception))

    def test_boolean_field_given_a_string(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(pre_existing="no")])), SCHEMA)
        self.assertIn("pre_existing", str(cm.exception))

    def test_empty_evidence_array(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([finding(evidence=[])])), SCHEMA)
        self.assertIn("evidence", str(cm.exception))

    def test_top_level_array_given_a_string(self):
        obj: dict[str, Any] = json.loads(answer([finding()]))
        obj["residual_risks"] = "none"
        with self.assertRaises(SystemExit) as cm:
            validator.validate(obj, SCHEMA)
        self.assertIn("residual_risks", str(cm.exception))

    def test_well_typed_finding_passes(self):
        good = finding(line=12, evidence=["f:12 -- x"], pre_existing=False)
        self.assertEqual(validator.validate(json.loads(answer([good])), SCHEMA), 1)


class TestSchemaShape(unittest.TestCase):
    """An unexpected schema is a failed run with a reason, never a traceback.

    The schema comes from a plugin cache this repo does not control, so its shape is an
    input, not an invariant. The subscripts used to raise KeyError straight out.
    """

    def test_missing_items(self):
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([])), {"properties": {}})
        self.assertIn("properties.findings.items", str(cm.exception))

    def test_items_without_properties(self):
        schema: dict[str, Any] = {"properties": {"findings": {"items": {"required": ["title"]}}}}
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([])), schema)
        self.assertIn("items.properties", str(cm.exception))

    def test_items_without_required(self):
        schema: dict[str, Any] = {
            "properties": {"findings": {"items": {"properties": {"severity": {}}}}}
        }
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([])), schema)
        self.assertIn("required", str(cm.exception))

    def test_items_without_any_enum(self):
        schema: dict[str, Any] = {
            "properties": {"findings": {"items": {"required": ["title"], "properties": {"t": {}}}}}
        }
        with self.assertRaises(SystemExit) as cm:
            validator.validate(json.loads(answer([])), schema)
        self.assertIn("enums", str(cm.exception))


class TestFlagsParser(unittest.TestCase):
    """The CLI-drift parser, which the gated flags check cannot exercise offline.

    Regression: grok_argv tokenised past the shell pipeline, so a valueless flag in last
    position took `|` as its successor, looked value-taking, and dropped out of the probe
    set -- leaving the check green with zero coverage.
    """

    INVOCATION = (
        'grok --prompt-file "$PROMPT_FILE" \\\n'
        "\t--verbatim \\\n"
        '\t--model "$MODEL" \\\n'
        '\t--cwd "$REPO" \\\n'
        "\t--sandbox read-only \\\n"
        '\t--output-format plain | tee "$OUT_FILE" || STATUS=$?\n'
    )

    def test_argv_stops_at_the_pipeline(self):
        argv = flags.grok_argv(self.INVOCATION)
        self.assertNotIn("|", argv)
        self.assertNotIn("tee", argv)
        self.assertEqual(argv[-1], "plain")

    def test_valueless_flag_last_before_the_pipe_is_still_valueless(self):
        moved = (
            'grok --prompt-file "$PROMPT_FILE" \\\n'
            "\t--output-format plain \\\n"
            '\t--verbatim | tee "$OUT_FILE"\n'
        )
        self.assertIn("--verbatim", flags.valueless_flags(flags.grok_argv(moved)))

    def test_valued_flags_include_the_ones_with_values(self):
        pairs = dict(flags.valued_flags(flags.grok_argv(self.INVOCATION)))
        self.assertIn("--cwd", pairs)
        self.assertIn("--model", pairs)
        self.assertNotIn("--verbatim", pairs)

    def test_real_wrapper_yields_a_non_empty_probe_set(self):
        text = (BIN / "ce-grok-persona").read_text(encoding="utf-8")
        argv = flags.grok_argv(text)
        self.assertTrue(flags.valueless_flags(argv), "probe set must never be empty")
        self.assertIn("--cwd", dict(flags.valued_flags(argv)))

    def test_flag_values_come_from_argv_not_from_a_comment(self):
        # Regression: wrapper_value() regexed the whole file, so `--sandbox` matched the
        # explanatory comment above the invocation. Comment and flag agreed by luck; a
        # retargeted flag would have been certified against the prose.
        wrapper = (
            "# we pass --sandbox danger-full-access here because prose is not code\n"
            'grok --prompt-file "$P" \\\n'
            "\t--sandbox read-only \\\n"
            "\t--output-format plain\n"
        )
        argv = flags.grok_argv(wrapper)
        self.assertEqual(flags.argv_value(argv, "--sandbox"), "read-only")

    def test_missing_flag_in_argv_is_a_failure_not_a_comment_match(self):
        wrapper = "# --sandbox read-only\ngrok --output-format plain\n"
        with self.assertRaises(SystemExit):
            flags.argv_value(flags.grok_argv(wrapper), "--sandbox")

    def test_resolve_wanted_ignores_comments_for_every_asserted_flag(self):
        # This is the function main() asserts against, so it is where a regression back
        # to whole-file matching would land.
        wrapper = (
            "# historical: --sandbox danger-full-access --output-format json\n"
            "# --permission-mode ask\n"
            'grok --prompt-file "$P" \\\n'
            "\t--permission-mode bypassPermissions \\\n"
            "\t--sandbox read-only \\\n"
            '\t--output-format plain | tee "$OUT"\n'
        )
        self.assertEqual(
            flags.resolve_wanted(wrapper),
            {
                "--permission-mode": "bypassPermissions",
                "--output-format": "plain",
                "--sandbox": "read-only",
            },
        )

    def test_resolve_wanted_on_the_real_wrapper(self):
        text = (BIN / "ce-grok-persona").read_text(encoding="utf-8")
        self.assertEqual(flags.resolve_wanted(text)["--sandbox"], "read-only")


class TestProvenance(unittest.TestCase):
    """Provenance is JSON, so Python writes it.

    The first version built it with a shell heredoc interpolating paths into JSON
    strings, which produces invalid JSON as soon as a path contains a quote or a
    backslash — the reason structured data does not belong in shell.
    """

    def test_pairs_and_hashes_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            brief = Path(tmp) / "persona.md"
            brief.write_text("brief body", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validator.write_provenance(
                str(out),
                ["provider=grok", "model=grok-4.6", f"persona_file={brief}"],
                {"persona": str(brief)},
            )
            record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["provider"], "grok")
        self.assertEqual(
            record["persona_sha256"],
            hashlib.sha256(b"brief body").hexdigest(),
        )

    def test_a_path_with_a_quote_still_produces_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            nasty = Path(tmp) / 'we"ird\\path.md'
            nasty.write_text("x", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validator.write_provenance(str(out), [f"persona_file={nasty}"], {"persona": str(nasty)})
            record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["persona_file"], str(nasty))

    def test_unreadable_file_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            validator.write_provenance(str(out), [], {"schema": str(Path(tmp) / "absent.json")})
            record = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("unreadable", record["schema_sha256"])


class TestRetrievalTiers(unittest.TestCase):
    """The caller-facing projection: rows carry enough to judge, detail is opt-in.

    Over-tiering is the failure mode here — a caller that sees only titles will
    under-weight a real defect or fix it from the label. Every row therefore carries the
    quoted line and the confidence, which is the minimum that makes a title checkable.
    """

    def test_rows_carry_evidence_and_confidence(self):
        row = project.render_row(1, finding(confidence=75, first_evidence="f:1 -- x"))
        self.assertIn("P1", row)
        self.assertIn("f:1", row)
        self.assertIn("confidence 75", row)
        self.assertIn("f:1 -- x", row)

    def test_row_falls_back_to_the_first_evidence_item(self):
        row = project.render_row(1, finding(evidence=["quoted line"]))
        self.assertIn("quoted line", row)

    def test_detail_adds_why_fix_and_routing(self):
        detail = project.render_detail(
            1,
            finding(
                why_it_matters="users hit it",
                suggested_fix="guard the call",
                autofix_class="gated_auto",
                owner="downstream-resolver",
            ),
        )
        self.assertIn("users hit it", detail)
        self.assertIn("guard the call", detail)
        self.assertIn("autofix_class=gated_auto", detail)

    def test_ordering_is_by_severity_then_location(self):
        findings = [
            finding(severity="P3", file="a", line=1),
            finding(severity="P0", file="b", line=2),
            finding(severity="P1", file="a", line=9),
        ]
        order = [f["severity"] for _, f in project.ordered(findings)]
        self.assertEqual(order, ["P0", "P1", "P3"])

    def test_numbers_are_stable_under_reordering(self):
        # `--show N` must mean the same finding as the row labelled #N.
        findings = [finding(severity="P3"), finding(severity="P0")]
        numbered = project.ordered(findings)
        self.assertEqual(numbered[0][0], 2)  # the P0 sorts first but keeps its number


class TestWrapperInvariants(unittest.TestCase):
    """Properties of the wrappers that no offline runtime probe would catch.

    Asserted against the parsed invocation and the executed line, never a bare substring:
    the previous versions of these two tests passed with the guarded lines deleted.
    """

    def _wrapper(self, name: str) -> str:
        return (BIN / name).read_text(encoding="utf-8")

    def _code(self, name: str) -> list[str]:
        """Wrapper lines with comments removed, so a comment cannot satisfy a check."""
        return [
            line for line in self._wrapper(name).splitlines() if not line.lstrip().startswith("#")
        ]

    def test_grok_runs_read_only(self):
        argv = flags.grok_argv("\n".join(self._code("ce-grok-persona")))
        self.assertIn("--sandbox", argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")

    def test_codex_runs_read_only(self):
        exec_lines = [line for line in self._code("ce-codex-persona") if "codex exec " in line]
        self.assertTrue(exec_lines, "no `codex exec` invocation found")
        self.assertIn("-s read-only", exec_lines[0])

    def test_validator_is_the_last_command_in_both_wrappers(self):
        # Not merely present: last, and unguarded. A trailing `|| true`, a `&`, or an
        # early exit above it would keep a substring assertion green. Matched as a
        # pattern so adding a flag to the call does not require editing this test,
        # while a guard appended after it still fails.
        pattern = re.compile(r'^python3 -m persona_review\.validate .*"\$SCHEMA_FILE"$')
        for name in ("ce-grok-persona", "ce-codex-persona"):
            code = [line for line in self._code(name) if line.strip()]
            self.assertRegex(
                code[-1],
                pattern,
                f"{name}: findings gate must be the wrapper's last command, unguarded",
            )

    def test_grok_asks_for_schema_constrained_output(self):
        # Dropping --json-schema silently downgrades the arm to unstructured text: the
        # `result` event still arrives, just without `structured_output`, and the
        # validator's raw-text fallback would paper over it. Assert the request itself.
        argv = flags.grok_argv("\n".join(self._code("ce-grok-persona")))
        self.assertIn("--json-schema", argv)
        self.assertIn("--output-format", argv)
        self.assertEqual(argv[argv.index("--output-format") + 1], "streaming-messages-json")

    def test_codex_writes_only_the_final_message(self):
        code = "\n".join(self._code("ce-codex-persona"))
        self.assertIn('-o "$LAST_FILE"', code)

    def test_both_wrappers_send_the_rubric_with_the_brief(self):
        # 14 of the 16 briefs tell the reviewer to use a rubric from the subagent
        # template. Dropping the call sends a brief that references a rubric the model
        # never sees, and nothing in the output would look wrong.
        #
        # This asserts the CALL SITE only, which `return 0` as rubric()'s first line
        # survives. The prompt the stub actually receives is checked by the process
        # suite; both are kept because they fail differently.
        for name in ("ce-grok-persona", "ce-codex-persona"):
            calls = [line.strip() for line in self._code(name) if line.strip() == "rubric"]
            self.assertTrue(calls, f"{name}: build_prompt must call rubric")

    def test_neither_wrapper_redirects_the_runner_to_stdout(self):
        # The whole point of the contract: a calling agent pays one summary line, not the
        # model's reasoning. Assert the runner's stdout is REDIRECTED, rather than
        # forbidding `| tee`, the one historical leak: deleting `>"$EVENTS_FILE"` puts
        # the entire event stream on the caller's stdout and left the old test green.
        for name, pattern in (
            ("ce-grok-persona", re.compile(r'>"\$EVENTS_FILE"')),
            ("ce-codex-persona", re.compile(r'>"\$EVENTS_FILE"')),
        ):
            code = "\n".join(self._code(name))
            self.assertRegex(code, pattern, f"{name}: runner output must go to a file")
            self.assertNotIn("| tee ", code, f"{name}: transcript must not reach stdout")

    def test_validator_is_preflighted_before_the_model_runs(self):
        for name, runner in (
            ("ce-grok-persona", "grok --prompt-file"),
            ("ce-codex-persona", "codex exec "),
        ):
            code = "\n".join(self._code(name))
            self.assertLess(
                code.index("import persona_review.validate"),
                code.index(runner),
                f"{name}: the gate must be import-checked before the model call, not after",
            )

    def test_boundary_fixture_matches_what_the_wrappers_emit(self):
        # BOUNDARY is the validator's echo anchor. If build_prompt's final line changes
        # and this fixture does not, every echo test silently stops testing the echo.
        #
        # _code(), not _wrapper(): reading the raw text let the guarded printf be deleted
        # while a comment still carried the string, which is the exact shape of guard
        # this class exists to memorialise.
        for name in ("ce-grok-persona", "ce-codex-persona"):
            self.assertIn(BOUNDARY, "\n".join(self._code(name)), name)

    def test_both_wrappers_refuse_a_missing_persona_argument(self):
        # The one-liner this replaced could not reach its own `exit 2`, so a missing
        # argument printed the whole usage block to STDOUT and exited 0.
        for name in ("ce-grok-persona", "ce-codex-persona"):
            proc = subprocess.run([str(BIN / name)], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2, f"{name}: no persona must exit 2\n{proc.stderr}")
            self.assertEqual(proc.stdout, "", f"{name}: usage must not go to stdout")

    def test_both_wrappers_print_usage_for_help(self):
        for name in ("ce-grok-persona", "ce-codex-persona"):
            for flag in ("-h", "--help"):
                proc = subprocess.run(
                    [str(BIN / name), flag], capture_output=True, text=True, check=False
                )
                self.assertEqual(proc.returncode, 0, f"{name} {flag}")
                self.assertIn("Usage:", proc.stdout, f"{name} {flag}")

    def test_both_wrappers_disable_the_implicit_cwd_on_sys_path(self):
        # `python3 -c` and `python3 -m` put the CWD at sys.path[0], ahead of
        # PYTHONPATH -- so a persona_review/ directory in the repo under review replaced
        # the gate entirely. Asserted as text too, because the process test that proves
        # it needs a planted package and this one fails the moment the line is dropped.
        for name in ("ce-grok-persona", "ce-codex-persona"):
            code = "\n".join(self._code(name))
            self.assertIn("export PYTHONSAFEPATH=1", code, name)
            self.assertLess(
                code.index("export PYTHONSAFEPATH=1"),
                code.index("import persona_review.validate"),
                f"{name}: PYTHONSAFEPATH must be set before the preflight import",
            )

    def test_both_wrappers_clear_stale_artifacts_before_running(self):
        # Artifact paths are deterministic and $CE_PERSONA_RUN_DIR is meant to be reused,
        # so without this a failed run leaves the PREVIOUS run's findings and provenance
        # beside this run's fresh event stream.
        for name, expected in (
            ("ce-grok-persona", ['rm -f "$FINDINGS_FILE" "$PROV_FILE"']),
            ("ce-codex-persona", ['rm -f "$LAST_FILE" "$FINDINGS_FILE" "$PROV_FILE"']),
        ):
            code = "\n".join(self._code(name))
            for line in expected:
                self.assertIn(line, code, name)

    def test_both_wrappers_require_a_modern_bash(self):
        # bash 3.2 errors on "${EMPTY[@]}" under `set -u`, and with an EXIT trap
        # installed that abort exits 0 -- a wrapper that never called the model
        # reporting success.
        for name in ("ce-grok-persona", "ce-codex-persona"):
            code = "\n".join(self._code(name))
            self.assertIn("BASH_VERSINFO", code, f"{name}: must refuse a bash older than 4.4")


def _finding(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "t",
        "severity": "P1",
        "file": "f",
        "line": 1,
        "evidence": ["f:1 -- x"],
    }
    base.update(over)
    return base


def _artifact(*findings: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer": "a",
        "findings": list(findings),
        "residual_risks": [],
        "testing_gaps": [],
    }


def _schema_with(field: str, spec: dict[str, Any]) -> dict[str, Any]:
    """SCHEMA with one extra/replaced property on a finding."""
    schema: dict[str, Any] = json.loads(json.dumps(SCHEMA))
    schema["properties"]["findings"]["items"]["properties"][field] = spec
    return schema


class TestSchemaTypeRules(unittest.TestCase):
    """The rules the installed plugin schema actually uses.

    Every case here was PASSING before: the type check read `spec["type"]` only when it
    was a `str`, and ignored `minimum`, `maxLength` and `items` entirely -- so on the real
    3.21.4 schema it silently skipped `suggested_fix` (a ["string","null"] union), let a
    finding cite line 0, and accepted an evidence array of objects.
    """

    def test_union_type_accepts_both_members(self):
        schema = _schema_with("suggested_fix", {"type": ["string", "null"]})
        for value in ("do the thing", None):
            validator.validate(_artifact(_finding(suggested_fix=value)), schema)

    def test_union_type_still_rejects_a_non_member(self):
        schema = _schema_with("suggested_fix", {"type": ["string", "null"]})
        with self.assertRaises(SystemExit) as caught:
            validator.validate(_artifact(_finding(suggested_fix=42)), schema)
        self.assertIn("string or null", str(caught.exception))

    def test_a_type_name_the_gate_cannot_check_fails_loudly(self):
        # Rather than skipping the field: passing a rule it does not understand is how a
        # validator certifies a shape nobody validated.
        schema = _schema_with("title", {"type": "sTrInG"})
        with self.assertRaises(SystemExit) as caught:
            validator.validate(_artifact(_finding()), schema)
        self.assertIn("cannot check", str(caught.exception))

    def test_minimum_is_enforced(self):
        schema = _schema_with("line", {"type": "integer", "minimum": 1})
        validator.validate(_artifact(_finding(line=1)), schema)
        with self.assertRaises(SystemExit) as caught:
            validator.validate(_artifact(_finding(line=0)), schema)
        self.assertIn(">= 1", str(caught.exception))

    def test_max_length_is_enforced(self):
        schema = _schema_with("title", {"type": "string", "maxLength": 100})
        validator.validate(_artifact(_finding(title="x" * 100)), schema)
        with self.assertRaises(SystemExit) as caught:
            validator.validate(_artifact(_finding(title="x" * 101)), schema)
        self.assertIn("maxLength", str(caught.exception))

    def test_array_item_types_are_enforced(self):
        # `evidence: [{"a": 1}]` satisfied minItems and then rendered as the quote-the-line
        # evidence a reader is supposed to be able to check against the code.
        schema = _schema_with(
            "evidence", {"type": "array", "minItems": 1, "items": {"type": "string"}}
        )
        validator.validate(_artifact(_finding(evidence=["f:1 -- x"])), schema)
        for bad in ([{"a": 1}], [None], ["ok", 3]):
            with self.assertRaises(SystemExit):
                validator.validate(_artifact(_finding(evidence=bad)), schema)

    def test_a_union_that_admits_booleans_still_takes_one(self):
        schema = _schema_with("pre_existing", {"type": ["boolean", "null"]})
        validator.validate(_artifact(_finding(pre_existing=True)), schema)

    def test_top_level_fields_get_the_same_rules(self):
        schema: dict[str, Any] = json.loads(json.dumps(SCHEMA))
        schema["properties"]["reviewer"] = {"type": ["string", "null"]}
        validator.validate({**_artifact(), "reviewer": None}, schema)
        with self.assertRaises(SystemExit):
            validator.validate({**_artifact(), "reviewer": 7}, schema)


class TestGrokRunGating(unittest.TestCase):
    """grok's terminal event, not just its payload.

    Schema-constrained decoding is why this matters: a run that hit the token ceiling or
    refused still returns a well-formed `{"findings": []}`, which reads exactly like a
    clean review.
    """

    def _events(self, **result: Any) -> str:
        event = {"type": "result", "is_error": False, "subtype": "success", **result}
        return '{"type":"system"}\n' + json.dumps(event) + "\n"

    def test_a_healthy_run_passes(self):
        # The shape of a real grok 1.0.4 success, which is what the gating is calibrated to.
        got = validator.from_grok_events(
            self._events(stop_reason="end_turn", structured_output=_artifact())
        )
        self.assertEqual(got["findings"], [])

    def test_the_last_result_event_wins(self):
        # "Terminal" is the documented semantics; a first-wins read would let an early
        # event certify a run that kept going and then failed.
        early = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "structured_output": _artifact(_finding(title="early")),
            }
        )
        late = json.dumps({"type": "result", "is_error": True, "stop_reason": "max_tokens"})
        with self.assertRaises(SystemExit) as caught:
            validator.from_grok_events(early + "\n" + late + "\n")
        self.assertIn("error result", str(caught.exception))

    def test_a_non_success_subtype_fails(self):
        with self.assertRaises(SystemExit) as caught:
            validator.from_grok_events(
                self._events(subtype="error_max_turns", structured_output=_artifact())
            )
        self.assertIn("error_max_turns", str(caught.exception))

    def test_an_early_stop_reason_fails_even_with_a_well_formed_answer(self):
        for stop in ("max_tokens", "refusal", "timeout"):
            with self.assertRaises(SystemExit) as caught:
                validator.from_grok_events(
                    self._events(stop_reason=stop, structured_output=_artifact())
                )
            self.assertIn(stop, str(caught.exception))

    def test_an_unfamiliar_stop_reason_does_not_fail_a_good_run(self):
        # Denylist, deliberately: stop_reason is an open vocabulary and a healthy but
        # unfamiliar value must not kill a completed review.
        got = validator.from_grok_events(
            self._events(stop_reason="finished_normally", structured_output=_artifact())
        )
        self.assertEqual(got["findings"], [])


class TestCodexFinalMessage(unittest.TestCase):
    """`--mode object` against what the prompt actually asks codex for.

    The prompt says to return the object "at the END of your reply (after any analysis)",
    and requiring the whole file to be bare JSON threw away complete reviews that did
    exactly that.
    """

    def test_a_bare_object_passes(self):
        got = validator.from_object_file(json.dumps(_artifact()))
        self.assertEqual(got["findings"], [])

    def test_analysis_then_object_passes(self):
        text = "I reviewed the change. One issue stood out.\n\n" + json.dumps(_artifact())
        self.assertEqual(validator.from_object_file(text)["findings"], [])

    def test_a_fenced_object_passes(self):
        text = "Here is what I found.\n\n```json\n" + json.dumps(_artifact()) + "\n```\n"
        self.assertEqual(validator.from_object_file(text)["findings"], [])

    def test_prose_with_no_object_still_fails(self):
        with self.assertRaises(SystemExit):
            validator.from_object_file("I could not complete the review.")

    def test_a_quoted_object_followed_by_prose_still_fails(self):
        # The end anchor is what stops a give-up answer that QUOTES a schema-shaped
        # object from validating.
        text = json.dumps(_artifact()) + "\n\nBut I could not actually verify any of it."
        with self.assertRaises(SystemExit) as caught:
            validator.from_object_file(text)
        self.assertIn("not at the end", str(caught.exception))

    def test_json_that_is_not_a_findings_object_fails(self):
        with self.assertRaises(SystemExit) as caught:
            validator.from_object_file('{"summary": "all good"}')
        self.assertIn("no findings key", str(caught.exception))


# The library root that was actually imported, so the CLI subprocesses below exercise the
# same copy this suite does -- the packaged one under the flake check.
LIB_ROOT = str(Path(validator.__file__).resolve().parent.parent)


class TestFindingsCli(unittest.TestCase):
    """`ce-persona-findings` end to end.

    Only its rendering helpers were covered, so deleting the artifact guard, breaking
    `--show`'s numbering, or making `--json` print nothing all left the suite green.
    """

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ, PYTHONPATH=LIB_ROOT, PYTHONSAFEPATH="1")
        return subprocess.run(
            [sys.executable, "-m", "persona_review.findings", *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "artifact.json")
        artifact = _artifact(
            _finding(title="a p2", severity="P2", file="b.py", line=2),
            _finding(title="a p0", severity="P0", file="a.py", line=1, confidence=100),
        )
        Path(self.path).write_text(json.dumps(artifact), encoding="utf-8")

    def test_default_tier_shows_p0_p1_and_hides_the_rest(self):
        proc = self._run(self.path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("a p0", proc.stdout)
        self.assertNotIn("a p2", proc.stdout)
        self.assertIn("hidden", proc.stdout)

    def test_all_shows_every_severity(self):
        proc = self._run(self.path, "--all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("a p0", proc.stdout)
        self.assertIn("a p2", proc.stdout)

    def test_list_is_accepted(self):
        # Named as tier 1 in the module docstring, and it used to exit 1 as an unknown option.
        proc = self._run(self.path, "--list")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("a p0", proc.stdout)

    def test_show_n_is_the_finding_numbered_n_in_the_listing(self):
        # The whole tier contract: `--show N` must mean the same finding as row #N, or a
        # caller reads detail for something it did not ask about.
        listing = self._run(self.path, "--all")
        rows = [ln for ln in listing.stdout.splitlines() if ln.startswith("#")]
        self.assertTrue(rows, listing.stdout)
        for row in rows:
            number = int(row.split()[0].lstrip("#"))
            title = row.split(" — ", 1)[1].split(" (confidence")[0]
            detail = self._run(self.path, "--show", str(number))
            self.assertEqual(detail.returncode, 0, detail.stderr)
            self.assertIn(title, detail.stdout.splitlines()[0])

    def test_json_prints_the_artifact_unchanged(self):
        proc = self._run(self.path, "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            json.loads(proc.stdout), json.loads(Path(self.path).read_text(encoding="utf-8"))
        )

    def test_a_json_file_that_is_not_a_findings_artifact_is_refused(self):
        other = str(Path(self.dir.name) / "other.json")
        Path(other).write_text('{"hello": "world"}', encoding="utf-8")
        proc = self._run(other)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not a findings artifact", proc.stderr)

    def test_no_argument_prints_usage_and_fails(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("usage:", proc.stderr)

    def test_show_out_of_range_fails(self):
        proc = self._run(self.path, "--show", "99")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no finding #99", proc.stderr)


if __name__ == "__main__":
    unittest.main()
