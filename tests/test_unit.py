#!/usr/bin/env python3
"""Unit tests for the findings gate, the schema rules, and the retrieval tiers.

Run: python3 tests/test_unit.py

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
import unittest
from pathlib import Path
from typing import Any

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


class TestObjectMode(unittest.TestCase):
    """`--mode object` is strict, and the strictness is the feature.

    Regression: loosening this to tolerate prose-then-object imported a hole object mode was
    immune to. Two independent reviewers demonstrated both halves of it.
    """

    def test_a_bare_object_passes(self):
        self.assertEqual(validate.from_object_file(EMPTY_EXAMPLE)["findings"], [])

    def test_a_give_up_message_quoting_the_example_is_refused(self):
        # The exact shape a reviewer reproduced: an explicit refusal, then the brief's own
        # schema-valid empty example. Under the loosened gate this exited 0, "0 findings".
        text = (
            "I could not inspect the repository. The requested output shape is:\n\n" + EMPTY_EXAMPLE
        )
        with self.assertRaises(validate.GateError) as caught:
            validate.from_object_file(text)
        self.assertIn("exactly one JSON object", str(caught.exception))

    def test_a_fenced_object_is_refused(self):
        text = "```json\n" + EMPTY_EXAMPLE + "\n```\n"
        with self.assertRaises(validate.GateError):
            validate.from_object_file(text)

    def test_real_findings_followed_by_the_example_are_refused(self):
        # The worse half: a completed review that also pastes the example last. Accepting
        # this discards real defects and reports clean.
        text = json.dumps(artifact(finding(severity="P0"))) + "\n\n" + EMPTY_EXAMPLE
        with self.assertRaises(validate.GateError):
            validate.from_object_file(text)

    def test_prose_only_is_refused(self):
        with self.assertRaises(validate.GateError):
            validate.from_object_file("I could not complete the review.")

    def test_json_without_a_findings_key_is_refused(self):
        with self.assertRaises(validate.GateError) as caught:
            validate.from_object_file('{"summary": "all good"}')
        self.assertIn("no findings key", str(caught.exception))


class TestTranscriptAnchors(unittest.TestCase):
    """The fallback mode's three anchors, each pinning a real false-pass."""

    BOUNDARY = assets.BOUNDARY
    PROMPT = f"# brief\n\n...body...\n{BOUNDARY}\n"

    def _echoed(self, tail: str) -> str:
        return f"# brief\n```json\n{EMPTY_EXAMPLE}\n```\n{self.BOUNDARY}\n{tail}"

    def test_the_echoed_persona_example_is_not_an_answer(self):
        with self.assertRaises(validate.GateError):
            validate.findings_object(self._echoed("I ran out of context."), self.PROMPT)

    def test_a_real_answer_after_the_echo_passes(self):
        got = validate.findings_object(
            self._echoed("here it is\n" + json.dumps(artifact(finding()))), self.PROMPT
        )
        self.assertEqual(len(findings_of(got)), 1)

    def test_a_quoted_object_followed_by_prose_is_not_an_answer(self):
        text = json.dumps(artifact(finding())) + "\n\nbut I could not verify any of it."
        with self.assertRaises(validate.GateError) as caught:
            validate.findings_object(text, self.PROMPT)
        self.assertIn("not at the end", str(caught.exception))

    def test_a_trailing_empty_object_must_not_wipe_a_real_one(self):
        # Last-wins is right for a model that restates its answer, and catastrophic when the
        # restatement is the brief's empty example: the real findings vanish and the run
        # reports clean.
        text = json.dumps(artifact(finding(severity="P0"))) + "\n\n" + EMPTY_EXAMPLE
        with self.assertRaises(validate.GateError) as caught:
            validate.findings_object(text, self.PROMPT)
        self.assertIn("ambiguous", str(caught.exception))

    def test_an_empty_boundary_does_not_slice_the_output_away(self):
        # `"abc".rfind("")` is 3, so a blank prompt would cut everything.
        self.assertEqual(validate.findings_object(EMPTY_EXAMPLE, "   \n")["findings"], [])


class TestSchemaRules(unittest.TestCase):
    """The rules the installed plugin schema actually uses."""

    def test_union_type_accepts_both_members(self):
        schema = schema_with("suggested_fix", {"type": ["string", "null"]})
        for value in ("do the thing", None):
            validate.validate(artifact(finding(suggested_fix=value)), schema)

    def test_union_type_rejects_a_non_member(self):
        schema = schema_with("suggested_fix", {"type": ["string", "null"]})
        with self.assertRaises(validate.GateError):
            validate.validate(artifact(finding(suggested_fix=42)), schema)

    def test_a_numeric_union_still_rejects_a_boolean(self):
        # `True` satisfies isinstance(x, int), so a union pairing a numeric type with a
        # non-numeric one must not lose the bool carve-out.
        schema = schema_with("line", {"type": ["integer", "null"]})
        validate.validate(artifact(finding(line=None)), schema)
        with self.assertRaises(validate.GateError):
            validate.validate(artifact(finding(line=True)), schema)

    def test_a_union_that_admits_booleans_takes_one(self):
        schema = schema_with("pre_existing", {"type": ["boolean", "null"]})
        validate.validate(artifact(finding(pre_existing=True)), schema)

    def test_minimum_and_maximum(self):
        validate.validate(artifact(finding(line=1)), SCHEMA)
        with self.assertRaises(validate.GateError) as caught:
            validate.validate(artifact(finding(line=0)), SCHEMA)
        self.assertIn(">= 1", str(caught.exception))

    def test_max_length(self):
        validate.validate(artifact(finding(title="x" * 100)), SCHEMA)
        with self.assertRaises(validate.GateError) as caught:
            validate.validate(artifact(finding(title="x" * 101)), SCHEMA)
        self.assertIn("maxLength", str(caught.exception))

    def test_array_item_types(self):
        validate.validate(artifact(finding(evidence=["f.py:1 -- x"])), SCHEMA)
        for bad in ([{"a": 1}], [None], ["ok", 3], []):
            with self.assertRaises(validate.GateError):
                validate.validate(artifact(finding(evidence=bad)), SCHEMA)

    def test_enums_are_enforced(self):
        for bad in ({"severity": "critical"}, {"confidence": 72}):
            with self.assertRaises(validate.GateError):
                validate.validate(artifact(finding(**bad)), SCHEMA)

    def test_a_type_name_the_gate_cannot_check_fails(self):
        with self.assertRaises(validate.GateError) as caught:
            validate.validate(artifact(finding()), schema_with("title", {"type": "sTrInG"}))
        self.assertIn("cannot check", str(caught.exception))

    def test_an_unimplemented_schema_keyword_fails_loudly(self):
        # Silently ignoring anyOf/oneOf/$ref/const certifies against rules nobody checked.
        for keyword in ("anyOf", "oneOf", "allOf", "$ref", "const", "pattern", "not"):
            with self.assertRaises(validate.GateError) as caught:
                validate.validate(artifact(finding()), schema_with("title", {keyword: "whatever"}))
            self.assertIn(keyword, str(caught.exception))

    def test_annotations_are_tolerated(self):
        schema = schema_with("title", {"type": "string", "description": "the title", "default": ""})
        validate.validate(artifact(finding()), schema)

    def test_a_non_object_property_spec_does_not_crash(self):
        schema = schema_with("title", "not-a-spec")  # type: ignore[arg-type]
        validate.validate(artifact(finding()), schema)

    def test_missing_required_keys(self):
        with self.assertRaises(validate.GateError) as caught:
            validate.validate({"findings": []}, SCHEMA)
        self.assertIn("missing required keys", str(caught.exception))


class TestGrokRunGating(unittest.TestCase):
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
        self.assertEqual(got["findings"], [])

    def test_the_last_result_event_wins(self):
        early = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "structured_output": artifact(finding(title="early")),
            }
        )
        late = json.dumps({"type": "result", "is_error": True, "stop_reason": "max_tokens"})
        with self.assertRaises(validate.GateError) as caught:
            validate.from_grok_events(early + "\n" + late + "\n")
        self.assertIn("error result", str(caught.exception))

    def test_a_non_success_subtype_fails(self):
        with self.assertRaises(validate.GateError):
            validate.from_grok_events(
                self._events(subtype="error_max_turns", structured_output=artifact())
            )

    def test_early_stop_reasons_fail_even_with_a_well_formed_answer(self):
        for stop in ("max_tokens", "length", "content_filter", "refusal", "timeout"):
            with self.assertRaises(validate.GateError) as caught:
                validate.from_grok_events(
                    self._events(stop_reason=stop, structured_output=artifact())
                )
            self.assertIn(stop, str(caught.exception))

    def test_a_mistyped_terminal_field_fails_closed(self):
        for bad in ({"subtype": 3}, {"stop_reason": ["end_turn"]}):
            with self.assertRaises(validate.GateError):
                validate.from_grok_events(self._events(structured_output=artifact(), **bad))

    def test_an_unfamiliar_stop_reason_does_not_fail_a_good_run(self):
        got = validate.from_grok_events(
            self._events(stop_reason="finished_normally", structured_output=artifact())
        )
        self.assertEqual(got["findings"], [])

    def test_no_result_event_fails(self):
        with self.assertRaises(validate.GateError):
            validate.from_grok_events('{"type":"system"}\n')

    def test_a_raw_text_result_goes_through_the_strict_object_reader(self):
        got = validate.from_grok_events(self._events(result=EMPTY_EXAMPLE))
        self.assertEqual(got["findings"], [])
        with self.assertRaises(validate.GateError):
            validate.from_grok_events(self._events(result="I gave up. " + EMPTY_EXAMPLE))


class TestAssets(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
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

    def test_a_capable_persona_resolves(self):
        name, brief = assets.resolve_persona(self.assets, "adversarial-reviewer")
        self.assertEqual(name, "adversarial-reviewer")
        self.assertTrue(brief.is_file())

    def test_the_ce_prefix_and_md_suffix_are_accepted(self):
        for spelling in ("ce-adversarial-reviewer", "adversarial-reviewer.md"):
            name, _ = assets.resolve_persona(self.assets, spelling)
            self.assertEqual(name, "adversarial-reviewer")

    def test_a_markdown_only_persona_is_refused(self):
        with self.assertRaises(assets.UsageError) as caught:
            assets.resolve_persona(self.assets, "agent-native-reviewer")
        self.assertIn("markdown output format", str(caught.exception))

    def test_a_persona_name_must_be_a_bare_brief_name(self):
        # The name reaches every artifact path; a separator would resolve a brief from
        # outside personas/ and place artifacts outside the run directory.
        for hostile in ("../outside", "/etc/passwd", ".hidden", "a/b", "", "."):
            with self.assertRaises(assets.UsageError) as caught:
                assets.resolve_persona(self.assets, hostile)
            self.assertIn("bare brief name", str(caught.exception), hostile)

    def test_the_traversal_fixture_would_otherwise_resolve(self):
        # Control for the test above: assets/outside.md really is a findings-capable brief,
        # so `../outside` is refused by the NAME check and not merely by being absent.
        self.assertTrue(assets.emits_findings(self.assets / "outside.md"))
        self.assertTrue((self.assets / "personas" / ".." / "outside.md").is_file())

    def test_plugin_roots_sort_numerically_not_lexically(self):
        # Lexical order puts 3.9 after 3.13, silently pinning an old brief set.
        made = [
            Path(self.tmp.name) / f"compound-engineering/{v}/skills/ce-code-review/references"
            for v in ("3.9.0", "3.13.1", "3.10.0")
        ]
        for path in made:
            path.mkdir(parents=True)
        self.assertEqual(sorted(made, key=assets.version_key)[-1], made[1])

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
        self.assertIn("Return findings matching the findings schema.", prompt)
        self.assertIn("Anchors 0 and 25 mean SUPPRESS", prompt)  # the fallback rubric
        self.assertIn("extra context here", prompt)
        self.assertIn("git diff HEAD~1..HEAD", prompt)
        # The strict output clause is what lets the gate stay strict.
        self.assertIn("exactly one JSON object", prompt)
        self.assertTrue(prompt.rstrip().endswith(assets.BOUNDARY))

    def test_the_plugin_rubric_replaces_the_fallback_when_present(self):
        (self.assets / "subagent-template.md").write_text(
            "**Schema conformance** — every finding carries file, line, evidence.\n"
            "RUBRIC-MARKER-FROM-TEMPLATE\n"
            "Example of a schema-valid finding:\n",
            encoding="utf-8",
        )
        text = assets.rubric(self.assets)
        self.assertIn("RUBRIC-MARKER-FROM-TEMPLATE", text)
        self.assertNotIn("Anchors 0 and 25", text)


class TestProviders(unittest.TestCase):
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
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertEqual(argv[argv.index("--output-format") + 1], "streaming-messages-json")
        self.assertIn("--json-schema", argv)

    def test_codex_runs_read_only_and_writes_only_the_final_message(self):
        argv = providers.CODEX.argv(self._inv())
        self.assertEqual(argv[argv.index("-s") + 1], "read-only")
        self.assertEqual(argv[argv.index("-o") + 1], "/tmp/last.json")
        self.assertNotIn("--output-schema", argv)
        self.assertEqual(argv[-1], "-")

    def test_codex_omits_the_effort_flag_when_it_is_unset(self):
        inv = providers.Invocation(
            model="m",
            effort="",
            repo=Path("/r"),
            prompt_file=Path("/p"),
            schema_text="{}",
            last_file=None,
        )
        self.assertNotIn("model_reasoning_effort", " ".join(providers.CODEX.argv(inv)))

    def test_every_provider_is_covered_by_the_flag_probe_shape(self):
        argv = flags.reference_argv()
        self.assertTrue(flags.valueless_flags(argv), "an empty probe set would cover nothing")
        self.assertIn("--verbatim", flags.valueless_flags(argv))
        self.assertIn("--json-schema", flags.valued_flags(argv))


class TestBudget(unittest.TestCase):
    def test_over_budget_counts_prompt_plus_diff(self):
        budget = runner.Budget(prompt_bytes=1200, diff_bytes=92_000, limit_tokens=2_000)
        self.assertEqual(budget.tokens, (1200 + 92_000) // 4)
        self.assertTrue(budget.over)

    def test_the_prompt_alone_is_not_enough_to_fire(self):
        self.assertFalse(runner.Budget(prompt_bytes=1200, diff_bytes=0, limit_tokens=2_000).over)


class TestFindingsRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "artifact.json")
        Path(self.path).write_text(
            json.dumps(
                artifact(
                    finding(title="a p2", severity="P2", file="b.py", line=2),
                    finding(title="a p0", severity="P0", file="a.py", line=1, confidence=100),
                )
            ),
            encoding="utf-8",
        )

    def _run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = findings.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_default_tier_shows_p0_p1_and_hides_the_rest(self):
        code, out, _ = self._run(self.path)
        self.assertEqual(code, 0)
        self.assertIn("a p0", out)
        self.assertNotIn("a p2", out)
        self.assertIn("hidden", out)

    def test_all_shows_every_severity(self):
        _, out, _ = self._run(self.path, "--all")
        self.assertIn("a p0", out)
        self.assertIn("a p2", out)

    def test_list_is_accepted(self):
        code, out, _ = self._run(self.path, "--list")
        self.assertEqual(code, 0)
        self.assertIn("a p0", out)

    def test_show_n_is_the_finding_numbered_n_in_the_listing(self):
        _, listing, _ = self._run(self.path, "--all")
        rows = [ln for ln in listing.splitlines() if ln.startswith("#")]
        self.assertEqual(len(rows), 2)
        for row in rows:
            number = int(row.split()[0].lstrip("#"))
            title = row.split(" — ", 1)[1].split(" (confidence")[0]
            _, detail, _ = self._run(self.path, "--show", str(number))
            self.assertIn(title, detail)

    def test_rendered_output_is_fenced_as_untrusted(self):
        # Everything rendered here was written by a model and is being handed to another
        # agent as its input.
        _, out, _ = self._run(self.path, "--all")
        self.assertIn("BEGIN UNTRUSTED MODEL OUTPUT", out)
        self.assertIn("END UNTRUSTED MODEL OUTPUT", out)
        _, detail, _ = self._run(self.path, "--show", "1")
        self.assertIn("BEGIN UNTRUSTED MODEL OUTPUT", detail)

    def test_the_fence_nonce_differs_between_runs(self):
        # A fixed delimiter could be closed by the text inside it.
        _, first, _ = self._run(self.path, "--all")
        _, second, _ = self._run(self.path, "--all")
        self.assertNotEqual(first, second)

    def test_json_is_not_fenced_and_round_trips(self):
        code, out, _ = self._run(self.path, "--json")
        self.assertEqual(code, 0)
        self.assertNotIn("UNTRUSTED", out)
        self.assertEqual(json.loads(out), json.loads(Path(self.path).read_text(encoding="utf-8")))

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
        self.assertEqual(code, 1)
        self.assertIn("not a findings artifact", err)

        code, _, _ = self._run(str(Path(self.tmp.name) / "nope.json"))
        self.assertEqual(code, 1)

    def test_usage_mistakes_are_usage_errors(self):
        for args, expected in (
            ((), "artifact path is required"),
            (("--nope",), "unknown option"),
            # A trailing --show once fell through to the unknown-option arm and reported a
            # documented flag as unknown.
            ((self.path, "--show"), "wants a finding number"),
            ((self.path, "--show", "x"), "wants a finding number"),
            ((self.path, "--show", "99"), "no finding #99"),
        ):
            with self.subTest(args=args):
                code, _, err = self._run(*args)
                self.assertEqual(code, 2, err)
                self.assertIn(expected, err)
                self.assertIn("usage:", err)

    def test_help_exits_zero_and_documents_every_tier(self):
        # A documented flag that exits non-zero with "unknown option" is how an agent
        # concludes a tool is broken; --list already had that problem once.
        for flag in ("-h", "--help"):
            code, out, _ = self._run(flag)
            self.assertEqual(code, 0)
            for token in ("--list", "--all", "--show", "--json", "UNTRUSTED MODEL OUTPUT"):
                self.assertIn(token, out, flag)

    def test_the_fence_stays_cheap(self):
        # It wraps every tier-1 and tier-2 read, and tier 1 budgets ~20 tokens a finding.
        _, out, _ = self._run(self.path, "--show", "1")
        overhead = [ln for ln in out.splitlines() if "UNTRUSTED MODEL OUTPUT" in ln]
        self.assertEqual(len(overhead), 2, "the fence should cost two lines, not a paragraph")


class TestProvenance(unittest.TestCase):
    def test_paths_with_quotes_and_backslashes_round_trip(self):
        # Structured data, written by json.dump rather than interpolated into a heredoc,
        # which produced invalid JSON the moment a path held a quote or a backslash.
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / 'we"ird\\name.md'
            odd.write_text("brief", encoding="utf-8")
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, ["provider=grok", "base_ref="], {"persona": str(odd)})
            record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["provider"], "grok")
        self.assertEqual(record["base_ref"], "")
        self.assertEqual(record["persona_file"], str(odd))
        self.assertEqual(len(record["persona_sha256"]), 64)

    def test_an_unreadable_asset_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prov.json"
            validate.write_provenance(out, [], {"schema": str(Path(tmp) / "gone.json")})
            record = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(record["schema_sha256"].startswith("unreadable:"))


if __name__ == "__main__":
    unittest.main()
