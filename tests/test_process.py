#!/usr/bin/env python3
"""Process-level tests: the installed commands, driven against stub runners.

Run: python3 tests/test_process.py   (set PERSONA_REVIEW_BIN to test a built package)

The unit suite exercises the library. This runs what actually ships — entry-point
generation, wrapper environment, argument parsing, artifact lifecycle, watchdogs, and the
one-line stdout contract — with no network, no credentials and no real model: the stubs on
PATH are the runners.

**Every case runs for BOTH providers.** That is not thoroughness for its own sake: when the
two runners were separate scripts, guards were repeatedly added to both and tested against
only one, so deleting the second copy left the suite green. One code path plus a
parameterised suite makes that impossible rather than merely unlikely.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC))

from persona_review import assets  # noqa: E402

PROVIDERS = ("grok", "codex")

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
                    "title": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "severity": {"enum": ["P0", "P1", "P2", "P3"]},
                },
            }
        },
    },
}

FINDING = {"title": "t", "severity": "P1", "file": "f.py", "line": 1, "evidence": ["f.py:1 -- x"]}
ANSWER = json.dumps(
    {
        "reviewer": "adversarial-reviewer",
        "findings": [FINDING],
        "residual_risks": [],
        "testing_gaps": [],
    }
)
TYPE_INVALID = json.dumps(
    {
        "reviewer": "a",
        "findings": [{**FINDING, "line": "nope"}],
        "residual_risks": [],
        "testing_gaps": [],
    }
)

# A stub runner. It records the argv it was called with, honours codex's `-o`, and does
# whatever the spec file tells it — including going silent, so the watchdogs can be tested.
STUB = """#!/usr/bin/env python3
import json, os, sys, time
spec = json.load(open(os.environ["STUB_SPEC"]))
argv = sys.argv[1:]
with open(os.environ["STUB_ARGV"], "w") as fh:
    json.dump(argv, fh)
last = None
for i, a in enumerate(argv):
    if a == "-o" and i + 1 < len(argv):
        last = argv[i + 1]
if not sys.stdin.isatty():
    try:
        sys.stdin.read()
    except Exception:
        pass
if spec.get("stdout"):
    sys.stdout.write(spec["stdout"])
    sys.stdout.flush()
if last is not None and spec.get("last") is not None:
    with open(last, "w") as fh:
        fh.write(spec["last"])
if spec.get("heartbeat"):
    while True:
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(spec["heartbeat"])
if spec.get("silent_for"):
    time.sleep(spec["silent_for"])
sys.exit(spec.get("exit", 0))
"""


def grok_stream(payload: str | None = None, **result: Any) -> str:
    event: dict[str, Any] = {
        "type": "result",
        "is_error": False,
        "subtype": "success",
        "stop_reason": "end_turn",
    }
    if payload is not None:
        event["structured_output"] = json.loads(payload)
    event.update(result)
    return '{"type":"system","subtype":"init"}\n' + json.dumps(event) + "\n"


class Harness(unittest.TestCase):
    """Fixture assets, stub runners on PATH, and one entry point per provider."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

        self.assets = self.work / "assets"
        (self.assets / "personas").mkdir(parents=True)
        # Ends with a schema-valid EXAMPLE object, as 13 of the 16 real briefs do.
        (self.assets / "personas" / "adversarial-reviewer.md").write_text(
            "# Adversarial Reviewer\n\nBreak it. Return findings matching the findings schema.\n\n"
            '```json\n{"reviewer":"adversarial","findings":[],"residual_risks":[],"testing_gaps":[]}\n```\n',
            encoding="utf-8",
        )
        (self.assets / "personas" / "agent-native-reviewer.md").write_text(
            "# Agent Native Reviewer\n\n## Output Format\n\nA markdown capability table.\n",
            encoding="utf-8",
        )
        # A findings-capable brief OUTSIDE personas/. Without it a traversal test proves
        # nothing: `../outside` fails as "unknown persona" whether or not the name is
        # validated, so deleting the guard would leave the suite green.
        (self.assets / "outside.md").write_text(
            "# Outside\n\nReturn findings matching the findings schema.\n", encoding="utf-8"
        )
        (self.assets / "findings-schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")

        self.bindir = self.work / "bin"
        self.bindir.mkdir()
        # git is symlinked INTO the stub directory rather than reached through the ambient
        # PATH, because the suite runs with a PATH that deliberately excludes it. A developer
        # machine has the real `grok` and `codex` installed, and an early version of this
        # file removed a stub and then reached the genuine binary — which started a real,
        # billed model run from a unit test. No test here may be able to find one.
        real_git = shutil.which("git")
        if real_git:
            (self.bindir / "git").symlink_to(real_git)
        self.run_dir = self.work / "run"
        self.run_dir.mkdir()
        self.spec = self.work / "spec.json"
        self.argv_log = self.work / "argv.json"

        installed = os.environ.get("PERSONA_REVIEW_BIN")
        self.commands: dict[str, Path] = {}
        for provider in PROVIDERS:
            name = f"ce-{provider}-persona"
            if installed:
                self.commands[provider] = Path(installed) / name
            else:
                # Local development: stand in for the console script the packaging
                # generates, with the same "own directory on sys.path" property.
                shim = self.bindir / name
                shim.write_text(
                    # This interpreter, not `env python3`: the suite runs with a restricted
                    # PATH, and /usr/bin/python3 can predate the syntax this package uses.
                    f"#!{sys.executable}\n"
                    "import sys\n"
                    f"sys.path.insert(0, {str(SRC)!r})\n"
                    f"from persona_review.cli import {provider}_main\n"
                    f"sys.exit({provider}_main())\n",
                    encoding="utf-8",
                )
                shim.chmod(0o755)
                self.commands[provider] = shim
            stub = self.bindir / provider
            stub.write_text(STUB, encoding="utf-8")
            stub.chmod(0o755)
        self.findings_cmd = (
            Path(installed) / "ce-persona-findings"
            if installed
            else [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(SRC)!r}); "
                "from persona_review.findings import main; sys.exit(main())",
            ]
        )

    def set_spec(self, **spec: Any) -> None:
        self.spec.write_text(json.dumps(spec), encoding="utf-8")

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                # The stub directory and the base system only. NOT the ambient PATH: the
                # real provider CLIs live on it, and a test that removes a stub must find
                # nothing rather than the genuine binary.
                "PATH": os.pathsep.join([str(self.bindir), "/usr/bin", "/bin"]),
                "CE_REVIEW_ASSETS": str(self.assets),
                "CE_PERSONA_RUN_DIR": str(self.run_dir),
                "STUB_SPEC": str(self.spec),
                "STUB_ARGV": str(self.argv_log),
                # Generous by default; the watchdog tests override them.
                "CE_PERSONA_IDLE_SECS": "60",
                "CE_PERSONA_HARD_SECS": "120",
            }
        )
        env.update(extra)
        return env

    def review(
        self, provider: str, *args: str, timeout: float = 120, **envextra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.commands[provider]), *args],
            capture_output=True,
            text=True,
            env=self.env(**envextra),
            cwd=self.work,
            timeout=timeout,
            check=False,
        )

    def good_answer(self, provider: str) -> None:
        """Spec a stub that returns a valid review through this provider's own channel."""
        if provider == "grok":
            self.set_spec(stdout=grok_stream(ANSWER))
        else:
            self.set_spec(stdout="transcript the caller must never see\n", last=ANSWER)

    def artifact(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}.json"

    def provenance(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}-provenance.json"


class TestContract(Harness):
    def test_a_valid_review_exits_zero_with_one_stdout_line(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(provider, "adversarial-reviewer")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(len(proc.stdout.strip().splitlines()), 1, proc.stdout)
                self.assertIn("1 P1", proc.stdout)
                self.assertTrue(self.artifact(provider).is_file())

    def test_the_transcript_never_reaches_stdout(self):
        # A calling agent pays for every token of stdout; the event stream is ~1 MB.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(provider, "adversarial-reviewer")
                self.assertNotIn("must never see", proc.stdout)
                self.assertTrue(
                    (self.run_dir / f"adversarial-reviewer-{provider}-events.jsonl").is_file()
                )

    def test_provenance_attests_this_run_s_brief_by_hash(self):
        import hashlib

        brief = self.assets / "personas" / "adversarial-reviewer.md"
        want = hashlib.sha256(brief.read_bytes()).hexdigest()
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                self.review(provider, "adversarial-reviewer")
                record = json.loads(self.provenance(provider).read_text(encoding="utf-8"))
                self.assertEqual(record["persona_sha256"], want)
                self.assertEqual(record["provider"], provider)
                self.assertEqual(record["persona"], "adversarial-reviewer")
                self.assertEqual(record["runner_status"], "0")

    def test_the_prompt_carries_the_brief_rubric_boundary_and_strict_clause(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                self.review(provider, "adversarial-reviewer")
                prompt = (self.run_dir / f"adversarial-reviewer-{provider}-prompt.md").read_text(
                    encoding="utf-8"
                )
                self.assertIn("Break it. Return findings matching the findings schema.", prompt)
                self.assertIn("Anchors 0 and 25 mean SUPPRESS", prompt)
                self.assertIn("exactly one JSON object", prompt)
                self.assertTrue(prompt.rstrip().endswith(assets.BOUNDARY))

    def test_the_effort_flag_reaches_the_runner(self):
        for provider, expected in (("grok", "xhigh"), ("codex", 'model_reasoning_effort="xhigh"')):
            with self.subTest(provider=provider):
                self.good_answer(provider)
                self.review(provider, "adversarial-reviewer", "-e", "xhigh")
                argv = json.loads(self.argv_log.read_text(encoding="utf-8"))
                self.assertIn(expected, argv)


class TestGate(Harness):
    def test_prose_instead_of_findings_fails(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                if provider == "grok":
                    self.set_spec(stdout=grok_stream(result="I gave up."))
                else:
                    self.set_spec(stdout="", last="I gave up.")
                self.assertEqual(self.review(provider, "adversarial-reviewer").returncode, 1)

    def test_a_give_up_quoting_the_brief_example_fails(self):
        # The regression two model families found independently: a refusal plus the brief's
        # own empty example used to validate as a clean review.
        text = "I could not inspect the repository. The requested shape is:\n\n" + json.dumps(
            {"reviewer": "adversarial", "findings": [], "residual_risks": [], "testing_gaps": []}
        )
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                if provider == "grok":
                    self.set_spec(stdout=grok_stream(result=text))
                else:
                    self.set_spec(stdout="", last=text)
                proc = self.review(provider, "adversarial-reviewer")
                self.assertEqual(proc.returncode, 1, proc.stdout)

    def test_a_type_invalid_answer_fails(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                if provider == "grok":
                    self.set_spec(stdout=grok_stream(TYPE_INVALID))
                else:
                    self.set_spec(stdout="", last=TYPE_INVALID)
                self.assertEqual(self.review(provider, "adversarial-reviewer").returncode, 1)

    def test_grok_terminal_status_is_checked_not_just_the_payload(self):
        for bad in (
            {"stop_reason": "max_tokens"},
            {"subtype": "error_max_turns"},
            {"is_error": True},
        ):
            with self.subTest(bad=bad):
                self.set_spec(stdout=grok_stream(ANSWER, **bad))
                self.assertEqual(self.review("grok", "adversarial-reviewer").returncode, 1)

    def test_codex_with_no_final_message_fails(self):
        # And must not re-validate whatever a previous run left at the same path.
        self.good_answer("codex")
        self.assertEqual(self.review("codex", "adversarial-reviewer").returncode, 0)
        self.set_spec(stdout="no final message this time\n")
        proc = self.review("codex", "adversarial-reviewer")
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertFalse(self.artifact("codex").exists())


class TestExitStatus(Harness):
    def test_a_runner_failure_is_reported_as_4_not_propagated(self):
        # Propagating the runner's own status verbatim collides with the codes reserved for
        # usage (2) and over-budget (78), so a caller cannot tell them apart.
        for provider in PROVIDERS:
            for runner_exit in (1, 2, 3, 78):
                with self.subTest(provider=provider, runner_exit=runner_exit):
                    self.set_spec(stdout="boom\n", exit=runner_exit)
                    proc = self.review(provider, "adversarial-reviewer")
                    self.assertEqual(proc.returncode, 4, proc.stderr)

    def test_usage_errors_exit_2(self):
        for provider in PROVIDERS:
            self.good_answer(provider)
            for args in (
                [],  # no persona
                ["no-such-persona"],
                ["agent-native-reviewer"],  # markdown-only brief
                ["../outside"],  # not a bare brief name
                ["adversarial-reviewer", "-C", str(self.work / "nope")],
                ["adversarial-reviewer", "-b", "no-such-ref"],
                ["adversarial-reviewer", "-m"],  # option missing its value
            ):
                with self.subTest(provider=provider, args=args):
                    self.assertEqual(self.review(provider, *args).returncode, 2)

    def test_a_traversal_persona_name_is_refused_even_though_it_would_resolve(self):
        # The fixture deliberately places a findings-capable brief at assets/outside.md, so
        # `../outside` WOULD resolve if the name were not validated — and every artifact path
        # would then be built outside the run directory.
        escaped = self.work / "outside-{}.json"
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(provider, "../outside")
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("bare brief name", proc.stderr)
                self.assertFalse(
                    Path(str(escaped).format(provider)).exists(),
                    "an artifact was written outside the run directory",
                )

    def test_help_exits_zero_and_documents_the_exit_table(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                proc = self.review(provider, "--help")
                self.assertEqual(proc.returncode, 0)
                for code in ("0 ", "1 ", "2 ", "3 ", "4 ", "5 ", "78 "):
                    self.assertIn(code, proc.stdout)
                self.assertIn("CE_PERSONA_IDLE_SECS", proc.stdout)

    def test_an_unusable_run_dir_is_an_environment_error_not_a_usage_one(self):
        # A caller retrying with different arguments cannot fix an unwritable directory, so
        # it belongs with the missing-binary class, not with bad arguments.
        blocker = self.work / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(
                    provider, "adversarial-reviewer", CE_PERSONA_RUN_DIR=str(blocker / "sub")
                )
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertIn("CE_PERSONA_RUN_DIR", proc.stderr)

    def test_a_missing_runner_is_an_environment_error(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                (self.bindir / provider).unlink()
                proc = self.review(provider, "adversarial-reviewer")
                self.assertEqual(proc.returncode, 3)
                self.assertIn("not on PATH", proc.stderr)


class TestBudget(Harness):
    def _git_env(self) -> dict[str, str]:
        return dict(
            os.environ,
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.invalid",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.invalid",
        )

    def _git(self, repo: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            env=self._git_env(),
            check=True,
        )
        return proc.stdout.strip()

    def _repo(self) -> tuple[Path, str]:
        repo = self.work / "repo"
        repo.mkdir()
        self._git(repo, "init", "-q", "-b", "main")
        (repo / "f.txt").write_text("one\n", encoding="utf-8")
        self._git(repo, "add", "f.txt")
        self._git(repo, "commit", "-qm", "base")
        base = self._git(repo, "rev-parse", "HEAD")
        (repo / "f.txt").write_text("x" * 40000 + "\n", encoding="utf-8")
        self._git(repo, "add", "f.txt")
        self._git(repo, "commit", "-qm", "big")
        return repo, base

    def test_a_large_diff_trips_a_budget_the_prompt_alone_never_could(self):
        repo, base = self._repo()
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.set_spec(stdout="the model must never be called\n")
                proc = self.review(
                    provider,
                    "adversarial-reviewer",
                    "-C",
                    str(repo),
                    "-b",
                    base,
                    CE_PERSONA_MAX_PROMPT_TOKENS="2000",
                )
                self.assertEqual(proc.returncode, 78, proc.stderr)
                self.assertIn("diff bytes", proc.stderr)
                self.assertFalse(self.argv_log.exists(), "the runner must not be called")

    def test_the_same_budget_with_an_empty_diff_is_not_refused(self):
        # Control: without it the test above could pass for the wrong reason.
        repo, _ = self._repo()
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(
                    provider,
                    "adversarial-reviewer",
                    "-C",
                    str(repo),
                    "-b",
                    "HEAD",
                    CE_PERSONA_MAX_PROMPT_TOKENS="2000",
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_a_tiny_budget_refuses_before_the_runner_is_called(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.set_spec(stdout="the model must never be called\n")
                proc = self.review(
                    provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="1"
                )
                self.assertEqual(proc.returncode, 78)
                self.assertFalse(self.argv_log.exists())

    def test_a_non_numeric_budget_is_a_usage_error(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(
                    provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="abc"
                )
                self.assertEqual(proc.returncode, 2)
                self.assertIn("is not a number", proc.stderr)

    def test_the_reviewed_repo_cannot_execute_code_through_git_diff(self):
        # `git diff` runs commands named by the repository's own configuration, and this
        # preflight runs OUTSIDE the read-only sandbox the review itself gets. Both vectors
        # are covered because they are disabled by different flags.
        #
        # Written with `git config`, never by appending raw text: an earlier version of this
        # test appended a [diff] section containing quotes, git never parsed it into an
        # effective driver, and the assertion then held whether or not the mitigation was
        # there. It passed for the wrong reason and proved nothing.
        repo, base = self._repo()
        ext_marker = self.work / "EXTERNAL_DIFF_RAN"
        tc_marker = self.work / "TEXTCONV_RAN"
        self._git(repo, "config", "diff.external", f"sh -c 'touch {ext_marker}; exit 0' --")
        (repo / ".gitattributes").write_text("f.txt diff=pwn\n", encoding="utf-8")
        self._git(repo, "config", "diff.pwn.textconv", f"sh -c 'touch {tc_marker}; cat \"$1\"' --")
        self._git(repo, "add", ".gitattributes")
        self._git(repo, "commit", "-qm", "attrs")

        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                ext_marker.unlink(missing_ok=True)
                tc_marker.unlink(missing_ok=True)
                self.review(provider, "adversarial-reviewer", "-C", str(repo), "-b", base)
                self.assertFalse(ext_marker.exists(), "the repo's diff.external driver executed")
                self.assertFalse(tc_marker.exists(), "the repo's textconv driver executed")

    def test_the_hostile_repo_fixture_actually_arms_both_drivers(self):
        # A control for the test above. Without it, a fixture that fails to configure the
        # drivers would make that test pass no matter what the code does — which is exactly
        # what happened before, and what mutation testing caught.
        repo, base = self._repo()
        ext_marker = self.work / "CONTROL_EXTERNAL"
        tc_marker = self.work / "CONTROL_TEXTCONV"
        self._git(repo, "config", "diff.external", f"sh -c 'touch {ext_marker}; exit 0' --")
        subprocess.run(
            ["git", "-C", str(repo), "diff", f"{base}..HEAD"],
            capture_output=True,
            env=self._git_env(),
            check=True,
        )
        self.assertTrue(ext_marker.exists(), "fixture did not arm diff.external")

        self._git(repo, "config", "--unset", "diff.external")
        (repo / ".gitattributes").write_text("f.txt diff=pwn\n", encoding="utf-8")
        self._git(repo, "config", "diff.pwn.textconv", f"sh -c 'touch {tc_marker}; cat \"$1\"' --")
        self._git(repo, "add", ".gitattributes")
        self._git(repo, "commit", "-qm", "attrs")
        subprocess.run(
            ["git", "-C", str(repo), "diff", f"{base}..HEAD"],
            capture_output=True,
            env=self._git_env(),
            check=True,
        )
        self.assertTrue(tc_marker.exists(), "fixture did not arm textconv")


class TestArtifactLifecycle(Harness):
    def _seed(self, provider: str) -> None:
        self.good_answer(provider)
        self.assertEqual(self.review(provider, "adversarial-reviewer").returncode, 0)
        self.assertTrue(self.artifact(provider).is_file())

    def test_a_failed_run_leaves_no_stale_artifact(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self._seed(provider)
                self.set_spec(stdout="boom\n", exit=9)
                self.assertEqual(self.review(provider, "adversarial-reviewer").returncode, 4)
                self.assertFalse(self.artifact(provider).exists())
                self.assertFalse(self.provenance(provider).exists())

    def test_a_refusal_before_dispatch_leaves_no_stale_artifact(self):
        # A run refused at the budget or for a bad base never reaches the runner, and used to
        # leave the previous run's findings and provenance in place.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self._seed(provider)
                proc = self.review(
                    provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="1"
                )
                self.assertEqual(proc.returncode, 78)
                self.assertFalse(self.artifact(provider).exists())
                self.assertFalse(self.provenance(provider).exists())


class TestWatchdogs(Harness):
    def test_a_silent_runner_is_killed_and_reported_as_a_timeout(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.set_spec(stdout="starting\n", silent_for=90)
                started = time.monotonic()
                proc = self.review(
                    provider, "adversarial-reviewer", timeout=60, CE_PERSONA_IDLE_SECS="3"
                )
                elapsed = time.monotonic() - started
                self.assertEqual(proc.returncode, 5, proc.stderr)
                self.assertIn("no output for", proc.stderr)
                self.assertLess(elapsed, 40, "the idle watchdog did not fire promptly")

    def test_a_chatty_runner_still_hits_the_hard_deadline(self):
        # File growth keeps the idle watchdog happy forever, which is exactly why a wall-clock
        # cap has to exist too.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.set_spec(heartbeat=0.2)
                proc = self.review(
                    provider,
                    "adversarial-reviewer",
                    timeout=60,
                    CE_PERSONA_IDLE_SECS="30",
                    CE_PERSONA_HARD_SECS="4",
                )
                self.assertEqual(proc.returncode, 5, proc.stderr)
                self.assertIn("hard timeout", proc.stderr)

    def test_partial_output_is_kept_after_a_timeout(self):
        self.set_spec(stdout="partial evidence\n", silent_for=90)
        self.review("grok", "adversarial-reviewer", timeout=60, CE_PERSONA_IDLE_SECS="3")
        events = self.run_dir / "adversarial-reviewer-grok-events.jsonl"
        self.assertIn("partial evidence", events.read_text(encoding="utf-8"))


class TestFindingsCommand(Harness):
    def _findings(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = self.findings_cmd
        argv = [str(cmd), *args] if isinstance(cmd, Path) else [*cmd, *args]
        return subprocess.run(argv, capture_output=True, text=True, env=self.env(), check=False)

    def test_the_installed_command_reads_a_real_artifact(self):
        self.good_answer("grok")
        self.assertEqual(self.review("grok", "adversarial-reviewer").returncode, 0)
        proc = self._findings(str(self.artifact("grok")))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("BEGIN UNTRUSTED MODEL OUTPUT", proc.stdout)
        self.assertIn("#1 P1", proc.stdout)

    def test_json_round_trips_through_the_installed_command(self):
        self.good_answer("grok")
        self.review("grok", "adversarial-reviewer")
        proc = self._findings(str(self.artifact("grok")), "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["findings"][0]["severity"], "P1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
