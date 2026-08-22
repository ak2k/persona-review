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
# Persist what the runner was ACTUALLY given, from whichever channel this provider uses.
# Without this nothing asserts the model received the brief at all: dispatching codex with
# no prompt whatsoever left both suites green.
prompt = ""
for i, a in enumerate(argv):
    if a == "--prompt-file" and i + 1 < len(argv):
        with open(argv[i + 1]) as fh:
            prompt = fh.read()
if not sys.stdin.isatty():
    try:
        prompt = prompt or sys.stdin.read()
    except Exception:
        pass
with open(os.environ["STUB_PROMPT"], "w") as fh:
    fh.write(prompt)
if spec.get("stdout"):
    sys.stdout.write(spec["stdout"])
    sys.stdout.flush()
if last is not None and spec.get("last") is not None:
    with open(last, "w") as fh:
        fh.write(spec["last"])
if spec.get("stubborn_child"):
    # A helper that ignores SIGTERM, like the real provider CLIs' subprocesses. The parent
    # dying is not evidence the group did.
    import signal as _signal
    _pid = os.fork()
    if _pid == 0:
        _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)
        time.sleep(300)
        os._exit(0)
    with open(os.environ["STUB_CHILD"], "w") as fh:
        fh.write(str(_pid))
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
        self.child_pid_file = self.work / "child.pid"
        self.prompt_seen = self.work / "prompt-the-runner-received.md"

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
                "STUB_CHILD": str(self.child_pid_file),
                "STUB_PROMPT": str(self.prompt_seen),
                # Generous by default; the watchdog tests override them.
                "CE_PERSONA_IDLE_SECS": "60",
                "CE_PERSONA_HARD_SECS": "120",
            }
        )
        env.update(extra)
        return env

    def review(
        self,
        provider: str,
        *args: str,
        timeout: float = 120,
        env_extra: dict[str, str] | None = None,
        **envextra: str,
    ) -> subprocess.CompletedProcess[str]:
        merged = {**(env_extra or {}), **envextra}
        return subprocess.run(
            [str(self.commands[provider]), *args],
            capture_output=True,
            text=True,
            env=self.env(**merged),
            cwd=self.work,
            timeout=timeout,
            check=False,
        )

    def good_answer(self, provider: str) -> None:
        """Spec a stub that returns a valid review through this provider's own channel.

        BOTH streams carry the leak marker. The grok fixture used to be marker-free, so the
        "transcript never reaches stdout" assertion could not have failed for the grok arm
        however the event stream was routed.
        """
        if provider == "grok":
            self.set_spec(
                stdout='{"type":"assistant","text":"transcript the caller must never see"}\n'
                + grok_stream(ANSWER)
            )
        else:
            self.set_spec(stdout="transcript the caller must never see\n", last=ANSWER)

    def artifact(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}.json"

    def provenance(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}-provenance.json"

    def assert_run_dir_clean(self, provider: str) -> None:
        """No artifact from an earlier run survives.

        Named files are not enough: asserting only `.json` and `-provenance.json` left a
        stale `-stderr.log` — the one artifact the failure path reads back — sitting in a
        directory the code promises to have cleared, and the test stayed green.
        """
        stem = f"adversarial-reviewer-{provider}"
        leftovers = sorted(p.name for p in self.run_dir.iterdir() if p.name.startswith(stem))
        self.assertEqual(leftovers, [], f"stale artifacts left in the run dir: {leftovers}")


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

    def test_the_prompt_the_RUNNER_RECEIVED_carries_brief_rubric_and_clause(self):
        # Read from what the stub was handed, NOT from the -prompt.md the CLI wrote. Those
        # are different claims, and only the first one is the product: dispatching codex
        # with no prompt at all left the whole suite green when this read the written file.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.prompt_seen.unlink(missing_ok=True)
                self.good_answer(provider)
                self.review(provider, "adversarial-reviewer")
                prompt = self.prompt_seen.read_text(encoding="utf-8")
                self.assertIn("Break it. Return findings matching the findings schema.", prompt)
                self.assertIn("Anchors 0 and 25 mean SUPPRESS", prompt)
                self.assertIn("exactly one JSON object", prompt)
                self.assertTrue(prompt.rstrip().endswith(assets.BOUNDARY))

    def test_the_model_and_effort_flags_reach_the_runner(self):
        # NON-DEFAULT sentinels. Asserting `-e xhigh` reaches grok proves nothing, because
        # xhigh is grok's default: hardcoding the flag and ignoring -e kept the suite green.
        for provider, expected in (
            ("grok", ["--model", "sentinel-model", "--effort", "sentinel-effort"]),
            ("codex", ["-m", "sentinel-model", 'model_reasoning_effort="sentinel-effort"']),
        ):
            with self.subTest(provider=provider):
                self.good_answer(provider)
                self.review(
                    provider,
                    "adversarial-reviewer",
                    "-e",
                    "sentinel-effort",
                    "-m",
                    "sentinel-model",
                )
                argv = json.loads(self.argv_log.read_text(encoding="utf-8"))
                for token in expected:
                    self.assertIn(token, argv)


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
        #
        # For grok this exercises the RAW-TEXT fallback (`result`), not the `structured_output`
        # channel `--json-schema` actually uses. That distinction is not cosmetic — see
        # test_an_empty_structured_output_is_the_documented_gap below, which pins what the
        # production channel really does rather than letting this test imply it is covered.
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

    def test_an_empty_structured_output_is_the_documented_gap(self):
        # Executable documentation of the ONE hole this package does not close, on the exact
        # channel production uses: a healthy terminal event carrying a schema-valid EMPTY
        # findings object is accepted, because "found nothing" and "quietly gave up" are
        # indistinguishable without judging the transcript.
        #
        # It is asserted rather than left implicit so that closing it later fails loudly
        # here, and so no other test can be read as already covering it.
        empty = json.dumps(
            {"reviewer": "adversarial", "findings": [], "residual_risks": [], "testing_gaps": []}
        )
        self.set_spec(stdout=grok_stream(empty))
        proc = self.review("grok", "adversarial-reviewer")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("0 findings", proc.stdout)

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

    def test_a_persona_review_package_in_the_cwd_cannot_replace_the_gate(self):
        # THE headline P0 this port exists to close, and the bash suite's guard for it was
        # deleted with no replacement. `python3 -c/-m` put the caller's cwd at sys.path[0],
        # so a persona_review/ directory inside the repo under review answered for the gate:
        # a clean review reported for a run that never happened, and planted code executing
        # outside the read-only sandbox.
        #
        # Only meaningful against the INSTALLED console script, whose sys.path[0] is its own
        # bin directory. The local-dev shim hardcodes sys.path, so it would assert the
        # property into existence rather than test it.
        if not os.environ.get("PERSONA_REVIEW_BIN"):
            self.skipTest("needs the installed console scripts (set PERSONA_REVIEW_BIN)")
        planted = self.work / "persona_review"
        planted.mkdir()
        (planted / "__init__.py").write_text("", encoding="utf-8")
        (planted / "validate.py").write_text(
            "import sys\nprint('ce-persona: 0 findings -> HIJACKED')\nsys.exit(0)\n",
            encoding="utf-8",
        )
        # The planted package must shadow the module the entry point actually imports.
        # Planting only validate.py crashes on `import persona_review.cli` and looks like a
        # pass; planting cli.py is what silently answers for the run.
        (planted / "cli.py").write_text(
            "def grok_main() -> int:\n"
            "    print('ce-persona: 0 findings -> HIJACKED')\n"
            "    return 0\n"
            "def codex_main() -> int:\n"
            "    return grok_main()\n",
            encoding="utf-8",
        )
        # BOTH doors: sys.path[0] (cwd), and a caller-set PYTHONPATH, which the packaging
        # wrapper appends behind rather than in front of. `PYTHONPATH=.` is routine under
        # direnv, tox and CI images — and this repo's own devShell sets it.
        for provider in PROVIDERS:
            for pythonpath in (None, ".", "./"):
                with self.subTest(provider=provider, pythonpath=pythonpath):
                    # A runner that categorically did not review, so a clean result can only
                    # have come from the planted module.
                    self.set_spec(stdout=grok_stream(ANSWER, is_error=True))
                    if provider == "codex":
                        self.set_spec(stdout="", last="I could not review.")
                    extra = {} if pythonpath is None else {"PYTHONPATH": pythonpath}
                    proc = self.review(provider, "adversarial-reviewer", env_extra=extra)
                    self.assertNotIn("HIJACKED", proc.stdout)
                    self.assertNotEqual(proc.returncode, 0, "the planted gate answered for the run")

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

    def test_an_option_shaped_base_is_refused_not_parsed_as_an_option(self):
        # `git diff` parses an option-shaped base as an OPTION: `--output=<path>` makes git
        # write the diff to that path and exit 0 with empty stdout, so the "proof the range
        # resolves" reports a 0-byte diff for a range it never resolved — a CLEAN REVIEW of
        # nothing — while writing a file of the caller's choosing.
        #
        # Two details this test previously got wrong, both of which made it vacuous:
        #   * `--base=VALUE`, not `-b VALUE`. argparse rejects a `-b` value that starts with
        #     a dash, so the space form never reaches git and the test passed for that
        #     reason alone.
        #   * git writes `<path>..HEAD`, not `<path>`, because the whole range string is
        #     taken as the option's value.
        repo, _ = self._repo()
        target = self.work / "SHOULD_NOT_BE_WRITTEN"
        written = Path(f"{target}..HEAD")
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                written.unlink(missing_ok=True)
                proc = self.review(
                    provider, "adversarial-reviewer", "-C", str(repo), f"--base=--output={target}"
                )
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertFalse(written.exists(), "git took the base as an option and wrote it")

    def test_a_dash_prefixed_base_is_rejected_by_argument_parsing(self):
        # The space form is blocked one layer earlier. Asserted so the two mechanisms stay
        # distinguishable: if argparse ever accepted it, the test above is the net.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                proc = self.review(provider, "adversarial-reviewer", "-b", "--output=/tmp/x")
                self.assertEqual(proc.returncode, 2, proc.stderr)

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
                proc = self.review(provider, "adversarial-reviewer", "-C", str(repo), "-b", base)
                # Assert the run SUCCEEDED first. Without this the markers are also absent
                # when the guarded git call simply fails before dispatch, so the test would
                # pass while proving nothing about the mitigation.
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertFalse(ext_marker.exists(), "the repo's diff.external driver executed")
                self.assertFalse(tc_marker.exists(), "the repo's textconv driver executed")

    def test_a_repo_cannot_wedge_the_budget_preflight_with_stderr(self):
        # The measurement drains stdout to EOF. If stderr is a second PIPE, a git that fills
        # the ~64 KiB stderr buffer blocks in write() and never closes stdout — the reader
        # waits for an EOF that cannot come. A committed .gitattributes of a few thousand
        # malformed lines produces half a megabyte of stderr, so the hang is repo-controlled,
        # and this runs BEFORE the watchdogs in `execute` exist.
        #
        # A real .gitattributes rather than a stub git: it needs no fixture to be believed,
        # and it is how the reviewer found it.
        repo, base = self._repo()
        (repo / ".gitattributes").write_text('f.txt "bad00000\n' * 4000, encoding="utf-8")
        self._git(repo, "add", ".gitattributes")
        self._git(repo, "commit", "-qm", "malformed attributes")
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.good_answer(provider)
                started = time.monotonic()
                proc = self.review(
                    provider, "adversarial-reviewer", "-C", str(repo), "-b", base, timeout=90
                )
                elapsed = time.monotonic() - started
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertLess(elapsed, 60, "the budget preflight wedged on git's stderr")

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

    def test_every_early_failure_leaves_no_stale_artifact(self):
        # Not just the budget refusal. The run dir is cleared as soon as the persona is
        # known, so a failure at ANY later preflight — a missing binary, an unreadable
        # context file — cannot leave the previous run's findings at the path a caller reads.
        for provider in PROVIDERS:
            with self.subTest(provider=provider, case="missing binary"):
                self._seed(provider)
                (self.bindir / provider).unlink()
                self.assertEqual(self.review(provider, "adversarial-reviewer").returncode, 3)
                self.assert_run_dir_clean(provider)
                (self.bindir / provider).write_text(STUB, encoding="utf-8")
                (self.bindir / provider).chmod(0o755)

            with self.subTest(provider=provider, case="unreadable context"):
                self._seed(provider)
                proc = self.review(
                    provider, "adversarial-reviewer", "-c", str(self.work / "no-such-context.md")
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assert_run_dir_clean(provider)

    def test_a_non_finite_timeout_is_refused_rather_than_disabling_the_watchdog(self):
        # float() accepts nan and inf, and every deadline is `elapsed >= value`, which is
        # False forever against either — a typo would silently switch the watchdog off.
        for provider in PROVIDERS:
            for name in ("CE_PERSONA_IDLE_SECS", "CE_PERSONA_HARD_SECS"):
                for value in ("nan", "inf", "-inf"):
                    with self.subTest(provider=provider, var=name, value=value):
                        self.good_answer(provider)
                        proc = self.review(
                            provider, "adversarial-reviewer", env_extra={name: value}
                        )
                        self.assertEqual(proc.returncode, 2, proc.stderr)

    def test_a_non_finite_budget_is_a_usage_error_not_a_crash(self):
        for provider in PROVIDERS:
            for value in ("nan", "inf"):
                with self.subTest(provider=provider, value=value):
                    self.good_answer(provider)
                    proc = self.review(
                        provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS=value
                    )
                    self.assertEqual(proc.returncode, 2, proc.stderr)

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
                self.assert_run_dir_clean(provider)


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

    def test_a_timeout_kills_helpers_that_ignore_sigterm(self):
        # The direct child exiting is not proof the process group did. A parent with default
        # SIGTERM handling dies at once while a helper that ignores it keeps running, so a
        # watchdog that waits on the child alone reports a kill it did not perform — and the
        # provider keeps working, and keeps the artifact descriptors open.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.child_pid_file.unlink(missing_ok=True)
                self.set_spec(stdout="starting\n", stubborn_child=True, silent_for=120)
                proc = self.review(
                    provider, "adversarial-reviewer", timeout=90, CE_PERSONA_IDLE_SECS="3"
                )
                self.assertEqual(proc.returncode, 5, proc.stderr)
                pid = int(self.child_pid_file.read_text(encoding="utf-8").strip())
                deadline = time.monotonic() + 20
                alive = True
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except (ProcessLookupError, PermissionError):
                        alive = False
                        break
                    time.sleep(0.2)
                if alive:
                    os.kill(pid, 9)  # do not leak a 300s sleeper from a failed test
                self.assertFalse(alive, f"helper {pid} survived the timeout kill")

    def test_killing_the_wrapper_does_not_orphan_the_provider(self):
        # `start_new_session=True` is what lets the watchdog signal the provider's whole
        # group — and it also means the provider does NOT die with this command. Ctrl-C, a
        # supervising agent's own timeout, or a CI cancel would otherwise leave a
        # full-effort model run going with nothing watching it and nothing that will read
        # its output. The shell version had an EXIT trap; this asserts the replacement.
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.child_pid_file.unlink(missing_ok=True)
                self.set_spec(stdout="starting\n", stubborn_child=True, silent_for=300)
                wrapper = subprocess.Popen(
                    [str(self.commands[provider]), "adversarial-reviewer"],
                    env=self.env(),
                    cwd=self.work,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline and not self.child_pid_file.exists():
                        time.sleep(0.2)
                    self.assertTrue(self.child_pid_file.exists(), "the stub never started")
                    pid = int(self.child_pid_file.read_text(encoding="utf-8").strip())

                    wrapper.terminate()
                    wrapper.wait(timeout=60)

                    deadline = time.monotonic() + 30
                    alive = True
                    while time.monotonic() < deadline:
                        try:
                            os.kill(pid, 0)
                        except (ProcessLookupError, PermissionError):
                            alive = False
                            break
                        time.sleep(0.2)
                    if alive:
                        os.kill(pid, 9)
                    self.assertFalse(alive, f"provider helper {pid} outlived the killed wrapper")
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=10)

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
