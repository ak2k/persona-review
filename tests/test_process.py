#!/usr/bin/env python3
"""Process-level tests: the installed commands, driven against stub runners.

Run: pytest tests/test_process.py   (set PERSONA_REVIEW_BIN to test a built package)

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
from pathlib import Path
from typing import Any

import pytest

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
#
# This interpreter, not `env python3`, for the reason the shims below carry the same
# shebang: `env()` runs these under a PATH of <stubs>:/usr/bin:/bin, and a build sandbox
# that has no /usr/bin/python3 cannot exec them. The kernel reports a missing interpreter
# as ENOENT on the stub itself, which surfaces as the provider missing from PATH — so the
# suite fails everywhere the platform does not happen to ship a system python3.
STUB = f"""#!{sys.executable}
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


# One tool call, in each provider's own event vocabulary. A stream WITHOUT one is a run that
# inspected nothing, which the wrapper now refuses with exit 6 — so a fixture standing in for
# a real review has to carry one, and the fixtures that deliberately omit it are testing the
# refusal rather than forgetting to be realistic.
#
# Both shapes are copied from real runs: grok 1.0.13 (a `tool_use` content block inside an
# assistant message) and codex-cli 0.150.1 (`item.started`/`item.completed` around an item
# whose `type` names a tool).
GROK_TOOL_CALL = json.dumps(
    {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"p": "f.py"}}
            ]
        },
    }
)
CODEX_TOOL_CALL = "\n".join(
    json.dumps(
        {
            "type": kind,
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "git diff",
                "status": status,
            },
        }
    )
    for kind, status in (("item.started", "in_progress"), ("item.completed", "completed"))
)


def grok_stream(payload: str | None = None, *, tool_call: bool = False, **result: Any) -> str:
    event: dict[str, Any] = {
        "type": "result",
        "is_error": False,
        "subtype": "success",
        "stop_reason": "end_turn",
        "num_turns": 3,
        "usage": {"output_tokens": 4096},
    }
    if payload is not None:
        event["structured_output"] = json.loads(payload)
    event.update(result)
    head = '{"type":"system","subtype":"init"}\n'
    if tool_call:
        head += GROK_TOOL_CALL + "\n"
    return head + json.dumps(event) + "\n"


def codex_stream(*, tool_call: bool = False) -> str:
    """codex's `--json` stream: a thread, a turn, an agent message, optionally a tool call.

    codex's ANSWER never travels here — `-o` writes it to its own file — so this exists only
    as the evidence of what the run did. The agent message and the turn are always present,
    which is what makes the tool-call arm the single variable.
    """
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "th_1"}),
        json.dumps({"type": "turn.started"}),
    ]
    if tool_call:
        lines.append(CODEX_TOOL_CALL)
    lines.append(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "item_9", "type": "agent_message", "text": "done"},
            }
        )
    )
    lines.append(json.dumps({"type": "turn.completed", "usage": {"output_tokens": 4096}}))
    return "\n".join(lines) + "\n"


class Harness:
    """Fixture assets, stub runners on PATH, and one entry point per provider."""

    def setup_method(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
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

    def teardown_method(self) -> None:
        self.tmp.cleanup()

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

        Both also carry a TOOL CALL, because a run that made none is refused: a fixture
        without one would be a dud review, and every test built on this would be asserting
        against the refusal path by accident.
        """
        if provider == "grok":
            self.set_spec(
                stdout='{"type":"assistant","text":"transcript the caller must never see"}\n'
                + grok_stream(ANSWER, tool_call=True)
            )
        else:
            self.set_spec(
                stdout=codex_stream(tool_call=True) + "transcript the caller must never see\n",
                last=ANSWER,
            )

    def findings(self, *args: str) -> subprocess.CompletedProcess[str]:
        """The retrieval command, as installed. On Harness rather than on its own test class
        because the vacuous-run tests need it too: the refusal is only closed if BOTH the
        review command and the reader of its artifact honour it."""
        cmd = self.findings_cmd
        argv = [str(cmd), *args] if isinstance(cmd, Path) else [*cmd, *args]
        return subprocess.run(argv, capture_output=True, text=True, env=self.env(), check=False)

    def artifact(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}.json"

    def provenance(self, provider: str) -> Path:
        return self.run_dir / f"adversarial-reviewer-{provider}-provenance.json"

    # A real git repository, on Harness rather than on one test class: the size
    # preflight and the provenance record both need one, and duplicating the fixture is
    # how two copies of a guard end up tested in only one place.
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

    def assert_run_dir_clean(self, provider: str) -> None:
        """No artifact from an earlier run survives.

        Named files are not enough: asserting only `.json` and `-provenance.json` left a
        stale `-stderr.log` — the one artifact the failure path reads back — sitting in a
        directory the code promises to have cleared, and the test stayed green.
        """
        stem = f"adversarial-reviewer-{provider}"
        # The `.lock` file is coordination, not an artifact: it holds no run output, it is
        # created before the clear rather than by it, and unlinking it would race a waiter
        # that had already opened the path. Excluded BY EXACT NAME rather than by extension,
        # so this stays a whitelist of one and a new leftover cannot slip through it.
        allowed = {f"{stem}.lock"}
        leftovers = sorted(
            p.name
            for p in self.run_dir.iterdir()
            if p.name.startswith(stem) and p.name not in allowed
        )
        assert leftovers == [], f"stale artifacts left in the run dir: {leftovers}"


class TestContract(Harness):
    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_valid_review_exits_zero_with_one_stdout_line(self, provider: str):
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer")
        assert proc.returncode == 0, proc.stderr
        assert len(proc.stdout.strip().splitlines()) == 1, proc.stdout
        assert "1 P1" in proc.stdout
        assert self.artifact(provider).is_file()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_transcript_never_reaches_stdout(self, provider: str):
        # A calling agent pays for every token of stdout; the event stream is ~1 MB.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer")
        assert "must never see" not in proc.stdout
        assert (self.run_dir / f"adversarial-reviewer-{provider}-events.jsonl").is_file()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_provenance_attests_this_run_s_brief_by_hash(self, provider: str):
        import hashlib

        brief = self.assets / "personas" / "adversarial-reviewer.md"
        want = hashlib.sha256(brief.read_bytes()).hexdigest()
        self.good_answer(provider)
        self.review(provider, "adversarial-reviewer")
        record = json.loads(self.provenance(provider).read_text(encoding="utf-8"))
        assert record["persona_sha256"] == want
        assert record["provider"] == provider
        assert record["persona"] == "adversarial-reviewer"
        assert record["runner_status"] == "0"
        # A non-repo target is legitimate — the model reads files, not history — but the
        # record must SAY that rather than omit the field, so a consumer can tell "no commit"
        # from "this field was never written".
        assert record["repo"] == str(self.work.resolve())
        assert record["head_sha"].startswith("unresolved:"), record["head_sha"]

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_provenance_records_the_commit_that_was_reviewed(self, provider: str):
        # `base_ref` is the CALLER's string: `HEAD~1` names a different commit every day, so
        # on its own it cannot tie a finding at f.py:42 to the code it was about. Two runs of
        # the same command a week apart were indistinguishable in the record.
        repo, base = self._repo()
        head = self._git(repo, "rev-parse", "HEAD").strip()
        base_sha = self._git(repo, "rev-parse", base).strip()
        assert head != base_sha, "the fixture must have two commits or this proves nothing"

        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer", "-C", str(repo), "-b", base)
        assert proc.returncode == 0, proc.stderr
        record = json.loads(self.provenance(provider).read_text(encoding="utf-8"))
        assert record["head_sha"] == head
        assert record["base_sha"] == base_sha
        assert record["base_ref"] == base

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_prompt_the_RUNNER_RECEIVED_carries_brief_rubric_and_clause(self, provider: str):
        # Read from what the stub was handed, NOT from the -prompt.md the CLI wrote. Those
        # are different claims, and only the first one is the product: dispatching codex
        # with no prompt at all left the whole suite green when this read the written file.
        self.prompt_seen.unlink(missing_ok=True)
        self.good_answer(provider)
        self.review(provider, "adversarial-reviewer")
        prompt = self.prompt_seen.read_text(encoding="utf-8")
        assert "Break it. Return findings matching the findings schema." in prompt
        assert "Anchors 0 and 25 mean SUPPRESS" in prompt
        assert "exactly one JSON object" in prompt
        assert prompt.rstrip().endswith(assets.BOUNDARY)

    # NON-DEFAULT sentinels. Asserting `-e xhigh` reaches grok proves nothing, because xhigh
    # is grok's default: hardcoding the flag and ignoring -e kept the suite green.
    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            ("grok", ["--model", "sentinel-model", "--effort", "sentinel-effort"]),
            ("codex", ["-m", "sentinel-model", 'model_reasoning_effort="sentinel-effort"']),
        ],
    )
    def test_the_model_and_effort_flags_reach_the_runner(self, provider: str, expected: list[str]):
        self.good_answer(provider)
        self.review(
            provider, "adversarial-reviewer", "-e", "sentinel-effort", "-m", "sentinel-model"
        )
        argv = json.loads(self.argv_log.read_text(encoding="utf-8"))
        for token in expected:
            assert token in argv, argv


class TestGate(Harness):
    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_prose_instead_of_findings_fails(self, provider: str):
        if provider == "grok":
            self.set_spec(stdout=grok_stream(result="I gave up."))
        else:
            self.set_spec(stdout="", last="I gave up.")
        assert self.review(provider, "adversarial-reviewer").returncode == 1

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_give_up_quoting_the_brief_example_fails(self, provider: str):
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
        if provider == "grok":
            self.set_spec(stdout=grok_stream(result=text))
        else:
            self.set_spec(stdout="", last=text)
        proc = self.review(provider, "adversarial-reviewer")
        assert proc.returncode == 1, proc.stdout

    def test_an_empty_structured_output_is_the_documented_gap(self):
        # Executable documentation of the hole this package does not close, on the exact
        # channel production uses: a healthy terminal event carrying a schema-valid EMPTY
        # findings object is accepted, because "found nothing" and "looked, then gave up" are
        # indistinguishable without judging the transcript.
        #
        # It is asserted rather than left implicit so that closing it later fails loudly
        # here, and so no other test can be read as already covering it.
        #
        # The stream carries a TOOL CALL, and that is the whole remaining gap: a run that made
        # none is refused with 6 (TestVacuousRuns), so without one this would be documenting
        # the refusal rather than the gap and the hole would look closed when it is not.
        empty = json.dumps(
            {"reviewer": "adversarial", "findings": [], "residual_risks": [], "testing_gaps": []}
        )
        self.set_spec(stdout=grok_stream(empty, tool_call=True))
        proc = self.review("grok", "adversarial-reviewer")
        assert proc.returncode == 0, proc.stderr
        assert "0 findings" in proc.stdout

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_type_invalid_answer_fails(self, provider: str):
        if provider == "grok":
            self.set_spec(stdout=grok_stream(TYPE_INVALID))
        else:
            self.set_spec(stdout="", last=TYPE_INVALID)
        assert self.review(provider, "adversarial-reviewer").returncode == 1

    @pytest.mark.parametrize(
        "bad",
        [
            {"stop_reason": "max_tokens"},
            {"subtype": "error_max_turns"},
            {"is_error": True},
        ],
    )
    def test_grok_terminal_status_is_checked_not_just_the_payload(self, bad: dict[str, Any]):
        self.set_spec(stdout=grok_stream(ANSWER, **bad))
        assert self.review("grok", "adversarial-reviewer").returncode == 1

    def test_codex_with_no_final_message_fails(self):
        # And must not re-validate whatever a previous run left at the same path.
        self.good_answer("codex")
        assert self.review("codex", "adversarial-reviewer").returncode == 0
        self.set_spec(stdout="no final message this time\n")
        proc = self.review("codex", "adversarial-reviewer")
        assert proc.returncode == 1, proc.stdout
        assert not self.artifact("codex").exists()


class TestVacuousRuns(Harness):
    """A run that made no tool calls read nothing, so it certified nothing.

    From the production incident this closes: one turn, zero tool calls, thinking truncated
    mid-sentence, 151 output tokens, four and a half seconds — and a schema-valid
    `{"findings": []}` that exited 0. A gating caller branching on the documented contract
    ("0 = schema-valid findings, an empty array is valid") read that as CLEAN.
    """

    EMPTY = json.dumps(
        {"reviewer": "adversarial", "findings": [], "residual_risks": [], "testing_gaps": []}
    )

    def review_dud(self, provider: str, answer: str) -> subprocess.CompletedProcess[str]:
        """Run a provider that answers without touching anything: the incident's shape."""
        if provider == "grok":
            self.set_spec(stdout=grok_stream(answer, num_turns=1, usage={"output_tokens": 151}))
        else:
            self.set_spec(stdout=codex_stream(), last=answer)
        return self.review(provider, "adversarial-reviewer")

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_run_with_no_tool_calls_is_refused_although_its_answer_is_valid(self, provider: str):
        proc = self.review_dud(provider, self.EMPTY)
        assert proc.returncode == 6, proc.stdout + proc.stderr
        # NOT the summary line. "0 findings -> <path>" beside a refusal is the exact
        # ambiguity being closed, and a caller that reads stdout must see nothing to relay.
        assert proc.stdout.strip() == "", proc.stdout
        assert "no tool calls" in proc.stderr, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_findings_from_a_run_with_no_tool_calls_are_refused_too(self, provider: str):
        # The half that is easy to get wrong. Refusing only EMPTY findings would treat a
        # populated array as evidence the model worked — but a model that read nothing and
        # reported a P1 has hallucinated it, and that is worse than reporting nothing.
        proc = self.review_dud(provider, ANSWER)
        assert proc.returncode == 6, proc.stdout + proc.stderr
        assert proc.stdout.strip() == "", proc.stdout

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_refusal_names_the_evidence_and_keeps_the_artifacts(self, provider: str):
        # A refusal a caller cannot audit is a rumour. The counts go on stderr and the dud
        # itself stays on disk, because it is the only record of what was refused.
        proc = self.review_dud(provider, self.EMPTY)
        assert proc.returncode == 6
        assert "0 tool calls" in proc.stderr, proc.stderr
        assert self.artifact(provider).is_file(), "the refused artifact must be kept as evidence"
        record = json.loads(self.provenance(provider).read_text(encoding="utf-8"))
        assert record["run_stats"]["tool_calls"] == 0, record["run_stats"]

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_one_tool_call_is_enough_and_the_stats_are_recorded(self, provider: str):
        # THE CONTROL, and it is load-bearing twice over: without it every assertion above
        # would also hold for a wrapper that refused every run, and `tool_calls` could be
        # hardcoded to 0 with the whole class still green.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer")
        assert proc.returncode == 0, proc.stderr
        assert "1 P1" in proc.stdout
        stats = json.loads(self.provenance(provider).read_text(encoding="utf-8"))["run_stats"]
        assert stats["tool_calls"] == 1, stats
        assert stats["output_tokens"] == 4096, stats
        # Measured by the wrapper rather than read from the stream, so it is real for both
        # providers even though only one of them publishes a duration.
        assert isinstance(stats["duration_s"], float) and stats["duration_s"] > 0, stats

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_reader_will_not_render_the_artifact_the_review_refused(self, provider: str):
        # THE LAUNDERING ROUTE. exit 6 keeps the artifact as evidence, and an artifact on
        # disk is exactly what `ce-persona-findings` renders — so a caller that ignored the
        # status got the same dud back as an ordinary listing at exit 0, one command later,
        # through this package's own reader. Both ends have to honour the refusal.
        assert self.review_dud(provider, ANSWER).returncode == 6
        proc = self.findings(str(self.artifact(provider)))
        assert proc.returncode == 6, proc.stdout + proc.stderr
        assert proc.stdout.strip() == "", proc.stdout
        assert "no tool calls" in proc.stderr, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_reader_refuses_json_too(self, provider: str):
        # The mode a programmatic caller uses, and therefore the one most likely to be acted
        # on without a person ever reading it.
        assert self.review_dud(provider, ANSWER).returncode == 6
        proc = self.findings(str(self.artifact(provider)), "--json")
        assert proc.returncode == 6, proc.stdout
        assert proc.stdout.strip() == "", proc.stdout

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_reader_refuses_return_too(self, provider: str):
        # --return feeds a merge directly, with no person between the artifact and the
        # verdict, so it is the mode in which a laundered refusal does the most damage.
        assert self.review_dud(provider, ANSWER).returncode == 6
        proc = self.findings(str(self.artifact(provider)), "--return")
        assert proc.returncode == 6, proc.stdout
        assert proc.stdout.strip() == "", proc.stdout

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_reader_still_renders_a_review_that_happened(self, provider: str):
        # The control for the three above: without it they are all satisfied by a reader that
        # refuses every artifact.
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        proc = self.findings(str(self.artifact(provider)))
        assert proc.returncode == 0, proc.stderr
        assert "#1 P1" in proc.stdout


class TestTheMergeTierReturn(Harness):
    """`--return`, through the installed command, over an artifact a real run produced.

    The plugin's merge helper reads compact RETURNS and demotes a finding whose
    `first_evidence` is missing to confidence 50, where its gate suppresses it. The lens
    artifacts carry the quote in `evidence[0]`, so this is the projection that lets a merge
    be built from what a run actually wrote.
    """

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_return_carries_a_backfilled_quote_for_every_finding(self, provider: str):
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        proc = self.findings(str(self.artifact(provider)), "--return")
        assert proc.returncode == 0, proc.stderr
        obj = json.loads(proc.stdout)
        assert obj["reviewer"] == "adversarial-reviewer"
        assert [row["first_evidence"] for row in obj["findings"]] == ["f.py:1 -- x"]
        # The evidence array is an artifact field, not a return field.
        assert "evidence" not in obj["findings"][0]
        assert "1 first_evidence backfilled from evidence[0]" in proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_quote_the_tree_carries_survives_and_one_it_does_not_is_dropped(self, provider: str):
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        tree = self.work / "reviewed"
        tree.mkdir()
        (tree / "f.py").write_text("x\n", encoding="utf-8")
        kept = self.findings(
            str(self.artifact(provider)), "--return", "--verify-quotes", "-C", str(tree)
        )
        assert kept.returncode == 0, kept.stderr
        assert json.loads(kept.stdout)["findings"][0]["first_evidence"] == "f.py:1 -- x"
        assert "0 dropped by --verify-quotes" in kept.stderr

        (tree / "f.py").write_text("something else entirely\n", encoding="utf-8")
        dropped = self.findings(
            str(self.artifact(provider)), "--return", "--verify-quotes", "-C", str(tree)
        )
        # Still 0: the caller gets the return, minus a quote the tree does not support. The
        # helper demotes the finding on its own rule, and the artifact is untouched.
        assert dropped.returncode == 0, dropped.stderr
        assert "first_evidence" not in json.loads(dropped.stdout)["findings"][0]
        assert "quoted text is not on f.py:1" in dropped.stderr
        assert "1 dropped by --verify-quotes" in dropped.stderr
        assert json.loads(self.artifact(provider).read_text(encoding="utf-8"))["findings"][0][
            "evidence"
        ] == ["f.py:1 -- x"]

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_verify_quotes_without_a_usable_tree_is_a_usage_error(self, provider: str):
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        artifact = str(self.artifact(provider))
        for args in (
            (artifact, "--verify-quotes"),
            (artifact, "--return", "--verify-quotes"),
            (artifact, "--return", "--verify-quotes", "-C", artifact),
            (artifact, "--return", "--json"),
        ):
            proc = self.findings(*args)
            assert proc.returncode == 2, (args, proc.stdout, proc.stderr)
            assert proc.stdout.strip() == "", args


class TestVocabularyDriftIsNotBlamedOnTheModel(Harness):
    """A renamed provider vocabulary is a broken wrapper, and must not read as a bad model.

    After a codex upgrade renames its item kinds, every run counts zero tool calls. Refusing
    those with exit 6 and "the model never opened the diff" would be a falsehood repeated on
    every run, about the component that was working — a permanent outage that the README
    tells the caller to retry once and then blame the model for.
    """

    def _drifted(self) -> str:
        renamed = json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "item_1", "type": "shell_call_v2", "command": "git diff"},
            }
        )
        return (
            '{"type":"thread.started","thread_id":"th_1"}\n'
            '{"type":"turn.started"}\n' + renamed + "\n"
            '{"type":"turn.completed","usage":{"output_tokens":900}}\n'
        )

    def test_an_unrecognised_item_vocabulary_is_an_environment_error(self):
        self.set_spec(stdout=self._drifted(), last=ANSWER)
        proc = self.review("codex", "adversarial-reviewer")
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert "shell_call_v2" in proc.stderr, proc.stderr
        assert "drift" in proc.stderr, proc.stderr
        # And NOT the model's fault, said in those words: the message a person reads first
        # decides which component they go and look at.
        assert "never opened the diff" not in proc.stderr
        assert not self.artifact("codex").exists()

    def test_a_codex_run_that_emits_no_events_at_all_is_an_environment_error(self):
        # `codex exec --json` opens every run with a thread and a turn, so an empty stream is
        # a runner that did not run rather than a model that did nothing.
        self.set_spec(stdout="", last=ANSWER)
        proc = self.review("codex", "adversarial-reviewer")
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert "no events at all" in proc.stderr, proc.stderr

    def test_a_genuine_dud_is_still_the_model_s_doing(self):
        # THE CONTROL that keeps exit 6 alive: a real vacuous run emits agent_message and
        # nothing else, and those are kinds the wrapper knows and skips on purpose. If they
        # counted as unrecognised, every dud would report drift and the refusal would be dead.
        self.set_spec(stdout=codex_stream(), last=ANSWER)
        proc = self.review("codex", "adversarial-reviewer")
        assert proc.returncode == 6, proc.stdout + proc.stderr
        assert "no tool calls" in proc.stderr


class TestExitStatus(Harness):
    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_runner_failure_is_reported_as_4_not_propagated(self, provider: str):
        # Propagating the runner's own status verbatim collides with the codes reserved for
        # usage (2) and over-budget (78), so a caller cannot tell them apart.
        for runner_exit in (1, 2, 3, 78):
            self.set_spec(stdout="boom\n", exit=runner_exit)
            proc = self.review(provider, "adversarial-reviewer")
            assert proc.returncode == 4, (runner_exit, proc.stderr)

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_usage_errors_exit_2(self, provider: str):
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
            assert self.review(provider, *args).returncode == 2, args

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_persona_review_package_in_the_cwd_cannot_replace_the_gate(self, provider: str):
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
            pytest.skip("needs the installed console scripts (set PERSONA_REVIEW_BIN)")
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
        for pythonpath in (None, ".", "./"):
            # A runner that categorically did not review, so a clean result can only
            # have come from the planted module.
            self.set_spec(stdout=grok_stream(ANSWER, is_error=True))
            if provider == "codex":
                self.set_spec(stdout="", last="I could not review.")
            extra = {} if pythonpath is None else {"PYTHONPATH": pythonpath}
            proc = self.review(provider, "adversarial-reviewer", env_extra=extra)
            assert "HIJACKED" not in proc.stdout, pythonpath
            assert proc.returncode != 0, f"the planted gate answered, PYTHONPATH={pythonpath!r}"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_traversal_persona_name_is_refused_even_though_it_would_resolve(self, provider: str):
        # The fixture deliberately places a findings-capable brief at assets/outside.md, so
        # `../outside` WOULD resolve if the name were not validated — and every artifact path
        # would then be built outside the run directory.
        escaped = self.work / "outside-{}.json"
        self.good_answer(provider)
        proc = self.review(provider, "../outside")
        assert proc.returncode == 2, proc.stderr
        assert "bare brief name" in proc.stderr
        assert not Path(str(escaped).format(provider)).exists(), (
            "an artifact was written outside the run directory"
        )

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_help_exits_zero_and_documents_the_exit_table(self, provider: str):
        proc = self.review(provider, "--help")
        assert proc.returncode == 0
        for code in ("0 ", "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "78 "):
            assert code in proc.stdout
        assert "CE_PERSONA_IDLE_SECS" in proc.stdout

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_an_unusable_run_dir_is_an_environment_error_not_a_usage_one(self, provider: str):
        # A caller retrying with different arguments cannot fix an unwritable directory, so
        # it belongs with the missing-binary class, not with bad arguments.
        blocker = self.work / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        self.good_answer(provider)
        proc = self.review(
            provider, "adversarial-reviewer", CE_PERSONA_RUN_DIR=str(blocker / "sub")
        )
        assert proc.returncode == 3, proc.stderr
        assert "CE_PERSONA_RUN_DIR" in proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_missing_runner_is_an_environment_error(self, provider: str):
        (self.bindir / provider).unlink()
        proc = self.review(provider, "adversarial-reviewer")
        assert proc.returncode == 3
        assert "not on PATH" in proc.stderr


class TestBudget(Harness):
    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_large_diff_trips_a_budget_the_prompt_alone_never_could(self, provider: str):
        repo, base = self._repo()
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
        assert proc.returncode == 78, proc.stderr
        assert "diff bytes" in proc.stderr
        assert not self.argv_log.exists(), "the runner must not be called"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_same_budget_with_an_empty_diff_is_not_refused(self, provider: str):
        # Control: without it the test above could pass for the wrong reason.
        repo, _ = self._repo()
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
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_tiny_budget_refuses_before_the_runner_is_called(self, provider: str):
        self.set_spec(stdout="the model must never be called\n")
        proc = self.review(provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="1")
        assert proc.returncode == 78
        assert not self.argv_log.exists()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_an_option_shaped_base_is_refused_not_parsed_as_an_option(self, provider: str):
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
        self.good_answer(provider)
        written.unlink(missing_ok=True)
        proc = self.review(
            provider, "adversarial-reviewer", "-C", str(repo), f"--base=--output={target}"
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert not written.exists(), "git took the base as an option and wrote it"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_dash_prefixed_base_is_rejected_by_argument_parsing(self, provider: str):
        # The space form is blocked one layer earlier. Asserted so the two mechanisms stay
        # distinguishable: if argparse ever accepted it, the test above is the net.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer", "-b", "--output=/tmp/x")
        assert proc.returncode == 2, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_non_numeric_budget_is_a_usage_error(self, provider: str):
        # Asserts the status and that the message NAMES the variable, not its exact prose.
        # The wording is not the contract and changed once already (the budget is parsed as a
        # whole number of tokens now, so "is not a number" became "is not a whole number");
        # the status a gating caller branches on, and an error a human can act on, are.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="abc")
        assert proc.returncode == 2, proc.stderr
        assert "CE_PERSONA_MAX_PROMPT_TOKENS" in proc.stderr
        assert "abc" in proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_the_reviewed_repo_cannot_execute_code_through_git_diff(self, provider: str):
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

        self.good_answer(provider)
        ext_marker.unlink(missing_ok=True)
        tc_marker.unlink(missing_ok=True)
        proc = self.review(provider, "adversarial-reviewer", "-C", str(repo), "-b", base)
        # Assert the run SUCCEEDED first. Without this the markers are also absent
        # when the guarded git call simply fails before dispatch, so the test would
        # pass while proving nothing about the mitigation.
        assert proc.returncode == 0, proc.stderr
        assert not ext_marker.exists(), "the repo's diff.external driver executed"
        assert not tc_marker.exists(), "the repo's textconv driver executed"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_repo_cannot_wedge_the_budget_preflight_with_stderr(self, provider: str):
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
        self.good_answer(provider)
        started = time.monotonic()
        proc = self.review(
            provider, "adversarial-reviewer", "-C", str(repo), "-b", base, timeout=90
        )
        elapsed = time.monotonic() - started
        assert proc.returncode == 0, proc.stderr
        assert elapsed < 60, "the budget preflight wedged on git's stderr"

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
        assert ext_marker.exists(), "fixture did not arm diff.external"

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
        assert tc_marker.exists(), "fixture did not arm textconv"


class TestArtifactLifecycle(Harness):
    def _seed(self, provider: str) -> None:
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        assert self.artifact(provider).is_file()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_failed_run_leaves_no_stale_artifact(self, provider: str):
        self._seed(provider)
        self.set_spec(stdout="boom\n", exit=9)
        assert self.review(provider, "adversarial-reviewer").returncode == 4
        assert not self.artifact(provider).exists()
        assert not self.provenance(provider).exists()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_every_early_failure_leaves_no_stale_artifact(self, provider: str):
        # Not just the budget refusal. The run dir is cleared as soon as the persona is
        # known, so a failure at ANY later preflight — a missing binary, an unreadable
        # context file — cannot leave the previous run's findings at the path a caller reads.
        self._seed(provider)
        (self.bindir / provider).unlink()
        assert self.review(provider, "adversarial-reviewer").returncode == 3
        self.assert_run_dir_clean(provider)
        (self.bindir / provider).write_text(STUB, encoding="utf-8")
        (self.bindir / provider).chmod(0o755)

        self._seed(provider)
        proc = self.review(
            provider, "adversarial-reviewer", "-c", str(self.work / "no-such-context.md")
        )
        assert proc.returncode == 2, proc.stderr
        self.assert_run_dir_clean(provider)

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_non_finite_timeout_is_refused_rather_than_disabling_the_watchdog(
        self, provider: str
    ):
        # float() accepts nan and inf, and every deadline is `elapsed >= value`, which is
        # False forever against either — a typo would silently switch the watchdog off.
        for name in ("CE_PERSONA_IDLE_SECS", "CE_PERSONA_HARD_SECS"):
            for value in ("nan", "inf", "-inf"):
                self.good_answer(provider)
                proc = self.review(provider, "adversarial-reviewer", env_extra={name: value})
                assert proc.returncode == 2, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_non_finite_budget_is_a_usage_error_not_a_crash(self, provider: str):
        for value in ("nan", "inf"):
            self.good_answer(provider)
            proc = self.review(provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS=value)
            assert proc.returncode == 2, proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize("name", ["CE_PERSONA_IDLE_SECS", "CE_PERSONA_HARD_SECS"])
    def test_a_zero_timeout_is_refused_because_it_switched_the_watchdog_off(
        self, provider: str, name: str
    ):
        # `0` used to be ACCEPTED — the parser refused only negatives, and each deadline was
        # guarded by `if secs > 0`. A typo therefore left a full-effort model run with
        # nothing watching it and nothing that would ever read its output. Now the seconds
        # are strictly positive by construction and `_watch` has no guard left to fail.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer", env_extra={name: "0"})
        assert proc.returncode == 2, proc.stderr
        assert name in proc.stderr
        assert "greater than zero" in proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_misspelled_setting_is_an_error_rather_than_a_silent_default(self, provider: str):
        # The one idea taken from pydantic-settings' extra="forbid". Setting
        # CE_PERSONA_IDEL_SECS used to leave the real idle timeout at 600s and say nothing:
        # a silent misconfiguration, in a package whose whole argument is that silence is the
        # failure mode.
        self.good_answer(provider)
        proc = self.review(provider, "adversarial-reviewer", CE_PERSONA_IDEL_SECS="30")
        assert proc.returncode == 2, proc.stderr
        assert "CE_PERSONA_IDEL_SECS" in proc.stderr
        assert "CE_PERSONA_IDLE_SECS" in proc.stderr, "the message should name the real setting"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_an_unknown_home_directory_is_a_usage_error_not_a_traceback(self, provider: str):
        # `Path('~nobody/x').expanduser()` raises RuntimeError — not OSError, and not a type
        # any caller would think to catch — so it escaped to the top of the process as a
        # traceback with an unmapped exit status. Both paths that expand `~` are covered.
        self.good_answer(provider)
        by_flag = self.review(provider, "adversarial-reviewer", "-C", "~nosuchuser0987/repo")
        assert by_flag.returncode == 2, by_flag.stderr
        assert "Traceback" not in by_flag.stderr

        by_env = self.review(
            provider, "adversarial-reviewer", CE_PERSONA_RUN_DIR="~nosuchuser0987/run"
        )
        assert by_env.returncode == 2, by_env.stderr
        assert "Traceback" not in by_env.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_second_concurrent_run_is_refused_rather_than_interleaved(self, provider: str):
        # Artifact paths are deterministic and CE_PERSONA_RUN_DIR is documented as reusable,
        # so two runs of the same persona through the same provider address the same files.
        # What happened then was not a lost race but a silently wrong answer: the second
        # run's CLEAR unlinks the first's event stream while its provider is still writing,
        # both append to one -events.jsonl, and the gate validates an interleaving of two
        # transcripts. Refusing is the only fail-closed option.
        self.set_spec(stdout="starting\n", silent_for=30)
        with subprocess.Popen(
            [str(self.commands[provider]), "adversarial-reviewer"],
            env=self.env(),
            cwd=self.work,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as first:
            try:
                events = self.run_dir / f"adversarial-reviewer-{provider}-events.jsonl"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not events.exists():
                    time.sleep(0.2)
                assert events.exists(), "the first run never reached the runner"

                second = self.review(provider, "adversarial-reviewer", timeout=60)
                assert second.returncode == 2, second.stderr
                assert "already running" in second.stderr
                # The refusal must not have cleared the running run's stream on its way out.
                assert events.exists(), "the refused run deleted the live run's event stream"
            finally:
                first.kill()
                first.wait(timeout=10)

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_sequential_rerun_is_not_blocked_by_a_released_lock(self, provider: str):
        # The control. A lock that is never released would satisfy the test above while
        # breaking every ordinary repeated invocation — and the run directory is documented
        # as reusable, so repeated invocation is the normal case.
        self.good_answer(provider)
        assert self.review(provider, "adversarial-reviewer").returncode == 0
        again = self.review(provider, "adversarial-reviewer")
        assert again.returncode == 0, again.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_refusal_before_dispatch_leaves_no_stale_artifact(self, provider: str):
        # A run refused at the budget or for a bad base never reaches the runner, and used to
        # leave the previous run's findings and provenance in place.
        self._seed(provider)
        proc = self.review(provider, "adversarial-reviewer", CE_PERSONA_MAX_PROMPT_TOKENS="1")
        assert proc.returncode == 78
        self.assert_run_dir_clean(provider)


class TestWatchdogs(Harness):
    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_silent_runner_is_killed_and_reported_as_a_timeout(self, provider: str):
        self.set_spec(stdout="starting\n", silent_for=90)
        started = time.monotonic()
        proc = self.review(provider, "adversarial-reviewer", timeout=60, CE_PERSONA_IDLE_SECS="3")
        elapsed = time.monotonic() - started
        assert proc.returncode == 5, proc.stderr
        assert "no output for" in proc.stderr
        assert elapsed < 40, "the idle watchdog did not fire promptly"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_chatty_runner_still_hits_the_hard_deadline(self, provider: str):
        # File growth keeps the idle watchdog happy forever, which is exactly why a wall-clock
        # cap has to exist too.
        self.set_spec(heartbeat=0.2)
        proc = self.review(
            provider,
            "adversarial-reviewer",
            timeout=60,
            CE_PERSONA_IDLE_SECS="30",
            CE_PERSONA_HARD_SECS="4",
        )
        assert proc.returncode == 5, proc.stderr
        assert "hard timeout" in proc.stderr

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_a_timeout_kills_helpers_that_ignore_sigterm(self, provider: str):
        # The direct child exiting is not proof the process group did. A parent with default
        # SIGTERM handling dies at once while a helper that ignores it keeps running, so a
        # watchdog that waits on the child alone reports a kill it did not perform — and the
        # provider keeps working, and keeps the artifact descriptors open.
        self.child_pid_file.unlink(missing_ok=True)
        self.set_spec(stdout="starting\n", stubborn_child=True, silent_for=120)
        proc = self.review(provider, "adversarial-reviewer", timeout=90, CE_PERSONA_IDLE_SECS="3")
        assert proc.returncode == 5, proc.stderr
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
        assert not alive, f"helper {pid} survived the timeout kill"

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_killing_the_wrapper_does_not_orphan_the_provider(self, provider: str):
        # `start_new_session=True` is what lets the watchdog signal the provider's whole
        # group — and it also means the provider does NOT die with this command. Ctrl-C, a
        # supervising agent's own timeout, or a CI cancel would otherwise leave a
        # full-effort model run going with nothing watching it and nothing that will read
        # its output. The shell version had an EXIT trap; this asserts the replacement.
        self.child_pid_file.unlink(missing_ok=True)
        self.set_spec(stdout="starting\n", stubborn_child=True, silent_for=300)
        # DEVNULL, not PIPE, and `with` so the handles close: nothing here ever reads the
        # wrapper's output, and an undrained pipe is the hazard runner.py exists to avoid —
        # a provider helper that outlives its parent while holding the write end blocks on a
        # full buffer instead of exiting. Leaving them open also leaked two file objects,
        # which `filterwarnings = ["error"]` turned into a failure.
        with subprocess.Popen(
            [str(self.commands[provider]), "adversarial-reviewer"],
            env=self.env(),
            cwd=self.work,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as wrapper:
            # Nested inside the `with`, because Popen.__exit__ waits WITHOUT a timeout: an
            # assertion firing below would otherwise hang the suite on a stub sleeping 300s.
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not self.child_pid_file.exists():
                    time.sleep(0.2)
                assert self.child_pid_file.exists(), "the stub never started"
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
                assert not alive, f"provider helper {pid} outlived the killed wrapper"
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=10)

    def test_partial_output_is_kept_after_a_timeout(self):
        self.set_spec(stdout="partial evidence\n", silent_for=90)
        self.review("grok", "adversarial-reviewer", timeout=60, CE_PERSONA_IDLE_SECS="3")
        events = self.run_dir / "adversarial-reviewer-grok-events.jsonl"
        assert "partial evidence" in events.read_text(encoding="utf-8")


class TestFindingsCommand(Harness):
    def test_the_installed_command_reads_a_real_artifact(self):
        self.good_answer("grok")
        assert self.review("grok", "adversarial-reviewer").returncode == 0
        proc = self.findings(str(self.artifact("grok")))
        assert proc.returncode == 0, proc.stderr
        assert "BEGIN UNTRUSTED MODEL OUTPUT" in proc.stdout
        assert "#1 P1" in proc.stdout

    def test_json_round_trips_through_the_installed_command(self):
        self.good_answer("grok")
        self.review("grok", "adversarial-reviewer")
        proc = self.findings(str(self.artifact("grok")), "--json")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["findings"][0]["severity"] == "P1"
