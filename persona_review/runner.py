"""Running a provider CLI: artifacts, the size budget, and the watchdogs.

Two things here are less obvious than they look.

**stdout goes straight to a file, never through a pipe this process reads.** Provider CLIs
spawn helper processes that outlive the main one while still holding the inherited stdout,
so a parent reading the pipe to EOF blocks long after the review has finished — a completed
run that hangs until something kills it. Handing the child a file descriptor and watching
the file grow sidesteps that entirely, and gives liveness for both runners with one
mechanism, since both stream progressively.

**Liveness is file growth, not event parsing.** Nothing here needs to understand a
provider's event vocabulary to know it is still working, which is what keeps the watchdog
correct across CLI versions that rename their events.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

# How often the watchdog looks at the growing output file. Small enough to notice a stall
# promptly, large enough that a long review costs a negligible number of stat() calls.
POLL_SECS = 2.0

# Grace between asking the process group to stop and insisting. Provider CLIs are Node and
# Rust binaries that install their own SIGTERM handlers and can take a moment to unwind.
KILL_GRACE_SECS = 10.0

# ~4 chars/token, the same rough conversion the plugin's own large-diff preflight uses. It
# only has to be right to an order of magnitude.
CHARS_PER_TOKEN = 4


class RunError(Exception):
    """The request is wrong in a way the caller can act on. Maps to a usage exit."""


class EnvError(RunError):
    """The machine is wrong, not the invocation. Maps to the environment exit.

    A distinct type rather than a message the caller greps: deciding an exit status by
    matching on prose means a reworded error silently changes the status a gating caller
    branches on.
    """


class MissingTool(EnvError):
    """A required binary is absent."""


@dataclass(frozen=True)
class ExecResult:
    status: int
    timed_out: bool
    reason: str


def _git_env() -> dict[str, str]:
    """Environment for reading a repository we do not necessarily trust.

    `git diff` will run commands named by configuration: `diff.external`, and per-driver
    `textconv` selected through `.gitattributes`. This process runs OUTSIDE the read-only
    sandbox the review itself gets, so those would execute as the user with full privileges.
    System and global configuration are dropped and the drivers are disabled below.

    Residual, and worth knowing: a repository's own `.git/config` is still honoured, because
    git offers no way to ignore it. Cloning never transfers `.git/config`, so this bites only
    for a working directory handed over wholesale rather than cloned.
    """
    env = dict(os.environ)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _git_diff_argv(repo: Path, base: str) -> list[str]:
    # --no-ext-diff, not `-c diff.external=`: setting the config key to an empty string makes
    # git try to execute it and die with "external diff died", turning every review of a repo
    # that configures a driver into a bogus "base does not resolve". The flag disables both
    # `diff.external` and any driver selected through .gitattributes.
    return [
        "git",
        "-C",
        str(repo),
        "-c",
        "core.attributesFile=/dev/null",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        f"{base}..HEAD",
    ]


def diff_bytes(repo: Path, base: str) -> int:
    """Size of `git diff <base>..HEAD`, and proof the range actually resolves.

    One git call does both jobs. Validating the range by any other expression gets it wrong:
    peeling with `^{commit}` refuses baselines that `git diff` accepts, including a tree
    object (the standard empty-tree baseline for an initial commit) and `:/subject`
    selectors, because appending the suffix changes their grammar.
    """
    try:
        proc = subprocess.run(
            _git_diff_argv(repo, base),
            capture_output=True,
            env=_git_env(),
            check=False,
        )
    except FileNotFoundError as exc:
        raise MissingTool(f"git is required to weigh `-b {base}` but is not on PATH") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        hint = detail[-1] if detail else "no detail from git"
        raise RunError(
            f"base ref '{base}' does not resolve in {repo} ({hint}).\n"
            "  An unresolvable base is not a smaller review, it is an unscoped one: the model\n"
            "  would be told to diff a range that does not exist and would review something else."
        )
    return len(proc.stdout)


@dataclass(frozen=True)
class Budget:
    prompt_bytes: int
    diff_bytes: int
    limit_tokens: int

    @property
    def tokens(self) -> int:
        return (self.prompt_bytes + self.diff_bytes) // CHARS_PER_TOKEN

    @property
    def over(self) -> bool:
        return self.tokens > self.limit_tokens


def weigh(*, prompt: str, repo: Path, base: str, limit_tokens: int) -> Budget:
    """Weigh the review before dispatching it.

    The diff is counted, not just the prompt: the prompt does not CONTAIN the change, it
    tells the model to fetch it, so weighing the prompt alone measures a near-constant ~1.2KB
    and the budget can never fire.

    With no base ref there is genuinely nothing to weigh — the model chooses its own scope —
    so the budget then bounds the prompt only. That limit is documented rather than papered
    over with a guessed base.
    """
    return Budget(
        prompt_bytes=len(prompt.encode("utf-8")),
        diff_bytes=diff_bytes(repo, base) if base else 0,
        limit_tokens=limit_tokens,
    )


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop the whole process group, then insist.

    The group, not the process: provider CLIs spawn helpers, and signalling only the parent
    leaves them running and holding the output file open.
    """
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECS), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def execute(
    argv: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    stdin_path: Path | None,
    idle_secs: float,
    hard_secs: float,
) -> ExecResult:
    """Run a provider CLI under an idle watchdog and a hard deadline."""
    with contextlib.ExitStack() as stack:
        stdin_arg: int | IO[bytes] = subprocess.DEVNULL
        if stdin_path is not None:
            stdin_arg = stack.enter_context(stdin_path.open("rb"))
        out = stack.enter_context(stdout_path.open("wb"))
        err = stack.enter_context(stderr_path.open("wb"))
        try:
            proc = subprocess.Popen(
                argv,
                stdin=stdin_arg,
                stdout=out,
                stderr=err,
                # Its own process group, so the watchdog can stop the helpers too.
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RunError(f"{argv[0]} is not on PATH: {exc}") from exc

    started = time.monotonic()
    last_change = started
    last_size = -1

    while True:
        status = proc.poll()
        if status is not None:
            return ExecResult(status=status, timed_out=False, reason="")

        now = time.monotonic()
        try:
            size = stdout_path.stat().st_size
        except OSError:
            size = last_size
        if size != last_size:
            last_size, last_change = size, now

        if hard_secs > 0 and now - started >= hard_secs:
            _kill_group(proc)
            return ExecResult(
                status=proc.returncode or -1,
                timed_out=True,
                reason=(
                    f"hard timeout after {int(now - started)}s "
                    f"(CE_PERSONA_HARD_SECS={int(hard_secs)})"
                ),
            )
        if idle_secs > 0 and now - last_change >= idle_secs:
            _kill_group(proc)
            return ExecResult(
                status=proc.returncode or -1,
                timed_out=True,
                reason=(
                    f"no output for {int(now - last_change)}s "
                    f"(CE_PERSONA_IDLE_SECS={int(idle_secs)}); the run was killed, "
                    "partial output kept"
                ),
            )
        time.sleep(POLL_SECS)
