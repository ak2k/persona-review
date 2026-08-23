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
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import errors

# How often the watchdog looks at the growing output file. Small enough to notice a stall
# promptly, large enough that a long review costs a negligible number of stat() calls.
POLL_SECS = 2.0

# Grace between asking the process group to stop and insisting. Provider CLIs are Node and
# Rust binaries that install their own SIGTERM handlers and can take a moment to unwind.
KILL_GRACE_SECS = 10.0

# ~4 chars/token, the same rough conversion the plugin's own large-diff preflight uses. It
# only has to be right to an order of magnitude.
CHARS_PER_TOKEN = 4

# The budget preflight runs before `execute`, so the idle and hard watchdogs do not cover
# it. Measuring a diff is local work on an already-checked-out tree; a minute is generous
# for anything legitimate and bounds a repository that tries to wedge the measurement.
DIFF_TIMEOUT_SECS = 60.0


# Re-exported from errors.py, where each one's exit status lives. Distinct TYPES rather than
# messages the caller greps: deciding an exit status by matching on prose means a reworded
# error silently changes the status a gating caller branches on.
RunError = errors.UsageError
EnvError = errors.EnvError
MissingTool = errors.MissingTool


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
        # --end-of-options, or an option-shaped base is parsed as an OPTION rather than a
        # revision. `-b '--output=/some/path'` made git write the diff to that path, exit 0
        # with empty stdout, and this function then reported a 0-byte diff and a resolved
        # range — destroying the "proof the range resolves" guarantee AND writing a file of
        # the caller's choosing.
        "--end-of-options",
        f"{base}..HEAD",
    ]


def diff_bytes(repo: Path, base: str) -> int:
    """Size of `git diff <base>..HEAD`, and proof the range actually resolves.

    One git call does both jobs. Validating the range by any other expression gets it wrong:
    peeling with `^{commit}` refuses baselines that `git diff` accepts, including a tree
    object (the standard empty-tree baseline for an initial commit) and `:/subject`
    selectors, because appending the suffix changes their grammar.

    The diff goes to a temp file and the count is an `fstat`, so peak memory is independent
    of diff size — buffering it in the parent took RSS to roughly 2.4x a 209 MB diff, and
    this runs BEFORE the budget check, which made the guard against an oversized review the
    thing an oversized review broke.

    It is NOT free, and the cost moved rather than vanished: the whole diff lands in TMPDIR
    for the duration of the call (26 MB of diff, 26 MB of TMPDIR, measured). That is the
    deliberate trade — bounded memory, temporary disk — and it is why a write failure below
    is reported as an environment problem rather than a bad base ref.
    """
    # NEITHER stream is a pipe. That is the invariant this module runs on, and the reason
    # is that both ways of breaking it have already cost a defect: reading the runner's
    # stdout through a pipe hangs when a provider's helper holds the descriptor open, and
    # draining stdout to EOF while stderr was a pipe deadlocked whenever git filled the
    # ~64 KiB stderr buffer — repo-controlled, via a committed .gitattributes. Files have
    # neither failure mode, and the byte count is a stat rather than a read.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(
                _git_diff_argv(repo, base), stdout=out, stderr=err, env=_git_env()
            )
        except FileNotFoundError as exc:
            raise MissingTool(f"git is required to weigh `-b {base}` but is not on PATH") from exc

        # `with proc:` reaps on EVERY exit path, including the ones not enumerated here.
        # Without it, an exception raised DURING the wait — KeyboardInterrupt is the
        # realistic one, since the budget preflight runs before any signal handling is
        # installed — leaves git running, unreaped, still filling a temp file.
        with proc:
            try:
                proc.wait(timeout=DIFF_TIMEOUT_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                # Bounded. SIGKILL is uncatchable, but a child wedged in uninterruptible
                # sleep on a hung filesystem would otherwise block forever here — in the
                # one preflight that exists precisely to be time-bounded.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=KILL_GRACE_SECS)
                raise RunError(
                    f"`git diff {base}..HEAD` in {repo} did not finish within "
                    f"{int(DIFF_TIMEOUT_SECS)}s"
                ) from None

        total = os.fstat(out.fileno()).st_size
        err.seek(0)
        detail = err.read().decode("utf-8", "replace")

    if proc.returncode != 0:
        lines = detail.strip().splitlines()
        hint = lines[-1] if lines else "no detail from git"
        # git can now fail for OUTPUT-side reasons, because the diff goes to TMPDIR. Telling
        # a caller its base ref is bad when the disk is full is a false diagnosis, and it
        # crosses the type boundary this module keeps: RunError is the caller's mistake,
        # EnvError is the machine's.
        if "No space left on device" in detail or "write error" in detail:
            raise EnvError(
                f"could not write `git diff {base}..HEAD` output to {tempfile.gettempdir()}: {hint}"
            )
        raise RunError(
            f"base ref '{base}' does not resolve in {repo} ({hint}).\n"
            "  An unresolvable base is not a smaller review, it is an unscoped one: the model\n"
            "  would be told to diff a range that does not exist and would review something else."
        )
    return total


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


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop the whole process group, then insist.

    The group, not the process: provider CLIs spawn helpers, and signalling only the parent
    leaves them running and holding the output file open.

    The direct child exiting is NOT proof the group did. A parent with default SIGTERM
    handling dies immediately while a helper that ignores SIGTERM keeps running, so waiting
    on the child and returning leaves provider work alive after the caller has been told the
    run was killed. Escalate on the GROUP's liveness instead, and poll `proc.poll()` while
    doing it, because an unreaped zombie is still a group member and would make the probe
    report members forever.

    `proc.pid` is the group id only because `execute` starts the child with
    `start_new_session=True`. Without that these signals would reach this process's own
    group — the calling agent included.
    """
    pgid = proc.pid
    for sig, grace in ((signal.SIGTERM, KILL_GRACE_SECS), (signal.SIGKILL, 5.0)):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            proc.poll()
            if not _group_alive(pgid):
                return
            time.sleep(0.2)
    proc.poll()


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

    with _stop_child_when_this_process_dies(proc):
        return _watch(proc, stdout_path, idle_secs, hard_secs)


@contextlib.contextmanager
def _stop_child_when_this_process_dies(proc: subprocess.Popen[bytes]) -> Generator[None]:
    """Do not orphan a full-effort model run.

    `start_new_session=True` is what makes the watchdog able to signal the provider's whole
    group — and it also means the provider does NOT die with this process. Ctrl-C at a
    terminal, a supervising agent's own timeout, a CI job cancel, or a plain `kill` on this
    wrapper would otherwise leave the run going with nothing watching it and nothing that
    will ever read its output. The shell version this replaced had an EXIT trap; this is
    that trap.

    SIGKILL on this process still orphans the child, because nothing can run then.
    """

    def handler(signum: int, _frame: object) -> None:
        if proc.poll() is None:
            _kill_group(proc)
        raise SystemExit(128 + signum)

    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError):
            previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, old)  # pyright: ignore[reportArgumentType]
        # Covers every non-signal exit too: an exception in the loop, or a return path that
        # ever stops killing the group itself.
        if proc.poll() is None:
            _kill_group(proc)


def _watch(
    proc: subprocess.Popen[bytes], stdout_path: Path, idle_secs: float, hard_secs: float
) -> ExecResult:
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

        # No `> 0` guard on either deadline. config.Settings parses both as strictly
        # positive, so "the watchdog is armed" is a property of the type rather than a branch
        # that can be false — `CE_PERSONA_IDLE_SECS=0` used to switch it off silently, which
        # left a full-effort model run with nothing watching it.
        if now - started >= hard_secs:
            _kill_group(proc)
            return ExecResult(
                status=proc.returncode or -1,
                timed_out=True,
                reason=(
                    f"hard timeout after {int(now - started)}s "
                    f"(CE_PERSONA_HARD_SECS={int(hard_secs)})"
                ),
            )
        if now - last_change >= idle_secs:
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
