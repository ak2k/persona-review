"""Everything this package reads from the environment, parsed once, in one place.

`os.environ` is read HERE AND NOWHERE ELSE, and `tests/test_unit.py` asserts that by
grepping the package. It is the same structural move that keeps `subprocess.PIPE` greppably
absent from `runner.py`: a rule you can check with one command beats a rule you have to
remember at every call site. Scattered `os.environ.get` calls meant a malformed timeout was
discovered three quarters of the way through `main` — after the run directory had been
cleared and the prompt built — and that a variable read in two places could be validated in
one of them.

TWO INVARIANTS THE TYPES CARRY, RATHER THAN THE CALLERS

  * **A watchdog cannot be switched off.** `CE_PERSONA_IDLE_SECS=0` used to disable it
    outright: the parser refused only negatives, and `_watch` guarded each deadline with
    `if secs > 0`. A typo therefore left a full-effort model run with nothing watching it and
    nothing that would ever read its output. Seconds are parsed as strictly positive and
    finite, so `_watch` can drop the guard instead of trusting it.
  * **Nothing here is nan or inf.** `float()` returns both, and every deadline is
    `elapsed >= value`, which is False forever against either.

AND ONE STOLEN FROM pydantic-settings

`extra="forbid"`: an unrecognised `CE_PERSONA_*` name is an error, not a shrug. Setting
`CE_PERSONA_IDEL_SECS=30` used to leave the real idle timeout at its 600s default and say
nothing — a silent misconfiguration, in a package whose whole argument is that silence is
the failure mode. This package has no pydantic dependency (see AGENTS.md), so it is eleven
lines below; the idea is worth more than the library here.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import EnvError, UsageError

DEFAULT_MAX_TOKENS = 80_000
DEFAULT_IDLE_SECS = 600.0
DEFAULT_HARD_SECS = 2400.0

# The complete CE_PERSONA_* namespace. Anything else under the prefix is a typo.
KNOWN_VARS = frozenset(
    {
        "CE_PERSONA_RUN_DIR",
        "CE_PERSONA_MAX_PROMPT_TOKENS",
        "CE_PERSONA_IDLE_SECS",
        "CE_PERSONA_HARD_SECS",
    }
)
PREFIX = "CE_PERSONA_"


def _positive_secs(name: str, raw: str, default: float) -> float:
    """Strictly positive and finite, or a usage error naming the variable."""
    text = raw.strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as exc:
        raise UsageError(f"{name}={text!r} is not a number") from exc
    if not math.isfinite(value):
        # nan and inf both make `elapsed >= value` False forever, so this reads as a very
        # long timeout and behaves as no timeout at all.
        raise UsageError(f"{name}={text!r} must be a finite number of seconds")
    if value <= 0:
        raise UsageError(
            f"{name}={text!r} must be greater than zero. There is no way to switch a "
            "watchdog off: an unwatched run is a full-effort model run that nothing will "
            "stop and nothing will read. Set a large value if you want a long one."
        )
    return value


def _positive_int(name: str, raw: str, default: int) -> int:
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise UsageError(f"{name}={text!r} is not a whole number") from exc
    if value <= 0:
        raise UsageError(f"{name}={text!r} must be greater than zero")
    return value


def _expand(name: str, raw: str) -> Path:
    """Expand `~` without letting its failure escape as a traceback.

    `Path('~nosuchuser/x').expanduser()` raises RuntimeError — not OSError, and not any type
    a caller would think to catch — so an unknown user in a path reached the top of the
    process as a traceback with an unmapped exit status.
    """
    try:
        return Path(raw).expanduser()
    except RuntimeError as exc:
        raise UsageError(f"{name}={raw!r} names a home directory that does not exist") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """The parsed environment. Frozen: nothing re-reads or re-decides these mid-run."""

    assets_override: str | None
    run_dir: Path | None
    max_prompt_tokens: int
    idle_secs: float
    hard_secs: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Parse and validate. Raises UsageError; never returns a half-checked object.

        Takes the mapping as an argument so the suite can drive it directly. The default is
        the only `os.environ` read in the package outside the library-provenance check in
        `cli`, which must run before argument parsing and therefore before this exists.
        """
        source = os.environ if env is None else env

        unknown = sorted(k for k in source if k.startswith(PREFIX) and k not in KNOWN_VARS)
        if unknown:
            near = ", ".join(sorted(KNOWN_VARS))
            raise UsageError(
                f"unrecognised setting(s): {', '.join(unknown)}. "
                f"This package reads only {near}. A misspelled variable is silently ignored "
                "otherwise, leaving the setting it was meant to change at its default."
            )

        raw_run_dir = source.get("CE_PERSONA_RUN_DIR", "").strip()
        return cls(
            assets_override=source.get("CE_REVIEW_ASSETS") or None,
            run_dir=_expand("CE_PERSONA_RUN_DIR", raw_run_dir) if raw_run_dir else None,
            max_prompt_tokens=_positive_int(
                "CE_PERSONA_MAX_PROMPT_TOKENS",
                source.get("CE_PERSONA_MAX_PROMPT_TOKENS", ""),
                DEFAULT_MAX_TOKENS,
            ),
            idle_secs=_positive_secs(
                "CE_PERSONA_IDLE_SECS", source.get("CE_PERSONA_IDLE_SECS", ""), DEFAULT_IDLE_SECS
            ),
            hard_secs=_positive_secs(
                "CE_PERSONA_HARD_SECS", source.get("CE_PERSONA_HARD_SECS", ""), DEFAULT_HARD_SECS
            ),
        )

    def resolved_run_dir(self) -> Path:
        """The run directory, created. A fresh temp dir when unset.

        Resolution happens here rather than in `from_env` so that parsing the environment
        touches no filesystem: `main` clears the run directory before any other fallible
        step, and a settings object that could fail on disk would have to be built first.
        """
        if self.run_dir is None:
            return Path(tempfile.mkdtemp(prefix="ce-persona-run."))
        run_dir = self.run_dir.resolve()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # The machine, not the invocation: an unwritable or occupied path is the same
            # class of problem as a missing binary, and a caller retrying with different
            # arguments would not fix it.
            raise EnvError(f"cannot use CE_PERSONA_RUN_DIR={run_dir}: {exc}") from exc
        return run_dir
