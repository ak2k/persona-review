"""The error vocabulary, and the exit status that belongs to each kind.

WHY THE CODE LIVES ON THE EXCEPTION

`main` used to translate errors into statuses with a chain of six near-identical
`except X: print(...); return CODE` blocks, one per raising module. That shape has two
failure modes and this package has had both:

  * an error class raised somewhere `main` does not catch it escapes as a traceback and
    whatever status the interpreter picks — `~nosuchuser` did exactly that, via a
    `RuntimeError` from `Path.expanduser()`;
  * a class caught in one arm and not another gets two different statuses depending on which
    call raised it, which is how the runner's own status came to collide with the codes
    reserved for usage (2) and over-budget (78).

Carrying `exit_code` as a class attribute makes the mapping total by construction: a new
error class cannot be introduced without choosing its status, `main` maps every AppError the
same way, and the exit table in `--help` is generated from the classes rather than restated
beside them. `tests/test_unit.py` asserts the two lists agree.

The codes are a PUBLISHED CONTRACT. A caller gates on them, so they are literals here and
literals in the tests — an assertion written against these constants moves with them, and
mutation testing caught precisely that: redefining EXIT_USAGE to 1 left a suite green.
"""

from __future__ import annotations


class AppError(Exception):
    """Anything this package refuses to do, with a status and a legible reason.

    Never raised directly — `exit_code` is deliberately absent from the base so a subclass
    that forgets to set one fails at the point of use rather than silently reporting 1.
    """

    exit_code: int


class GateError(AppError):
    """The answer was not schema-valid findings. The review did not produce a verdict."""

    exit_code = 1


class UsageError(AppError):
    """The invocation was wrong: arguments, persona, base ref, or a CE_PERSONA_* value.

    Retrying with different arguments could succeed, which is what separates this from
    EnvError.
    """

    exit_code = 2


class EnvError(AppError):
    """The machine is not set up: a missing binary, unusable directory, or absent assets.

    No argument the caller could pass would fix it.
    """

    exit_code = 3


class MissingTool(EnvError):
    """The provider CLI, or git, is not on PATH."""


class RunnerError(AppError):
    """The provider CLI exited non-zero. Its own status is recorded in provenance.

    Not re-raised as itself: propagating it verbatim collides with the codes reserved above,
    so a caller could not tell "codex exited 2" from "you called this wrongly".
    """

    exit_code = 4


class RunTimeout(AppError):
    """An idle or hard watchdog fired. Partial output is kept.

    Named RunTimeout rather than TimeoutError so it cannot be confused with — or accidentally
    caught alongside — the builtin, which subprocess raises for its own reasons.
    """

    exit_code = 5


class BudgetError(AppError):
    """Over CE_PERSONA_MAX_PROMPT_TOKENS. Refused, never summarized.

    78 is EX_CONFIG from sysexits.h: the run was not attempted and no artifact exists.
    """

    exit_code = 78


# The published exit contract, in status order. `cli._epilog` RENDERS this into `--help`
# rather than restating it: two lists of the same numbers are one edit from disagreeing, and
# a caller branches on them. `{runner}` is filled in with the provider's binary name.
#
# Descriptions may span lines; continuation lines are indented by the renderer.
EXIT_TABLE: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("schema-valid findings (an empty findings array is valid)",)),
    (GateError.exit_code, ("the answer was not schema-valid findings",)),
    (
        UsageError.exit_code,
        (
            "usage error: bad arguments, unknown or markdown-only persona,",
            "bad -C directory, unresolvable -b base ref, malformed CE_PERSONA_* value",
        ),
    ),
    (
        EnvError.exit_code,
        (
            "environment error: {runner}, git or the plugin assets are missing,",
            "or CE_PERSONA_RUN_DIR cannot be created",
        ),
    ),
    (RunnerError.exit_code, ("{runner} itself exited non-zero",)),
    (RunTimeout.exit_code, ("idle or hard timeout; the run was killed and partial output kept",)),
    (BudgetError.exit_code, ("over CE_PERSONA_MAX_PROMPT_TOKENS; refused, never summarized",)),
)


def render_exit_table(runner: str, width: int = 4) -> str:
    """The exit table as `--help` shows it. One source, two readers: help text and the tests.

    `width` pads the status column so 78 and 0 line up; it is the only formatting decision,
    and getting it wrong is cosmetic rather than a wrong contract.
    """
    lines: list[str] = []
    for code, description in EXIT_TABLE:
        head, *rest = description
        lines.append(f"  {str(code).ljust(width)}{head.format(runner=runner)}")
        lines.extend(f"      {line.format(runner=runner)}" for line in rest)
    return "\n".join(lines)
