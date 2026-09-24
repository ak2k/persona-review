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

from collections.abc import Mapping


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

    No argument the caller could pass would fix it. Also the status for a provider CLI whose
    event vocabulary has moved out from under this build — the wrapper can no longer count
    what a run did, which is a defect in the wrapper and must not be reported as one in the
    model (see VacuousRun).
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


class VacuousRun(AppError):
    """The provider answered without making a single local tool call. It inspected nothing.

    Not a gate failure: the answer can be perfectly schema-valid, and usually is, because
    schema-constrained decoding produces a well-formed object whether or not the model read
    anything. A reviewer that never opened a file has no verdict to report — an empty
    findings array and a page of them are equally unfounded — so the run is refused instead
    of summarized.

    The one status a caller should consider retrying: the machine is fine, the invocation
    is fine, and the same command may well work next time.

    Reached only when the wrapper positively understood the stream and counted nothing in it.
    A stream it could not read, or one carrying event kinds it does not recognise, is an
    EnvError instead — "the model inspected nothing" would be a false statement told
    identically on every run, about the one component that was working.
    """

    exit_code = 6


class BudgetError(AppError):
    """Over CE_PERSONA_MAX_PROMPT_TOKENS. Refused, never summarized.

    78 is EX_CONFIG from sysexits.h: the run was not attempted and no artifact exists.
    """

    exit_code = 78


# The published exit contract, in status order. `cli._epilog` RENDERS this into `--help`
# rather than restating it: two lists of the same numbers are one edit from disagreeing, and
# a caller branches on them. `{runner}` is filled in with the provider's binary name.
#
# The statuses mean the same thing in both modes, but the NOUNS do not — a validation
# returns verdicts, and "unknown or markdown-only persona" describes an argument it does not
# take. Those words are slots too, filled from one of the sets below, so this stays one
# table: a second literal table for the validate commands would be the same two-lists
# problem the rendering exists to avoid.
#
# Descriptions may span lines; continuation lines are indented by the renderer.
EXIT_TABLE: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("schema-valid {answer} ({ok})",)),
    (GateError.exit_code, ("the answer was not schema-valid {answer}{also}",)),
    (
        UsageError.exit_code,
        (
            "usage error: bad arguments, {bad_argument},",
            "bad -C directory, unresolvable -b base ref, malformed CE_PERSONA_* value",
        ),
    ),
    (
        EnvError.exit_code,
        (
            "environment error: {runner}, git or the plugin assets are missing,",
            "CE_PERSONA_RUN_DIR cannot be created, or {runner}'s event vocabulary",
            "changed and this build can no longer count what a run did",
        ),
    ),
    (RunnerError.exit_code, ("{runner} itself exited non-zero",)),
    (RunTimeout.exit_code, ("idle or hard timeout; the run was killed and partial output kept",)),
    (
        VacuousRun.exit_code,
        (
            "the model answered without making a single local tool call: it inspected",
            "nothing, so its {answer} -- empty or not -- attest to nothing",
            "(through codex, a local call is a shell command or a file change or patch;",
            "a web search, MCP call or function call is not one)",
        ),
    ),
    (BudgetError.exit_code, ("over CE_PERSONA_MAX_PROMPT_TOKENS; refused, never summarized",)),
)

# The two word sets the table is rendered with. A flow picks one; nothing else varies.
REVIEW_WORDS: Mapping[str, str] = {
    "answer": "findings",
    "ok": "an empty findings array is valid",
    "bad_argument": "unknown or markdown-only persona",
    # What ELSE makes an answer unusable in this flow. Empty for a review: findings are the
    # shape it is meant to return, so there is no second list it could collide with.
    "also": "",
}
VALIDATE_WORDS: Mapping[str, str] = {
    "answer": "verdicts",
    "ok": "one verdict for every input #, exactly once",
    "bad_argument": "a batch that is not an array of findings carrying a `#` each",
    "also": ", or also carries a findings list",
}


def _words(line: str, words: Mapping[str, str]) -> str:
    """Fill the flow's nouns by REPLACEMENT, before `{runner}` is filled by `format`.

    Two passes rather than one `format` call, so a description reaching `format` with a
    noun still in it is impossible: `format` would raise KeyError at `--help` time, which
    is a crash in the one place a caller goes to read the contract.
    """
    for name, value in words.items():
        line = line.replace("{" + name + "}", value)
    return line


def render_exit_table(runner: str, words: Mapping[str, str] = REVIEW_WORDS, width: int = 4) -> str:
    """The exit table as `--help` shows it. One source, two readers: help text and the tests.

    `width` pads the status column so 78 and 0 line up; it is the only formatting decision,
    and getting it wrong is cosmetic rather than a wrong contract.
    """
    lines: list[str] = []
    for code, description in EXIT_TABLE:
        head, *rest = [_words(line, words) for line in description]
        lines.append(f"  {str(code).ljust(width)}{head.format(runner=runner)}")
        lines.extend(f"      {line.format(runner=runner)}" for line in rest)
    return "\n".join(lines)
