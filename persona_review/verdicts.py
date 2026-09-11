"""The validator mode's own rules: the batch it is handed, and the verdicts it must return.

A review asks a model what it finds. A VALIDATION asks it to judge findings somebody else
already made, one at a time, and the answer is only usable if it addresses each of them
exactly once — a batch of eight findings that comes back with seven verdicts has let one
finding through unvalidated while reading as a completed pass.

Separate from `cli.py` because the coverage gate measures the unit suite alone and cannot
see `cli.py` at all: rules that decide whether a run is believed belong where a unit test
can drive them directly. Separate from `validate.py` because that module holds the schema
walk both modes share, and this holds the one rule only a validation has.
"""

from __future__ import annotations

from pathlib import Path

from . import errors
from .validate import Artifact, fail, loads

# Shipped as package data rather than read from the plugin. `findings-schema.json` is the
# plugin's file and this package has no authority over it; the verdict shape is inlined in
# the plugin's prompt TEMPLATE and ships as no file at all, so there is nothing to read and
# the schema the gate enforces has to be ours. AGENTS.md records the departure.
SCHEMA_FILE = "verdicts-schema.json"


def schema_path() -> Path:
    """The verdicts schema on disk, beside the module that owns it."""
    return Path(__file__).resolve().parent / SCHEMA_FILE


def parse_batch(text: str, source: str) -> list[int]:
    """The stable `#` of every finding in the plugin's validator batch, in input order.

    Refused rather than repaired, because every malformed shape here breaks the one thing
    the verdicts are matched on. A duplicate `#` makes a verdict ambiguous — two findings
    claim the same address, so "validated" cannot be attributed — and a missing or
    non-integer one leaves a finding with no address at all, which the coverage check would
    then report as an extra verdict rather than as a bad batch.
    """
    try:
        decoded = loads(text)
    except ValueError as exc:
        raise errors.UsageError(f"{source} is not JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise errors.UsageError(
            f"{source} must be a JSON array of findings, got {type(decoded).__name__}. "
            "This is the plugin's validator batch: one object per finding, each with a `#`."
        )
    numbers: list[int] = []
    seen: set[int] = set()
    for position, element in enumerate(decoded, 1):
        if not isinstance(element, dict):
            raise errors.UsageError(
                f"{source} element {position} is {type(element).__name__}, not a finding object"
            )
        number = element.get("#")
        # `isinstance(True, int)` is True in Python, so a batch carrying `"#": true` would
        # otherwise pass as the number 1 and silently address another finding's verdict.
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise errors.UsageError(
                f"{source} element {position} has #={number!r}. Every finding needs an "
                "integer stable number of 1 or more; the verdicts are matched to it."
            )
        if number in seen:
            raise errors.UsageError(
                f"{source} element {position} repeats #{number}. The numbers address the "
                "findings one for one, so a repeat leaves a verdict belonging to neither."
            )
        seen.add(number)
        numbers.append(number)
    if not numbers:
        # An empty batch would produce a full-effort model run whose only correct answer is
        # an empty verdicts array. Refusing costs the caller nothing and says why.
        raise errors.UsageError(f"{source} is an empty array: there is nothing to validate")
    return numbers


def _numbers(found: Artifact) -> dict[int, int]:
    """How many times each `#` appears in the answer's verdicts."""
    entries = found.get("verdicts")
    counts: dict[int, int] = {}
    for entry in entries if isinstance(entries, list) else []:
        number = entry.get("#") if isinstance(entry, dict) else None
        if isinstance(number, int) and not isinstance(number, bool):
            counts[number] = counts.get(number, 0) + 1
    return counts


def _listed(label: str, numbers: list[int]) -> str:
    return f"{label} {', '.join(f'#{n}' for n in numbers)}"


def check_coverage(found: Artifact, expected: list[int]) -> None:
    """One verdict for every input `#`, exactly once. Raises `GateError` naming the gaps.

    The template's own rule, enforced here because the failure is silent otherwise: a batch
    that comes back a verdict short reads as a completed validation, and the finding nobody
    judged is then carried as if it had been. Extra and duplicated numbers are named too —
    a verdict addressed to a finding that was never sent means the model was answering about
    something else, which says the same thing about the run as a missing one.
    """
    counts = _numbers(found)
    wanted = set(expected)
    missing = sorted(wanted - set(counts))
    extra = sorted(set(counts) - wanted)
    duplicated = sorted(number for number, count in counts.items() if count > 1)
    problems: list[str] = []
    if missing:
        problems.append(_listed("no verdict for", missing))
    if extra:
        problems.append(_listed("a verdict for findings that were not sent:", extra))
    if duplicated:
        problems.append(_listed("more than one verdict for", duplicated))
    if problems:
        fail(
            f"the verdicts do not cover the batch one for one ({'; '.join(problems)}). "
            f"The batch carried {len(wanted)} finding(s); a finding with no verdict would "
            "be carried as validated by a run that never judged it."
        )


def summarize(found: Artifact, count: int) -> str:
    """`k validated, m rejected` — the split a caller acts on, for the one stdout line.

    Counted from `validated` rather than derived from `count`, so a verdict whose flag is
    neither true nor false is absent from both tallies instead of being folded into one.
    The schema refuses that shape before this runs; the arithmetic does not rely on it.
    """
    entries = found.get("verdicts")
    validated = 0
    rejected = 0
    for entry in entries if isinstance(entries, list) else []:
        flag = entry.get("validated") if isinstance(entry, dict) else None
        if flag is True:
            validated += 1
        elif flag is False:
            rejected += 1
    return f"{validated} validated, {rejected} rejected"
