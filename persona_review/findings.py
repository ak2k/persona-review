#!/usr/bin/env python3
"""Read a persona run's findings artifact at the detail level you actually need.

The wrappers write a findings JSON artifact and print one summary line. This projects that
artifact for a caller — usually an agent — so nobody pays for detail they will not use.

    ce-persona-findings <artifact>              # triage rows, P0/P1 by default
    ce-persona-findings <artifact> --all        # every severity
    ce-persona-findings <artifact> --show 3     # one finding, in full
    ce-persona-findings <artifact> --json       # the raw object, unchanged

THREE TIERS, AND WHY
--------------------
Tier 0 is the wrapper's own summary line plus its exit status: a gating caller — CI, a
pre-push hook, a loop deciding whether to iterate — branches on that and reads nothing.

Tier 1 is the default listing (`--list` names it explicitly and changes nothing):
severity, file:line, title, confidence, and the ONE quoted line that
motivated the finding. That last field is what makes a title trustworthy without the full
evidence array, and it is why CE's own subagents return `first_evidence` in their compact
return while the rest stays in the artifact. Roughly 20 tokens a finding.

Tier 2 is `--show N`: why_it_matters, the full evidence array, the suggested fix. Pay it
for the finding you are about to act on, not for the four you are not.

The failure mode this guards against is over-tiering: a caller that sees only titles will
under-weight a real defect or fix it wrongly from the label. Carrying evidence and
confidence in tier 1 is the hedge.
"""

from __future__ import annotations

import json
import sys
from typing import Any, NoReturn

SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
DEFAULT_SEVERITIES = ("P0", "P1")

# A literal, not `__doc__`: `python -OO` strips docstrings, and reading `.strip()` off the
# None that leaves behind turns a missing-argument message into an AttributeError.
USAGE = "usage: ce-persona-findings <artifact> [--list] [--all] [--show N] [--json]"

# A finding, and the artifact that holds them. Free-form by design: the schema lives in
# the compound-engineering plugin, is read at run time, and may add fields we do not know
# about — so the value type is Any, explicitly, rather than a shape we would be inventing.
Finding = dict[str, Any]
Artifact = dict[str, Any]


def fail(message: str) -> NoReturn:
    sys.exit(f"ce-persona-findings: {message}")


def load(path: str) -> Artifact:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError) as exc:
        fail(f"cannot read findings artifact {path}: {exc}")
    if not isinstance(obj, dict) or not isinstance(obj.get("findings"), list):
        fail(f"{path} is not a findings artifact")
    return obj


def ordered(findings: list[Finding]) -> list[tuple[int, Finding]]:
    """Stable numbering by severity then file:line, so `--show N` means one thing."""
    numbered = list(enumerate(findings, 1))
    return sorted(
        numbered,
        key=lambda pair: (
            SEVERITY_ORDER.get(str(pair[1].get("severity")), 9),
            str(pair[1].get("file", "")),
            pair[1].get("line") if isinstance(pair[1].get("line"), int) else 0,
        ),
    )


def render_row(n: int, finding: Finding) -> str:
    where = f"{finding.get('file', '?')}:{finding.get('line', '?')}"
    conf = finding.get("confidence")
    head = (
        f"#{n} {finding.get('severity', '??')} {where} — {finding.get('title', '')}"
        f"{f' (confidence {conf})' if conf is not None else ''}"
    )
    evidence = finding.get("first_evidence")
    if not evidence:
        items = finding.get("evidence")
        evidence = items[0] if isinstance(items, list) and items else None
    return f"{head}\n    {evidence}" if evidence else head


def render_detail(n: int, finding: Finding) -> str:
    lines = [render_row(n, finding)]
    if finding.get("why_it_matters"):
        lines.append(f"\nwhy: {finding['why_it_matters']}")
    items = finding.get("evidence")
    if isinstance(items, list) and items:
        lines.append("\nevidence:")
        lines += [f"  - {item}" for item in items]
    if finding.get("suggested_fix"):
        lines.append(f"\nfix: {finding['suggested_fix']}")
    routing = [
        f"{key}={finding[key]}"
        for key in ("autofix_class", "owner", "requires_verification", "pre_existing")
        if key in finding
    ]
    if routing:
        lines.append("\nrouting: " + ", ".join(routing))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    path = None
    show = None
    want_all = False
    as_json = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--show" and i + 1 < len(argv):
            try:
                show = int(argv[i + 1])
            except ValueError:
                fail(f"--show wants a finding number, got {argv[i + 1]!r}")
            i += 2
        elif arg == "--all":
            want_all = True
            i += 1
        elif arg == "--json":
            as_json = True
            i += 1
        elif arg == "--list":
            # The default already IS the tier-1 listing. Accepted as a no-op because the
            # docstring names `--list` as that tier, and a documented flag that exits 1
            # with "unknown option" is worse than a redundant one.
            i += 1
        elif arg.startswith("-"):
            fail(f"unknown option {arg}")
        else:
            path = arg
            i += 1

    if path is None:
        sys.exit(USAGE)

    obj = load(path)
    findings = obj["findings"]

    if as_json:
        json.dump(obj, sys.stdout, indent=1)
        print()
        return 0

    if show is not None:
        for n, finding in enumerate(findings, 1):
            if n == show:
                print(render_detail(n, finding))
                return 0
        fail(f"no finding #{show} (artifact has {len(findings)})")

    rows = ordered(findings)
    if not want_all:
        rows = [(n, f) for n, f in rows if f.get("severity") in DEFAULT_SEVERITIES]
    if not rows:
        total = len(findings)
        if total and not want_all:
            print(f"no P0/P1 findings ({total} total; --all to see the rest)")
        else:
            print("no findings")
        return 0
    for n, finding in rows:
        print(render_row(n, finding))
    hidden = len(findings) - len(rows)
    if hidden > 0 and not want_all:
        print(f"({hidden} lower-severity finding(s) hidden; --all to see them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
