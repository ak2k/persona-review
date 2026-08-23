#!/usr/bin/env python3
"""Prove the guards can fail: revert each fix and require a test to die.

Run: python3 tests/test_mutations.py

This exists because the recurring defect in this package is not a wrong guard, it is a
guard that CANNOT fail. Seven of them shipped green across a single review cycle — a
traversal fixture with no brief to traverse to, a hostile git config git never parsed,
exit-code assertions written against the module's own constants, a prompt test reading the
file the CLI wrote rather than what the runner received, an effort assertion whose expected
value was the default. Each was found by a reviewer running mutations by hand.

So the harness lives in the repository and runs as a check. Every entry below is a
one-line reversion of a real fix; the suite fails if the unit tests survive any of them.

Two properties make this trustworthy rather than decorative:

  * Each mutation asserts its own pattern applied EXACTLY once. A probe that silently stops
    matching would otherwise report a false "killed" — which happened twice while this was
    still a scratch script, and once produced a survivor that was really a broken probe.
  * The reverted line is quoted in the table, so a reader can see what is being undone.

Only mutations the UNIT suite can kill live here, so the check stays seconds rather than
hours. Guards that need the process suite — watchdogs, artifact lifecycle, the gate
replacement doors — are exercised by `tests/test_process.py` itself and swept manually.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    pattern: str
    replacement: str


MUTATIONS: list[Mutation] = [
    Mutation(
        # The P0 two model families found independently: allow anything around the answer
        # and a give-up that appends the brief's own empty example reads as a clean review.
        "object mode accepts prose around the answer again",
        "persona_review/validate.py",
        r"    try:\n        decoded = loads\(text\)\n    except ValueError as exc:",
        "    import re as _re\n"
        "    _m = _re.search(r'\\{.*\\}', text, _re.S)\n"
        "    text = _m.group(0) if _m else text\n"
        "    try:\n        decoded = loads(text)\n    except ValueError as exc:",
    ),
    Mutation(
        "numeric union loses the bool carve-out",
        "persona_review/validate.py",
        r'        numeric = any\(name in \("integer", "number"\) for name in names\)',
        '        numeric = all(name in ("integer", "number") for name in names)',
    ),
    Mutation(
        "unimplemented schema keywords silently ignored",
        "persona_review/validate.py",
        r"    unknown = sorted\(set\(spec\) - ENFORCED_KEYWORDS - IGNORED_KEYWORDS\)",
        "    unknown = []",
    ),
    Mutation(
        "nested object rules accepted but never enforced",
        "persona_review/validate.py",
        r'            for deeper in \("required", "properties"\):',
        "            for deeper in ():",
    ),
    Mutation(
        "tuple-form items accepted",
        "persona_review/validate.py",
        r"    if items is not None and not isinstance\(items, dict\):",
        "    if False:",
    ),
    Mutation(
        "minimum and maximum ignored",
        "persona_review/validate.py",
        r'        low_n, high_n = spec\.get\("minimum"\), spec\.get\("maximum"\)',
        "        low_n, high_n = None, None",
    ),
    Mutation(
        "array item types ignored",
        "persona_review/validate.py",
        r'        items = spec\.get\("items"\)\n        if isinstance\(items, dict\):',
        '        items = spec.get("items")\n        if False:',
    ),
    Mutation(
        "per-finding required keys unenforced",
        "persona_review/validate.py",
        r"        absent = \[k for k in item_required "
        r"if isinstance\(k, str\) and k not in finding\]",
        "        absent = []",
    ),
    Mutation(
        "grok subtype treated as optional",
        "persona_review/validate.py",
        r"    if not isinstance\(subtype, str\) or subtype not in GOOD_SUBTYPES:",
        "    if subtype is not None and subtype not in GOOD_SUBTYPES:",
    ),
    Mutation(
        "grok stop_reason treated as optional",
        "persona_review/validate.py",
        r"    if not isinstance\(stop, str\):",
        "    if False:",
    ),
    Mutation(
        "grok is_error accepted when absent",
        "persona_review/validate.py",
        r'    if result\.get\("is_error"\) is not False:',
        '    if result.get("is_error"):',
    ),
    Mutation(
        "grok result event first-wins instead of last",
        "persona_review/validate.py",
        r'        if isinstance\(event, dict\) and event\.get\("type"\) == "result":'
        r"\n            result = event",
        '        if isinstance(event, dict) and event.get("type") == "result":'
        "\n            result = result or event",
    ),
    Mutation(
        "persona name may be a path again",
        "persona_review/assets.py",
        r'    if not persona or persona != Path\(persona\)\.name or persona\.startswith\("\."\):',
        "    if False:",
    ),
    Mutation(
        "version ties fall back to filesystem order",
        "persona_review/assets.py",
        r"    return \(numeric, pure, name\)",
        '    return (numeric, 0, "")',
    ),
    Mutation(
        "a digit-like non-integer version crashes again",
        "persona_review/assets.py",
        r"    return int\(part\) if part\.isascii\(\) and part\.isdigit\(\) else 0",
        "    return int(part) if part.isdigit() else 0",
    ),
    Mutation(
        "the rubric never reaches the prompt",
        "persona_review/assets.py",
        r"    parts\.append\(rubric\(assets\)\)",
        '    parts.append("")',
    ),
    Mutation(
        "tier 2 collapses into tier 1",
        "persona_review/findings.py",
        r"def render_detail\(n: int, finding: Finding\) -> str:"
        r"\n    lines = \[render_row\(n, finding\)\]",
        "def render_detail(n: int, finding: Finding) -> str:\n"
        "    return render_row(n, finding)\n    lines = [render_row(n, finding)]",
    ),
    Mutation(
        "tier 1 drops the quoted motivating line",
        "persona_review/findings.py",
        r'    return f"\{head\}\\n    \{_text\(evidence\)\}" if evidence else head',
        "    return head",
    ),
    Mutation(
        "severity ordering removed from the listing",
        "persona_review/findings.py",
        r"    return sorted\(rows, key=key\)",
        "    return rows",
    ),
    Mutation(
        "any JSON file accepted as a findings artifact",
        "persona_review/findings.py",
        r'    if not isinstance\(raw, dict\) or not isinstance\(raw\.get\("findings"\), list\):',
        "    if False:",
    ),
    Mutation(
        "--show off by one",
        "persona_review/findings.py",
        r"            if n == show:",
        "            if n == show + 1:",
    ),
    Mutation(
        "usage errors collapse back onto the data exit code",
        "persona_review/findings.py",
        r"^EXIT_USAGE = 2$",
        "EXIT_USAGE = 1",
    ),
    Mutation(
        "the untrusted-output fence is dropped",
        "persona_review/findings.py",
        r"    nonce = secrets\.token_hex\(8\)\n    return \(",
        '    nonce = ""\n    return "" + (',
    ),
]


class TestGuardsCanFail(unittest.TestCase):
    def test_every_reverted_fix_kills_a_test(self):
        survivors: list[str] = []
        broken: list[str] = []
        for mutation in MUTATIONS:
            with self.subTest(mutation=mutation.name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(SRC / "persona_review", root / "persona_review")
                shutil.copy(SRC / "tests" / "test_unit.py", root / "test_unit.py")
                # The source may be a read-only Nix store path, and copytree preserves mode.
                for path in root.rglob("*"):
                    path.chmod(path.stat().st_mode | stat.S_IWUSR)

                target = root / mutation.path.replace("persona_review/", "persona_review/")
                text = target.read_text(encoding="utf-8")
                mutated, count = re.subn(
                    mutation.pattern, mutation.replacement, text, count=1, flags=re.M
                )
                if count != 1:
                    # A probe that stopped matching reports a false "killed", which is the
                    # same class of lie this whole file exists to catch.
                    broken.append(mutation.name)
                    continue
                target.write_text(mutated, encoding="utf-8")

                proc = subprocess.run(
                    [sys.executable, "test_unit.py"],
                    cwd=root,
                    capture_output=True,
                    env={"PATH": "/usr/bin:/bin", "HOME": str(root), "PYTHONSAFEPATH": "1"},
                    check=False,
                )
                if proc.returncode == 0:
                    survivors.append(mutation.name)

        self.assertEqual(broken, [], f"mutation probes stopped matching the source: {broken}")
        self.assertEqual(
            survivors,
            [],
            "the unit suite survived these reverted fixes, so nothing guards them: "
            + "; ".join(survivors),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
