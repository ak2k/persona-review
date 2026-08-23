#!/usr/bin/env python3
"""Prove the guards can fail: revert each fix and require a test to die.

Run: pytest tests/test_mutations.py

This exists because the recurring defect in this package is not a wrong guard, it is a
guard that CANNOT fail. Seven of them shipped green across a single review cycle — a
traversal fixture with no brief to traverse to, a hostile git config git never parsed,
exit-code assertions written against the module's own constants, a prompt test reading the
file the CLI wrote rather than what the runner received, an effort assertion whose expected
value was the default. Each was found by a reviewer running mutations by hand.

So the harness lives in the repository and runs as a check. Every entry below is a
one-line reversion of a real fix; the suite fails if the unit tests survive any of them.

THREE controls make this trustworthy rather than decorative, and it needs all three because
the first version of this file had none of them and was worthless in a way that read as a
perfect score — 23 kills in three seconds, every one a ModuleNotFoundError, because the
suite was copied to a path its own import bootstrap could not resolve. Two independent
reviewers found it. A harness that cannot fail is the exact defect it exists to catch, so:

  * **The scratch suite must PASS unmutated** before any kill is believed
    (`test_the_scratch_suite_runs_before_any_mutation_is_believed`). Without this, anything
    that stops the suite from running at all counts as every guard working.
  * **A mutation that merely breaks the code is not a kill**
    (`test_a_mutation_that_merely_breaks_the_code_is_not_counted_as_a_kill`). A syntax error
    exits non-zero without any guard firing; counting that would let a table of garbage
    report a perfect score. A run is a kill only if it actually executed tests and failed.
  * **Each probe must match EXACTLY once.** A pattern that silently stops matching reports a
    false kill; that happened twice while this was a scratch script, and once produced a
    "survivor" that was really a broken probe.

The reverted line is quoted in each entry, so a reader can see what is being undone.

Only mutations the UNIT suite can kill live here, so the check stays seconds rather than
hours. Guards that need the process suite — watchdogs, artifact lifecycle, the gate
replacement doors — are exercised by `tests/test_process.py` itself and swept manually.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
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
        # Two HEALTHY terminal events: the second silently became the verdict, so a real P0
        # review reported clean. Every status check passes because it genuinely is healthy.
        "multiple grok result events resolve to one instead of being refused",
        "persona_review/validate.py",
        r"    if len\(results\) > 1:",
        "    if False:",
    ),
    Mutation(
        "grok is_error accepted when absent",
        "persona_review/validate.py",
        r'    if result\.get\("is_error"\) is not False:',
        '    if result.get("is_error"):',
    ),
    Mutation(
        # Found by asking which guards had no entry here: mutating this left all 64 unit
        # tests green. An unguarded guard is exactly what this table exists to surface.
        "unknown extraction mode falls through to a real mode",
        "persona_review/validate.py",
        r'    fail\(f"unknown extraction mode \{mode!r\}"\)',
        "    return from_object_file(text)",
    ),
    Mutation(
        "a malformed structured_output falls through to the raw text",
        "persona_review/validate.py",
        r'        if not isinstance\(obj, dict\) or "findings" not in obj:',
        "        if False:",
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
    Mutation(
        # `0` was accepted, and `_watch` guarded each deadline with `if secs > 0`, so a typo
        # left a full-effort model run with nothing watching it and nothing to read it.
        "a zero timeout is accepted again, switching the watchdog off",
        "persona_review/config.py",
        r"    if value <= 0:\n        raise UsageError\(\n"
        r"            f\"\{name\}=\{text!r\} must be greater than zero",
        "    if value < 0:\n        raise UsageError(\n"
        '            f"{name}={text!r} must be greater than zero',
    ),
    Mutation(
        "a non-finite timeout is accepted again",
        "persona_review/config.py",
        r"    if not math\.isfinite\(value\):",
        "    if False:",
    ),
    Mutation(
        # The idea taken from pydantic-settings' extra="forbid". Without it a misspelled
        # variable is ignored and the setting it was meant to change keeps its default.
        "a misspelled CE_PERSONA_* setting is silently ignored again",
        "persona_review/config.py",
        r"        unknown = sorted\(k for k in source "
        r"if k\.startswith\(PREFIX\) and k not in KNOWN_VARS\)",
        "        unknown = []",
    ),
    Mutation(
        # RuntimeError, not OSError: `~nosuchuser` reached the top of the process as a
        # traceback with an unmapped exit status instead of a usage error.
        "an unknown home directory escapes as a traceback again",
        "persona_review/config.py",
        r"    except RuntimeError as exc:\n"
        r"        raise UsageError\(f\"\{name\}=\{raw!r\} names a home directory "
        r"that does not exist\"\) from exc",
        "    except ValueError as exc:\n"
        '        raise UsageError(f"{name}={raw!r} names a home directory that does not exist")'
        " from exc",
    ),
    Mutation(
        # The exit status lives on the exception so the mapping cannot drift. Collapsing two
        # kinds onto one code is what made "codex exited 2" indistinguishable from "you
        # called this wrongly".
        "two error kinds collapse onto one exit status",
        "persona_review/errors.py",
        r"^    exit_code = 3$",
        "    exit_code = 2",
    ),
    Mutation(
        # --help renders this table rather than restating it. A stale second copy is how a
        # published contract and its documentation come apart.
        "the documented exit table drifts from the classes",
        "persona_review/errors.py",
        r'    \(RunnerError\.exit_code, \("\{runner\} itself exited non-zero",\)\),',
        '    (RunnerError.exit_code, ("{runner} exited with some other status",)),',
    ),
    Mutation(
        # Without substitution the table still lists every number, so only a control that
        # looks for the provider name catches it.
        "the provider name is never substituted into the exit table",
        "persona_review/errors.py",
        r"\{head\.format\(runner=runner\)\}",
        "{head}",
    ),
    Mutation(
        # findings-schema.json is the plugin's file, read fresh every run, so its top-level
        # shape is an input. Dropping this guard turns a schema that is an array or null into
        # an AttributeError traceback and an unmapped exit status instead of a refusal.
        # The guard shipped with no test at all until a property test went looking.
        "the findings schema is used without checking it is an object",
        "persona_review/validate.py",
        r'            schema = _as_object\(loads\(schema_path\.read_text\(encoding="utf-8"\)\),'
        r' "findings schema"\)',
        '            schema = cast(JSONObject, loads(schema_path.read_text(encoding="utf-8")))',
    ),
]


@dataclass(frozen=True)
class Run:
    """What happened when the scratch suite ran."""

    code: int
    output: str

    @property
    def executed_tests(self) -> bool:
        """pytest collected tests and reached a verdict on them.

        Read from the exit STATUS, not from the output text. pytest documents 0 as "all
        passed" and 1 as "tests failed"; every other code is the suite failing to judge
        anything — 2 interrupted, 3 internal error, 4 usage error, 5 nothing collected. A
        mutation that produces one of those must not be counted as a guard firing, which is
        precisely the lie this file exists to catch.
        """
        return self.code in (0, 1)

    @property
    def passed(self) -> bool:
        return self.code == 0


def _scratch_tree(tmp: Path) -> Path:
    """A runnable copy of the package and its unit suite.

    The suite bootstraps its import with `Path(__file__).resolve().parent.parent`, so it
    MUST sit at `<root>/tests/test_unit.py` for that to land on the scratch tree. Copying it
    flat to `<root>/test_unit.py` pointed the bootstrap one directory too high, every run
    died with ModuleNotFoundError, and every mutation was recorded as killed — the harness
    built to catch guards that cannot fail could not itself fail.
    """
    shutil.copytree(SRC / "persona_review", tmp / "persona_review")
    (tmp / "tests").mkdir()
    shutil.copy(SRC / "tests" / "test_unit.py", tmp / "tests" / "test_unit.py")
    # The source may be a read-only Nix store path, and copytree preserves mode.
    for path in tmp.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return tmp


def _run_suite(root: Path) -> Run:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root)}
    # Inherited so `-m pytest` resolves the same pytest this harness is running under; the
    # scratch tree deliberately carries no pyproject.toml, so nothing else supplies it.
    for passthrough in ("PYTHONPATH", "VIRTUAL_ENV"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    proc = subprocess.run(
        # -p no:cacheprovider so parallel scratch trees never contend over a cache dir, and
        # --no-header -q to keep a 24-mutation sweep's captured output small.
        # No PYTHONSAFEPATH here. The scratch tree is ours, not a repository under review,
        # and dropping sys.path[0] is what broke the import bootstrap.
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_unit.py",
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:cov",
        ],
        cwd=root,
        capture_output=True,
        env=env,
        check=False,
    )
    return Run(proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace"))


class TestGuardsCanFail:
    def test_the_scratch_suite_runs_before_any_mutation_is_believed(self):
        """The control. Without it every 'kill' could be an import error.

        This is not ceremony: the first version of this harness failed exactly here and
        reported 23 kills that were all ModuleNotFoundError.
        """
        with tempfile.TemporaryDirectory() as tmp:
            control = _run_suite(_scratch_tree(Path(tmp)))
        assert control.passed, (
            "the unmutated scratch suite does not pass, so no kill below means anything:\n"
            + control.output[-2000:]
        )

    def test_every_reverted_fix_kills_a_test(self):
        survivors: list[str] = []
        broken: list[str] = []
        for mutation in MUTATIONS:
            with tempfile.TemporaryDirectory() as tmp:
                root = _scratch_tree(Path(tmp))
                target = root / mutation.path
                text = target.read_text(encoding="utf-8")
                mutated, count = re.subn(
                    mutation.pattern, mutation.replacement, text, count=1, flags=re.M
                )
                if count != 1:
                    # A probe that stopped matching reports a false "killed", which is the
                    # same class of lie this whole file exists to catch.
                    broken.append(f"{mutation.name} (pattern matched {count} times)")
                    continue
                target.write_text(mutated, encoding="utf-8")

                run = _run_suite(root)
                if run.code == 0:
                    survivors.append(mutation.name)
                elif not run.executed_tests:
                    # Non-zero, but the suite never ran a test — a syntax error or a broken
                    # import, not a guard firing. Counting that as a kill is how a harness
                    # flatters itself.
                    broken.append(f"{mutation.name} (suite did not run: no test executed)")

        assert broken == [], f"mutations that proved nothing: {broken}"
        assert survivors == [], (
            "the unit suite survived these reverted fixes, so nothing guards them: "
            + "; ".join(survivors)
        )

    def test_a_mutation_that_merely_breaks_the_code_is_not_counted_as_a_kill(self):
        """The classifier's own control.

        A syntax error makes the suite exit non-zero without any guard firing. If that
        counted, a mutation table full of garbage would report a perfect score.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = _scratch_tree(Path(tmp))
            target = root / "persona_review/validate.py"
            target.write_text(target.read_text(encoding="utf-8") + "\nthis is not python(\n")
            run = _run_suite(root)
        assert run.code != 0, "a syntax error should stop the suite"
        assert not run.executed_tests, "a syntax error must not be classified as a kill"
