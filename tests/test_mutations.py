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

TWO TIERS, because one tier was quietly covering two thirds of the package.

This file used to hold only mutations the UNIT suite could kill, and said the rest were
"swept manually". That left `cli.py` and `runner.py` — the run lock, the artifact clear,
both watchdogs, the process-group kill — with ZERO entries between them, backed by a
sentence in a docstring. A claim about coverage, in the one file whose whole job is to
replace claims about coverage with checks. Four modules had no entry at all.

  * **unit** — the whole unit suite per mutation, seconds each, run serially.
  * **process** — `tests/test_process.py` for the tests named in the entry's `selector`.
    Slower, because these mutations disable the very watchdogs whose tests then wait out
    their own timeouts, so the tier runs concurrently: it is bounded by stub sleeps rather
    than by CPU, and the mutations are fully independent. A selector matching nothing makes
    pytest exit 5, which `executed_tests` classifies as proving nothing rather than as a
    kill.

`test_every_module_is_represented_or_explicitly_exempt` stops a module having no entries
unnoticed: one with neither an entry nor a written exemption in `UNMUTATED_MODULES` fails
the suite. Two modules are exempt and say why.

WHAT THAT CHECK DOES NOT CLAIM. It is MODULE granularity, not guard granularity. cli.py has
3 entries against a dozen raise sites; validate.py has 15 against three dozen. So "every
module is represented" is enforced, and "every guard can fail" is not — the table is a
growing floor, not a proof of completeness, and the honest way to extend it is still to ask
which guard has no entry and write one. Saying otherwise here would be the same
coverage-shaped claim this file exists to disbelieve.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent

# Which copy of the library gets mutated. The unit and process checks both run the BUILT
# package, so a harness that mutates the source tree proves the source's guards can fail —
# a claim about a different set of files from the ones that ship. They are the same modulo
# packaging, which is why this was only a P3, but "modulo packaging" is an assumption and
# this file is where assumptions go to be checked.
#
# PERSONA_REVIEW_PKG points at the installed site-packages; the flake checks set it. Falling
# back to the source tree keeps a bare `pytest` working in a checkout, and
# `test_the_mutated_library_is_the_one_that_ships` asserts the flake's copy is in use
# whenever the variable is set, so the fallback cannot silently become the default in CI.
PACKAGE_ROOT = Path(os.environ.get("PERSONA_REVIEW_PKG") or SRC)


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    pattern: str
    replacement: str
    # Which suite is expected to kill it, and — for the process tier — the `-k` expression
    # naming the tests that should. A process mutation runs the whole process suite only for
    # the selected tests, because running all 85 per mutation would take hours.
    suite: str = "unit"
    selector: str = ""


# Modules with no mutation entry, and why. Checked by `test_every_module_is_represented`, so
# "swept manually" stops being a claim in a docstring and becomes a list someone had to
# write down. A new module with no entries fails that test until it is covered or listed.
UNMUTATED_MODULES: dict[str, str] = {
    # Probes a real authenticated `grok` binary for CLI drift. No automated check drives it —
    # the same reason it is omitted from the coverage gate — so there is no test to kill.
    "flags.py": "manual CLI-drift probe; no automated check exercises it",
    # Pure data: the Provider dataclass and its argv builders, with no guard to revert. Its
    # argv shape IS asserted, by test_process.py's model/effort tests and by flags.py.
    "providers.py": "declarative provider table; no branch that can fail open",
}


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
        r"        absent = \[k for k in required_fields "
        r"if isinstance\(k, str\) and k not in entry\]",
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
        r"        if not isinstance\(obj, dict\) or key not in obj:",
        "        if False:",
    ),
    Mutation(
        # The production incident: one turn, zero tool calls, 151 output tokens, a
        # schema-valid empty findings array, exit 0. Every schema rule above passes on that
        # answer, because the answer is fine — it is the RUN that never happened.
        "a run that made no tool calls certifies a clean review again",
        "persona_review/validate.py",
        r"    if stats\.tool_calls == 0:",
        "    if False:",
    ),
    Mutation(
        # Counting every content block rather than the tool_use ones counts thinking and
        # text as evidence of work, which is exactly what the dud run produced.
        "any content block counts as a tool call",
        "persona_review/validate.py",
        r'                    if isinstance\(block, dict\) and block\.get\("type"\) '
        r"== GROK_TOOL_BLOCK",
        "                    if isinstance(block, dict)",
    ),
    Mutation(
        # `agent_message` and `reasoning` are items too. Taking every item as a tool call
        # certifies a codex run that only ever thought and answered.
        "every codex item counts as a tool call, not just the tool ones",
        "persona_review/validate.py",
        r"            if item_kind not in CODEX_TOOL_ITEMS:",
        "            if False:",
    ),
    Mutation(
        # codex emits `item.started` and `item.completed` for the same call, so counting
        # events rather than items doubles every total — a difference that only matters when
        # the real answer is small, which is the case this whole guard is about.
        "codex tool calls are counted twice, once per event",
        "persona_review/validate.py",
        r"                if ident not in seen:",
        "                if True:",
    ),
    Mutation(
        # The other half of that rule, and the one that used to fail OPEN: an item with no
        # usable id fell past the dedupe entirely and was counted once per event.
        "an id-less codex item is counted on both of its events",
        "persona_review/validate.py",
        r'            elif kind == "item\.completed":',
        "            else:",
    ),
    Mutation(
        # Without this the wrapper reports "the model never opened the diff" on every run
        # after a provider-CLI rename — a falsehood about the one component that was working.
        "a renamed codex vocabulary is blamed on the model instead of the wrapper",
        "persona_review/validate.py",
        r"    if calls == 0 and unknown:",
        "    if False:",
    ),
    Mutation(
        # The control side of the same guard: if the kinds we skip on purpose counted as
        # unrecognised, every genuine dud would report drift and exit 6 would be dead code.
        "kinds this wrapper skips on purpose are reported as drift",
        "persona_review/validate.py",
        r"                if item_kind not in CODEX_QUIET_ITEMS:",
        "                if True:",
    ),
    Mutation(
        "an empty codex stream reads as a model that did nothing",
        "persona_review/validate.py",
        r"    if total == 0:",
        "    if False:",
    ),
    Mutation(
        # grok's answer arrives inside its event stream, so the gate counts from the text it
        # has already read. Reverting this reads the same ~1.4 MB a second time; the unit
        # test proves it by making the file source unusable.
        "the gate opens grok's stream a second time to count it",
        "persona_review/validate.py",
        r"        if evidence\.events_file == answer_file",
        "        if False",
    ),
    Mutation(
        # The reader's half of the refusal. Without it the package launders its own verdict:
        # the artifact exit 6 kept as EVIDENCE comes back as an ordinary listing at exit 0.
        "a refused artifact is rendered anyway by the retrieval command",
        "persona_review/findings.py",
        r"    vacuous = validate\.refused_run\(Path\(path\)\)",
        "    vacuous = None",
    ),
    Mutation(
        # Only a POSITIVE reading of zero refuses. Reverting this refuses every artifact that
        # has a sidecar at all, which is every artifact this package writes.
        "the reader refuses on any provenance rather than on a counted zero",
        "persona_review/validate.py",
        r"    if calls != 0:",
        "    if False:",
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
        # The whole reason --return exists: the merge helper demotes a 75/100 finding with no
        # `first_evidence` to 50, where its confidence gate suppresses it, so a lens that
        # filled only the evidence array reads as having found nothing.
        "a lens that filled only evidence[0] is projected as having quoted nothing",
        "persona_review/findings.py",
        r"    return _quote\(items\[0\]\) if isinstance\(items, list\) and items else None",
        "    return None",
    ),
    Mutation(
        # The helper drops a malformed return WITH every finding in it and says nothing, so
        # emitting one turns a whole reviewer's pass into silence downstream.
        "a return the merge helper would drop whole is emitted anyway",
        "persona_review/findings.py",
        r"    if not isinstance\(reviewer, str\) or not reviewer\.strip\(\):",
        "    if False:",
    ),
    Mutation(
        # Verification that cannot fail is worse than none: it certifies quotes it never
        # checked, and the merge then treats a fabricated line as founded evidence.
        "--verify-quotes keeps a quote the reviewed tree does not carry",
        "persona_review/findings.py",
        r'    if quoted not in _normalized\("\\n"\.join\(window\)\):',
        "    if False:",
    ),
    Mutation(
        # A quote is model-written text, so the path inside it is an input and not a
        # destination. Reverting this lets `/etc/passwd:1 -- root:x:0:0` make the reader
        # confirm a line in a file the caller never pointed it at.
        #
        # On the RESOLVED path, and the kill cases say why: the string check this replaced
        # refused `..` and absolute paths — the only two vectors the old kill cases used — and
        # admitted a symlink inside the tree pointing out of it, so this very entry passed
        # with the escape wide open.
        "a quote can steer the verifier at a file outside the reviewed tree",
        "persona_review/findings.py",
        r"                if not target\.is_relative_to\(repo\):",
        "                if False:",
    ),
    Mutation(
        # `_resolve`'s contract is that an unresolvable citation is dropped, never fatal. The
        # line number is model-written text, and CPython refuses to int() 5000 digits; the
        # narrow except let that leave as a traceback and exit 1, which already means "not a
        # findings artifact".
        "a citation the reader cannot parse crashes instead of dropping",
        "persona_review/findings.py",
        r"            except \(OSError, ValueError\):",
        "            except OSError:",
    ),
    Mutation(
        # The backticked span is read only when the remainder IS one. Searching for it
        # anywhere chose the branch by the presence of a token: a parenthesized aside was
        # checked instead of the quote, certifying prose and dropping verbatim lines.
        "a backtick anywhere in the quote redirects the check at a fragment",
        "persona_review/findings.py",
        r"    span = _BACKTICKED\.fullmatch\(rest\)",
        "    span = _BACKTICKED.search(rest)",
    ),
    Mutation(
        # The evidence contract asks for the motivating LINE(S). A quote spanning the two
        # lines the finding turns on, tested against one line, can never match — and the
        # helper then demotes the finding to 50, where its gate suppresses it.
        "a multi-line quote is tested against a single line again",
        "persona_review/findings.py",
        r'    span = compared\.count\("\\n"\) \+ 1',
        "    span = 1",
    ),
    Mutation(
        # Without the floor, `account` and `r` both "verify" against `return bill(account)`,
        # and a surviving first_evidence is what unlocks cross-model promotion.
        "the substring test loses its specificity floor",
        "persona_review/findings.py",
        r"^_QUOTE_FLOOR = 12$",
        "_QUOTE_FLOOR = 0",
    ),
    Mutation(
        # Markdown decoration and a `:col` suffix are shapes lenses write every day. Without
        # them in the match, the path carried `**` or a backtick, nothing resolved, and a true
        # quote was dropped with a reason indistinguishable from a fabricated citation.
        "a decorated citation is unparseable again",
        "persona_review/findings.py",
        r"^_REFERENCE = re\.compile\(.*$",
        r"""_REFERENCE = re.compile(r"([^\\s`'\\"]+?):(\\d+)\\b")""",
    ),
    Mutation(
        # Committing to the first citation that resolves drops a quote the tree does carry
        # when a lens annotates it with a second one.
        "only the first resolving citation may corroborate a quote",
        "persona_review/findings.py",
        r"    for cite in cited:",
        "    for cite in cited[:1]:",
    ),
    Mutation(
        # A surviving first_evidence is what makes the finding's LOCATION trustworthy, and
        # every tree holds some real twelve-character line elsewhere. Without the own-file
        # rule a quote of any README line founds a finding reported in another file.
        "a citation of any file may found a finding reported in another one",
        "persona_review/findings.py",
        r"    if not isinstance\(own, str\):\n        return True",
        "    if True:\n        return True",
    ),
    Mutation(
        # A citation of the finding's own file by an in-tree absolute path or through `..`
        # names some other location when the two are compared lexically, so a quote verbatim
        # from the file the finding is reported at is dropped as founding somewhere else.
        "a citation founds a finding only when it spells the path the same way",
        "persona_review/findings.py",
        r"        return \(repo / ref\.path\)\.resolve\(strict=True\) == "
        r"\(repo / own\)\.resolve\(strict=True\)",
        "        return PurePosixPath(ref.path) == PurePosixPath(own)",
    ),
    Mutation(
        # `__init__.py` names one file per package and the bare one at the repository root
        # resolves first, so the citation is checked against a file the finding is not at
        # and a quote verbatim from the finding's own file is dropped.
        "a basename citation is resolved against the root before the finding's own file",
        "persona_review/findings.py",
        r"            candidates = \[own, cited\]",
        "            candidates = [cited, own]",
    ),
    Mutation(
        # Each snippet of a multi-citation quote is a claim about its own line. Compared as
        # one remainder, every snippet carries the others' text too, so a quote whose
        # snippets are all true on their lines is dropped.
        "a quote citing several locations is compared as one claim again",
        "persona_review/findings.py",
        r"    if len\(cited\) < 2:\n        return None",
        "    if True:\n        return None",
    ),
    Mutation(
        # The whole quote is one claim about the location it cites. Read as segments first,
        # a quoted source line that is itself citation-shaped -- a fixture, a log line -- is
        # split, and the finding's own citation is left a fragment under the floor.
        "a quote is split on its citations before being read whole",
        "persona_review/findings.py",
        r"        if ref is None or not _founds\(ref, own, repo\):",
        "        if True:",
    ),
    Mutation(
        # Segmenting on the citations that resolved skips the one that resolved to nothing,
        # so a fabricated location prefixed to true snippets is never checked at all.
        "a citation that resolves to nothing is left out of the segments",
        "persona_review/findings.py",
        r"    segments = _segments\(quote, cited\)",
        "    segments = _segments(quote, [c for c in cited if c.ref is not None])",
    ),
    Mutation(
        # What makes a quote segmented: each citation owning a snippet long enough to check,
        # and nothing outside them. Without it `a.py:1 and b.py:2 -- code` is segmented on
        # the connector `and`, which decides the quote on text that answers for nothing.
        "a connector between two citations counts as a snippet of its own",
        "persona_review/findings.py",
        r"    if _segment\(outside\) or "
        r"any\(len\(_normalized\(t\)\) < _QUOTE_FLOOR for _, t in segments\):\n        return None",
        "    if False:\n        return None",
    ),
    Mutation(
        # A location cited twice is one claim about it. Removing only the citation being
        # checked leaves its twin in the remainder, where the line does not carry it.
        "a doubled citation is two claims, so each leaves the other in the remainder",
        "persona_review/findings.py",
        r"    same = \[c for c in cited if c\.path == target\.path and c\.line == target\.line\]",
        "    same = [target]",
    ),
    Mutation(
        # A backticked path closes before the colon in a shape lenses write, and without the
        # optional backtick the whole citation resolves to nothing.
        "a backtick between the path and the line number is unparseable again",
        "persona_review/findings.py",
        r"^_REFERENCE = re\.compile\(.*`\?:.*$",
        r'''_REFERENCE = re.compile(r"""[(\\[*<`]*([^\\s`'"(\\[*<]+?):'''
        r'''(\\d+)(?::\\d+)?\\b[*)\\]>`]*""")''',
    ),
    Mutation(
        # A newline inside the backticks decorates the quote rather than belonging to it.
        # Counted as a line, it widens the window past the line the citation names, and the
        # text on the NEXT line then certifies the citation.
        "padding inside the backticks widens the window past the cited line",
        "persona_review/findings.py",
        r"    return span\.group\(1\)\.strip\(\) if span else rest",
        "    return span.group(1) if span else rest",
    ),
    Mutation(
        # The twin of the reviewer-name guard above. A non-list `residual_risks` makes the
        # helper drop the whole return into a silent counter, so a reviewer's entire pass
        # becomes silence downstream.
        "a return whose list fields are the wrong shape is emitted anyway",
        "persona_review/findings.py",
        r"        if not isinstance\(out\.setdefault\(key, \[\]\), list\):",
        "        if False:",
    ),
    Mutation(
        # Merge state is the orchestrator's to stamp on its own reconciled returns. A truthy
        # `settled_conflict` copied out of a lens artifact exempts the finding from the
        # helper's confidence gate — carrying one whose quote --verify-quotes just dropped
        # past the very gate this projection exists to feed.
        "merge state is copied out of a lens artifact again",
        "persona_review/findings.py",
        r'^    "first_evidence",$',
        '    "first_evidence", "settled_conflict",',
    ),
    Mutation(
        # Silently discarding a -C nobody could use emits the plain projection at exit 0,
        # byte-identical to a verified one, on the mode whose only reader is a machine.
        "-C without --verify-quotes is discarded instead of refused",
        "persona_review/findings.py",
        r"    elif repo_spec is not None:",
        "    elif False:",
    ),
    Mutation(
        # --return and the rendering modes project the same artifact for different readers,
        # so ranking them silently hands a caller a shape it did not ask for.
        "--return together with --json or --show is ranked instead of refused",
        "persona_review/findings.py",
        r"    if as_return and \(as_json or show is not None\):",
        "    if False:",
    ),
    Mutation(
        "--verify-quotes is accepted without the --return it modifies",
        "persona_review/findings.py",
        r"        if not as_return:",
        "        if False:",
    ),
    Mutation(
        # Defaulting the reviewed tree to the working directory verifies the quotes against
        # whatever happens to be checked out there, and says nothing.
        "--verify-quotes falls back to the current directory when -C is missing",
        "persona_review/findings.py",
        r'^            return _usage_error\("--verify-quotes wants -C.*$',
        '            repo_spec = "."',
    ),
    Mutation(
        "a -C that is not a directory is accepted",
        "persona_review/findings.py",
        r"        if not repo\.is_dir\(\):",
        "        if False:",
    ),
    Mutation(
        # `Path("~nosuchuser/x").expanduser()` raises RuntimeError — not OSError, and not a
        # type a caller would think to catch — so an unknown user leaves as a traceback and an
        # unmapped status instead of exit 2.
        "an unknown home directory in -C leaves as a traceback",
        "persona_review/findings.py",
        r"    except RuntimeError:",
        "    except SystemError:",
    ),
    Mutation(
        # The laundering route in the mode a machine consumes. --return feeds a merge
        # directly, so a refused run reaching it is the refusal being undone by a reader.
        "the vacuous-run refusal stops preceding --return",
        "persona_review/findings.py",
        r"^    vacuous = validate\.refused_run\(Path\(path\)\)$",
        "    vacuous = None if as_return else validate.refused_run(Path(path))",
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
        r' f"\{noun\} schema"\)',
        '            schema = cast(JSONObject, loads(schema_path.read_text(encoding="utf-8")))',
    ),
    Mutation(
        # The rule the whole mode exists for. A batch that comes back one verdict short
        # reads as a completed validation, and the finding nobody judged is then carried as
        # if it had been -- which is the validation-degraded state this replaces.
        "a batch with a finding nobody judged passes as a complete validation",
        "persona_review/verdicts.py",
        r"    if problems:",
        "    if False:",
    ),
    Mutation(
        # `#` is the only address a verdict has. Two findings claiming one number makes a
        # verdict unattributable, and the coverage check above would then report a count
        # that happens to match while one finding is unjudged.
        "a batch may address two findings with one number",
        "persona_review/verdicts.py",
        r"        if number in seen:",
        "        if False:",
    ),
    Mutation(
        # An empty batch buys a full-effort model run whose only correct answer is `[]`.
        "an empty batch is dispatched instead of refused",
        "persona_review/verdicts.py",
        r"    if not numbers:",
        "    if False:",
    ),
    Mutation(
        # The codex/object site of the same fail-open guard the grok entry above covers, and
        # a separate one because neither site's pattern can reach the other: without the key
        # check the gate believes whatever top-level key the answer happens to carry.
        "the object answer's top-level key is never checked",
        "persona_review/validate.py",
        r"    if not isinstance\(decoded, dict\) or key not in decoded:",
        "    if False:",
    ),
    # ----------------------------------------------------------------------------------
    # PROCESS TIER. cli.py and runner.py had no entries at all, because the unit suite
    # cannot reach them: both are driven through subprocesses by tests/test_process.py. That
    # made the module docstring's "swept manually" the only thing standing behind four
    # modules — which is a claim, not a mechanism, and this file exists because claims about
    # coverage are exactly what keeps turning out to be false here.
    #
    # Each entry names the tests that should kill it, so one mutation runs a handful of
    # process tests rather than all 85. The selector is itself checked: a mutation whose
    # tests all pass is a survivor, and a selector matching NOTHING is reported as broken.
    Mutation(
        # The documented way to consume --return pipes it into `jq -s .`. Without the handler
        # a reader that stops early leaves the interpreter's shutdown flush to raise where
        # nothing can catch it, reporting failure for a projection that completed.
        "a broken pipe turns a completed --return into a failure",
        "persona_review/findings.py",
        r"        except BrokenPipeError:",
        "        except SystemError:",
        suite="process",
        selector="piped_into_a_reader_that_stops_early",
    ),
    Mutation(
        # Two concurrent runs of one persona through one provider address the same files.
        # Not a lost race — a silently wrong answer, which is the failure this package is for.
        "concurrent runs are allowed to interleave again",
        "persona_review/runner.py",
        r"            fcntl\.flock\(handle\.fileno\(\), fcntl\.LOCK_EX \| fcntl\.LOCK_NB\)",
        "            pass",
        suite="process",
        selector="(concurrent or sequential_rerun) and grok",
    ),
    Mutation(
        # Ordering, not presence: locking after the clear still lets the second run delete
        # the first run's event stream out from under a provider still writing to it.
        "the run lock is taken after the clear instead of before",
        "persona_review/cli.py",
        r"    with runner\.exclusive_run\(stem\):\n        _clear_run_dir\(run_dir, stem\)",
        "    _clear_run_dir(run_dir, stem)\n"
        "    with runner.exclusive_run(stem):\n        pass\n"
        "    if True:",
        suite="process",
        selector="(concurrent or sequential_rerun) and grok",
    ),
    Mutation(
        "the run directory is never cleared, so stale findings survive a refusal",
        "persona_review/cli.py",
        r"    for suffix in ARTIFACT_SUFFIXES:",
        "    for suffix in ():",
        suite="process",
        selector="refusal_before_dispatch and grok",
    ),
    Mutation(
        # The idle watchdog. Its `if secs > 0` guard is gone because config makes the value
        # positive, so what remains to revert is the comparison itself.
        "the idle watchdog never fires",
        "persona_review/runner.py",
        r"        if now - last_change >= idle_secs:",
        "        if False:",
        suite="process",
        selector="silent_runner_is_killed and grok",
    ),
    Mutation(
        "the hard deadline never fires",
        "persona_review/runner.py",
        r"        if now - started >= hard_secs:",
        "        if False:",
        suite="process",
        selector="chatty_runner and grok",
    ),
    Mutation(
        # Provenance recorded WHICH brief ran but never WHAT IT RAN AGAINST, so a finding at
        # f.py:42 could not be tied to the code it was about.
        "provenance stops recording the commit that was reviewed",
        "persona_review/cli.py",
        r"^            f\"head_sha=\{runner\.resolve_revision\(repo, 'HEAD'\)\}\",$",
        '            f"head_sha=",',
        suite="process",
        selector="provenance and grok",
    ),
    Mutation(
        # The wiring, not the counter: the gate can count correctly and still count the wrong
        # thing if the CLI points it at another file. The unit tier cannot reach this — it
        # drives `run_stats` and `gate` directly.
        "the gate is pointed at the wrong file for its evidence",
        "persona_review/cli.py",
        r"        events_file=events_file, mode=provider\.events_mode, duration_s=elapsed",
        "        events_file=prompt_file, mode=provider.events_mode, duration_s=elapsed",
        suite="process",
        selector="one_tool_call_is_enough and grok",
    ),
    Mutation(
        # start_new_session is what lets the watchdog signal the provider's whole group. A
        # provider CLI spawns helpers that ignore SIGTERM; killing only the parent leaves a
        # full-effort model run going with nothing watching it.
        "the provider is no longer run in its own process group",
        "persona_review/runner.py",
        # Anchored to the CODE line. Unanchored, this matched the docstring above it first.
        r"^                start_new_session=True,$",
        "                start_new_session=False,",
        suite="process",
        selector="ignore_sigterm and grok",
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


SUITE_FILE = {"unit": "test_unit.py", "process": "test_process.py"}

# Generous: a mutation that disables a watchdog makes that watchdog's own tests wait out
# their own timeouts, which is the slowest legitimate case here and lands around 90s.
RUN_TIMEOUT_SECS = 300.0

# Not a pytest status. pytest uses 0-5, so this cannot be mistaken for one, and
# `executed_tests` (which admits only 0 and 1) classifies it as proving nothing.
TIMED_OUT = -1

# The process tier is bounded by wall clock, not CPU: nearly all of it is spent waiting for
# stub runners to sleep. The mutations are fully independent — separate scratch trees,
# separate temp dirs, separate processes — so running them concurrently turns the sum of
# their timeouts into the slowest single one. Four at a time, because each spawns a pytest
# that itself spawns review subprocesses that fork helpers.
PROCESS_WORKERS = 4


def _scratch_tree(tmp: Path, suite: str = "unit") -> Path:
    """A runnable copy of the package and one of its suites.

    The suite bootstraps its import with `Path(__file__).resolve().parent.parent`, so it
    MUST sit at `<root>/tests/<file>` for that to land on the scratch tree. Copying it
    flat to `<root>/test_unit.py` pointed the bootstrap one directory too high, every run
    died with ModuleNotFoundError, and every mutation was recorded as killed — the harness
    built to catch guards that cannot fail could not itself fail.
    """
    # The LIBRARY comes from PACKAGE_ROOT (the built package under the flake checks); the
    # SUITE always comes from the source tree, because the package does not ship its tests.
    shutil.copytree(PACKAGE_ROOT / "persona_review", tmp / "persona_review")
    (tmp / "tests").mkdir()
    name = SUITE_FILE[suite]
    shutil.copy(SRC / "tests" / name, tmp / "tests" / name)
    # The source may be a read-only Nix store path, and copytree preserves mode.
    for path in tmp.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return tmp


def _run_suite(root: Path, suite: str = "unit", selector: str = "") -> Run:
    if suite == "unit":
        # A closed environment: the unit suite touches no binary, so nothing needs to leak in.
        env = {"PATH": "/usr/bin:/bin", "HOME": str(root)}
        # Inherited so `-m pytest` resolves the same pytest this harness is running under;
        # the scratch tree deliberately carries no pyproject.toml, so nothing else supplies it.
        for passthrough in ("PYTHONPATH", "VIRTUAL_ENV"):
            if passthrough in os.environ:
                env[passthrough] = os.environ[passthrough]
    else:
        # The process suite builds its own restricted PATH for every review it launches, and
        # symlinks a real `git` into its stub directory — so it needs to find one here. It
        # never reaches a real `grok` or `codex`: the PATH it hands each subprocess contains
        # only its stubs, which is what stops a unit test starting a billed model run.
        env = {**os.environ, "HOME": str(root)}
        env.pop("PERSONA_REVIEW_BIN", None)  # drive the local shim, not an installed build

    argv = [
        sys.executable,
        "-m",
        "pytest",
        f"tests/{SUITE_FILE[suite]}",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:cov",
    ]
    # -x only for the unit tier. In the process tier a selector picks a handful of tests and
    # the run is short, while stopping at the first failure would hide which of them died.
    if suite == "unit":
        # -x only for the unit tier: it runs the whole suite, and the first failure is a
        # kill. The process tier already runs a handful of selected tests.
        argv.append("-x")
    if selector:
        argv += ["-k", selector]
    # No PYTHONSAFEPATH here. The scratch tree is ours, not a repository under review, and
    # dropping sys.path[0] is what broke the import bootstrap.
    #
    # The timeout is reported as its own outcome rather than raised. Disabling a watchdog
    # makes that watchdog's own tests run to THEIR timeouts, so slowness here is expected;
    # what must not happen is a hang being silently counted either way. A run that does not
    # finish is neither a kill nor a survivor — it is a probe nobody can conclude from.
    try:
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, env=env, check=False, timeout=RUN_TIMEOUT_SECS
        )
    except subprocess.TimeoutExpired:
        return Run(TIMED_OUT, f"the scratch suite did not finish within {RUN_TIMEOUT_SECS}s")
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

    @staticmethod
    def _verdict(mutation: Mutation, suite: str) -> tuple[str, str] | None:
        """Revert one fix and classify the result. Returns (kind, detail) or None for a kill."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _scratch_tree(Path(tmp), suite)
            target = root / mutation.path
            text = target.read_text(encoding="utf-8")
            # COUNT FIRST, with findall. `subn(..., count=1)` can never report more than one,
            # so the "exactly once" rule this file claims to enforce was never enforced at
            # all: a pattern matching five places substituted the first and reported 1.
            #
            # That is not hypothetical. The `start_new_session=True` probe matched the
            # DOCSTRING three dozen lines above the code, mutated a sentence, changed no
            # behaviour, and was duly reported as a surviving guard. The finding was a
            # harness defect wearing the costume of a code defect.
            matches = len(re.findall(mutation.pattern, text, flags=re.M))
            if matches != 1:
                # A probe that stopped matching reports a false "killed"; one that matches
                # several places mutates whichever came first. Both are the same class of lie
                # this whole file exists to catch.
                return (
                    "broken",
                    f"{mutation.name} (pattern matched {matches} times, want exactly 1)",
                )
            mutated = re.sub(mutation.pattern, mutation.replacement, text, count=1, flags=re.M)
            assert mutated != text, f"{mutation.name}: substitution changed nothing"
            target.write_text(mutated, encoding="utf-8")

            run = _run_suite(root, suite, mutation.selector)
            if run.code == 0:
                return ("survivor", mutation.name)
            if not run.executed_tests:
                # Non-zero, but the suite never reached a verdict — a syntax error, a broken
                # import, a selector matching nothing (pytest exits 5), or a timeout.
                # Counting any of those as a kill is how a harness flatters itself.
                return ("broken", f"{mutation.name} (suite reached no verdict: {run.code})")
            return None

    def _sweep(self, suite: str) -> tuple[list[str], list[str]]:
        """Revert each fix in this tier and report (survivors, broken)."""
        entries = [m for m in MUTATIONS if m.suite == suite]
        assert entries, f"no mutations in the {suite!r} tier, so this sweep proves nothing"

        def verdict(mutation: Mutation) -> tuple[str, str] | None:
            """A named function, not a lambda: strict mode cannot infer a lambda's parameter."""
            return self._verdict(mutation, suite)

        if suite == "unit":
            results = [verdict(m) for m in entries]
        else:
            with ThreadPoolExecutor(max_workers=PROCESS_WORKERS) as pool:
                results = list(pool.map(verdict, entries))

        survivors = [d for kind, d in filter(None, results) if kind == "survivor"]
        broken = [d for kind, d in filter(None, results) if kind == "broken"]
        return survivors, broken

    def test_every_reverted_fix_kills_a_unit_test(self):
        survivors, broken = self._sweep("unit")
        assert broken == [], f"mutations that proved nothing: {broken}"
        assert survivors == [], (
            "the unit suite survived these reverted fixes, so nothing guards them: "
            + "; ".join(survivors)
        )

    def test_every_reverted_fix_kills_a_process_test(self):
        """cli.py and runner.py, which the unit suite cannot reach.

        Both are driven through subprocesses, so before this tier existed they had ZERO
        entries between them while the module docstring said their guards were "swept
        manually" — a claim, not a mechanism, in the one file whose entire job is to replace
        claims about coverage with checks.
        """
        survivors, broken = self._sweep("process")
        assert broken == [], f"mutations that proved nothing: {broken}"
        assert survivors == [], (
            "the process suite survived these reverted fixes, so nothing guards them: "
            + "; ".join(survivors)
        )

    def test_the_mutated_library_is_the_one_that_ships(self):
        """When the flake sets PERSONA_REVIEW_PKG, the built package must be what is mutated.

        Otherwise this harness proves the SOURCE tree's guards can fail while the unit and
        process checks run the built package — the same "tested a different copy from the one
        that ships" defect the unit check's own negative control exists to catch. The
        source-tree fallback is for a bare `pytest` in a checkout, and this makes it
        impossible for that fallback to be silently in force under CI.
        """
        declared = os.environ.get("PERSONA_REVIEW_PKG")
        if not declared:
            pytest.skip("no PERSONA_REVIEW_PKG; running against the source tree")
        assert Path(declared) == PACKAGE_ROOT
        assert (PACKAGE_ROOT / "persona_review" / "validate.py").is_file()
        assert PACKAGE_ROOT != SRC, "PERSONA_REVIEW_PKG points back at the source tree"

    def test_every_module_is_represented_or_explicitly_exempt(self):
        """No module may quietly have no entries.

        Four did: cli.py, runner.py, providers.py and flags.py. Two now have a tier, and two
        are listed in UNMUTATED_MODULES with a written reason. The point is that adding a
        module with guards and no entries fails HERE rather than going unnoticed for a
        release, which is how the first four accumulated.
        """
        covered = {m.path.split("/")[-1] for m in MUTATIONS}
        modules = {p.name for p in (PACKAGE_ROOT / "persona_review").glob("*.py")} - {"__init__.py"}
        unexplained = sorted(modules - covered - set(UNMUTATED_MODULES))
        assert unexplained == [], (
            f"modules with no mutation entry and no stated exemption: {unexplained}. "
            "Add an entry, or add the module to UNMUTATED_MODULES with the reason."
        )
        stale = sorted(set(UNMUTATED_MODULES) - modules)
        assert stale == [], f"UNMUTATED_MODULES names modules that no longer exist: {stale}"
        both = sorted(covered & set(UNMUTATED_MODULES))
        assert both == [], f"listed as exempt but also mutated: {both}"

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
