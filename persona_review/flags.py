#!/usr/bin/env python3
"""Assert the grok flags bin/ce-grok-persona passes are still accepted.

`grok` arrives via ungated llm-agents digest bumps, so its CLI can move without any
PR here naming it. That already bit once: the wrapper shipped `--effort max`, which
grok rejects, and every default run died before the review started.

WHAT THIS COVERS, AND WHAT IT CANNOT
------------------------------------
grok validates its arguments in two places, and only the first is reachable offline:

  * clap, at parse time. `--permission-mode` and `--output-format` are ValueEnums, so
    an invalid value exits 2 and prints the full accepted set. `--sandbox` is looked up
    against the profile registry and an unknown profile is refused. Valueless flags
    (`--verbatim`) are covered too: clap rejects an unknown one, so the wrapper's own set
    can be probed for existence. All of these are checkable in a Nix build sandbox with
    no network and no credentials.

  * the model catalog, fetched from xAI after sign-in. `--model` and `--effort` resolve
    there (`resolve_effort_for_model`), so the accepted effort set is per-model and
    per-account. This check CANNOT cover them. Worse, the offline fallback accepts a
    superset: with no catalog loaded, grok falls back to `parse_canonical_effort_token`,
    and `max` IS a canonical token -- so the exact value that broke the wrapper parses
    clean here. A green run means the three clap-checkable flags survived a grok bump;
    it says nothing about effort. Only an authenticated end-to-end run covers that.

Values come from the wrapper itself, never a second copy, so the two cannot drift apart.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CANARY = "__ce_grok_persona_canary__"

# Flags whose accepted set clap prints on rejection, so the wrapper's value can be
# checked against grok's own answer rather than a list duplicated here.
ENUM_FLAGS = ("--permission-mode", "--output-format")


def resolve_wanted(text: str) -> dict[str, str]:
    """Every flag value this check will assert, resolved from the invocation.

    One path, so `main()` cannot quietly go back to reading the file: the values are
    taken from the parsed argv here, and the unit suite exercises this function against a
    wrapper whose comment deliberately disagrees with its flag.
    """
    argv = grok_argv(text)
    return {flag: argv_value(argv, flag) for flag in (*ENUM_FLAGS, "--sandbox")}


def argv_value(argv: list[str], flag: str) -> str:
    """The literal value the wrapper PASSES for `flag`, read from the invocation.

    Never from a regex over the whole file: the wrappers explain their flags in comments,
    so a file-wide search returns whatever the prose says. That version read the comment
    ABOVE the grok invocation rather than the flag inside it, and the two agreeing was
    luck -- retarget the flag while the comment stands and this check would certify the
    documentation.
    """
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    sys.exit(f"FAIL: {flag} not found in the wrapper's grok invocation -- did it change?")


# The invocation ends at the shell, not at the end of the line. Without this the
# redirect tail (`>"$EVENTS_FILE" 2>"$ERR_FILE" || STATUS=$?`) is tokenised as argv, and a
# valueless flag in last position takes the redirect as its successor -- so it looks
# value-taking, drops out of coverage, and the probe loop silently becomes a no-op.
SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", ">", ">>", "2>", "&"})

# Set membership is not enough: whitespace tokenisation yields `>"$EVENTS_FILE"` as ONE
# token, not a bare `>`, so a redirect with its target attached matched nothing and the
# tail stayed in argv. Anything starting with a redirect (optionally fd-prefixed) ends it.
REDIRECT = re.compile(r"^\d*[<>]")


def is_operator(tok: str) -> bool:
    return tok in SHELL_OPERATORS or REDIRECT.match(tok) is not None


def grok_argv(text: str) -> list[str]:
    """The wrapper's grok invocation, tokenised, backslash continuations joined."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("grok "):
            continue
        block = [line]
        while block[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            block.append(lines[i])
        argv = " ".join(b.rstrip().rstrip("\\") for b in block).split()
        for n, tok in enumerate(argv):
            if is_operator(tok):
                return argv[:n]
        return argv
    sys.exit("FAIL: no `grok ...` invocation found in the wrapper.")


def _is_terminator(tok: str | None) -> bool:
    return tok is None or tok.startswith("--") or is_operator(tok)


def valueless_flags(argv: list[str]) -> list[str]:
    """Long flags the wrapper passes with no value of their own.

    A flag whose successor is another flag, a shell operator, or nothing takes no value.
    Derived from the wrapper rather than listed here, so a flag added to the invocation
    is covered without touching this file -- which is the failure this check exists to
    prevent.
    """
    return [
        tok
        for i, tok in enumerate(argv)
        if tok.startswith("--") and _is_terminator(argv[i + 1] if i + 1 < len(argv) else None)
    ]


def valued_flags(argv: list[str]) -> list[tuple[str, str]]:
    """Long flags the wrapper passes WITH a value, as (flag, value) pairs.

    These need an existence probe too. `--cwd` arrived in the same change that added the
    valueless probe and was invisible to it, which is exactly the drift this file exists
    to catch.
    """
    return [
        (tok, argv[i + 1])
        for i, tok in enumerate(argv)
        if tok.startswith("--") and i + 1 < len(argv) and not _is_terminator(argv[i + 1])
    ]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["grok", *args], capture_output=True, text=True, timeout=120, check=False)


def accepted_values(flag: str) -> set[str]:
    """Ask grok for `flag`'s accepted set by feeding it a value that cannot be valid."""
    proc = run([flag, CANARY, "--help"])
    blob = proc.stdout + proc.stderr
    # Liveness: grok must actually have rejected the canary. Without this an exec
    # failure, or a future grok that quietly ignores unknown values, yields an empty
    # set that trivially "contains" nothing and the check would false-green.
    if proc.returncode != 2 or f"invalid value '{CANARY}'" not in blob:
        sys.exit(
            f"FAIL: {flag} no longer rejects an invalid value at parse time "
            f"(exit {proc.returncode}). grok's CLI contract moved; re-verify the wrapper "
            f"by hand.\n{blob.strip()[:500]}"
        )
    m = re.search(r"\[possible values: ([^\]]+)\]", blob)
    if not m:
        sys.exit(f"FAIL: {flag} rejected the canary but printed no possible-values list.")
    return {v.strip() for v in m.group(1).split(",")}


def main() -> int:
    # Default to the wrapper next to this package, not a path relative to the caller's
    # cwd. Run by hand (`python3 -m persona_review.flags [wrapper]`) and not by any flake
    # check: every probe needs a real `grok` binary, which a Nix build sandbox has no
    # network or credentials to provide.
    default = Path(__file__).resolve().parent.parent / "bin" / "ce-grok-persona"
    wrapper = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    text = wrapper.read_text()
    argv = grok_argv(text)
    wanted = resolve_wanted(text)

    for flag in ENUM_FLAGS:
        want = wanted[flag]
        allowed = accepted_values(flag)
        if want not in allowed:
            sys.exit(
                f"FAIL: ce-grok-persona passes `{flag} {want}`, which grok no longer "
                f"accepts. Accepted: {', '.join(sorted(allowed))}."
            )
        print(f"ok: {flag} {want}")

    # --sandbox is not a ValueEnum; an unknown profile is refused by the profile
    # registry instead. Prove the canary is refused (liveness), then that the
    # wrapper's profile is not refused the same way.
    # --help exits before the sandbox is ever applied, so these probes must be real
    # invocations. A prompt file that cannot exist makes them fail in ~30ms, after the
    # sandbox is resolved but before anything needs the network or credentials.
    profile = wanted["--sandbox"]
    probe = [
        "--prompt-file",
        "/nonexistent/ce-grok-persona-flag-check",
        "-m",
        "grok-4.6",
    ]
    canary = run(["--sandbox", CANARY, *probe])
    if f"'{CANARY}' not found" not in canary.stdout + canary.stderr:
        sys.exit(
            "FAIL: --sandbox no longer refuses an unknown profile, so this check "
            f"cannot prove `--sandbox {profile}` is real.\n"
            f"{(canary.stdout + canary.stderr).strip()[:500]}"
        )
    real = run(["--sandbox", profile, *probe])
    if f"'{profile}' not found" in real.stdout + real.stderr:
        sys.exit(f"FAIL: ce-grok-persona passes `--sandbox {profile}`, which grok rejects.")
    print(f"ok: --sandbox {profile}")

    # Valueless flags carry no enum to interrogate, so existence is the whole contract:
    # grok must reject one it does not know (liveness), and accept each one the wrapper
    # passes. Without the liveness half, a grok that ignored unknown flags would green.
    if run([f"--{CANARY}", "--help"]).returncode == 0:
        sys.exit(
            "FAIL: grok accepts an unknown valueless flag, so probing the wrapper's "
            "valueless flags proves nothing."
        )
    valueless = valueless_flags(argv)
    if not valueless:
        sys.exit(
            "FAIL: no valueless flags derived from the wrapper's grok invocation. "
            "Either the invocation moved or the parser stopped seeing it; an empty "
            "probe set would green this check while covering nothing."
        )
    for flag in valueless:
        proc = run([flag, "--help"])
        if proc.returncode != 0:
            sys.exit(
                f"FAIL: ce-grok-persona passes `{flag}`, which grok no longer accepts "
                f"(exit {proc.returncode}).\n{(proc.stdout + proc.stderr).strip()[:500]}"
            )
        print(f"ok: {flag}")

    # Value-taking flags need the same existence probe. Their VALUES are not checked
    # here: --model/--effort resolve against the authenticated catalog (see the module
    # docstring), and --sandbox/--permission-mode/--output-format were already checked
    # above against grok's own accepted sets. A benign literal keeps the probe offline.
    checked = {*ENUM_FLAGS, "--sandbox"}
    for flag, _ in valued_flags(argv):
        if flag in checked:
            continue
        proc = run([flag, "ce-grok-persona-flag-check", "--help"])
        if proc.returncode != 0:
            sys.exit(
                f"FAIL: ce-grok-persona passes `{flag} <value>`, which grok no longer "
                f"accepts (exit {proc.returncode}).\n"
                f"{(proc.stdout + proc.stderr).strip()[:500]}"
            )
        print(f"ok: {flag} <value>")

    print("note: --model/--effort need an authenticated model catalog and are NOT covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
