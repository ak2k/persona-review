"""Assert the grok flags this package passes are still accepted by the installed grok.

`grok` arrives through ungated dependency bumps, so its CLI can move without any change
here naming it. That has already bitten once: the wrapper shipped `--effort max`, which grok
rejects, and every default run died before the review started.

The flags are read from `providers.GROK.argv()` — the same function that builds the real
invocation — so the two cannot drift apart. An earlier version parsed the shell script's
text, which meant the parser itself could silently stop finding the invocation, and it once
read a value out of a comment rather than the flag ten lines below it.

WHAT THIS COVERS, AND WHAT IT CANNOT
------------------------------------
grok validates its arguments in two places, and only the first is reachable offline:

  * clap, at parse time. `--permission-mode` and `--output-format` are ValueEnums, so an
    invalid value exits 2 and prints the full accepted set. `--sandbox` is looked up against
    the profile registry and an unknown profile is refused. Valueless flags are covered too:
    clap rejects one it does not know, so existence is probeable. All of this works with no
    network and no credentials.

  * the model catalog, fetched from xAI after sign-in. `--model` and `--effort` resolve
    there, so the accepted effort set is per-model and per-account. This CANNOT cover them,
    and the offline fallback accepts a superset — `max` parses clean here while a real run
    rejects it. A green run means the clap-checkable flags survived a grok bump; it says
    nothing about effort.

Run by hand (`python3 -m persona_review.flags`); no flake check drives it, because every
probe needs a real authenticated `grok` that a build sandbox has neither network nor
credentials to provide.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .providers import GROK, Invocation

CANARY = "__ce_grok_persona_canary__"

# Flags whose accepted set clap prints on rejection, so the value can be checked against
# grok's own answer rather than a list duplicated here.
ENUM_FLAGS = ("--permission-mode", "--output-format")

# Checked against grok's own accepted sets above, so they are skipped by the generic
# existence probe for value-taking flags.
_VALUE_CHECKED = frozenset({*ENUM_FLAGS, "--sandbox"})


def reference_argv() -> list[str]:
    """The real invocation, with placeholder values."""
    return GROK.argv(
        Invocation(
            model=GROK.default_model,
            effort=GROK.default_effort,
            repo=Path("."),
            prompt_file=Path("/nonexistent/ce-grok-persona-flag-check"),
            schema_text="{}",
            last_file=None,
        )
    )


def argv_value(argv: list[str], flag: str) -> str:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    sys.exit(f"FAIL: {flag} is not in the grok invocation any more -- did providers.py change?")


def valueless_flags(argv: list[str]) -> list[str]:
    """Long flags passed with no value of their own.

    Derived from the invocation rather than listed here, so a flag added to the argv builder
    is covered without touching this file — which is the drift this check exists to catch.
    """
    return [
        tok
        for i, tok in enumerate(argv)
        if tok.startswith("--") and (i + 1 >= len(argv) or argv[i + 1].startswith("--"))
    ]


def valued_flags(argv: list[str]) -> list[str]:
    return [
        tok
        for i, tok in enumerate(argv)
        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--")
    ]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["grok", *args], capture_output=True, text=True, timeout=120, check=False)


def accepted_values(flag: str) -> set[str]:
    """Ask grok for `flag`'s accepted set by feeding it a value that cannot be valid."""
    proc = run([flag, CANARY, "--help"])
    blob = proc.stdout + proc.stderr
    # Liveness: grok must actually have rejected the canary. Without this an exec failure, or
    # a future grok that quietly ignores unknown values, yields an empty set that trivially
    # "contains" nothing and the check would false-green.
    if proc.returncode != 2 or f"invalid value '{CANARY}'" not in blob:
        sys.exit(
            f"FAIL: {flag} no longer rejects an invalid value at parse time "
            f"(exit {proc.returncode}). grok's CLI contract moved; re-verify by hand.\n"
            f"{blob.strip()[:500]}"
        )
    match = re.search(r"\[possible values: ([^\]]+)\]", blob)
    if not match:
        sys.exit(f"FAIL: {flag} rejected the canary but printed no possible-values list.")
    return {v.strip() for v in match.group(1).split(",")}


def main() -> int:
    argv = reference_argv()

    for flag in ENUM_FLAGS:
        want = argv_value(argv, flag)
        allowed = accepted_values(flag)
        if want not in allowed:
            sys.exit(
                f"FAIL: this package passes `{flag} {want}`, which grok no longer accepts. "
                f"Accepted: {', '.join(sorted(allowed))}."
            )
        print(f"ok: {flag} {want}")

    # --sandbox is not a ValueEnum; an unknown profile is refused by the profile registry.
    # --help exits before the sandbox is applied, so these probes must be real invocations.
    # A prompt file that cannot exist makes them fail in ~30ms, after the sandbox resolves
    # but before anything needs network or credentials.
    profile = argv_value(argv, "--sandbox")
    probe = ["--prompt-file", "/nonexistent/ce-grok-persona-flag-check", "-m", GROK.default_model]
    canary = run(["--sandbox", CANARY, *probe])
    if f"'{CANARY}' not found" not in canary.stdout + canary.stderr:
        sys.exit(
            "FAIL: --sandbox no longer refuses an unknown profile, so this check cannot "
            f"prove `--sandbox {profile}` is real.\n{(canary.stdout + canary.stderr).strip()[:500]}"
        )
    real = run(["--sandbox", profile, *probe])
    if f"'{profile}' not found" in real.stdout + real.stderr:
        sys.exit(f"FAIL: this package passes `--sandbox {profile}`, which grok rejects.")
    print(f"ok: --sandbox {profile}")

    # Valueless flags carry no enum to interrogate, so existence is the whole contract: grok
    # must reject one it does not know (liveness), and accept each one passed.
    if run([f"--{CANARY}", "--help"]).returncode == 0:
        sys.exit("FAIL: grok accepts an unknown valueless flag, so probing proves nothing.")
    valueless = valueless_flags(argv)
    if not valueless:
        sys.exit(
            "FAIL: no valueless flags derived from the grok invocation. An empty probe set "
            "would green this check while covering nothing."
        )
    for flag in valueless:
        proc = run([flag, "--help"])
        if proc.returncode != 0:
            sys.exit(
                f"FAIL: this package passes `{flag}`, which grok no longer accepts "
                f"(exit {proc.returncode}).\n{(proc.stdout + proc.stderr).strip()[:500]}"
            )
        print(f"ok: {flag}")

    # Value-taking flags need the same existence probe. Their VALUES are not checked here:
    # --model/--effort resolve against the authenticated catalog, and the rest were checked
    # above against grok's own accepted sets. A benign literal keeps the probe offline.
    for flag in valued_flags(argv):
        if flag in _VALUE_CHECKED:
            continue
        proc = run([flag, "ce-grok-persona-flag-check", "--help"])
        if proc.returncode != 0:
            sys.exit(
                f"FAIL: this package passes `{flag} <value>`, which grok no longer accepts "
                f"(exit {proc.returncode}).\n{(proc.stdout + proc.stderr).strip()[:500]}"
            )
        print(f"ok: {flag} <value>")

    print("note: --model/--effort need an authenticated model catalog and are NOT covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
