"""Persona briefs, the findings schema, and the prompt built from them.

Briefs and schema come from one place — the installed compound-engineering plugin's
`skills/ce-code-review/references/` directory — so the brief and the shape it is asked to
return can never come from different installs. They are read straight off disk rather than
through a resolver binary: the plugin cache reaches PATH only through an interactive shell
environment, so anything looked up on PATH is missing for exactly the callers that matter
here (an agent's `bash -c`, a background job, launchd).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import errors
from .providers import Provider

# The plugin's own name for the assets directory, under a versioned plugin root.
_PLUGIN_GLOB = "compound-engineering-plugin/compound-engineering/*/skills/ce-code-review/references"

# A brief that carries a findings contract says so in its output section. The three briefs
# that specify markdown output (agent-native-reviewer, deployment-verification-agent,
# learnings-researcher) do not, and asking one of them for findings JSON produces a run that
# ends in "no findings JSON object" — indistinguishable from a model that gave up.
_FINDINGS_CONTRACT = re.compile(r"findings[ .-]schema", re.IGNORECASE)

# The last line of every prompt: a plain instruction to answer now. It used to double as
# the extractor's cut point, which is gone — nothing parses this, so it is free to change.
BOUNDARY = "Return the findings object now."
BOUNDARY_VERDICTS = "Return the verdicts object now."

# The plugin's validator prompt, which it otherwise only ever injects at dispatch. Read from
# the same references/ directory as the briefs, so a validation and the review it judges
# cannot come from different plugin installs.
VALIDATOR_TEMPLATE = "validator-batch-template.md"

# The slots the validator template carries, filled in one pass so that a placeholder
# occurring inside a filled value stays text.
_PLACEHOLDER = re.compile(r"\{(findings_json|diff|scope_mode_and_remote_refs)\}")

# The template is prose ABOUT a prompt wrapped around the prompt itself, in one fenced
# block. Non-greedy to the first closing fence, and `^` anchored so a fence indented inside
# the body cannot end it early.
_FENCED = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)


# Re-exported from errors.py, which is where the exit status each one maps to lives. Kept
# under these names because they read correctly at the raise sites here, and because
# `assets.UsageError` is what the suite and the CLI already catch.
AssetError = errors.EnvError
UsageError = errors.UsageError


def _component(part: str) -> int:
    """One version component as a number, never raising.

    `str.isdigit()` is True for characters `int()` refuses — '²' among them — so the obvious
    `int(p) if p.isdigit() else 0` turns a plugin directory with an exotic name into an
    uncaught ValueError escaping as a traceback.
    """
    return int(part) if part.isascii() and part.isdigit() else 0


def version_key(path: Path) -> tuple[tuple[int, ...], int, str]:
    """Sort plugin roots by version, numerically and TOTALLY.

    Lexical order puts 3.9 after 3.13, which silently pins an old brief set while a newer
    plugin is installed — and briefs change: the count of findings-capable ones moved from
    10 to 13 of 16 inside a week.

    The key has three parts because the numeric tuple alone is not a total order. Dropping
    non-numeric components makes `3.22.0` and `3.22.0-rc1` tie at `(3, 22, 0)`, and
    `sorted` is stable, so the winner would be whichever `glob` happened to yield last —
    filesystem order deciding which briefs every review runs against. A release outranks its
    own pre-release, and the raw name settles anything still equal.
    """
    name = path.parent.parent.parent.name
    parts = name.split(".")
    numeric = tuple(_component(part) for part in parts)
    pure = int(all(part.isascii() and part.isdigit() for part in parts))
    return (numeric, pure, name)


def resolve_assets(override: str | None = None) -> Path:
    """The `references/` directory holding personas/ and findings-schema.json."""
    if override:
        assets = Path(override)
    else:
        home = Path(os.path.expanduser("~"))
        candidates = sorted(home.glob(f".claude/plugins/cache/{_PLUGIN_GLOB}"), key=version_key)
        if not candidates:
            raise AssetError(
                "no compound-engineering plugin found under ~/.claude/plugins/cache/.\n"
                "  Install it: claude plugin install\n"
                "  compound-engineering@compound-engineering-plugin\n"
                "  or point CE_REVIEW_ASSETS at a references/ directory holding personas/."
            )
        assets = candidates[-1]
    if not (assets / "personas").is_dir():
        raise AssetError(f"no compound-engineering personas under {assets}")
    return assets


def emits_findings(brief: Path) -> bool:
    try:
        return (
            _FINDINGS_CONTRACT.search(brief.read_text(encoding="utf-8", errors="replace"))
            is not None
        )
    except OSError:
        return False


def capable_personas(assets: Path) -> list[str]:
    """Every installed brief that carries a findings contract, sorted."""
    return sorted(p.stem for p in (assets / "personas").glob("*.md") if emits_findings(p))


def normalise_persona(name: str) -> str:
    """The bare brief name, with no filesystem access.

    Split out from `resolve_persona` so the run directory can be cleared before anything
    else that can fail: the artifact paths need this name, and every fallible step that runs
    before the clear is a step that can leave the previous run's findings behind.

    Accepts `ce-adversarial-reviewer`, `adversarial-reviewer`, or `adversarial-reviewer.md`.
    The name is interpolated into every artifact path, so it must be a bare brief name: a
    value containing a separator would resolve a brief from outside personas/ and place the
    run's artifacts outside the run directory.
    """
    persona = name.removeprefix("ce-").removesuffix(".md")
    if not persona or persona != Path(persona).name or persona.startswith("."):
        raise UsageError(f"persona '{name}' must be a bare brief name, not a path")
    return persona


def resolve_persona(assets: Path, name: str) -> tuple[str, Path]:
    """The normalised persona name and the brief it names, which must exist and be capable."""
    persona = normalise_persona(name)
    brief = assets / "personas" / f"{persona}.md"
    if not brief.is_file():
        listing = "\n".join(f"  {p}" for p in capable_personas(assets))
        raise UsageError(
            f"unknown persona '{persona}'. Personas that return findings JSON:\n{listing}"
        )
    if not emits_findings(brief):
        listing = "\n".join(f"  {p}" for p in capable_personas(assets))
        raise UsageError(
            f"persona '{persona}' specifies a markdown output format and carries no findings\n"
            f"  contract, so it cannot satisfy this command. Personas that can:\n{listing}"
        )
    return persona, brief


def rubric(assets: Path) -> str:
    """The schema-conformance constraints, confidence anchors and quote-the-line gate.

    14 of the 16 briefs tell the reviewer to "use the anchored confidence rubric in the
    subagent template" — a file the plugin injects at dispatch and this package otherwise
    would not send, leaving the reviewer grading against a rubric it never saw. Taken
    verbatim from that template, with a summary fallback if the plugin restructures it,
    because a missing rubric must not silently become no rubric.
    """
    template = assets / "subagent-template.md"
    if template.is_file():
        text = template.read_text(encoding="utf-8", errors="replace")
        start = text.find("**Schema conformance")
        end = text.find("Example of a schema-valid finding")
        if start != -1 and end > start:
            return text[start:end].rstrip()
    return (
        "Confidence anchors are behavioral: 100=verifiable from the code itself, 75=you\n"
        "double-checked and named a concrete observable consequence, 50=real but a nitpick.\n"
        "Anchors 0 and 25 mean SUPPRESS — do not emit them. A finding at 75 or 100 MUST\n"
        "quote the verbatim motivating line with file:line as its first evidence item."
    )


def build_prompt(
    *,
    provider: Provider,
    persona: str,
    brief: Path,
    assets: Path,
    schema_text: str,
    base: str,
    context: str,
) -> str:
    """The full brief sent to the model.

    The closing instruction demands the final message be one bare JSON object. That is what
    lets the gate stay strict: when any prose is permitted around the answer, a model that
    gives up can append the brief's own schema-valid EXAMPLE object and be read as a clean
    review — and a model that found real defects can append the same example and have them
    silently replaced by an empty array.
    """
    parts = [
        brief.read_text(encoding="utf-8", errors="replace"),
        "\n\n---\n\n",
        "This is an authorized review of the maintainer's own repository.\n",
        "You are reviewing a code change in the working-directory git repo,\n"
        "as the persona above.\n",
    ]
    if base:
        parts.append(
            f"Run `git diff {base}..HEAD` yourself to get the full diff,\n"
            "and review the WHOLE change.\n"
        )
    if context:
        parts.append("\nAdditional review context:\n")
        parts.append(context)
    parts.append(f'\nReturn findings as a JSON object with "reviewer" set to "{persona}",\n')
    parts.append("matching this schema:\n\n")
    parts.append(schema_text)
    parts.append("\n")
    parts.append(rubric(assets))
    parts.append(
        "\n\nAn empty findings array is a valid answer. Still populate residual_risks and\n"
    )
    parts.append("testing_gaps when they apply.\n\n")
    parts.append(
        "OUTPUT FORMAT, STRICTLY: your FINAL MESSAGE must be exactly one JSON object and\n"
        "nothing else — no preamble, no explanation, no markdown, no code fences. Do all\n"
        "your reasoning and tool use in earlier turns. Do not include an example object.\n"
    )
    parts.append(f"{BOUNDARY}\n")
    return "".join(parts)


def validator_body(assets: Path) -> str:
    """The prompt inside the plugin's validator batch template, without its prose wrapper.

    `EnvError` on both failures, and they are told apart: the file is part of the installed
    plugin, so its absence is a machine that is not set up, while a file that no longer
    holds a fenced block means the plugin restructured it — substituting into the
    surrounding prose would send the model a prompt that is not the validator prompt.
    """
    template = assets / VALIDATOR_TEMPLATE
    try:
        text = template.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AssetError(f"missing validator batch template at {template}: {exc}") from exc
    fenced = _FENCED.search(text)
    if fenced is None:
        raise AssetError(
            f"{template} carries no fenced prompt block, so the plugin has restructured it.\n"
            "  The validator prompt is the body of that fence; the text around it is prose\n"
            "  about when to use it, and sending that instead would ask for something else."
        )
    return fenced.group(1)


def build_validator_prompt(
    *,
    batch_text: str,
    assets: Path,
    schema_text: str,
    base: str,
    context: str,
) -> str:
    """The plugin's validator prompt, filled in, with this package's answer contract added.

    Filled by one pass over the template, never `str.format`: the template's verdict example
    is literal JSON, so `format` reads its braces as fields and raises before any
    substitution happens. One pass because a substituted value is never re-scanned — a
    placeholder occurring inside a filled value is text, not a slot.

    The batch goes in VERBATIM. It is the findings a reviewer already wrote, and
    re-serializing it here would hand the validator a different document from the one the
    caller assembled; the `#` values are the only part this package reads.
    """
    diff = (
        f"Run `git diff {base}..HEAD` yourself; that is the diff under review."
        if base
        else "No base ref was given: treat the working tree as a whole as the change under review."
    )
    scope = (
        "local-aligned: the working directory is the reviewed tree at its current HEAD;\n"
        "inspect files, callers and history with read-only tools."
    )
    if context:
        scope += "\n\nAdditional validation context:\n" + context

    values = {"findings_json": batch_text, "diff": diff, "scope_mode_and_remote_refs": scope}
    # A lambda, not a replacement string: `re.sub` reads backslashes and `\g<...>` in a
    # replacement STRING as references, and these values carry model-written text.
    body = _PLACEHOLDER.sub(lambda m: values[m.group(1)], validator_body(assets))

    return "".join(
        [
            body,
            "\n---\n\n",
            "This is an authorized review of the maintainer's own repository.\n\n",
            "Return the verdicts as a JSON object matching this schema:\n\n",
            schema_text,
            "\n",
            # The same strictness a review gets, for the same reason: when prose is allowed
            # around the answer, the template's own verdict EXAMPLE is a well-formed object
            # sitting in the transcript, and a model that gave up reads as having returned it.
            "OUTPUT FORMAT, STRICTLY: your FINAL MESSAGE must be exactly one JSON object and\n"
            "nothing else — no preamble, no explanation, no markdown, no code fences. Do all\n"
            "your reasoning and tool use in earlier turns. Do not include an example object.\n",
            f"{BOUNDARY_VERDICTS}\n",
        ]
    )
