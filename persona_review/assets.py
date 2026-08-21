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

from .providers import Provider

# The plugin's own name for the assets directory, under a versioned plugin root.
_PLUGIN_GLOB = "compound-engineering-plugin/compound-engineering/*/skills/ce-code-review/references"

# A brief that carries a findings contract says so in its output section. The three briefs
# that specify markdown output (agent-native-reviewer, deployment-verification-agent,
# learnings-researcher) do not, and asking one of them for findings JSON produces a run that
# ends in "no findings JSON object" — indistinguishable from a model that gave up.
_FINDINGS_CONTRACT = re.compile(r"findings[ .-]schema", re.IGNORECASE)

# The last line of every prompt. The transcript-mode extractor cuts on it to tell the
# runner's echo of the prompt from the model's own answer, so it must be stable and it must
# be the final line build_prompt emits.
BOUNDARY = "Return the findings object now."


class AssetError(Exception):
    """The plugin assets are missing or unusable. Not the user's argument error."""


class UsageError(Exception):
    """The caller asked for something that cannot be done. Names what to do instead."""


def version_key(path: Path) -> tuple[int, ...]:
    """Sort plugin roots by version numerically.

    Lexical order puts 3.9 after 3.13, which silently pins an old brief set while a newer
    plugin is installed — and briefs change: the count of findings-capable ones moved from
    10 to 13 of 16 inside a week.
    """
    name = path.parent.parent.parent.name
    return tuple(int(part) if part.isdigit() else 0 for part in name.split("."))


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


def resolve_persona(assets: Path, name: str) -> tuple[str, Path]:
    """Normalise a persona name and return it with its brief path.

    Accepts `ce-adversarial-reviewer`, `adversarial-reviewer`, or `adversarial-reviewer.md`.
    The name is interpolated into every artifact path, so it must be a bare brief name: a
    value containing a separator would resolve a brief from outside personas/ and place the
    run's artifacts outside the run directory.
    """
    persona = name.removeprefix("ce-").removesuffix(".md")
    if not persona or persona != Path(persona).name or persona.startswith("."):
        raise UsageError(f"persona '{name}' must be a bare brief name, not a path")

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
