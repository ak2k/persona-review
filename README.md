# persona-review

Run a [compound-engineering](https://github.com/EveryInc/compound-engineering-plugin) reviewer
persona through xAI's `grok` or OpenAI's `codex`, and get **schema-valid findings** back — not a
transcript.

Built for a calling agent. One invocation costs the caller a single line of stdout:

```console
$ ce-grok-persona adversarial-reviewer -b origin/main
ce-persona: 5 findings (1 P1, 3 P2, 1 P3) -> /tmp/.../adversarial-reviewer-grok.json
```

The model's reasoning, its tool calls, and the full event stream go to files. A 1 MB event stream
and a 100-byte findings artifact is a real measurement from a real run — that ratio is the point.

## Why this exists

Compound-engineering already runs a cross-model adversarial pass, and if that is what you want, use
it: `/ce-code-review` does this and merges the results. This exists for the three things it cannot
do — any persona (its worker hardcodes one), any model and reasoning effort (it pins one per
provider), and **streaming on the grok route** (it constrains decoding in a way that precludes it).

## The contract

| Tier | Cost | What you get |
|------|------|--------------|
| 0 | one line + exit status | counts by severity, artifact path. A gating caller reads nothing else. |
| 1 | ~20 tokens/finding | `ce-persona-findings <artifact>` — severity, `file:line`, title, confidence, and the one quoted line that motivates it |
| 2 | per finding | `ce-persona-findings <artifact> --show N` — why it matters, full evidence, suggested fix |

Exit status is the verdict:

| Exit | Meaning |
|------|---------|
| `0` | schema-valid findings (an empty findings array is a valid answer) |
| `1` | the runner failed, or the answer was not schema-valid findings |
| `2` | usage error: no persona, a persona that cannot return findings, a base ref that does not resolve |
| `78` | the review is over `CE_PERSONA_MAX_PROMPT_TOKENS` — refused, never summarized |

The budget counts the prompt **plus** `git diff <base>..HEAD`, because the prompt does not contain
the diff — it tells the agent to fetch it, and that is the input that overruns a context.

## How the findings are extracted

Not by scanning prose. That heuristic produced two silent false-passes — the persona brief's own
example object, echoed back inside a replayed prompt, validating as a completed review.

- **grok** — `--json-schema` combined with `--output-format streaming-messages-json`. This composes,
  despite grok's help text saying the schema implies non-streaming JSON: the terminal `result` event
  carries `structured_output` already parsed, *and* the stream still grows for liveness.
- **codex** — `-o` writes only the agent's final message. Not `--output-schema`: OpenAI's strict mode
  rejects the plugin's schema outright and would force every optional field, changing what a finding
  means.

A transcript-scanning mode survives as a fallback, with both of its hard-won anchors intact.

## Requirements

- `grok` or `codex` on PATH, **already logged in**. They are deliberately not dependencies of this
  package: it uses the subscription you already have rather than an API key of its own.
- The compound-engineering plugin installed, for the persona briefs and the findings schema. Point
  `$CE_REVIEW_ASSETS` at a `references/` directory to override.
- `python3` 3.11 or newer (the wrappers set `PYTHONSAFEPATH`).
- `bash` 4.4 or newer. macOS ships 3.2, where expanding an empty array under `set -u` aborts — and
  with an `EXIT` trap installed, that abort exits **0**. The wrappers refuse to run on it rather
  than risk reporting success for a review that never happened. The Nix package patches the shebang,
  so this only affects a source checkout.
- `git`, for the diff the size preflight weighs.

## Install

```nix
inputs.persona-review.url = "github:ak2k/persona-review";
# then add inputs.persona-review.packages.${system}.default to your profile
```

## Environment

| Variable | Effect |
|----------|--------|
| `CE_REVIEW_ASSETS` | the plugin `references/` directory to read briefs and schema from |
| `CE_PERSONA_RUN_DIR` | where artifacts land; defaults to a fresh temp dir |
| `CE_PERSONA_MAX_PROMPT_TOKENS` | prompt budget, default 80000. Over it, the run is refused — never summarized, because a review of a summarized diff produces findings nobody can check |
| `PERSONA_REVIEW_PYTHONPATH` | root containing `persona_review/`; set by the packaged build |

## Development

```console
nix flake check   # package build, lint, strict types, unit tests, packaged-wrapper process tests
nix develop
```

The unit suite asserts the **authored** wrappers against the **packaged** library; the process suite
drives the **packaged** wrappers against stub runners. Both matter, and they are not the same thing —
`makeWrapper` shims carry none of the source text the invariants check. The unit suite asserts which
library it actually imported, because the earlier version put its own source root ahead of
`PYTHONPATH` and passed with every packaged module replaced by `raise RuntimeError`.

Type checking is `basedpyright` in **strict** mode with no baseline, over `persona_review/` and
`tests/`. Four rules are off, in `pyrightconfig.json` and with reasons: every value here originates
in `json.load` of a schema this package does not own, so `reportUnknown*` fires on the whole program
and the usual remedy — a `TypedDict` — would assert a shape the compound-engineering plugin is free
to change. Everything else stays on. The `types` check ends with a **negative control**: it injects
`return x + None` into `persona_review/validate.py` and fails if the checker accepts it, because
this check once passed while analysing exactly one file and none of the code that ships.

## Licence

Apache-2.0. Persona briefs and the findings schema are read at run time from the
compound-engineering plugin, which is MIT — see `NOTICE`.
