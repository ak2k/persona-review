# persona-review

Run a [compound-engineering](https://github.com/EveryInc/compound-engineering-plugin) reviewer
persona through xAI's `grok` or OpenAI's `codex`, and get **schema-valid findings** back — not a
transcript.

Built for a calling agent. One invocation costs the caller a single line of stdout:

```console
$ ce-grok-persona adversarial-reviewer -b origin/main
ce-grok-persona: 5 findings (1 P1, 3 P2, 1 P3) -> /tmp/.../adversarial-reviewer-grok.json
```

The model's reasoning, its tool calls, and the full event stream go to files. A 1.4 MB event
stream and a 10 KB findings artifact is a real measurement from a real run — that ratio is the
point.

## Why this exists

Compound-engineering already runs a cross-model adversarial pass, and if that is what you want, use
it: `/ce-code-review` does this and merges the results. This exists for the three things it cannot
do — any persona (its worker hardcodes one), any model and reasoning effort (it pins one per
provider), and **streaming on the grok route** (it constrains decoding in a way that precludes it).

## The contract

| Tier | Cost | What you get |
|------|------|--------------|
| 0 | one line + exit status | counts by severity, artifact path. A gating caller reads nothing else. |
| 1 | ~20 tokens/finding | `ce-persona-findings <artifact>` — severity, `file:line`, title, confidence, and the one quoted line that motivates it. P0/P1 only; `--all` for every severity |
| 2 | per finding | `ce-persona-findings <artifact> --show N` — why it matters, full evidence, suggested fix |
| — | whole artifact | `ce-persona-findings <artifact> --json` — the raw object, unchanged and unfenced, for a programmatic caller |

`N` in `--show N` is the number rendered as `#N` in the listing. Numbering follows the
artifact's own order; the listing is *displayed* most-severe-first, so `#1` is not
necessarily the top row.

**Exit status is the verdict.** For `ce-grok-persona` / `ce-codex-persona`:

| Exit | Meaning |
|------|---------|
| `0` | schema-valid findings (an empty findings array is valid) |
| `1` | the answer was not schema-valid findings — the gate refused |
| `2` | usage error: bad arguments, unknown or markdown-only persona, bad `-C`, unresolvable `-b`, malformed `CE_PERSONA_*` value |
| `3` | environment error: the runner, `git` or the plugin assets are missing, or `CE_PERSONA_RUN_DIR` cannot be created |
| `4` | the runner itself exited non-zero |
| `5` | idle or hard timeout; the run was killed and partial output kept |
| `78` | over `CE_PERSONA_MAX_PROMPT_TOKENS` — refused, never summarized |

`ce-persona-findings` uses the same vocabulary, narrowed to what it can hit: `0` rendered,
`1` the file is unreadable or is not a findings artifact, `2` usage error.

Everything `ce-persona-findings` renders is wrapped in `BEGIN/END UNTRUSTED MODEL OUTPUT` with a
per-run nonce. It is text a model wrote about a repository it read, being handed to another agent
as *its* input; the fence is what lets the consumer tell data from instructions. `--json` is
unfenced, because a programmatic caller parses it rather than reading it.

## How the findings are extracted

Not by scanning prose. That heuristic produced three silent false-passes.

- **grok** — `--json-schema` combined with `--output-format streaming-messages-json`. This composes,
  despite grok's help text saying the schema implies non-streaming JSON: the terminal `result` event
  carries `structured_output` already parsed, *and* the stream still grows for liveness. The run's
  own terminal status (`is_error`, `subtype`, `stop_reason`) is checked before its answer is
  believed, because schema-constrained decoding means a truncated run still returns a well-formed
  empty findings array.
- **codex** — `-o` writes only the agent's final message, and the prompt requires that message to be
  exactly one JSON object. Not `--output-schema`: OpenAI's strict mode rejects the plugin's schema
  outright and would force every optional field, changing what a finding means.

The strictness is load-bearing. When prose is allowed around the answer, a model that gives up can
append the brief's own schema-valid **example** object and read as a clean review — and a model that
found real defects can append the same example and have them silently replaced by an empty array.
Both were demonstrated by independent reviewers against a version that permitted it.

## Requirements

- `grok` or `codex` on PATH, **already logged in**. They are deliberately not dependencies of this
  package: it uses the subscription you already have rather than an API key of its own.
- The compound-engineering plugin installed, for the persona briefs and the findings schema. Point
  `$CE_REVIEW_ASSETS` at a `references/` directory to override.
- `python3` 3.12 or newer.
- `git`, only when `-b` is passed.

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
| `CE_PERSONA_MAX_PROMPT_TOKENS` | budget, default 80000. Counts the prompt **plus** `git diff <base>..HEAD` |
| `CE_PERSONA_IDLE_SECS` | kill after this long with no output, default 600. Must be **> 0** |
| `CE_PERSONA_HARD_SECS` | kill after this long overall, default 2400. Must be **> 0** |

Every value is parsed and validated before any work happens, and a bad one exits `2` naming the
variable. Two rules are worth knowing because they refuse things you might expect to work:

- **A watchdog cannot be switched off.** `0` is rejected, not treated as "no timeout". An unwatched
  run is a full-effort model run that nothing will stop and nothing will read. Set a large value if
  you want a long one.
- **An unrecognised `CE_PERSONA_*` name is an error.** `CE_PERSONA_IDEL_SECS=30` would otherwise be
  ignored in silence while the real idle timeout stayed at its default — a misconfiguration that
  looks exactly like a working one.

**The budget only means something with `-b`.** Without a base ref the prompt does not contain the
change and does not name a range — the model picks its own scope — so there is genuinely nothing to
weigh, and the budget bounds the prompt alone. That is a real limit, stated rather than papered over
with a guessed base.

## Concurrency

Artifact paths are deterministic and `CE_PERSONA_RUN_DIR` is reusable, which together mean two runs
of the same persona through the same provider in the same directory address the same files. A run
takes an exclusive lock on its artifact stem; a second one **exits `2` rather than interleaving**.
Wait, or point `CE_PERSONA_RUN_DIR` somewhere else. Different personas, different providers and the
default fresh temp dir never contend.

This is a refusal rather than a queue because the alternative is not a lost race: the second run's
directory clear unlinks the first run's event stream while its provider is still writing, both
append to one events file, and the gate then validates an interleaving of two transcripts.

## Provenance

Each run writes `<persona>-<provider>-provenance.json` beside its findings, recording what produced
them: the provider, model and effort; the persona brief, the findings schema and **the exact prompt
the model received**, each by SHA-256; and the repository with the resolved `head_sha` and `base_sha`.

The hashes are the point. Briefs live in a plugin cache that updates underneath you, so two runs are
only comparable if they ran the same brief — and `base_ref=HEAD~1` names a different commit every
day, so without the resolved SHAs a finding reading `f.py:42` cannot be tied to the code it was
about. Reviewing a directory that is not a git repository is fine; the SHA fields record
`unresolved:` rather than going missing.

## Breaking changes in 0.2.0

The implementation moved from two shell scripts to one Python package. The commands and their flags
are unchanged; the exit statuses are not:

- A **missing persona argument** now exits `2` with usage on stderr. It previously printed usage to
  **stdout and exited 0** — success, for an invocation that reviewed nothing.
- A **runner failure** is now `4` rather than the runner's own status. Propagating it verbatim
  collided with the codes reserved for usage (`2`) and over-budget (`78`), so a caller could not
  tell them apart.
- New refusals: `3` for a missing runner/git/plugin, `5` for a timeout, `2` for a persona name that
  is not a bare brief name or a `-b` that does not resolve.
- `CE_PERSONA_MAX_PROMPT_TOKENS` now counts the diff as well as the prompt, so an existing tuned
  value bounds more than it used to.
- `ce-persona-findings` gained `-h`/`--help` and now uses `2` for usage errors rather than `1` for
  everything, so a caller can tell a bad invocation from an unusable artifact.
- Both review commands now enforce timeouts. A wedged provider previously hung forever and took the
  calling agent with it; `CE_PERSONA_IDLE_SECS` and `CE_PERSONA_HARD_SECS` bound that.
- **`CE_PERSONA_*` values are validated strictly.** A timeout of `0`, a non-finite or negative
  number, and an unrecognised `CE_PERSONA_*` name all exit `2`. Previously `0` disabled the watchdog
  and a misspelled name was ignored in silence.
- **Concurrent runs of the same persona and provider in one run directory are refused** with `2`
  rather than overwriting each other. See Concurrency above.

## Development

```console
nix flake check   # package build, lint, strict types, unit + process suites, mutation harness
nix develop
```

Five checks, and the fifth is the unusual one. `tests/test_mutations.py` reverts each fix in a
scratch copy of the **built** package and requires a test to die — because the recurring defect here
has never been a wrong guard, it has been a guard that *cannot* fail. Seven shipped green in a
single review cycle. A new guard without an entry in that table is not finished.

`AGENTS.md` records the house-template divergences (no pydantic, no rich, no structlog) with the
measurements behind each, and the structural invariants worth not breaking.

The unit suite exercises the packaged library and asserts which copy it imported — an earlier
version put its own source root ahead of `PYTHONPATH` and passed with every packaged module replaced
by `raise RuntimeError`. The process suite drives the **installed console scripts** against stub
runners, with a PATH that deliberately cannot reach a real `grok` or `codex`, and every case runs
for **both** providers: when the two runners were separate scripts, guards were repeatedly added to
both and tested against only one.

Type checking is `basedpyright` in **strict** mode, with no baseline and no rule suppressions. The
JSON boundary is typed (`validate.JSONValue`) rather than silenced. The `types` check ends with a
negative control that injects `return x + None` into the library and fails if the checker accepts
it — because this check once passed while analysing exactly one file and none of the code that
ships.

`python3 -m persona_review.flags` probes the installed `grok` for CLI drift. It needs a real
authenticated binary, so no flake check runs it.

## Licence

Apache-2.0. Persona briefs and the findings schema are read at run time from the
compound-engineering plugin, which is MIT — see `NOTICE`.
