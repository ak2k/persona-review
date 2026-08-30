# persona-review — working notes

Conventions follow the house Python template (`ak2k/python-starter`), Profile A: a
distributable CLI. This file records only where this package **departs** from it, and why.
Anything not listed here follows the template.

## The one thing to understand before changing anything

This package exists so that **a review which did not happen cannot report a clean result.**
Every design choice below follows from that, and the recurring defect here has never been a
wrong guard — it has been a guard that *cannot fail*. Seven shipped green in a single review
cycle. Two independent reviewers found a mutation harness that reported 23 kills, every one
a `ModuleNotFoundError`.

So the standing rule is: **watch a guard fail before believing it.** `tests/test_mutations.py`
enforces that mechanically — every fix has an entry that reverts it, and a test must die. A
new guard without an entry is not finished. Its own three controls exist because the first
version of that file had none and was worthless in a way that read as a perfect score.

## Divergences from the template

### 1. No runtime dependencies — so no pydantic / pydantic-settings

**Template says:** pydantic v2 + pydantic-settings at every boundary, `extra="forbid"`,
`frozen=True`.

**Here:** `dependencies = []`, and `config.py` is a frozen stdlib dataclass.

**Why:** the settings surface is six scalars. The decisive argument is not closure size, and
it is *not* security — `PYTHONSAFEPATH` plus an unset `PYTHONPATH` already closes the
module-shadowing route whether or not there are dependencies. It is verification cost.
Every guard in this package has to be *seen* to fail before it is believed and then pinned in
the mutation table. Declarative validation moves those guards into a library, which would
mean first establishing empirically that `PositiveFloat` rejects exactly `0`, that
`allow_inf_nan=False` applies to floats coerced from environment strings, and that
`extra="forbid"` sees environment variables rather than only init kwargs. Forty lines of
stdlib are pinned directly, by machinery already in the repo.

The place pydantic *would* earn its keep is parsing `findings-schema.json` into typed
models — and the design deliberately rules that out. The schema belongs to the
compound-engineering plugin and is free to grow fields this package has no authority to fix,
so `validate.JSONValue` plus `isinstance` narrowing is the documented choice. See the note
at the top of `validate.py`.

**Taken from the template anyway:** `extra="forbid"`. An unrecognised `CE_PERSONA_*` name is
an error, not a shrug — `CE_PERSONA_IDEL_SECS=30` used to leave the real idle timeout at its
default and say nothing. Eleven lines in `config.py`. The idea was worth more than the
library here.

### 2. `print()` to stdout/stderr, not structlog

**Template says:** structlog; `print()` is banned.

**Why:** stdout is a **product contract**, not logging. It is exactly one line — finding
counts by severity and the artifact path — because a calling agent pays for every token of
it, and the event stream this wraps is around a megabyte. stderr carries one legible reason
for a refusal. Neither is a log stream, neither is consumed by a log aggregator, and JSON
event output would make both worse for the only two consumers that exist: an agent reading
one line, and a person reading one error.

`tests/test_process.py` asserts the one-line contract directly.

### 3. No rich

**Template says:** rich for Profile A.

**Why — measured, not assumed** (rich 15.0.0, Python 3.14):

The obvious objection, colour, is **not** the reason: rich auto-detects a non-TTY and emits
no ANSI when piped. Two real ones replace it.

**It wraps the payload.** Rich falls back to **80 columns** when it cannot detect a terminal.
The real stdout line is 137 characters, so writing it through a `Console` to a pipe produces
three lines with the artifact path split mid-token:

```
'ce-grok-persona: 3 findings (1 P0, 2 P1) -> '
'/nix/store/143mqb7h2n08nrw4nlqnlxjn80qaabby-persona-review-run/adversarial-revie'
'wer-grok.json'
```

That breaks the one-line contract *and* corrupts the one field the caller has to use. A path
is data, not prose, and rich has no way to know the difference.

**TTY detection is defeatable by the environment.** With `FORCE_COLOR=1`, which plenty of CI
images set, rich emits ANSI even when piped. The caller does not control that variable.

### 4. argparse, not typer

**Template says:** typer for Profile A.

**Status: open, and deprioritised — not blocked.** Two objections previously recorded here
were wrong, and are corrected rather than deleted so they are not re-derived:

- ~~A missing option value would exit 1, not 2.~~ **False.** `click.UsageError.exit_code`
  is 2, verified end to end for both a missing option value and a missing required argument.
  That fear was the bash `shift 2` bug projected onto a library that does not have it.
- ~~Help formatting would destroy the generated exit table.~~ **Real by default, but
  solvable.** Typer rewraps an epilog into a paragraph and loses the indented continuation
  lines. Click's documented `\b` preformat marker preserves the structure exactly, modulo two
  spaces of added indent.

What actually remains, and it is weaker than the above:

1. **Consistency with the pydantic decision.** typer pulls click, and its default help path
   pulls rich — a larger closure than pydantic, for a smaller win, in a package where
   argparse already covers five stable flags.
2. **Rich is typer's default** (`rich_markup_mode="rich"`), so `--help` renders boxes that
   agents read and pay tokens for. Disabling it removes typer's main advantage and leaves
   parameter parsing that already exists.
3. **It would create a second exit frame.** Click exits internally in standalone mode, so
   parse errors would leave through click while every `AppError` leaves through `main`'s one
   handler. That partly undoes the invariant `errors.py` was built to establish.

None of these is a reason typer is wrong; together they are a reason it is not next.

### 4. `unittest` → pytest: done, no divergence

Recorded only because the migration is recent. `pytest` + `hypothesis`, per the template.
Coverage is gated at the template's 60.

**Read the coverage number correctly.** It measures the *unit* suite alone. The process and
mutation suites drive the library through subprocesses that the parent's tracer cannot see,
so `cli.py` and `runner.py` report low while being the most heavily exercised modules here.
Neither is omitted from the report: hiding them to flatter the total is precisely the
guard-shaped-nothing this repo keeps finding. The number is a floor against regression, not a
claim about how much of this code is tested.

## Structural invariants, and the tests that hold them

These are properties you can check with one command, which beats a rule you have to remember
at every call site. Each has a test; breaking one should fail loudly rather than quietly.

| Invariant | Held by |
|---|---|
| `subprocess.PIPE` appears nowhere in `runner.py` | a provider CLI spawns helpers that outlive the parent while holding stdout, so a pipe reader hangs on a *completed* run. The child writes to a file; the parent polls it. |
| `os.environ` is read only in `config.py` | `tests/test_unit.py::TestTheEnvironmentIsReadInOnePlace`, with a control proving the detector fires |
| Every error class carries its own `exit_code` | `TestErrorVocabulary`. A class cannot be added without choosing a status, so `main`'s mapping is total by construction. |
| Timeouts are strictly positive | `config.Settings`, so `_watch` has no `if secs > 0` left to fail. `0` used to switch the watchdog off. |
| The gate that ran came from the install | `PERSONA_REVIEW_LIB`, asserted before argument parsing. Replaces an enumeration of doors that was losing to each new door. |
| Every guard can fail | `tests/test_mutations.py` |

## Things that look like bugs and are not

- **A schema-valid but EMPTY findings array is accepted — from a run that inspected
  something.** This is the hole the package does not close, and it is asserted explicitly in
  `test_an_empty_structured_output_is_the_documented_gap` so that closing it later fails
  loudly and so no other test can be read as already covering it. "Found nothing" and
  "looked, then gave up" are indistinguishable without judging the transcript.

  What is no longer in the hole: a run that made **zero** tool calls. It read nothing, so its
  findings are unfounded whether the array is empty or full, and it exits `6` with no summary
  line. Exactly zero, with no configurable floor — "did this run inspect anything" has an
  answer, "did it inspect enough" is a judgement this package is not entitled to make. The
  fixture for the gap test therefore carries a tool call, because without one it would be
  testing the refusal instead.
- **The two tool-call adapters are shaped differently on purpose.** grok names a tool call
  structurally — a `tool_use` content block — so any of them counts and there is no list to
  go stale. codex names it by an item KIND, so that adapter carries a list, and a list can
  fall out of date. Hence the drift check on the codex side only: a stream whose item kinds
  are all unrecognised exits `3` naming them, never `6`. Getting that wrong would report
  "the model never opened the diff" identically on every run after a provider upgrade — a
  permanent outage, misdiagnosed as a bad model, in the direction the README tells callers to
  retry. `CODEX_QUIET_ITEMS` is what keeps "a kind we skip on purpose" and "a kind we have
  never heard of" different facts, and its control test is what keeps exit `6` reachable.
- **`ce-persona-findings` refuses an artifact whose provenance records zero tool calls.** It
  looks like a reader reaching into a sidecar it has no business reading. It is the other
  half of the refusal: exit `6` KEEPS the artifact as evidence, and an artifact on disk is
  exactly what this command renders — so without the check the package laundered its own
  verdict into an ordinary listing at exit `0`, one command later. Only a positive reading of
  zero refuses; no sidecar, or a malformed one, renders as before.
- **`git` is symlinked into the stub directory in `test_process.py`.** The suite runs with a
  PATH that deliberately excludes the real `grok` and `codex`: an early version removed a
  stub, reached the genuine binary, and started a real billed model run from a unit test.
- **Exit-code assertions use literals, not the module's constants.** An assertion written
  against `EXIT_USAGE` moves with it. Mutation testing caught exactly that — redefining
  `EXIT_USAGE` to `1` left the suite green.
