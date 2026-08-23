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

### 3. argparse, not typer + rich

**Template says:** typer + rich for Profile A.

**Status: open, not settled.** argparse is what the bash-to-Python port carried over. rich is
a clear no — this output is read by agents and by CI, where colour and boxes are noise. typer
is a genuine question: it would remove hand-rolled argument handling, and the flags are
already stable and simple (`<persona> [-C dir] [-b ref] [-m model] [-e effort] [-c file]`).

Two things have to survive any switch, and they are why it has not happened yet:

- **The exit table in `--help` is generated from `errors.EXIT_TABLE`**, not restated beside
  it. Two lists of the same numbers are one edit from disagreeing, and these numbers are a
  published contract a gating caller branches on.
- **A missing option value must exit 2**, not 1. That specific collision is a bug this
  package already shipped once, in bash, via `shift 2` under `set -e`.

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

- **A schema-valid but EMPTY findings array is accepted.** This is the one hole the package
  does not close, and it is asserted explicitly in
  `test_an_empty_structured_output_is_the_documented_gap` so that closing it later fails
  loudly and so no other test can be read as already covering it. "Found nothing" and
  "quietly gave up" are indistinguishable without judging the transcript.
- **`git` is symlinked into the stub directory in `test_process.py`.** The suite runs with a
  PATH that deliberately excludes the real `grok` and `codex`: an early version removed a
  stub, reached the genuine binary, and started a real billed model run from a unit test.
- **Exit-code assertions use literals, not the module's constants.** An assertion written
  against `EXIT_USAGE` moves with it. Mutation testing caught exactly that — redefining
  `EXIT_USAGE` to `1` left the suite green.
