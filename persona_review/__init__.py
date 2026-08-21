"""Schema-validated code review through compound-engineering persona briefs.

Three commands, installed as console entry points:

- `ce-grok-persona` / `ce-codex-persona` — run one reviewer persona through xAI's `grok` or
  OpenAI's `codex` and hold the answer to the plugin's findings schema. stdout is one
  summary line; the exit status is the verdict.
- `ce-persona-findings` — tiered retrieval over the resulting artifact, so a caller pays for
  detail only when it has a reason to read it.

The two review commands are the same code path. `providers.py` holds the entire difference
between them, as data: when they were separate scripts, the same defects had to be found and
fixed twice, and a guard added to one while tested against the other read as coverage while
the second copy was free to be deleted.

Entry points rather than `python -m`: `-m` and `-c` both put the CALLER'S working directory
first on `sys.path`, ahead of everything else, so a `persona_review/` directory in the
repository under review would replace this package — reporting a clean review for a run that
never happened, and executing planted code outside the read-only sandbox. A generated
console script gets its own directory on `sys.path` instead, which closes that by
construction rather than by an environment variable.
"""
