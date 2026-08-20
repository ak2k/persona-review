"""Schema-validated code review through compound-engineering persona briefs.

Three modules, each also runnable as `python -m persona_review.<name>`:

- `validate` — the findings gate. Extracts the model's own findings object (from grok's
  NDJSON `result` event, from codex's `-o` final message, or by scanning a transcript as
  a fallback) and holds it to the plugin's findings schema. A run that returned prose,
  or an object of the wrong shape, exits non-zero rather than reading as a clean review.
- `findings` — tiered retrieval over a findings artifact, so a caller pays for detail
  only when it has a reason to read it.
- `flags` — a CLI-drift gate for the grok wrapper's flags, derived from the wrapper's own
  invocation so the two cannot disagree.

These are a package rather than loose executables because the type checker must be able
to import them: hyphenated, suffix-less scripts can only be loaded dynamically, and
everything a dynamic loader returns is untyped by construction.

No `__all__`: nothing does `from persona_review import *`, and the version here listed
three submodules this file never imports, so it declared an export surface that did not
exist.
"""
