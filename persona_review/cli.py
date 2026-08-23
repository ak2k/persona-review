"""The command shared by every provider.

One entry point, bound to a `Provider` at import time. The output contract is built for a
CALLING AGENT rather than a terminal: stdout is one summary line naming the finding counts
by severity and the artifact path, the event stream and validated findings go to a run
directory, and the exit status is the verdict. A gating caller reads the status and nothing
else.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

from . import assets, errors, runner, validate
from .config import DEFAULT_HARD_SECS, DEFAULT_IDLE_SECS, DEFAULT_MAX_TOKENS, Settings
from .providers import CODEX, GROK, Invocation, Provider

# The verdict vocabulary, read from the exception classes that carry it. Distinct codes,
# because a caller has to tell "the review found nothing" from "the review never happened",
# and because propagating a runner's own status verbatim collides with the meanings reserved
# here. Aliased rather than restated: two lists of the same numbers is one edit away from
# disagreeing, and these numbers are a published contract.
EXIT_OK = 0
EXIT_GATE = errors.GateError.exit_code
EXIT_USAGE = errors.UsageError.exit_code
EXIT_ENV = errors.EnvError.exit_code
EXIT_RUNNER = errors.RunnerError.exit_code
EXIT_TIMEOUT = errors.RunTimeout.exit_code
EXIT_BUDGET = errors.BudgetError.exit_code


def _assert_running_the_installed_library() -> str | None:
    """The gate must be the code that was installed, whatever put something else first.

    This replaces an enumeration that was losing. The hole is one mechanism — anything that
    lands earlier on `sys.path` shadows `persona_review`, because the packaging wrapper
    APPENDS its own site-packages — and the caller's working directory, `PYTHONPATH` and
    `PYTHONHOME` are three instances of it, not the set. Two of those were closed one at a
    time, in a commit whose comment said "TWO doors, and both have to be shut"; a reviewer
    then found the third. Enumerating instances of a mechanism loses to the next instance.

    So assert the property instead: the module that is running came from where the build put
    it. `PERSONA_REVIEW_LIB` is set by the packaging wrapper, so a repository under review
    cannot clear it; when it is absent — a source checkout, a developer — there is nothing to
    compare against and the check stands down rather than guessing.
    """
    expected = os.environ.get("PERSONA_REVIEW_LIB", "").strip()
    if not expected:
        return None
    actual = Path(validate.__file__).resolve()
    if actual.is_relative_to(Path(expected).resolve()):
        return None
    return (
        f"refusing to run: the findings gate was loaded from {actual}, not from the "
        f"installed library at {expected}.\n"
        "  Something placed another `persona_review` earlier on sys.path — the repository "
        "under review is the one that matters.\n"
        "  A gate that is not this package's gate can report any verdict it likes."
    )


def _epilog(provider: Provider) -> str:
    runner_name = provider.binary
    return f"""\
exit status
{errors.render_exit_table(runner_name)}

environment
  CE_REVIEW_ASSETS      plugin references/ dir to read briefs and schema from
  CE_PERSONA_RUN_DIR    where artifacts land (default: a fresh temp dir)
  CE_PERSONA_MAX_PROMPT_TOKENS
      budget, default {DEFAULT_MAX_TOKENS}. Counts the prompt PLUS `git diff <base>..HEAD`.
      WITHOUT -b there is no diff to weigh and the budget bounds the prompt
      only -- the model picks its own scope, so there is nothing to bound.
      Over budget the run is refused, never summarized: findings from a
      summarized diff cannot be checked against the code.
  CE_PERSONA_IDLE_SECS  kill after this long with no output (default {int(DEFAULT_IDLE_SECS)})
  CE_PERSONA_HARD_SECS  kill after this long overall (default {int(DEFAULT_HARD_SECS)})

  Both timeouts must be greater than zero: there is no way to switch a watchdog
  off, because an unwatched run is a full-effort model run that nothing will stop
  and nothing will read. An unrecognised CE_PERSONA_* name is an error, not a
  shrug -- a misspelled one would otherwise leave the real setting at its default.
"""


def _parser(provider: Provider) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=provider.command,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"Run a compound-engineering review persona through {provider.binary} and return\n"
            "schema-valid findings, not a transcript.\n\n"
            "stdout is ONE line: finding counts by severity and the artifact path. Read detail\n"
            "with `ce-persona-findings <artifact>`; a gating caller reads only the exit status."
        ),
        epilog=_epilog(provider),
    )
    parser.add_argument(
        "persona",
        help="brief name, e.g. adversarial-reviewer; a leading `ce-` and `.md` are accepted",
    )
    parser.add_argument(
        "-C", "--cd", dest="repo", default=".", help="repo to review in (default: cwd)"
    )
    parser.add_argument(
        "-b",
        "--base",
        default="",
        help="base ref; the persona is told to `git diff <base>..HEAD`",
    )
    parser.add_argument("-m", "--model", default=provider.default_model, help=provider.model_help)
    parser.add_argument(
        "-e", "--effort", default=provider.default_effort, help=provider.effort_help
    )
    parser.add_argument(
        "-c",
        "--context",
        default="",
        help="file whose contents are appended as extra review context (- for stdin)",
    )
    return parser


def _read_context(spec: str) -> str:
    if not spec:
        return ""
    if spec == "-":
        return sys.stdin.read()
    try:
        return Path(spec).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise assets.UsageError(f"cannot read context file {spec}: {exc}") from exc


def _expand_repo(spec: str) -> Path:
    """`-C` with `~` expanded, without letting the expansion escape as a traceback.

    `Path('~nosuchuser/repo').expanduser()` raises RuntimeError — not OSError, and not a type
    any caller would think to catch — so an unknown user in a path reached the top of the
    process as a traceback and an unmapped exit status rather than exit 2.
    """
    try:
        return Path(spec).expanduser()
    except RuntimeError as exc:
        raise errors.UsageError(f"-C '{spec}' names a home directory that does not exist") from exc


# Everything a run writes, relative to its stem. `.lock` is NOT here: it is coordination,
# not an artifact, it is created before the clear, and unlinking it would race a waiter.
ARTIFACT_SUFFIXES = (
    ".json",
    "-provenance.json",
    "-last.json",
    "-events.jsonl",
    "-prompt.md",
    # The failure path reads this one back, so a survivor from an earlier run is the stale
    # artifact most likely to be believed.
    "-stderr.log",
)


def _clear_run_dir(run_dir: Path, stem: Path) -> None:
    """Remove this stem's artifacts. Call only while holding the stem's exclusive lock.

    Artifact paths are deterministic and the run dir is documented as reusable, so a run that
    fails before the gate must not leave the PREVIOUS run's findings sitting next to this
    run's fresh event stream — the provenance sidecar exists to attest which brief produced
    these findings, and a stale one attests the wrong run. Cleared before the budget
    preflight, so an early refusal leaves nothing behind either.
    """
    for suffix in ARTIFACT_SUFFIXES:
        path = Path(f"{stem}{suffix}")
        if path.parent != run_dir:  # a persona name is a bare brief name; assert it stayed one
            raise errors.UsageError(f"refusing to touch {path}, which is outside {run_dir}")
        path.unlink(missing_ok=True)


def main(provider: Provider, argv: list[str] | None = None) -> int:
    """Map every AppError to its own exit status, once.

    The body raises; this frame is the only place that decides a status. It used to be six
    near-identical `except X: print; return CODE` blocks, and the two things that shape
    got wrong are both in this package's history: an error class raised where `main` did not
    catch it escaped as a traceback with an unmapped status, and a class caught in one arm
    but not another got different statuses depending on which call raised it.
    """
    # Before anything else, including argument parsing: if the gate is not this package's
    # gate, nothing it goes on to report means anything.
    wrong_library = _assert_running_the_installed_library()
    if wrong_library:
        print(f"{provider.command}: {wrong_library}", file=sys.stderr)
        return EXIT_ENV

    args = _parser(provider).parse_args(argv)
    try:
        return _review(provider, args)
    except errors.AppError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return exc.exit_code


def _review(provider: Provider, args: argparse.Namespace) -> int:
    # The environment is parsed FIRST and in full, so a malformed CE_PERSONA_* value is
    # refused before any work: it used to be read three quarters of the way down, after the
    # run directory had been cleared and the prompt built.
    settings = Settings.from_env()

    # Only the persona name is needed to know the artifact paths, and normalising it touches
    # no filesystem — which is why it is split from resolution.
    persona = assets.normalise_persona(args.persona)
    run_dir = settings.resolved_run_dir()
    stem = run_dir / f"{persona}-{provider.name}"

    # The lock is taken BEFORE the clear and held through the gate. Taken after, a second
    # concurrent run would already have deleted the first run's event stream out from under
    # a provider still writing to it.
    with runner.exclusive_run(stem):
        _clear_run_dir(run_dir, stem)
        return _review_locked(provider, args, settings, persona, stem)


def _review_locked(
    provider: Provider,
    args: argparse.Namespace,
    settings: Settings,
    persona: str,
    stem: Path,
) -> int:
    """The run itself. Every path under `stem` is this process's alone for the duration."""
    repo = _expand_repo(args.repo)
    if not repo.is_dir():
        raise errors.UsageError(f"-C '{args.repo}' is not a directory")
    repo = repo.resolve()

    asset_dir = assets.resolve_assets(settings.assets_override)
    schema_file = asset_dir / "findings-schema.json"
    if not schema_file.is_file():
        raise errors.EnvError(f"missing findings schema at {schema_file}")
    resolved, brief = assets.resolve_persona(asset_dir, args.persona)
    if resolved != persona:
        # The lock and the clear were taken against the NORMALISED name, before any
        # filesystem access; the artifacts are written under the RESOLVED one. Both call
        # `normalise_persona`, so they agree — but if they ever stopped agreeing, this run
        # would write to paths it does not hold the lock on, and a concurrent run would
        # overwrite them. Assert it rather than rely on it.
        raise errors.UsageError(
            f"persona resolved to '{resolved}' but the run is locked as '{persona}'"
        )

    if shutil.which(provider.binary) is None:
        raise errors.MissingTool(f"{provider.binary} not on PATH ({provider.install_hint})")

    context = _read_context(args.context)
    schema_text = schema_file.read_text(encoding="utf-8")
    prompt = assets.build_prompt(
        provider=provider,
        persona=persona,
        brief=brief,
        assets=asset_dir,
        schema_text=schema_text,
        base=args.base,
        context=context,
    )

    limit = settings.max_prompt_tokens
    budget = runner.weigh(prompt=prompt, repo=repo, base=args.base, limit_tokens=limit)
    if budget.over:
        raise errors.BudgetError(
            f"review is ~{budget.tokens} tokens "
            f"({budget.prompt_bytes} prompt + {budget.diff_bytes} diff bytes), "
            f"over the {limit}-token budget.\n"
            "  Narrow the review (-b a nearer base, or a smaller -c context)\n"
            "  rather than sending it.\n"
            "  Raise CE_PERSONA_MAX_PROMPT_TOKENS only if the provider can genuinely take it."
        )

    prompt_file = Path(f"{stem}-prompt.md")
    events_file = Path(f"{stem}-events.jsonl")
    err_file = Path(f"{stem}-stderr.log")
    findings_file = Path(f"{stem}.json")
    prov_file = Path(f"{stem}-provenance.json")
    last_file = Path(f"{stem}-last.json") if provider.writes_last_message else None
    prompt_file.write_text(prompt, encoding="utf-8")

    inv = Invocation(
        model=args.model,
        effort=args.effort,
        repo=repo,
        prompt_file=prompt_file,
        schema_text=schema_text,
        last_file=last_file,
    )
    started_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = runner.execute(
        provider.argv(inv),
        stdout_path=events_file,
        stderr_path=err_file,
        stdin_path=prompt_file if provider.prompt_on_stdin else None,
        idle_secs=settings.idle_secs,
        hard_secs=settings.hard_secs,
    )

    if result.timed_out:
        raise errors.RunTimeout(f"{result.reason}\n  partial output: {events_file}")
    if result.status != 0:
        tail = "".join(
            f"\n  {line}" for line in _tail(err_file if err_file.stat().st_size else events_file, 5)
        )
        raise errors.RunnerError(f"{provider.binary} exited {result.status}{tail}")

    answer_file = last_file if last_file is not None else events_file
    if last_file is not None and not (last_file.exists() and last_file.stat().st_size):
        raise errors.GateError(f"{provider.binary} wrote no final message to {last_file}")

    # The gate has the last word: a run that returned prose, or an object of the wrong
    # shape, is a failed review that otherwise reads as a clean one.
    return validate.gate(
        answer_file=answer_file,
        schema_path=schema_file,
        mode=provider.mode,
        findings_out=findings_file,
        provenance_out=prov_file,
        prov_pairs=[
            f"provider={provider.name}",
            f"model={args.model}",
            f"effort={args.effort or 'config-default'}",
            f"persona={persona}",
            f"assets_dir={asset_dir}",
            f"base_ref={args.base}",
            f"started_at={started_at}",
            f"runner_status={result.status}",
            # WHAT WAS REVIEWED, not just that a review happened. `base_ref` is the caller's
            # string — `HEAD~1` names a different commit every day — so without these a
            # finding reading `f.py:42` cannot be tied back to the code it was about, and
            # two runs a week apart are indistinguishable in the record. Resolved after the
            # run rather than before, so the recorded state is the one the model saw for the
            # whole of it. `unresolved:` when the target is not a git repository, which is
            # legitimate: the model reads files, not history.
            f"repo={repo}",
            f"head_sha={runner.resolve_revision(repo, 'HEAD')}",
            f"base_sha={runner.resolve_revision(repo, args.base) if args.base else ''}",
        ],
        prov_files={"persona": str(brief), "schema": str(schema_file)},
        label=provider.command,
    )


def _tail(path: Path, count: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


def grok_main() -> int:
    return main(GROK)


def codex_main() -> int:
    return main(CODEX)


if __name__ == "__main__":  # pragma: no cover - module is driven through the entry points
    raise SystemExit(main(GROK))
