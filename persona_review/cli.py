"""The command shared by every provider, in each of its two modes.

One entry point, bound to a `Provider` and a `Flow` at import time. The output contract is
built for a CALLING AGENT rather than a terminal: stdout is one summary line naming what
came back and the artifact path, the event stream and the validated answer go to a run
directory, and the exit status is the verdict. A gating caller reads the status and nothing
else.

A REVIEW asks a model for findings; a VALIDATION hands it findings somebody else made and
asks for a verdict on each. Both run the same frame — the installed-library assertion, the
one AppError mapping, the lock, the clear, the dispatch, the gate — because every guard
this package exists for lives there, and a second copy is a second place to forget one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import assets, errors, runner, validate, verdicts
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
EXIT_VACUOUS = errors.VacuousRun.exit_code
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


def _epilog(provider: Provider, words: Mapping[str, str]) -> str:
    """The shared help tail. `words` is the flow's noun set: the statuses mean the same
    thing in both modes, the things they are about do not."""
    runner_name = provider.binary
    return f"""\
exit status
{errors.render_exit_table(runner_name, words)}

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
        epilog=_epilog(provider, errors.REVIEW_WORDS),
    )
    parser.add_argument(
        "persona",
        help="brief name, e.g. adversarial-reviewer; a leading `ce-` and `.md` are accepted",
    )
    return _shared_options(parser, provider)


def _validate_parser(provider: Provider) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=provider.validate_command,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"Judge a compound-engineering findings batch through {provider.binary} and return\n"
            "one schema-valid verdict per finding, not a transcript.\n\n"
            "stdout is ONE line: the validated/rejected split and the artifact path. Read detail\n"
            "with `ce-persona-findings <artifact>`; a gating caller reads only the exit status."
        ),
        epilog=_epilog(provider, errors.VALIDATE_WORDS),
    )
    parser.add_argument(
        "batch",
        help="the plugin's validator batch: a JSON array of findings, each carrying a `#`",
    )
    return _shared_options(parser, provider)


def _shared_options(parser: argparse.ArgumentParser, provider: Provider) -> argparse.ArgumentParser:
    """Everything after the positional. One copy, because a flag that reached only one mode
    would be a second contract nobody documented."""
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


def _run(provider: Provider, args: argparse.Namespace, flow: Flow) -> int:
    # The environment is parsed FIRST and in full, so a malformed CE_PERSONA_* value is
    # refused before any work: it used to be read three quarters of the way down, after the
    # run directory had been cleared and the prompt built.
    settings = Settings.from_env()

    # Only the argv is needed to know the artifact paths, and deriving the stem touches no
    # filesystem — which is why it is split from resolution.
    run_dir = settings.resolved_run_dir()
    stem = flow.stem(provider, args, run_dir)

    # The lock is taken BEFORE the clear and held through the gate. Taken after, a second
    # concurrent run would already have deleted the first run's event stream out from under
    # a provider still writing to it. ONE frame for both modes: a second copy is a second
    # place for the ordering to be got wrong, and this one has been got wrong before.
    with runner.exclusive_run(stem):
        _clear_run_dir(run_dir, stem)
        return flow.locked(provider, args, settings, stem)


@dataclass(frozen=True)
class Dispatched:
    """What a completed provider run leaves behind for the gate to judge."""

    answer_file: Path
    evidence: validate.Evidence
    started_at: str
    status: int
    artifact_out: Path
    provenance_out: Path
    prompt_file: Path


def _dispatch(
    provider: Provider,
    args: argparse.Namespace,
    settings: Settings,
    stem: Path,
    *,
    prompt: str,
    schema_text: str,
    repo: Path,
) -> Dispatched:
    """Weigh a prompt, run the provider under the watchdogs, and hand back the evidence.

    Shared by both modes because nothing here is about what was asked. The budget refusal
    is inside rather than beside it so its wording has ONE copy: the two modes disagreeing
    about how an over-budget run is described would be two contracts for one status.
    """
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
    # Monotonic, not wall clock: this measures how long the run took, and a clock stepped by
    # ntp mid-review must not turn a four-second dud into a plausible-looking one.
    started = time.monotonic()
    result = runner.execute(
        provider.argv(inv),
        stdout_path=events_file,
        stderr_path=err_file,
        stdin_path=prompt_file if provider.prompt_on_stdin else None,
        idle_secs=settings.idle_secs,
        hard_secs=settings.hard_secs,
    )
    elapsed = time.monotonic() - started

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

    # WHERE THE RUN'S OWN ACCOUNT OF ITSELF LIVES. The events file for BOTH providers, even
    # though codex's answer arrives elsewhere: the question is what the run inspected, and
    # only the event stream records that. Handed to the gate rather than counted here, so the
    # counting happens where the stream has already been read.
    evidence = validate.Evidence(
        events_file=events_file, mode=provider.events_mode, duration_s=elapsed
    )
    return Dispatched(
        answer_file=answer_file,
        evidence=evidence,
        started_at=started_at,
        status=result.status,
        artifact_out=findings_file,
        provenance_out=prov_file,
        prompt_file=prompt_file,
    )


def _target_pairs(repo: Path, base: str) -> list[str]:
    """WHAT WAS EXAMINED, not just that something was. `base_ref` is the caller's string —
    `HEAD~1` names a different commit every day — so without these a finding reading
    `f.py:42` cannot be tied back to the code it was about, and two runs a week apart are
    indistinguishable in the record. Resolved after the run rather than before, so the
    recorded state is the one the model saw for the whole of it. `unresolved:` when the
    target is not a git repository, which is legitimate: the model reads files, not history.

    One copy for both modes: a validation that recorded a different set of facts about the
    tree would not be comparable with the review whose findings it judged.
    """
    return [
        f"repo={repo}",
        f"head_sha={runner.resolve_revision(repo, 'HEAD')}",
        f"base_sha={runner.resolve_revision(repo, base) if base else ''}",
    ]


def _review_stem(provider: Provider, args: argparse.Namespace, run_dir: Path) -> Path:
    return run_dir / f"{assets.normalise_persona(args.persona)}-{provider.name}"


def _review_locked(
    provider: Provider,
    args: argparse.Namespace,
    settings: Settings,
    stem: Path,
) -> int:
    """The run itself. Every path under `stem` is this process's alone for the duration."""
    persona = assets.normalise_persona(args.persona)
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

    sent = _dispatch(
        provider, args, settings, stem, prompt=prompt, schema_text=schema_text, repo=repo
    )

    # The gate has the last word: a run that returned prose, or an object of the wrong
    # shape, is a failed review that otherwise reads as a clean one. So is a run that
    # answered without reading anything, and `evidence` is what lets the gate see it.
    return validate.gate(
        answer_file=sent.answer_file,
        schema_path=schema_file,
        mode=provider.mode,
        findings_out=sent.artifact_out,
        provenance_out=sent.provenance_out,
        prov_pairs=[
            f"provider={provider.name}",
            f"model={args.model}",
            f"effort={args.effort or 'config-default'}",
            f"persona={persona}",
            f"assets_dir={asset_dir}",
            f"base_ref={args.base}",
            f"started_at={sent.started_at}",
            f"runner_status={sent.status}",
            *_target_pairs(repo, args.base),
        ],
        # The PROMPT, not just its ingredients. Hashing the brief and the schema attests two
        # of the inputs; what the model was actually told also carries the rubric — read from
        # a plugin file that updates underneath us — the -c context, and the diff instruction.
        # Without this, "what exactly was this model asked?" is unanswerable from the record,
        # and -prompt.md is no substitute: the next run's clear deletes it.
        prov_files={
            "persona": str(brief),
            "schema": str(schema_file),
            "prompt": str(sent.prompt_file),
        },
        evidence=sent.evidence,
        label=provider.command,
    )


def _validate_stem(provider: Provider, args: argparse.Namespace, run_dir: Path) -> Path:
    # No name from the caller goes into this one: a validation is per provider, so the stem
    # is fixed and there is nothing here that could address a path outside the run dir.
    del args
    return run_dir / f"validator-{provider.name}"


def _validate_locked(
    provider: Provider,
    args: argparse.Namespace,
    settings: Settings,
    stem: Path,
) -> int:
    """One validation run: judge a findings batch, gate the verdicts it comes back with."""
    repo = _expand_repo(args.repo)
    if not repo.is_dir():
        raise errors.UsageError(f"-C '{args.repo}' is not a directory")
    repo = repo.resolve()

    asset_dir = assets.resolve_assets(settings.assets_override)
    schema_file = verdicts.schema_path()

    # Parsed HERE, after the clear rather than beside the argument parsing: the clear is the
    # first thing inside the lock precisely so that no early refusal can leave a previous
    # run's artifacts beside a fresh stream. A malformed batch is an early refusal.
    batch_file = Path(args.batch).expanduser()
    try:
        batch_text = batch_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise errors.UsageError(f"cannot read validator batch {args.batch}: {exc}") from exc
    numbers = verdicts.parse_batch(batch_text, str(batch_file))

    if shutil.which(provider.binary) is None:
        raise errors.MissingTool(f"{provider.binary} not on PATH ({provider.install_hint})")

    context = _read_context(args.context)
    schema_text = schema_file.read_text(encoding="utf-8")
    prompt = assets.build_validator_prompt(
        batch_text=batch_text,
        assets=asset_dir,
        schema_text=schema_text,
        base=args.base,
        context=context,
    )

    sent = _dispatch(
        provider, args, settings, stem, prompt=prompt, schema_text=schema_text, repo=repo
    )

    def check(found: validate.Artifact, schema: validate.JSONObject) -> int:
        # The schema first, so a verdict whose `#` is not an integer is refused as a bad
        # shape rather than counted as a number the batch never carried.
        count = validate.check_object(found, schema, "verdicts")
        verdicts.check_coverage(found, numbers)
        return count

    return validate.gate(
        answer_file=sent.answer_file,
        schema_path=schema_file,
        mode=provider.mode,
        findings_out=sent.artifact_out,
        provenance_out=sent.provenance_out,
        prov_pairs=[
            f"provider={provider.name}",
            f"model={args.model}",
            f"effort={args.effort or 'config-default'}",
            # WHICH MODE WROTE THIS. The artifact stem already says it, but a sidecar is
            # read on its own and a consumer must not have to parse a filename to learn
            # whether these are findings or judgments of somebody else's.
            "kind=validator",
            f"batch={batch_file}",
            f"assets_dir={asset_dir}",
            f"base_ref={args.base}",
            f"started_at={sent.started_at}",
            f"runner_status={sent.status}",
            *_target_pairs(repo, args.base),
        ],
        prov_files={
            "template": str(asset_dir / assets.VALIDATOR_TEMPLATE),
            "batch": str(batch_file),
            "schema": str(schema_file),
            "prompt": str(sent.prompt_file),
        },
        evidence=sent.evidence,
        label=provider.validate_command,
        key="verdicts",
        check=check,
        summarize=verdicts.summarize,
        noun="verdicts",
    )


def _tail(path: Path, count: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


@dataclass(frozen=True)
class Flow:
    """A mode of this command: what it is called, what it parses, where it writes, what it does.

    Four plain functions rather than a subclass or a pile of `if mode ==` branches. The frame
    around them — the installed-library assertion, the one AppError mapping, the lock, the
    clear, the dispatch — is the part neither mode is allowed its own copy of, because every
    guard this package exists for lives in it.
    """

    command: Callable[[Provider], str]
    parser: Callable[[Provider], argparse.ArgumentParser]
    stem: Callable[[Provider, argparse.Namespace, Path], Path]
    locked: Callable[[Provider, argparse.Namespace, Settings, Path], int]


def _review_command(provider: Provider) -> str:
    return provider.command


def _validate_command(provider: Provider) -> str:
    return provider.validate_command


REVIEW = Flow(command=_review_command, parser=_parser, stem=_review_stem, locked=_review_locked)
VALIDATE = Flow(
    command=_validate_command,
    parser=_validate_parser,
    stem=_validate_stem,
    locked=_validate_locked,
)


def main(provider: Provider, argv: list[str] | None = None, flow: Flow = REVIEW) -> int:
    """Map every AppError to its own exit status, once.

    The body raises; this frame is the only place that decides a status. It used to be six
    near-identical `except X: print; return CODE` blocks, and the two things that shape
    got wrong are both in this package's history: an error class raised where `main` did not
    catch it escaped as a traceback with an unmapped status, and a class caught in one arm
    but not another got different statuses depending on which call raised it.

    `flow` defaults to the review, so the two-argument call this frame has always taken
    still means what it did.
    """
    command = flow.command(provider)

    # Before anything else, including argument parsing: if the gate is not this package's
    # gate, nothing it goes on to report means anything.
    wrong_library = _assert_running_the_installed_library()
    if wrong_library:
        print(f"{command}: {wrong_library}", file=sys.stderr)
        return EXIT_ENV

    args = flow.parser(provider).parse_args(argv)
    try:
        return _run(provider, args, flow)
    except errors.AppError as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return exc.exit_code


def grok_main() -> int:
    return main(GROK)


def codex_main() -> int:
    return main(CODEX)


def grok_validate_main() -> int:
    return main(GROK, flow=VALIDATE)


def codex_validate_main() -> int:
    return main(CODEX, flow=VALIDATE)


if __name__ == "__main__":  # pragma: no cover - module is driven through the entry points
    raise SystemExit(main(GROK))
