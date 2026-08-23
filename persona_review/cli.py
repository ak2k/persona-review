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
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import assets, runner, validate
from .providers import CODEX, GROK, Invocation, Provider

# The verdict vocabulary. Distinct codes, because a caller has to tell "the review found
# nothing" from "the review never happened", and because propagating a runner's own status
# verbatim collides with the meanings reserved here.
EXIT_OK = 0
EXIT_GATE = 1
EXIT_USAGE = 2
EXIT_ENV = 3
EXIT_RUNNER = 4
EXIT_TIMEOUT = 5
EXIT_BUDGET = 78

DEFAULT_MAX_TOKENS = 80_000
DEFAULT_IDLE_SECS = 600.0
DEFAULT_HARD_SECS = 2400.0


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


def _env_number(name: str, default: float) -> float:
    """A finite, non-negative number from the environment, or a usage error.

    `float()` happily returns `nan` and `inf`, and every deadline here is checked with
    `elapsed >= value` — a comparison that is False forever against either. A typo in a
    timeout would silently disable the watchdog it was meant to set.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise runner.RunError(f"{name}={raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise runner.RunError(f"{name}={raw!r} must be a finite number")
    if value < 0:
        raise runner.RunError(f"{name}={raw!r} must not be negative")
    return value


def _epilog(provider: Provider) -> str:
    runner_name = provider.binary
    return f"""\
exit status
  {EXIT_OK}   schema-valid findings (an empty findings array is valid)
  {EXIT_GATE}   the answer was not schema-valid findings
  {EXIT_USAGE}   usage error: bad arguments, unknown or markdown-only persona,
      bad -C directory, unresolvable -b base ref, malformed CE_PERSONA_* value
  {EXIT_ENV}   environment error: {runner_name}, git or the plugin assets are missing,
      or CE_PERSONA_RUN_DIR cannot be created
  {EXIT_RUNNER}   {runner_name} itself exited non-zero
  {EXIT_TIMEOUT}   idle or hard timeout; the run was killed and partial output kept
  {EXIT_BUDGET}  over CE_PERSONA_MAX_PROMPT_TOKENS; refused, never summarized

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


def _run_dir(persona: str, provider: Provider) -> Path:
    raw = os.environ.get("CE_PERSONA_RUN_DIR", "").strip()
    run_dir = (
        Path(raw).expanduser().resolve()
        if raw
        else Path(tempfile.mkdtemp(prefix="ce-persona-run."))
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # The machine, not the invocation: an unwritable or occupied path is the same class
        # of problem as a missing binary, and a caller retrying with different arguments
        # would not fix it.
        raise runner.EnvError(f"cannot use CE_PERSONA_RUN_DIR={run_dir}: {exc}") from exc

    # Artifact paths are deterministic and the run dir is documented as reusable, so a run
    # that fails before the gate must not leave the PREVIOUS run's findings sitting next to
    # this run's fresh event stream — the provenance sidecar exists to attest which brief
    # produced these findings, and a stale one attests the wrong run. Cleared here, before
    # the budget preflight, so an early refusal leaves nothing behind either.
    for suffix in (
        ".json",
        "-provenance.json",
        "-last.json",
        "-events.jsonl",
        "-prompt.md",
        # The failure path reads this one back, so a survivor from an earlier run is the
        # stale artifact most likely to be believed.
        "-stderr.log",
    ):
        path = run_dir / f"{persona}-{provider.name}{suffix}"
        if path.parent != run_dir:  # a persona name is a bare brief name; assert it stayed one
            raise runner.RunError(f"refusing to touch {path}, which is outside {run_dir}")
        path.unlink(missing_ok=True)
    return run_dir


def main(provider: Provider, argv: list[str] | None = None) -> int:
    # Before anything else, including argument parsing: if the gate is not this package's
    # gate, nothing it goes on to report means anything.
    wrong_library = _assert_running_the_installed_library()
    if wrong_library:
        print(f"{provider.command}: {wrong_library}", file=sys.stderr)
        return EXIT_ENV

    args = _parser(provider).parse_args(argv)

    # Clear the run dir FIRST, before anything else that can fail. The artifact paths are
    # deterministic and the directory is documented as reusable, so every fallible step
    # ahead of the clear is a step that can leave the PREVIOUS run's findings sitting at the
    # path a caller reads. Only the persona name is needed to know those paths, and
    # normalising it touches no filesystem — which is why it is split from resolution.
    try:
        persona = assets.normalise_persona(args.persona)
        run_dir = _run_dir(persona, provider)
    except assets.UsageError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except runner.EnvError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_ENV
    except runner.RunError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    repo = Path(args.repo).expanduser()
    if not repo.is_dir():
        print(f"{provider.command}: -C '{args.repo}' is not a directory", file=sys.stderr)
        return EXIT_USAGE
    repo = repo.resolve()

    try:
        asset_dir = assets.resolve_assets(os.environ.get("CE_REVIEW_ASSETS"))
        schema_file = asset_dir / "findings-schema.json"
        if not schema_file.is_file():
            raise assets.AssetError(f"missing findings schema at {schema_file}")
        persona, brief = assets.resolve_persona(asset_dir, args.persona)
    except assets.AssetError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_ENV
    except assets.UsageError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if shutil.which(provider.binary) is None:
        print(
            f"{provider.command}: {provider.binary} not on PATH ({provider.install_hint})",
            file=sys.stderr,
        )
        return EXIT_ENV

    try:
        context = _read_context(args.context)
    except assets.UsageError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE

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

    try:
        limit = int(_env_number("CE_PERSONA_MAX_PROMPT_TOKENS", DEFAULT_MAX_TOKENS))
        idle = _env_number("CE_PERSONA_IDLE_SECS", DEFAULT_IDLE_SECS)
        hard = _env_number("CE_PERSONA_HARD_SECS", DEFAULT_HARD_SECS)
        budget = runner.weigh(prompt=prompt, repo=repo, base=args.base, limit_tokens=limit)
    except runner.EnvError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_ENV
    except runner.RunError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if budget.over:
        print(
            f"{provider.command}: review is ~{budget.tokens} tokens "
            f"({budget.prompt_bytes} prompt + {budget.diff_bytes} diff bytes), "
            f"over the {limit}-token budget.\n"
            "  Narrow the review (-b a nearer base, or a smaller -c context)\n"
            "  rather than sending it.\n"
            "  Raise CE_PERSONA_MAX_PROMPT_TOKENS only if the provider can genuinely take it.",
            file=sys.stderr,
        )
        return EXIT_BUDGET

    stem = run_dir / f"{persona}-{provider.name}"
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
    try:
        result = runner.execute(
            provider.argv(inv),
            stdout_path=events_file,
            stderr_path=err_file,
            stdin_path=prompt_file if provider.prompt_on_stdin else None,
            idle_secs=idle,
            hard_secs=hard,
        )
    except runner.RunError as exc:
        print(f"{provider.command}: {exc}", file=sys.stderr)
        return EXIT_ENV

    if result.timed_out:
        print(f"{provider.command}: {result.reason}", file=sys.stderr)
        print(f"  partial output: {events_file}", file=sys.stderr)
        return EXIT_TIMEOUT
    if result.status != 0:
        print(f"{provider.command}: {provider.binary} exited {result.status}", file=sys.stderr)
        for line in _tail(err_file if err_file.stat().st_size else events_file, 5):
            print(f"  {line}", file=sys.stderr)
        return EXIT_RUNNER

    answer_file = last_file if last_file is not None else events_file
    if last_file is not None and not (last_file.exists() and last_file.stat().st_size):
        print(
            f"{provider.command}: {provider.binary} wrote no final message to {last_file}",
            file=sys.stderr,
        )
        return EXIT_GATE

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
