"""The two runners, as data.

This module is the ENTIRE difference between reviewing through xAI's `grok` and through
OpenAI's `codex`. Everything else — argument parsing, asset resolution, prompt assembly,
the size budget, artifact lifecycle, timeouts, the findings gate — is one code path shared
by both.

That is a deliberate structural property, not tidiness. When the two runners were separate
scripts, a guard added to one and tested against the other read as coverage while the
second copy was free to be deleted; the same defects had to be found and fixed twice; and
the two copies drifted far enough that the same guard carried two different explanations,
one of which had stopped being true. A provider is a dataclass here so that adding a third
one cannot reintroduce any of that.

Neither binary is a package dependency. They are the user's own subscription-authenticated
CLIs, resolved from PATH at run time, so a review uses the login they already have rather
than an API key this package would need to hold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# How the model's answer is recovered. `grok-events` reads the terminal `result` event of
# grok's NDJSON stream; `object` reads a file that holds the final message and nothing else.
MODE_GROK_EVENTS = "grok-events"
MODE_OBJECT = "object"

# How the run's own EVENTS are read, to count what it actually did. A second vocabulary
# rather than a reuse of the one above, because the two questions have different answers for
# codex: its ANSWER is a file written by `-o`, while its EVIDENCE is the `codex exec --json`
# stream.
#
# EVERY VALUE HERE DIFFERS FROM EVERY VALUE ABOVE, deliberately. `MODE_GROK_EVENTS` and this
# once shared the string "grok-events", which meant passing an answer mode where an events
# mode belongs was caught for codex and silently accepted for grok — the two vocabularies
# would then be one mis-wiring away from agreeing by accident. The names say what each
# stream is made of instead: grok's is messages carrying content blocks, codex's is items
# carrying kinds.
EVENTS_GROK = "grok-messages"
EVENTS_CODEX = "codex-items"


@dataclass(frozen=True)
class Invocation:
    """Everything a runner needs to be given for one review."""

    model: str
    effort: str
    repo: Path
    prompt_file: Path
    schema_text: str
    last_file: Path | None


def _grok_argv(inv: Invocation) -> list[str]:
    # `--sandbox read-only` is the real boundary; `--permission-mode bypassPermissions` only
    # suppresses the approval prompt, which a headless run has no way to answer. `--verbatim`
    # stops grok offloading a large prompt into a session file.
    #
    # `--json-schema` composes with `--output-format streaming-messages-json`, contradicting
    # grok's own help text ("--json-schema ... Implies --output-format json"). Verified
    # against 1.0.4 and 1.0.5: the terminal `result` event carries `structured_output` as a
    # parsed object, so the answer needs no transcript scanning, AND the stream still grows
    # message by message, which is what the idle watchdog measures.
    return [
        "grok",
        "--prompt-file",
        str(inv.prompt_file),
        "--verbatim",
        "--model",
        inv.model,
        "--effort",
        inv.effort,
        "--cwd",
        str(inv.repo),
        "--sandbox",
        "read-only",
        "--permission-mode",
        "bypassPermissions",
        "--disable-web-search",
        "--json-schema",
        inv.schema_text,
        "--output-format",
        "streaming-messages-json",
    ]


def _codex_argv(inv: Invocation) -> list[str]:
    # `-o` writes ONLY the agent's final message, so there is no transcript to scan. `--json`
    # puts the event stream on stdout, which the runner redirects to a file for liveness.
    #
    # Deliberately NOT `--output-schema`: OpenAI's strict mode rejects the plugin's findings
    # schema outright ("'additionalProperties' is required to be supplied and to be false")
    # and would additionally force every optional field — suggested_fix on every finding —
    # changing what a finding means. The schema travels in the prompt and the gate enforces
    # it afterwards.
    argv = ["codex", "exec", "-C", str(inv.repo), "-s", "read-only", "-m", inv.model]
    argv += ["-c", f'model_reasoning_effort="{inv.effort}"']
    argv += ["--ephemeral", "--color", "never", "--json"]
    if inv.last_file is not None:
        argv += ["-o", str(inv.last_file)]
    # Trailing `-` reads the prompt from stdin, which keeps a large brief clear of ARG_MAX.
    argv.append("-")
    return argv


@dataclass(frozen=True)
class Provider:
    name: str
    binary: str
    default_model: str
    default_effort: str
    # The levels the provider CLI parses. `-e` is checked against these before anything runs:
    # codex parses an unknown level as a custom one and writes it into a `-c` config override,
    # so an unchecked value is config injection. Whether a given model supports a level is
    # still the CLI's call.
    effort_levels: tuple[str, ...]
    # Whether the event stream names the model that actually served the run. When it should
    # and does not, that is worth a warning; when it never does, silence is the only answer.
    reports_served_model: bool
    mode: str
    # Which event vocabulary the run's stdout stream speaks, so the wrapper can count the
    # tool calls it made. Both providers write one; the schemas share nothing.
    events_mode: str
    argv: Callable[[Invocation], list[str]]
    # codex takes its prompt on stdin; grok is given a path and opens the file itself.
    prompt_on_stdin: bool
    # codex writes its final message to a separate file via -o; grok's answer is in the stream.
    writes_last_message: bool
    install_hint: str

    @property
    def model_help(self) -> str:
        return f"{self.name} model (default: {self.default_model})"

    @property
    def effort_help(self) -> str:
        return f"reasoning effort: {'|'.join(self.effort_levels)} (default: {self.default_effort})"

    @property
    def command(self) -> str:
        """The console command that drives this provider."""
        return f"ce-{self.name}-persona"

    @property
    def validate_command(self) -> str:
        """The console command that validates a findings batch through this provider.

        A LABEL only. `argv` is unchanged between the two modes: `Invocation.schema_text`
        carries whichever schema is being enforced, and only grok's `--json-schema` reads
        it, so a validation is the same invocation with a different prompt and a different
        schema rather than a second runner to keep in step.
        """
        return f"ce-{self.name}-validate"


GROK = Provider(
    name="grok",
    binary="grok",
    default_model="grok-4.7",
    # The top tier; the TUI labels it "Deep / Maximum reasoning". Not `max`: grok's
    # ReasoningEffort enum carries it and clap parses it, but the builds reject it during
    # validation, so it is not a level here either. A level a newer build accepts needs a
    # release of this package, the same channel as a change of default.
    default_effort="xhigh",
    effort_levels=("low", "medium", "high", "xhigh"),
    # The keys of the terminal event's `modelUsage`. The `model` on the init and assistant
    # events echoes the requested id, so it is no evidence of what served.
    reports_served_model=True,
    mode=MODE_GROK_EVENTS,
    events_mode=EVENTS_GROK,
    argv=_grok_argv,
    prompt_on_stdin=False,
    writes_last_message=False,
    install_hint="see https://github.com/xai-org/grok-build",
)

CODEX = Provider(
    name="codex",
    binary="codex",
    default_model="gpt-6.1-sol",
    # Explicit rather than ~/.codex/config.toml's, so every machine runs the same review.
    # high, not xhigh: on gpt-6.1-sol it found 16 of 21 planted defects to xhigh's 15, in
    # half the time.
    default_effort="high",
    effort_levels=("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"),
    # `codex exec --json` names no model anywhere in its stream.
    reports_served_model=False,
    mode=MODE_OBJECT,
    events_mode=EVENTS_CODEX,
    argv=_codex_argv,
    prompt_on_stdin=True,
    writes_last_message=True,
    install_hint="install the OpenAI codex CLI and run `codex login`",
)

PROVIDERS: dict[str, Provider] = {p.name: p for p in (GROK, CODEX)}
