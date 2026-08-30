"""The findings gate: hold a model's answer to the plugin's findings-schema.json.

This is the one component whose failure mode is SILENT. Everything else exits non-zero and
says why; a gate that wrongly passes reports a clean review of a change nobody reviewed. It
has been wrong that way three times, and every one was found by a reviewer rather than by
this package's own tests, so the rules below are written to fail closed and the regressions
are pinned by name in the test suite.

TWO EXTRACTION MODES, BECAUSE THE RUNNERS DIFFER
------------------------------------------------
`grok-events` reads grok's NDJSON and takes `structured_output` from the terminal `result`
event: `--json-schema` composes with `--output-format streaming-messages-json`, so the
answer arrives parsed AND the stream still grows for liveness.

`object` reads a file that IS the answer — `codex exec -o` writes only the final message.
It requires that file to be one JSON object and nothing else. That strictness is the point,
not an inconvenience: when prose is allowed around the answer, a model that gives up can
append the brief's own schema-valid EXAMPLE object and read as a clean review, and a model
that found real defects can append the same example and have them silently replaced by an
empty array. The prompt asks for a bare object precisely so this can stay strict.

THERE IS NO THIRD MODE, DELIBERATELY
------------------------------------
A prose-scanning fallback used to exist for a hypothetical runner offering only plain text.
No shipped provider ever used it, and it accounted for SIX separate false-passes: the
persona brief's echoed example validating as a clean review, a quoted object validating
after the boundary cut, a trailing empty object wiping a real answer, an unanchored cut
slicing into a real object, a replayed boundary hiding the answer from the ambiguity check,
and a non-empty object supplied by the reviewed repository replacing the model's findings.

Each fix was correct and each exposed the next, because the premise is unsound: you cannot
reliably tell "the model's answer" from "text the model quoted" by scanning output. Both
remaining modes read a channel the RUNNER delimits, so the question never arises. A runner
that cannot provide one is unsupported rather than supported badly.

ONE QUESTION ABOUT THE ANSWER, ONE ABOUT THE RUN
------------------------------------------------
The schema rules judge what the model SAID. `run_stats` counts what it DID, from the event
stream the wrapper already keeps, and `gate` refuses a run that made zero tool calls: a
reviewer that never opened a file cannot certify anything, and its findings are unfounded
whether the array is empty or full. Exactly zero is the threshold, with no configurable
floor — "did this run inspect anything" has an answer, while "did it inspect enough" is a
judgement this package is not entitled to make.

WHAT THIS DOES NOT COVER
------------------------
A well-formed but EMPTY findings array from a run that DID inspect the code is schema-valid
and stays valid: distinguishing "found nothing" from "looked, then gave up" needs a
judgement about the transcript this does not attempt. Two checks narrow that gap from either
side and neither closes it — the grok path reads the run's own terminal status before
believing its answer, but `stop_reason` is an open vocabulary and the denylist cannot be
exhaustive; the tool-call count catches a run that inspected nothing at all, but one tool
call is not diligence.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from . import errors

# Parsed JSON, spelled out. The schema is owned by the compound-engineering plugin, read at
# run time, and free to grow fields this package has never heard of — so the value type is
# "some JSON", narrowed by isinstance where it is used, rather than a TypedDict asserting a
# shape this package has no authority to fix.
type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]
Artifact = JSONObject


# Re-exported from errors.py, where its exit status lives. `validate.GateError` is the name
# every raise site and every test already uses, and it reads correctly here.
GateError = errors.GateError


def fail(message: str) -> NoReturn:
    raise GateError(message)


def loads(text: str) -> JSONValue:
    return cast(JSONValue, json.loads(text))


def _as_object(value: JSONValue, what: str) -> JSONObject:
    if not isinstance(value, dict):
        fail(f"{what} is not a JSON object")
    return value


# JSON Schema type -> what Python accepts. bool is excluded from the numeric types because
# `True` is an int in Python and a model emitting `"line": true` must not pass.
JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

# The subset of JSON Schema this gate implements. Declaring it is the point: the schema is
# owned by the plugin and can grow, and a validator that silently ignores a keyword it does
# not implement certifies against rules nobody checked. Anything outside these two sets is
# a hard failure with a message naming the keyword.
ENFORCED_KEYWORDS = frozenset(
    {
        "type",
        "enum",
        "required",
        "properties",
        "items",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
    }
)
# Annotations and metadata: they constrain nothing, so ignoring them is honest.
IGNORED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        # NOT additionalProperties. It is a constraint, not an annotation, and it is the
        # likeliest keyword for the plugin to add — the day it does, filing it here would
        # keep this gate returning 0 while enforcing nothing about extra keys, which is
        # precisely the silent certification the mechanism exists to prevent. Left out of
        # BOTH sets so it hard-fails on arrival rather than passing unchecked.
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)


def check_schema_supported(spec: JSONObject, label: str = "schema") -> None:
    """Refuse a schema using rules this gate does not implement."""
    unknown = sorted(set(spec) - ENFORCED_KEYWORDS - IGNORED_KEYWORDS)
    if unknown:
        fail(
            f"{label} uses JSON Schema keyword(s) this gate does not implement: "
            f"{', '.join(unknown)}. Passing a rule it cannot check is how a validator "
            f"certifies a shape nobody validated."
        )
    # Positional, not just global membership. The accept-list says WHICH keywords exist;
    # enforcement happens at specific positions, and the two disagreeing is how a schema
    # passes the gate and is then never checked. Two shapes did exactly that: tuple-form
    # `items` (a list of per-element schemas) and object rules nested inside a property.
    items = spec.get("items")
    if items is not None and not isinstance(items, dict):
        fail(
            f"{label}.items is not a single schema object. Tuple-form items is a rule this "
            f"gate does not implement, and accepting it would check nothing."
        )
    if isinstance(items, dict):
        check_schema_supported(items, f"{label}.items")

    properties = spec.get("properties")
    if isinstance(properties, dict):
        for key, sub in properties.items():
            if not isinstance(sub, dict):
                fail(f"{label}.properties.{key} is not a schema object")
            # Only the top level and a finding's items carry rules this gate enforces.
            # Anything deeper declaring `required` or `properties` would be silently
            # ignored, so refuse it by name instead.
            for deeper in ("required", "properties"):
                if deeper in sub:
                    fail(
                        f"{label}.properties.{key} declares `{deeper}`, which this gate "
                        f"enforces only at the top level and on a finding. Accepting it "
                        f"would certify against a rule nobody checked."
                    )
            check_schema_supported(sub, f"{label}.{key}")


def _type_names(label: str, spec: JSONObject) -> list[str]:
    """The declared type(s), normalised to a list of known names.

    JSON Schema lets `type` be a LIST, and the plugin schema uses that form:
    `"suggested_fix": {"type": ["string", "null"]}`. Reading only the `str` case skips that
    field's check entirely.
    """
    declared = spec.get("type")
    if declared is None:
        return []
    raw: list[JSONValue] = declared if isinstance(declared, list) else [declared]
    names = [name for name in raw if isinstance(name, str) and name in JSON_TYPES]
    if len(names) != len(raw):
        fail(f"{label} declares a type this gate cannot check: {declared!r}")
    return names


def _check_type(label: str, value: JSONValue, spec: JSONObject) -> None:
    """Enforce the declared type and the schema's bounds for one field."""
    names = _type_names(label, spec)
    if names:
        allowed = tuple(t for name in names for t in JSON_TYPES[name])
        # A bool satisfies `isinstance(x, int)`, so any union admitting a numeric type must
        # reject booleans unless it also admits `boolean` outright. Keying this on "every
        # member is numeric" let `["integer", "null"]` accept `true`.
        numeric = any(name in ("integer", "number") for name in names)
        ok = isinstance(value, allowed) and not (
            numeric and "boolean" not in names and isinstance(value, bool)
        )
        if not ok:
            fail(f"{label} must be {' or '.join(names)}, got {type(value).__name__}")

    if isinstance(value, list):
        low, high = spec.get("minItems"), spec.get("maxItems")
        if isinstance(low, int) and not isinstance(low, bool) and len(value) < low:
            fail(f"{label} needs at least {low} item(s), got {len(value)}")
        if isinstance(high, int) and not isinstance(high, bool) and len(value) > high:
            fail(f"{label} allows at most {high} item(s), got {len(value)}")
        items = spec.get("items")
        if isinstance(items, dict):
            for n, element in enumerate(value):
                _check_type(f"{label}[{n}]", element, items)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low_n, high_n = spec.get("minimum"), spec.get("maximum")
        if isinstance(low_n, (int, float)) and not isinstance(low_n, bool) and value < low_n:
            fail(f"{label} must be >= {low_n}, got {value}")
        if isinstance(high_n, (int, float)) and not isinstance(high_n, bool) and value > high_n:
            fail(f"{label} must be <= {high_n}, got {value}")

    if isinstance(value, str):
        min_len, max_len = spec.get("minLength"), spec.get("maxLength")
        if isinstance(min_len, int) and not isinstance(min_len, bool) and len(value) < min_len:
            fail(f"{label} is {len(value)} characters, under the schema's minLength of {min_len}")
        if isinstance(max_len, int) and not isinstance(max_len, bool) and len(value) > max_len:
            fail(f"{label} is {len(value)} characters, over the schema's maxLength of {max_len}")


def validate(found: Artifact, schema: JSONObject) -> int:
    check_schema_supported(schema)

    required = schema.get("required")
    if isinstance(required, list):
        missing = [k for k in required if isinstance(k, str) and k not in found]
        if missing:
            fail(f"findings JSON missing required keys: {', '.join(missing)}")
    entries = found.get("findings")
    if not isinstance(entries, list):
        fail("findings must be an array")

    top_properties = schema.get("properties")
    if not isinstance(top_properties, dict):
        fail("findings schema has no properties object")
    for key, spec in top_properties.items():
        if key != "findings" and key in found and isinstance(spec, dict):
            _check_type(key, found[key], spec)

    findings_spec = _as_object(top_properties.get("findings"), "findings schema")
    item = findings_spec.get("items")
    if not isinstance(item, dict):
        fail("findings schema has no properties.findings.items object")
    properties = item.get("properties")
    if not isinstance(properties, dict) or not properties:
        fail("findings schema has no properties.findings.items.properties object")
    item_required = item.get("required")
    if not isinstance(item_required, list) or not item_required:
        fail("findings schema declares no required finding fields")

    # Every enum the schema declares, not just severity. The shape this replaced taught
    # `autofix_class: safe_auto` and `owner: review-fixer`, neither of which the current
    # schema allows, so a model repeating either must not pass.
    enums: dict[str, list[JSONValue]] = {}
    for key, spec in properties.items():
        if isinstance(spec, dict):
            allowed = spec.get("enum")
            if isinstance(allowed, list) and allowed:
                enums[key] = allowed
    if not enums:
        fail("findings schema declares no finding enums")

    for n, finding in enumerate(entries, 1):
        if not isinstance(finding, dict):
            fail(f"finding {n} is not an object")
        absent = [k for k in item_required if isinstance(k, str) and k not in finding]
        if absent:
            fail(f"finding {n} missing {', '.join(absent)}")
        for key, allowed in enums.items():
            if key in finding and finding[key] not in allowed:
                fail(f"finding {n} has {key}={finding[key]!r}, not one of {allowed}")
        for key, spec in properties.items():
            # Guarded the same way the top-level loop is: a non-object property spec is a
            # malformed schema, not something to walk into.
            if key in finding and isinstance(spec, dict):
                _check_type(f"finding {n} field {key}", finding[key], spec)
    return len(entries)


def from_object_file(text: str) -> Artifact:
    """The findings object from a file that is the final message and nothing else.

    Strict by design. `codex exec -o` writes only the final message and the prompt demands
    that message be one bare JSON object, so anything around it means the model did not
    answer the way it was asked — and the shapes that arrive when it does not are exactly
    the ones that read as a clean review while being nothing of the kind.
    """
    try:
        decoded = loads(text)
    except ValueError as exc:
        fail(
            f"final message is not a single JSON object: {exc}. The prompt requires the "
            f"final message to be exactly one JSON object with no prose or code fences."
        )
    if not isinstance(decoded, dict) or "findings" not in decoded:
        fail("final message is JSON but carries no findings key")
    return decoded


# `subtype` is a closed vocabulary, so an allowlist is safe. `stop_reason` is open, so the
# denylist is deliberately not exhaustive: an unfamiliar but healthy value must not fail a
# completed review. Calibrated against real grok 1.0.4/1.0.5 success events, which carry
# subtype="success" and stop_reason="end_turn".
GOOD_SUBTYPES = frozenset({"success"})
BAD_STOP_REASONS = frozenset(
    {
        "max_tokens",
        "max_output_tokens",
        "max_turns",
        "length",
        "content_filter",
        "refusal",
        "error",
        "timeout",
        "aborted",
        "cancelled",
        "canceled",
    }
)


def from_grok_events(text: str) -> Artifact:
    """The findings object from grok's NDJSON `result` event.

    The run's own terminal status is checked before its answer is believed. That matters
    because schema-constrained decoding means a truncated or refused run STILL returns a
    well-formed `{"findings": []}`, which is indistinguishable from a clean review by
    looking at the payload alone.
    """
    results: list[JSONObject] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            results.append(event)

    if not results:
        fail("no `result` event in grok's output stream")
    if len(results) > 1:
        # REFUSE, rather than preferring either end. "Last wins" looks safe — a late failure
        # should not be certified by an early success — but it is only safe when the extra
        # events are unhealthy. Two HEALTHY terminal events silently hand the run to the
        # second: a stream carrying a real P0 review followed by `findings: []` exits 0 and
        # reports clean, and every terminal-status check passes because the surviving event
        # genuinely is healthy. A run has one verdict; more than one is not a verdict.
        fail(
            f"grok's output stream carries {len(results)} `result` events; a run has one "
            "verdict, and choosing among them would let a second event overwrite the answer"
        )
    result = results[0]

    # PRESENT and healthy, not "absent is fine". A terminal event that carries none of these
    # is not evidence of a completed run, and schema-constrained decoding means the payload
    # beside it is a well-formed empty findings array either way — so treating absence as
    # success accepts exactly the shape this gate exists to reject. Real grok 1.0.4 and
    # 1.0.5 success events carry all three, so requiring them costs nothing and a future
    # build that drops one fails loudly rather than silently certifying.
    if result.get("is_error") is not False:
        fail(
            f"grok's result event does not report is_error=false "
            f"(is_error={result.get('is_error')!r}, stop_reason={result.get('stop_reason')!r})"
        )
    subtype = result.get("subtype")
    if not isinstance(subtype, str) or subtype not in GOOD_SUBTYPES:
        fail(f"grok's run ended with subtype={subtype!r}, which is not a completed review")
    stop = result.get("stop_reason")
    if not isinstance(stop, str):
        fail(f"grok's result event carries no stop_reason (got {stop!r})")
    if stop in BAD_STOP_REASONS:
        fail(f"grok stopped early (stop_reason={stop!r}); the answer is truncated or refused")

    obj = result.get("structured_output")
    if obj is not None:
        # PRESENT but wrong is a failure, not a reason to look elsewhere. Falling through to
        # the raw text when `--json-schema` was in force means the schema-constrained channel
        # produced something unexpected and the gate quietly used a different one instead —
        # the answer it returns is then from a channel nobody asked for.
        if not isinstance(obj, dict) or "findings" not in obj:
            fail(
                "grok's result event carries a structured_output that is not a findings "
                f"object ({type(obj).__name__}); --json-schema was requested, so this is a "
                "malformed answer rather than a reason to read the raw text"
            )
        return obj
    # No structured output at all: the run was not schema-constrained, so the final text is
    # the only channel left. Strict, like codex's.
    raw = result.get("result")
    if isinstance(raw, str) and raw.strip():
        return from_object_file(raw)
    fail("grok's result event carried no structured output")


@dataclass(frozen=True)
class RunStats:
    """What a run DID, as opposed to what it said.

    `tool_calls` is an int and is never None, because it is the only field here that gates
    anything: "the adapter could not tell" is not an answer this package is entitled to
    give. A stream carrying nothing this module recognises counts zero and the run is
    refused; a stream that cannot be READ is an environment error rather than a quiet zero,
    because the two have different causes and only one of them is about the model. The rest
    are provenance — best effort, and None where a provider publishes no comparable number.
    """

    tool_calls: int
    turns: int | None
    output_tokens: int | None
    duration_s: float | None


# A tool call in grok's stream is a `tool_use` content block inside an ASSISTANT message.
# The matching `tool_result` comes back in a `user` message, so counting blocks without
# looking at the event type would double every total. Verified against grok 1.0.13: one real
# review carried 99 `tool_use` blocks across 31 turns.
GROK_TOOL_BLOCK = "tool_use"

# codex's `--json` stream is items rather than messages: one `item.started` and one
# `item.completed` per call, both carrying the same `item.id`. These are the kinds that
# reach outside the model; `agent_message` and `reasoning` are the model talking to itself
# and are deliberately absent. Verified against codex-cli 0.150.1.
#
# An unrecognised kind is NOT counted, and that is the fail-closed direction on purpose: a
# provider that renames its vocabulary makes every run refuse loudly, rather than certifying
# a review nobody can show happened.
CODEX_TOOL_ITEMS = frozenset(
    {
        "command_execution",
        "custom_tool_call",
        "file_change",
        "function_call",
        "local_shell_call",
        "mcp_tool_call",
        "patch_apply",
        "todo_list",
        "web_search",
    }
)


def _stream(events_file: Path) -> Iterator[JSONObject]:
    """Every JSON object in a run's event stream, one line at a time.

    Streamed rather than read whole: a real stream is around a megabyte and its size is the
    provider's decision, not this package's. A line that does not parse is skipped, because
    the stream is append-only and a killed run leaves a partial last line — a condition the
    watchdogs have already judged, and not one to re-decide here.
    """
    try:
        with events_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    event = loads(text)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError as exc:
        # The machine, not the answer: this process wrote the file moments ago, so failing
        # to read it back is the same class of problem as an unwritable run directory.
        raise errors.EnvError(f"cannot read the event stream {events_file}: {exc}") from exc


def _whole_number(value: JSONValue) -> int | None:
    """A count a provider reported, or None. `True` is an `int` in Python and is not one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _grok_stats(events_file: Path) -> tuple[int, int | None, int | None]:
    calls = 0
    turns: int | None = None
    output_tokens: int | None = None
    for event in _stream(events_file):
        kind = event.get("type")
        if kind == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                calls += sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == GROK_TOOL_BLOCK
                )
        elif kind == "result":
            turns = _whole_number(event.get("num_turns"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = _whole_number(usage.get("output_tokens"))
    return calls, turns, output_tokens


def _codex_stats(events_file: Path) -> tuple[int, int | None, int | None]:
    seen: set[str] = set()
    calls = 0
    # codex's own turn accounting, which counts one per `exec` turn rather than one per
    # model round-trip. It is not comparable with grok's `num_turns` and is recorded because
    # it is the number codex publishes, not because the two mean the same thing.
    turns = 0
    output_tokens: int | None = None
    for event in _stream(events_file):
        kind = event.get("type")
        if kind == "turn.started":
            turns += 1
        elif kind == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = _whole_number(usage.get("output_tokens"))
        elif kind in ("item.started", "item.completed"):
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in CODEX_TOOL_ITEMS:
                continue
            # One call, two events. Deduped on the item's own id rather than counted from
            # `item.completed` alone, so a call the run never finished still counts as
            # evidence that something was inspected.
            ident = item.get("id")
            if isinstance(ident, str):
                if ident in seen:
                    continue
                seen.add(ident)
            calls += 1
    return calls, turns, output_tokens


def run_stats(mode: str, events_file: Path, duration_s: float | None) -> RunStats:
    """Count what the run did, through the event vocabulary its runner actually speaks.

    Named modes rather than sniffing the stream's shape. Guessing which schema a file is in
    is the same undecidable move as scanning prose for the model's answer, and the two event
    vocabularies share nothing: one is messages carrying content blocks, the other is items
    carrying kinds.

    `duration_s` is measured by the caller rather than read from the stream. Only one of the
    two providers publishes a duration, and a field that means a different thing depending on
    who produced it is worse than one that means the same thing everywhere.
    """
    if mode == "grok-events":
        calls, turns, tokens = _grok_stats(events_file)
    elif mode == "codex-events":
        calls, turns, tokens = _codex_stats(events_file)
    else:
        fail(f"unknown event mode {mode!r}")
    return RunStats(tool_calls=calls, turns=turns, output_tokens=tokens, duration_s=duration_s)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _evidence(stats: RunStats) -> str:
    """The counts behind a refusal, in one clause. A refusal nobody can check is a rumour."""
    parts: list[str] = []
    if stats.turns is not None:
        parts.append(_plural(stats.turns, "turn"))
    parts.append(_plural(stats.tool_calls, "tool call"))
    if stats.duration_s is not None:
        parts.append(f"{stats.duration_s:.1f}s")
    if stats.output_tokens is not None:
        parts.append(f"{stats.output_tokens} output tokens")
    return ", ".join(parts)


def _sha256(filename: str) -> str:
    """Content hash, or a recorded reason. An unreadable asset is provenance too, and must
    not abort a review that already ran."""
    try:
        return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


def write_provenance(path: Path, pairs: list[str], files: dict[str, str], stats: RunStats) -> None:
    """Record which brief and schema this run used, by content hash, and what it did.

    Two arms are only comparable if they ran the same brief, and briefs live in a plugin
    cache that updates underneath us, so without this the drift between two runs is
    invisible and a comparison silently stops meaning anything. A sidecar, so the findings
    artifact keeps exactly the shape the plugin's fold-in expects.

    `stats` is required rather than optional: it is what the exit status turns on, and a
    caller that could forget to pass it would silently get a record attesting nothing about
    whether the run inspected anything.
    """
    record: JSONObject = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        record[key] = value
    for key, filename in files.items():
        record[f"{key}_file"] = filename
        record[f"{key}_sha256"] = _sha256(filename)
    # WHAT THE RUN DID, beside what produced it. `tool_calls` decides the exit status, so it
    # is written down rather than only acted on: a refusal a caller cannot audit afterwards
    # is one it has to take on trust.
    record["run_stats"] = {
        "tool_calls": stats.tool_calls,
        "turns": stats.turns,
        "output_tokens": stats.output_tokens,
        "duration_s": stats.duration_s,
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


def extract(mode: str, text: str) -> Artifact:
    """Recover the model's answer through the channel its runner actually provides.

    Both modes read a channel the RUNNER defines — grok's terminal event, codex's
    final-message file — so neither has to decide which part of a transcript is the answer.
    There is deliberately no prose-scanning mode: telling "the model's answer" from "text
    the model quoted" by scanning output is not a decidable problem, and the attempt hosted
    six separate false-passes before it was removed. A runner offering only plain text is
    not supported rather than supported badly.
    """
    if mode == "object":
        return from_object_file(text)
    if mode == "grok-events":
        return from_grok_events(text)
    fail(f"unknown extraction mode {mode!r}")


def gate(
    *,
    answer_file: Path,
    schema_path: Path,
    mode: str,
    findings_out: Path,
    provenance_out: Path,
    prov_pairs: list[str],
    prov_files: dict[str, str],
    stats: RunStats,
    label: str,
) -> int:
    """Validate a run's answer and write its artifacts. Returns the process exit status.

    Raises `VacuousRun` when the run made no tool calls — after the artifacts are written,
    because that dud IS the evidence and a caller has to be able to look at what it refused.
    """
    try:
        text = answer_file.read_text(encoding="utf-8", errors="replace")
        try:
            schema = _as_object(loads(schema_path.read_text(encoding="utf-8")), "findings schema")
        except (OSError, ValueError) as exc:
            fail(f"cannot read findings schema {schema_path}: {exc}")
        found = extract(mode, text)
        count = validate(found, schema)
    except GateError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return 1

    # Findings first, provenance second: the sidecar's job is attesting THIS artifact, so it
    # must never be the only thing on disk.
    findings_out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    write_provenance(provenance_out, prov_pairs, prov_files, stats)

    # A reviewer that made ZERO tool calls never opened the diff, so it has no verdict to
    # summarize: empty findings and a page of them are equally unfounded. Decided here, ahead
    # of the summary the rest of this function builds — a gating caller reads only the status,
    # and a line saying "0 findings" beside a status saying "refused" is the exact ambiguity
    # being closed.
    #
    # Deliberately no retry here. Whether to spend another full-effort model run is the
    # calling agent's decision and its budget; this command's job is to refuse to certify,
    # and a wrapper that silently re-ran would hide how often this happens.
    if stats.tool_calls == 0:
        raise errors.VacuousRun(
            f"refusing to report {count} findings from a run that made no tool calls "
            f"({_evidence(stats)}). The model never opened the diff, so nothing it "
            f"reported is founded; the artifacts are kept at {findings_out} as evidence."
        )

    entries = found.get("findings")
    tally: dict[str, int] = {}
    if isinstance(entries, list):
        for finding in entries:
            severity = finding.get("severity") if isinstance(finding, dict) else None
            key = severity if isinstance(severity, str) else "?"
            tally[key] = tally.get(key, 0) + 1
    breakdown = ", ".join(f"{n} {sev}" for sev, n in sorted(tally.items()))

    # stdout is the caller's context: one line, never the transcript. Severity counts first
    # so a caller can triage without opening the artifact at all.
    #
    # Flushed here, and a broken pipe swallowed: the one-line contract invites
    # `ce-grok-persona ... | head -1`, which closes the pipe. The interpreter's shutdown
    # flush would then raise where no handler can catch it and exit 120 — reporting failure
    # for a review that succeeded and whose artifacts are already on disk.
    try:
        print(
            f"{label}: {count} findings{f' ({breakdown})' if breakdown else ''} -> {findings_out}"
        )
        sys.stdout.flush()
    except BrokenPipeError:
        # Point the interpreter's shutdown flush at /dev/null, closing the fd we opened to
        # do it: dup2 duplicates, it does not consume.
        with contextlib.suppress(OSError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, sys.stdout.fileno())
            finally:
                os.close(devnull)
    return 0
