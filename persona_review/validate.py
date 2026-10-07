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
stream the wrapper already keeps, and `gate` refuses a run with zero successful local tool
calls: a reviewer not shown to have read a file cannot certify anything, and its findings
are unfounded whether the array is empty or full. A web search, or a tool that does not say
where it runs, is a call but not a local one; a local call that failed shows no read either,
since one that never started and one that ran and matched nothing look alike. A run that
attempted no local call exits 6, and one whose every attempt failed exits 3, because only
the first is the model's doing. Exactly zero is the threshold, with no configurable
floor — "did this run inspect anything" has an answer, while "did it inspect enough" is a
judgment this package is not entitled to make.

WHAT THIS DOES NOT COVER
------------------------
A well-formed but EMPTY findings array from a run that DID inspect the code is schema-valid
and stays valid: distinguishing "found nothing" from "looked, then gave up" needs a
judgment about the transcript this does not attempt. Two checks narrow that gap from either
side and neither closes it — the grok path reads the run's own terminal status before
believing its answer, but `stop_reason` is an open vocabulary and the denylist cannot be
exhaustive; the tool-call count catches a run that inspected nothing at all, but one
successful tool call is not diligence.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
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


def objects(lines: Iterable[str]) -> Iterator[JSONObject]:
    """Every JSON object in an NDJSON stream, in order.

    THE ONE PLACE THE SKIP RULES LIVE. Both readers of a provider's event stream — the
    answer extractor and the tool-call counter — need "one JSON object per line, ignore
    what is not one", and a second copy of those four rules is a second thing to drift.

    A line that does not parse is skipped rather than fatal: the stream is append-only, a
    killed run leaves a partial last line, and that is a condition the watchdogs have
    already judged rather than one to re-decide here.
    """
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            event = loads(text)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def ndjson_lines(text: str) -> list[str]:
    """A stream's records. NDJSON ends one at "\\n" and nowhere else: `str.splitlines` also
    breaks at U+2028, U+2029 and U+0085, which JSON carries raw inside a string, and a record
    cut there parses as neither half."""
    return text.split("\n")


def file_objects(events_file: Path) -> Iterator[JSONObject]:
    """The same, streamed from a file a line at a time.

    Never `read_text`: a real event stream is around a megabyte and its size is the
    provider's decision, not this package's.

    Only the OPEN is guarded. A failure there means the file this process wrote moments ago
    is not there or not readable, which is the machine's problem; an OSError part-way
    through a read of a local file is a different and far stranger animal, and swallowing it
    into the same message would report the wrong cause.
    """
    try:
        handle = events_file.open(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise errors.EnvError(f"cannot open the event stream {events_file}: {exc}") from exc
    with handle:
        yield from objects(handle)


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
    """The declared type(s), normalized to a list of known names.

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


def check_object(
    found: Artifact, schema: JSONObject, key: str = "findings", *, demand_item_rules: bool = False
) -> int:
    """Hold one answer object to one schema, and return how many entries it carried.

    The generic walk, shared by the findings gate and the verdicts gate: the schema's own
    top-level `required`, the types of every top-level property, and each element of the
    `key` array against `properties.<key>.items`. Nothing here knows what a finding or a
    verdict means — the rules come from the schema in every case.

    `demand_item_rules` is the findings gate's extra demand and is off by default, because
    the two assertions behind it (`required` and `enum` declared on an item) are statements
    about the PLUGIN's schema rather than about JSON Schema: the verdicts schema declares no
    enums at all, so a walk that insisted on them would fail closed on every validation.
    """
    # What ONE element is called, for messages a person reads: "finding 3 missing title",
    # "verdict 3 missing reason". Both keys this gate is given are regular plurals.
    item_noun = key.removesuffix("s") or key
    check_schema_supported(schema)

    required = schema.get("required")
    if isinstance(required, list):
        missing = [k for k in required if isinstance(k, str) and k not in found]
        if missing:
            fail(f"{key} JSON missing required keys: {', '.join(missing)}")
    entries = found.get(key)
    if not isinstance(entries, list):
        fail(f"{key} must be an array")

    top_properties = schema.get("properties")
    if not isinstance(top_properties, dict):
        fail(f"{key} schema has no properties object")
    for name, spec in top_properties.items():
        if name != key and name in found and isinstance(spec, dict):
            _check_type(name, found[name], spec)

    entries_spec = _as_object(top_properties.get(key), f"{key} schema")
    item = entries_spec.get("items")
    if not isinstance(item, dict):
        fail(f"{key} schema has no properties.{key}.items object")
    properties = item.get("properties")
    if not isinstance(properties, dict) or not properties:
        fail(f"{key} schema has no properties.{key}.items.properties object")
    item_required = item.get("required")
    if demand_item_rules and (not isinstance(item_required, list) or not item_required):
        fail("findings schema declares no required finding fields")
    required_fields = item_required if isinstance(item_required, list) else []

    # Every enum the schema declares, not just severity. The shape this replaced taught
    # `autofix_class: safe_auto` and `owner: review-fixer`, neither of which the current
    # schema allows, so a model repeating either must not pass.
    enums: dict[str, list[JSONValue]] = {}
    for name, spec in properties.items():
        if isinstance(spec, dict):
            allowed = spec.get("enum")
            if isinstance(allowed, list) and allowed:
                enums[name] = allowed
    if demand_item_rules and not enums:
        fail("findings schema declares no finding enums")

    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            fail(f"{item_noun} {n} is not an object")
        absent = [k for k in required_fields if isinstance(k, str) and k not in entry]
        if absent:
            fail(f"{item_noun} {n} missing {', '.join(absent)}")
        for name, allowed in enums.items():
            if name in entry and entry[name] not in allowed:
                fail(f"{item_noun} {n} has {name}={entry[name]!r}, not one of {allowed}")
        for name, spec in properties.items():
            # Guarded the same way the top-level loop is: a non-object property spec is a
            # malformed schema, not something to walk into.
            if name in entry and isinstance(spec, dict):
                _check_type(f"{item_noun} {n} field {name}", entry[name], spec)
    return len(entries)


def validate(found: Artifact, schema: JSONObject) -> int:
    """The findings gate: the generic walk, plus what the plugin's schema must declare.

    Kept as its own name because it is `gate`'s default `check` and three call sites already
    pass it; the demands it adds are the ones that only make sense about findings.
    """
    return check_object(found, schema, "findings", demand_item_rules=True)


def from_object_file(text: str, key: str = "findings") -> Artifact:
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
    if not isinstance(decoded, dict) or key not in decoded:
        fail(f"final message is JSON but carries no {key} key")
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


def from_grok_events(text: str, key: str = "findings") -> Artifact:
    """The findings object from grok's NDJSON `result` event.

    The run's own terminal status is checked before its answer is believed. That matters
    because schema-constrained decoding means a truncated or refused run STILL returns a
    well-formed `{"findings": []}`, which is indistinguishable from a clean review by
    looking at the payload alone.
    """
    results = [event for event in objects(ndjson_lines(text)) if event.get("type") == "result"]
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
        if not isinstance(obj, dict) or key not in obj:
            fail(
                f"grok's result event carries a structured_output that is not a {key} "
                f"object ({type(obj).__name__}); --json-schema was requested, so this is a "
                "malformed answer rather than a reason to read the raw text"
            )
        return obj
    # No structured output at all: the run was not schema-constrained, so the final text is
    # the only channel left. Strict, like codex's.
    raw = result.get("result")
    if isinstance(raw, str) and raw.strip():
        return from_object_file(raw, key)
    fail("grok's result event carried no structured output")


@dataclass(frozen=True)
class RunStats:
    """What a run DID, as opposed to what it said.

    `local_tool_calls` is the field that gates: the calls that can read the machine the
    review ran on, which is where the repository is, AND that the stream shows succeeded. A
    run whose only calls were web searches read the internet and not the diff, one whose only
    calls were edits wrote to the tree without reading it, and a command that could not start
    read nothing at all, so none of them has certified anything. A call counts only when its
    id pairs it with its outcome unambiguously: an id two different tools or item kinds
    share, or one reported both succeeding and failing, counts for none of them.
    `local_tool_attempts` is every local call, succeeded or not, and it is what tells the
    two refusals apart: none attempted is the model's doing, every one failing is not.
    `tool_calls` is every call, local or not, recorded so the refusal can say what the run
    did instead. All three are ints when
    counted from a stream, because "the adapter could not tell" is not an answer this package
    is entitled to give. A stream carrying nothing this module recognizes counts zero and the
    run is refused; a stream that cannot be READ is an environment error rather than a quiet
    zero, because the two have different causes and only one of them is about the model. The
    rest are provenance — best effort, and None where a provider publishes no comparable
    number.

    `local_tool_attempts` is None where it was never recorded: a sidecar written before
    0.3.5, whose `local_tool_calls` counted attempts rather than successes. `first_failure`
    is the first line the first failed local call printed, `first_status` the status it
    reported, as JSON text, when the success rule refuses that status, and `ambiguous_id`
    the first id whose reported success did not count because the id was ambiguous, all for
    the refusal to quote. None is written to provenance, because the event stream kept
    beside the sidecar holds all of it.

    `served_models` is every model id the stream says served the run, sorted, or None when
    it names none: the provider does not report one, or this stream carried no usable one.
    """

    tool_calls: int
    local_tool_calls: int
    turns: int | None
    output_tokens: int | None
    duration_s: float | None
    local_tool_attempts: int | None = None
    first_failure: str | None = None
    ambiguous_id: str | None = None
    first_status: str | None = None
    served_models: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Evidence:
    """Where a run's own account of itself lives, and how long the run took.

    Passed to `gate` instead of a finished `RunStats` so the counting happens where the
    stream has already been read. grok's answer arrives INSIDE its event stream, so
    `events_file` is then the same path as the answer file and the gate counts from the text
    it already holds rather than opening a megabyte twice.

    `model` is the id the run asked for, and `reports_model` whether its provider's stream
    is expected to name the model that served it, so the gate can say when the two differ.
    """

    events_file: Path
    mode: str
    duration_s: float | None
    model: str = ""
    reports_model: bool = False


# BOTH ADAPTERS ASK TWO QUESTIONS: did the model reach outside itself, and did what it
# reached for work? They answer them differently because the streams differ in kind, and
# the difference is worth stating rather than discovering.
#
# grok names a tool call structurally — a `tool_use` content block — and codex by an item
# KIND. Both adapters still carry a list of what reads the tree, grok's by tool name, because
# some calls that succeed read nothing; and a list can go out of date, so both carry the
# drift detection below.
#
# A call SUCCEEDED only when the provider reports that it ran and worked. Neither a
# `failed` status nor a non-zero exit code separates a command that never started from one
# that ran and found nothing, since both read the same; only exit code 0 says a command ran.
# So a codex command counts on `exit_code` 0 beside no status but "completed", and a grok
# call on a `tool_result` reporting no error whose content is a JSON object in which no
# report names an error or carries a status but "completed" or an exit code but 0 — and, for
# a poll of several background commands, at least one of them completed with 0.

# A `tool_use` block, inside an ASSISTANT message. Restricting to assistant events is
# defense in depth rather than a fix for an observed shape: grok returns tool RESULTS in
# `user` events as `tool_result` blocks, which the block-type test already excludes. If a
# future build ever echoed a `tool_use` block back, this stops it being counted twice.
# Verified against grok 1.0.13: one real review carried 99 `tool_use` blocks over 31 turns.
GROK_TOOL_BLOCK = "tool_use"

# A call's outcome: a `tool_result` block inside a `user` event, naming the call it answers
# by `tool_use_id`. Its `content` is a JSON object serialized as a string, and a command's
# carries its `exit_code` there, beside `is_error` false even when the command failed.
GROK_RESULT_BLOCK = "tool_result"

# The tools whose success means the working tree was read, by the `name` on the `tool_use`
# block. A result is not enough on its own: the plan grok writes itself (`todo_write`) comes
# back with no error and no exit code, so a run whose every command failed passed as
# inspected once it also wrote a todo. Verified against 185 local grok-4.7 streams, where
# every call used one of the names in this list or the next.
GROK_INSPECTING_TOOLS = frozenset(
    {"get_command_or_subagent_output", "grep", "list_dir", "read_file", "run_terminal_command"}
)

# The inspecting tools that report an exit code: a command, a search, and a background
# command's output. Only exit code 0 says one ran, so a result of theirs with none, or with the
# field renamed, does not count, as through codex. Verified against 473 local grok event
# streams, where every counted result of these three carried an integer `exit_code`.
GROK_COMMAND_TOOLS = frozenset({"get_command_or_subagent_output", "grep", "run_terminal_command"})

# Known tools that read nothing: bookkeeping, and edits. Named, like `CODEX_QUIET_ITEMS`, so
# that "a tool we skip on purpose" and "a tool we have never heard of" stay different facts —
# a list of names can go stale, and the drift check in `_grok_stats` is what notices.
GROK_QUIET_TOOLS = frozenset({"kill_command_or_subagent", "search_replace", "todo_write", "write"})

# A command grok moved to the background reports `status` "running" at once, and its real
# outcome arrives later as a `TaskOutput` with the command's own `status` and `exit_code`
# one level down, under this key.
GROK_NESTED_REPORT = "Result"

# A poll of several background commands at once has no status or exit code of its own: each
# command's report is in a list under this key's `results`.
GROK_BATCH_REPORT = "MultiResult"

# The one `status` a report may carry and still count; a report may also carry none.
GROK_COMPLETED = "completed"

# A report that names an error, by this key or by this type, is no success whatever
# `is_error` says. grok reports a tool that could not execute as `{"error": ...,
# "message": ...}`; no report of a real success carries either.
GROK_ERROR_KEY = "error"
GROK_ERROR_TYPE = "Error"

# codex's `--json` stream is items rather than messages: one `item.started` and one
# `item.completed` per call, both carrying the same `item.id`. These are the kinds that reach
# outside the model. Verified against codex-cli 0.150.1.
#
# `todo_list` is deliberately NOT here. It is a tool invocation, but it reaches nothing: a
# run whose only "tool call" was writing itself a plan has inspected exactly as much as one
# that made none.
CODEX_TOOL_ITEMS = frozenset(
    {
        "command_execution",
        "custom_tool_call",
        "file_change",
        "function_call",
        "local_shell_call",
        "mcp_tool_call",
        "patch_apply",
        "web_search",
    }
)

# The kinds that can read the working directory by what they are, which is what the refusal
# turns on. Listed rather than derived from `CODEX_TOOL_ITEMS`, so a kind added there is not
# local until someone decides it is.
#
# `web_search` reads the internet. `function_call`, `custom_tool_call` and `mcp_tool_call`
# name a tool without saying where it runs, so one of them may be a remote server's.
# `file_change` and `patch_apply` write to the tree, and a write shows no read of it, as
# grok's edits do not. None of them proves the tree was read, so they count in `tool_calls`
# and not here. A codex build that does read the tree through one of them is then refused
# visibly, with exit 6, rather than an MCP-only or edit-only run being passed silently.
CODEX_LOCAL_TOOL_ITEMS = frozenset(
    {
        "command_execution",
        "local_shell_call",
    }
)

# The local kinds that succeed by exit code, which is every one: `_codex_succeeded` reads
# only that and the status, so a local kind that carries no exit code needs its own rule
# there first. Verified against codex-cli 0.152.1 and 0.156.1: every completed
# `command_execution` carries an integer `exit_code`.
CODEX_COMMAND_ITEMS = frozenset({"command_execution", "local_shell_call"})

# The one `status` a completed command may carry and still count, as for grok; it may also
# carry none. An exit code of 0 beside any other status is not taken for success.
CODEX_COMPLETED = "completed"

# Kinds this wrapper knows about and deliberately does not count: the model talking to
# itself, and the plan it writes for itself. Named explicitly so that "a kind we chose to
# skip" and "a kind we have never heard of" stay different facts — which is the whole of the
# drift check in `_codex_stats`.
CODEX_QUIET_ITEMS = frozenset({"agent_message", "reasoning", "todo_list", "error"})


def _whole_number(value: JSONValue) -> int | None:
    """A count a provider reported, or None. `True` is an `int` in Python and is not one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# How much of a failed call's output the exit-3 refusal quotes. One line, because stderr
# carries one legible reason; cut, because the output is the provider's and any length.
FAILURE_LINE_CHARS = 200


def _text(value: JSONValue) -> str:
    """Output as text. grok reports a command's output as an array of byte values."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        octets = [b for b in value if isinstance(b, int) and not isinstance(b, bool)]
        if len(octets) == len(value) and all(0 <= b < 256 for b in octets):
            return bytes(octets).decode("utf-8", errors="replace")
    return ""


def _cut(line: str) -> str:
    return line if len(line) <= FAILURE_LINE_CHARS else line[:FAILURE_LINE_CHARS] + "..."


def _first_line(value: JSONValue) -> str:
    return _cut(next((line.strip() for line in _text(value).splitlines() if line.strip()), ""))


def _refused_status(report: JSONObject, allowed: str) -> str | None:
    """A report's status as JSON text, which keeps it one line, when it is not `allowed`."""
    if report.get("status", allowed) == allowed:
        return None
    return _cut(json.dumps(report["status"]))


def _grok_reports(block: JSONObject) -> list[JSONObject]:
    """The objects a `tool_result` reports a call's outcome in: its content, and the report
    nested in it when the call was a background command's. None at all when either is not
    an object, because it then reports nothing to judge."""
    content = block.get("content")
    if isinstance(content, str):
        try:
            content = loads(content)
        except ValueError:
            return []
    if not isinstance(content, dict):
        return []
    if GROK_NESTED_REPORT not in content:
        return [content]
    nested = content[GROK_NESTED_REPORT]
    return [content, nested] if isinstance(nested, dict) else []


def _grok_report_ok(report: JSONObject) -> bool:
    """Whether one report says its call finished and, if it names an exit code, exited 0.

    The status is an allowlist: grok reports "running" for a command it moved to the
    background and "failed" for one that did not work, and a status it has never sent is not
    taken for success. A null exit code is a command that has not finished. A report that
    names an error never passes.
    """
    if GROK_ERROR_KEY in report or report.get("type") == GROK_ERROR_TYPE:
        return False
    if report.get("status", GROK_COMPLETED) != GROK_COMPLETED:
        return False
    return "exit_code" not in report or _whole_number(report["exit_code"]) == 0


def _grok_batch_ok(batch: JSONValue) -> bool:
    """Whether a poll of several background commands holds one that completed with exit 0."""
    results = batch.get("results") if isinstance(batch, dict) else None
    return isinstance(results, list) and any(
        isinstance(child, dict)
        and _whole_number(child.get("exit_code")) == 0
        and _grok_report_ok(child)
        for child in results
    )


def _grok_succeeded(block: JSONObject, command: bool) -> bool:
    """Whether a `tool_result` reports a call that ran and worked.

    `is_error` false alone is not that: grok returns a command that exited 2 with `is_error`
    false and the exit code in the content. The content must hold a JSON object, because
    content that is not one reports nothing to judge, and every report in it must pass
    `_grok_report_ok`. A `command` result must also carry exit code 0 in one of its reports;
    a result with no status and no exit code, a file read, rests on `is_error`.

    A batch poll, whichever report carries it, worked when any command in it completed with
    exit code 0: that one inspected something, and a sibling still running or failed does
    not undo it. The reports around the batch still have to pass, and any other batch shape
    fails closed.
    """
    if block.get("is_error") is not False:
        return False
    reports = _grok_reports(block)
    if not reports:
        return False
    if not all(_grok_report_ok(report) for report in reports):
        return False
    batches = [report[GROK_BATCH_REPORT] for report in reports if GROK_BATCH_REPORT in report]
    if command and not batches and all(_whole_number(r.get("exit_code")) != 0 for r in reports):
        return False
    return all(_grok_batch_ok(batch) for batch in batches)


def _grok_status(block: JSONObject) -> str | None:
    """The first status in a result's reports that the success rule refuses."""
    for report in _grok_reports(block):
        status = _refused_status(report, GROK_COMPLETED)
        if status is not None:
            return status
    return None


def _grok_output(block: JSONObject) -> str:
    """What a failed call printed, for the refusal to quote.

    The content as it came stands in when no report printed anything, unless a report names
    a refused status: that status is the reason, and a file read's content is the file.
    """
    for report in _grok_reports(block):
        for key in ("output", "stderr", "stdout"):
            text = _text(report.get(key))
            if text.strip():
                return text
    if _grok_status(block) is not None:
        return ""
    content = block.get("content")
    return content if isinstance(content, str) else ""


def _grok_stats(events: Iterable[JSONObject]) -> RunStats:
    calls = 0
    attempts = 0
    local = 0
    failure: str | None = None
    failure_status: str | None = None
    ambiguous: str | None = None
    unknown: set[str] = set()
    # Every call's id with the tool each call under it named, and the outcome of each result
    # answering it once called. A result for an id not yet called is evidence of nothing.
    called: dict[str, list[str]] = {}
    answered: dict[str, list[bool]] = {}
    turns: int | None = None
    output_tokens: int | None = None
    served: tuple[str, ...] | None = None
    for event in events:
        kind = event.get("type")
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
        if kind == "assistant":
            for block in blocks:
                if block.get("type") == GROK_TOOL_BLOCK:
                    calls += 1
                    name = block.get("name")
                    tool = name if isinstance(name, str) else ""
                    ident = block.get("id")
                    if isinstance(ident, str):
                        called.setdefault(ident, []).append(tool)
                    if tool not in GROK_INSPECTING_TOOLS:
                        if tool not in GROK_QUIET_TOOLS:
                            unknown.add(tool or "<unnamed>")
                        continue
                    attempts += 1
        elif kind == "user":
            for block in blocks:
                ident = block.get("tool_use_id")
                if block.get("type") != GROK_RESULT_BLOCK or not isinstance(ident, str):
                    continue
                if ident not in called:
                    continue
                succeeded = _grok_succeeded(
                    block, bool(GROK_COMMAND_TOOLS.intersection(called[ident]))
                )
                answered.setdefault(ident, []).append(succeeded)
                if succeeded or failure is not None:
                    continue
                if any(tool in GROK_INSPECTING_TOOLS for tool in called[ident]):
                    failure = _first_line(_grok_output(block))
                    failure_status = _grok_status(block)
        elif kind == "result":
            # LAST wins, where the extractor refuses a stream carrying more than one. The
            # laxity is unreachable rather than a disagreement: `gate` extracts and validates
            # the answer BEFORE it asks for these numbers, so a multi-`result` stream has
            # already failed the gate and no caller ever sees the stats taken from it.
            turns = _whole_number(event.get("num_turns"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = _whole_number(usage.get("output_tokens"))
            served = _served_models(event.get("modelUsage"))
    for ident, outcomes in answered.items():
        # ONE tool under the id, an inspecting one, and every result a success. An id two
        # tools share, or one answered both ways, does not say which call worked; one tool
        # named under it twice, as an echo would, leaves no doubt.
        tools = set(called[ident])
        if len(tools) == 1 and tools <= GROK_INSPECTING_TOOLS and all(outcomes):
            local += 1
        elif tools & GROK_INSPECTING_TOOLS and any(outcomes) and ambiguous is None:
            ambiguous = ident
    if attempts == 0 and unknown:
        # The codex drift check's reason, for a list of names instead of kinds: a renamed
        # inspecting tool would otherwise make every run exit 6, blaming the model. An MCP
        # server's tool lands here too when nothing else was called, since grok's events name
        # a tool without saying where it runs.
        raise errors.EnvError(
            "counted no inspecting tool calls, but grok's stream carries tool name(s) this "
            f"wrapper does not recognize: {', '.join(sorted(unknown))}. That is provider CLI "
            "drift or an unfamiliar tool, not model behavior, and no run through grok can be "
            "believed until each name is added to validate.GROK_INSPECTING_TOOLS if it reads "
            "the working tree, or validate.GROK_QUIET_TOOLS if it does not."
        )
    return RunStats(
        tool_calls=calls,
        local_tool_calls=local,
        turns=turns,
        output_tokens=output_tokens,
        duration_s=None,
        local_tool_attempts=attempts,
        first_failure=failure,
        ambiguous_id=ambiguous,
        first_status=failure_status,
        served_models=served,
    )


def _served_models(usage_by_model: JSONValue) -> tuple[str, ...] | None:
    """The model ids keying grok's `modelUsage`, which names what served rather than what
    was asked for. Anything that is not an object of named entries reports nothing: this is
    provenance, so a malformed one is neither a crash nor a refusal."""
    if not isinstance(usage_by_model, dict):
        return None
    served = tuple(sorted(name for name in usage_by_model if name))
    return served or None


def served_model_match(requested: str, served: tuple[str, ...] | None) -> bool | None:
    """Whether a served id is the requested one, or that id with a build suffix.

    grok serves `grok-4.7` as `grok-4.7-build`, so a suffix after `-` matches; `grok-4.75`
    is another model and does not. The suffix rule also accepts a sibling such as
    `grok-4.7-mini`, and one matching id does not clear a run that also used another, which
    is why every served id is recorded. None when the stream named none.
    """
    if served is None:
        return None
    return any(name == requested or name.startswith(requested + "-") for name in served)


def served_model_warning(
    label: str, evidence: Evidence, stats: RunStats, match: bool | None
) -> str | None:
    """The one stderr line for a run that served another model, or did not say which."""
    if match is False or (evidence.reports_model and stats.served_models is None):
        named = stats.served_models
        served = ", ".join(_printable(name) for name in named) if named else "not reported"
        return f"{label}: warning: requested {_printable(evidence.model)}, served {served}"
    return None


def _printable(text: str) -> str:
    """`text` with every non-printable character escaped. An id the provider chose must not
    be able to break the warning across lines, and a normal id prints as itself."""
    return "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text
    )


def _warn(line: str) -> None:
    """`line` on stderr, or nowhere: never on stdout, and never failing the run it warns about.

    With fd 2 closed at startup sys.stderr is None, and print would fall back to stdout. With
    fd 2 open read-only, as a launcher script can leave it when stderr is closed, the write
    fails and the line stays buffered for the shutdown flush, which would fail too and exit
    120; pointing fd 2 at /dev/null lets that flush succeed.
    """
    if sys.stderr is None:
        return
    try:
        print(line, file=sys.stderr, flush=True)
    except OSError:
        with contextlib.suppress(OSError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, sys.stderr.fileno())
            finally:
                os.close(devnull)


def _codex_succeeded(item: JSONObject) -> bool:
    """Whether a local item's `item.completed` reports that it ran and worked."""
    if item.get("status", CODEX_COMPLETED) != CODEX_COMPLETED:
        return False
    return _whole_number(item.get("exit_code")) == 0


def _codex_stats(events: Iterable[JSONObject]) -> RunStats:
    # One call is the item id and kind its two events share.
    seen: set[tuple[str, str]] = set()
    # Per id, the kinds of call made under it and the outcome of each local completion.
    kinds: dict[str, set[str]] = {}
    completed: dict[str, list[bool]] = {}
    unknown: set[str] = set()
    total = 0
    calls = 0
    attempts = 0
    local = 0
    failure: str | None = None
    failure_status: str | None = None
    ambiguous: str | None = None
    # codex's own turn accounting, which counts one per `exec` turn rather than one per
    # model round-trip. It is not comparable with grok's `num_turns` and is recorded because
    # it is the number codex publishes, not because the two mean the same thing.
    turns = 0
    output_tokens: int | None = None
    for event in events:
        total += 1
        kind = event.get("type")
        if kind == "turn.started":
            turns += 1
        elif kind == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = _whole_number(usage.get("output_tokens"))
        elif kind in ("item.started", "item.completed"):
            item = event.get("item")
            item_kind = item.get("type") if isinstance(item, dict) else None
            if not isinstance(item, dict) or not isinstance(item_kind, str):
                continue
            if item_kind not in CODEX_TOOL_ITEMS:
                if item_kind not in CODEX_QUIET_ITEMS:
                    unknown.add(item_kind)
                continue
            # ONE CALL, TWO EVENTS, and the count sits inside whichever rule applies rather
            # than after both — an item with no usable id used to fall past the dedupe and be
            # counted once per event, which is the fail-OPEN direction.
            ident = item.get("id")
            is_local = item_kind in CODEX_LOCAL_TOOL_ITEMS
            if isinstance(ident, str):
                # Deduped on the item's own id and kind, so a call the run started and never
                # finished is still an attempt, and a completion of another kind is another call.
                kinds.setdefault(ident, set()).add(item_kind)
                if (ident, item_kind) not in seen:
                    seen.add((ident, item_kind))
                    calls += 1
                    attempts += is_local
            elif kind == "item.completed":
                # No id, so the two events cannot be paired. Count the terminal one only:
                # counting both would double a single call.
                calls += 1
                attempts += is_local
            # Read at completion only: every `item.started` carries `exit_code` null and
            # `status` "in_progress", whatever the call goes on to do.
            if not is_local or kind != "item.completed":
                continue
            succeeded = _codex_succeeded(item)
            if not succeeded and failure is None:
                failure = _first_line(item.get("aggregated_output"))
                failure_status = _refused_status(item, CODEX_COMPLETED)
            if isinstance(ident, str):
                completed.setdefault(ident, []).append(succeeded)
            elif succeeded:
                local += 1
    for ident, outcomes in completed.items():
        # One kind of call under the id, and every completion a success. An id two kinds
        # share, or one completed both ways, does not say which call worked.
        if len(kinds[ident]) == 1 and all(outcomes):
            local += 1
        elif any(outcomes) and ambiguous is None:
            ambiguous = ident

    if total == 0:
        # `codex exec --json` opens every run with `thread.started` and `turn.started`, so a
        # stream with nothing in it is not a model that did nothing — it is a runner that did
        # not run, or one whose output shape moved. Saying "the model never opened the diff"
        # about that would be a diagnosis of the wrong component.
        raise errors.EnvError(
            "codex produced no events at all. `codex exec --json` opens every run with a "
            "thread and a turn, so an empty stream means the runner did not run or its "
            "output format has changed -- this is not something the model did."
        )
    if attempts == 0 and unknown:
        # THE MISDIAGNOSIS THIS PREVENTS. After a provider-CLI upgrade renames its item
        # kinds, every run counts zero and would otherwise be refused as "the model never
        # opened the diff" — a falsehood, told identically on every run, about a component
        # that is working. The wrapper is the broken part and the message says so. Keyed on
        # LOCAL ATTEMPTS because that is what separates exit 6 from the rest: a web search
        # beside a renamed local kind would otherwise leave the rename to be reported as
        # exit 6, and recognized commands that all failed are not drift.
        #
        # The recovery names BOTH lists: a renamed tree-reading kind added only to the first
        # silences this check while every run still counts zero local calls and exits 6.
        raise errors.EnvError(
            "counted no local tool calls, but codex's stream carries item kind(s) this wrapper "
            f"does not recognize: {', '.join(sorted(unknown))}. That is provider CLI drift, "
            "not model behavior, and no run through codex can be believed until each kind "
            "is added to validate.CODEX_TOOL_ITEMS. "
            "A kind that reads the working tree also goes in "
            "validate.CODEX_LOCAL_TOOL_ITEMS: added only to the first, it silences this error "
            "and every run then exits 6."
        )
    return RunStats(
        tool_calls=calls,
        local_tool_calls=local,
        turns=turns,
        output_tokens=output_tokens,
        duration_s=None,
        local_tool_attempts=attempts,
        first_failure=failure,
        ambiguous_id=ambiguous,
        first_status=failure_status,
    )


def run_stats(mode: str, events: Iterable[JSONObject], duration_s: float | None) -> RunStats:
    """Count what the run did, through the event vocabulary its runner actually speaks.

    Named modes rather than sniffing the stream's shape. Guessing which schema a file is in
    is the same undecidable move as scanning prose for the model's answer, and the two event
    vocabularies share nothing: one is messages carrying content blocks, the other is items
    carrying kinds.

    Takes already-parsed objects rather than a path, so the caller decides where they come
    from: `gate` reuses the text it has already read when the answer and the evidence are the
    same file, and streams from disk when they are not.

    `duration_s` is measured by the caller rather than read from the stream. Only one of the
    two providers publishes a duration, and a field that means a different thing depending on
    who produced it is worse than one that means the same thing everywhere.
    """
    if mode == "grok-messages":
        counted = _grok_stats(events)
    elif mode == "codex-items":
        counted = _codex_stats(events)
    else:
        # The MACHINE, not the model. An events mode this module does not implement is a
        # mis-wired build, and reporting it as a gate failure would blame the answer for a
        # defect in the wrapper — the same misdiagnosis the drift check above exists to stop.
        raise errors.EnvError(
            f"unknown event mode {mode!r}; this build cannot count what its own provider did"
        )
    return replace(counted, duration_s=duration_s)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def describe_run(stats: RunStats) -> str:
    """The counts behind a refusal, in one clause. A refusal nobody can check is a rumor."""
    parts: list[str] = []
    if stats.turns is not None:
        parts.append(_plural(stats.turns, "turn"))
    calls = _plural(stats.tool_calls, "tool call")
    attempts = stats.local_tool_attempts
    if attempts is not None and attempts > stats.local_tool_calls:
        # A run whose local calls failed says how many it made, or "0 succeeded" would read
        # as a run that never tried.
        calls += f" ({attempts} local, {stats.local_tool_calls} succeeded)"
    elif stats.local_tool_calls != stats.tool_calls:
        # Only when they differ, so the common case reads as it always has and a run that
        # searched the web but touched nothing says so in the clause that refuses it.
        calls += f" ({stats.local_tool_calls} local)"
    parts.append(calls)
    if stats.duration_s is not None:
        parts.append(f"{stats.duration_s:.1f}s")
    if stats.output_tokens is not None:
        parts.append(f"{stats.output_tokens} output tokens")
    return ", ".join(parts)


def _first_failure(stats: RunStats) -> str:
    """What the first failed local call printed, quoted as data: it is the provider's text.

    A status the success rule refused is named before it, since after a provider renames a
    status that is the only reason every run fails. A success an ambiguous id kept from
    counting is named by that id, because a run whose only success it was would otherwise
    read as one whose calls never finished.
    """
    if stats.first_failure is None:
        said = ""
    else:
        printed = f"printed {stats.first_failure!r}" if stats.first_failure else "printed nothing"
        if stats.first_status is not None:
            printed = f"reported status {stats.first_status} and {printed}"
        said = f"the first to fail {printed}"
    if stats.ambiguous_id is None:
        return said or "none of them finished"
    unpaired = (
        f"a success under id {stats.ambiguous_id!r} does not count, because that id names "
        "more than one call or also reported a failure"
    )
    return f"{said}, and {unpaired}" if said else unpaired


def _sha256(filename: str) -> str:
    """Content hash, or a recorded reason. An unreadable asset is provenance too, and must
    not abort a review that already ran."""
    try:
        return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


def digest_bytes(data: bytes) -> str:
    """The provenance hash of bytes a caller already holds, in the same form `_sha256` writes.

    So a caller that read an input can attest THOSE bytes rather than the path they came
    from, without a second hashing convention to keep in step with this one.
    """
    return hashlib.sha256(data).hexdigest()


def write_provenance(
    path: Path,
    pairs: list[str],
    files: dict[str, str],
    stats: RunStats,
    digests: dict[str, str] | None = None,
    served_match: bool | None = None,
) -> None:
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
        # A digest the caller computed when it READ the input wins over re-reading the path:
        # the record must say what the run used, not what happens to be there now.
        record[f"{key}_sha256"] = (digests or {}).get(key) or _sha256(filename)
    # WHAT THE RUN DID, beside what produced it. `local_tool_calls` and `local_tool_attempts`
    # decide the exit status, so they are written down rather than only acted on: a refusal a
    # caller cannot audit afterwards is one it has to take on trust.
    record["run_stats"] = {
        "tool_calls": stats.tool_calls,
        "local_tool_calls": stats.local_tool_calls,
        "local_tool_attempts": stats.local_tool_attempts,
        "turns": stats.turns,
        "output_tokens": stats.output_tokens,
        "duration_s": stats.duration_s,
    }
    # Beside `model`, which records only what was asked for.
    served = stats.served_models
    record["served_models"] = list(served) if served is not None else None
    record["served_model_match"] = served_match
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


# The sidecar's name, derived from the findings artifact's. `cli.ARTIFACT_SUFFIXES` builds
# the same path from the other end; both sit on `<stem>` and this is the only place that has
# to turn one into the other.
PROVENANCE_SUFFIX = "-provenance.json"


def _provenance(artifact: Path) -> JSONObject | None:
    """The record beside this artifact, or None when there is no readable object there."""
    sidecar = artifact.with_name(artifact.stem + PROVENANCE_SUFFIX)
    try:
        record = loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


# A git object id, SHA-1 or SHA-256, as `rev-parse` prints one.
_OBJECT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def reviewed_head(artifact: Path) -> str:
    """The commit the run behind this artifact reviewed, or an `unresolved:` reason.

    Read from the provenance record rather than asked of git, because what is checked out
    now need not be what the model saw. A caller placing findings on a pull request
    compares this with the head it posts against, so anything but an object id -- a
    missing record, or a tree that was not a git repository -- reads as unresolved rather
    than as a head that happens to differ.
    """
    record = _provenance(artifact)
    if record is None:
        return "unresolved: no provenance"
    head = record.get("head_sha")
    if not isinstance(head, str) or _OBJECT_ID.fullmatch(head) is None:
        return "unresolved: the provenance records no head commit"
    return head


def refused_run(artifact: Path) -> RunStats | None:
    """The run behind this artifact, IF its provenance records no local tool calls.

    The reading counterpart to `write_provenance`, here so one module owns the sidecar's
    shape from both ends rather than two agreeing by memory.

    This exists because a refusal has to survive being handed on. `gate` refuses a run with no
    successful local call — exit 6 when it attempted none, exit 3 when every one failed — and
    keeps the artifact as evidence, and an artifact on disk is exactly what the retrieval
    command renders, cheerfully and at exit 0. The refusal was laundered by this package's
    own reader, so the reader has to be able to see it too.

    `None` means "nothing here says this run was vacuous": no sidecar, an unreadable or
    malformed one, or one recording at least one local tool call. Only a POSITIVE reading
    refuses, because the sidecar is a record rather than a gate — an artifact written before
    this field existed, or one a person assembled by hand, must still render.

    Zero `tool_calls` refuses on its own, which is the whole rule for a sidecar written
    before `local_tool_calls` existed. Otherwise zero `local_tool_calls` refuses: since 0.3.5
    that field counts the calls that succeeded, so a run whose every local call failed is
    refused too. A sidecar written by 0.3.3 or 0.3.4 counted attempts there, and is read by
    that rule rather than reinterpreted. A count that is present but not a whole
    non-negative number is not a reading of zero.
    """
    record = _provenance(artifact)
    stats = record.get("run_stats") if record is not None else None
    if not isinstance(stats, dict):
        return None
    # Both counts are checked as whole numbers before either is believed: a zero local
    # count beside a total that is not a count is a malformed record, not a reading.
    calls = _whole_number(stats.get("tool_calls"))
    if calls is None or calls < 0:
        return None
    local = _whole_number(stats.get("local_tool_calls")) if "local_tool_calls" in stats else calls
    # A total of zero refuses whatever the local count says, so no sidecar the reader
    # refused before `local_tool_calls` existed renders now.
    if calls != 0 and local != 0:
        return None
    attempts = _whole_number(stats.get("local_tool_attempts"))
    return RunStats(
        tool_calls=calls,
        local_tool_calls=0,
        turns=_whole_number(stats.get("turns")),
        output_tokens=_whole_number(stats.get("output_tokens")),
        duration_s=_seconds(stats.get("duration_s")),
        # Absent before 0.3.5, when a zero `local_tool_calls` meant no local call was even
        # attempted. None keeps that reading; zero or the total would claim a count nobody
        # made.
        local_tool_attempts=attempts if attempts is not None and attempts >= 0 else None,
    )


def _seconds(value: JSONValue) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def extract(mode: str, text: str, key: str = "findings") -> Artifact:
    """Recover the model's answer through the channel its runner actually provides.

    Both modes read a channel the RUNNER defines — grok's terminal event, codex's
    final-message file — so neither has to decide which part of a transcript is the answer.
    There is deliberately no prose-scanning mode: telling "the model's answer" from "text
    the model quoted" by scanning output is not a decidable problem, and the attempt hosted
    six separate false-passes before it was removed. A runner offering only plain text is
    not supported rather than supported badly.

    `key` is the top-level key the answer must carry. It is a parameter rather than a
    constant because a validation's answer is `{"verdicts": [...]}`: without it, a perfectly
    good verdicts object was refused at exit 1 on BOTH providers, before the schema that
    would have judged it was ever applied.
    """
    if mode == "object":
        return from_object_file(text, key)
    if mode == "grok-events":
        return from_grok_events(text, key)
    fail(f"unknown extraction mode {mode!r}")


def severity_tally(found: Artifact, count: int) -> str:
    """Findings by severity for the one stdout line: `1 P0, 2 P1`, or empty when there are none.

    `count` is unused here and is part of the signature anyway, because this and the verdicts
    summary are the two values of one `gate` parameter: a summary that had to be called
    differently depending on the mode would put the mode back inside `gate`.
    """
    entries = found.get("findings")
    tally: dict[str, int] = {}
    if isinstance(entries, list):
        for finding in entries:
            severity = finding.get("severity") if isinstance(finding, dict) else None
            name = severity if isinstance(severity, str) else "?"
            tally[name] = tally.get(name, 0) + 1
    return ", ".join(f"{n} {sev}" for sev, n in sorted(tally.items()))


def gate(
    *,
    answer_file: Path,
    schema_path: Path,
    mode: str,
    findings_out: Path,
    provenance_out: Path,
    prov_pairs: list[str],
    prov_files: dict[str, str],
    prov_digests: dict[str, str] | None = None,
    evidence: Evidence,
    label: str,
    key: str = "findings",
    check: Callable[[Artifact, JSONObject], int] = validate,
    summarize: Callable[[Artifact, int], str] = severity_tally,
    noun: str = "findings",
) -> int:
    """Validate a run's answer and write its artifacts. Returns the process exit status.

    Raises `VacuousRun` when the run attempted no local tool call, and `EnvError` when it
    attempted some and none succeeded — both after the artifacts are written, because that dud
    IS the evidence and a caller has to be able to look at what it refused.
    Raises `EnvError` before writing anything when the stream cannot be counted at all, which
    is a fact about this build or the provider CLI rather than about the model, and must not
    be told as one.

    `key`, `check`, `summarize` and `noun` are what a validation varies: the top-level key its
    answer carries, the rules its answer is held to, the one-line summary and the word for what
    it returned. All four default to the review, because the run frame around them — the status
    checks, the artifact order, the vacuous-run refusal — is the part neither mode may have its
    own copy of. Two gates would mean two places for the refusal to be forgotten in.
    """
    try:
        text = answer_file.read_text(encoding="utf-8", errors="replace")
        try:
            schema = _as_object(loads(schema_path.read_text(encoding="utf-8")), f"{noun} schema")
        except (OSError, ValueError) as exc:
            fail(f"cannot read {noun} schema {schema_path}: {exc}")
        found = extract(mode, text, key)
        count = check(found, schema)
    except GateError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return 1

    # WHAT THE RUN DID. Counted after the answer is validated, so a stream carrying two
    # terminal events has already been refused, and BEFORE any artifact is written, so a
    # stream this build cannot count leaves nothing behind that a caller could believe.
    #
    # grok's answer arrives inside its event stream, which is why `answer_file` and
    # `events_file` are then one path: the text above IS the whole stream, and counting from
    # it costs one read of a ~1.4 MB file rather than two.
    stats = run_stats(
        evidence.mode,
        objects(ndjson_lines(text))
        if evidence.events_file == answer_file
        else file_objects(evidence.events_file),
        evidence.duration_s,
    )

    # Findings first, provenance second: the sidecar's job is attesting THIS artifact, so it
    # must never be the only thing on disk.
    findings_out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    match = served_model_match(evidence.model, stats.served_models)
    write_provenance(provenance_out, prov_pairs, prov_files, stats, prov_digests, match)

    # A reviewer with ZERO successful local tool calls has shown no read of the diff, so it
    # has no verdict to summarize: empty findings and a page of them are equally unfounded. A
    # web search or an MCP call read something, but was not shown to read the repository, and
    # a command that failed shows nothing either way, since one that never started and one
    # that ran and matched nothing look alike. Decided here, ahead of the summary the rest of
    # this function builds — a gating caller reads only the status, and a line saying "0
    # findings" beside a status saying "refused" is the exact ambiguity being closed.
    #
    # Deliberately no retry here. Whether to spend another full-effort model run is the
    # calling agent's decision and its budget; this command's job is to refuse to certify,
    # and a wrapper that silently re-ran would hide how often this happens.
    if stats.local_tool_calls == 0:
        # The PROVENANCE sidecar, not the findings file. Naming the findings file here was a
        # laundering route: it invited the caller to open the very artifact just refused, and
        # `ce-persona-findings` would render it as an ordinary review. The sidecar is the
        # record of WHY it was refused, and it is the only one of the two safe to read.
        if stats.local_tool_attempts:
            # Not 6, the status that blames the model and invites a retry: the model tried to
            # read the tree and every call failed, which the same command in the same place
            # will most likely do again. The first failure is quoted, not diagnosed — a
            # command that could not start and one that ran and exited non-zero land here
            # alike.
            raise errors.EnvError(
                f"refusing to report {count} {noun} from a run none of whose local tool calls "
                f"succeeded ({describe_run(stats)}); {_first_failure(stats)}. A call that could "
                "not start and a call that ran and exited non-zero both count as failed, so no "
                "successful local call stands behind anything the run reported; the evidence "
                f"is {provenance_out}."
            )
        raise errors.VacuousRun(
            f"refusing to report {count} {noun} from a run that made no local tool calls "
            f"({describe_run(stats)}). The model never opened the diff, so nothing it "
            f"reported is founded; the evidence is {provenance_out}."
        )

    # After the refusals, so a refused run keeps its one reason; on stderr, so stdout stays
    # one line. A warning rather than a refusal: the review did happen.
    warning = served_model_warning(label, evidence, stats, match)
    if warning is not None:
        _warn(warning)

    breakdown = summarize(found, count)

    # stdout is the caller's context: one line, never the transcript. Severity counts first
    # so a caller can triage without opening the artifact at all.
    #
    # Flushed here, and a broken pipe swallowed: the one-line contract invites
    # `ce-grok-persona ... | head -1`, which closes the pipe. The interpreter's shutdown
    # flush would then raise where no handler can catch it and exit 120 — reporting failure
    # for a review that succeeded and whose artifacts are already on disk.
    try:
        print(f"{label}: {count} {noun}{f' ({breakdown})' if breakdown else ''} -> {findings_out}")
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
