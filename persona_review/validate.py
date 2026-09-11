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
judgment this package is not entitled to make.

WHAT THIS DOES NOT COVER
------------------------
A well-formed but EMPTY findings array from a run that DID inspect the code is schema-valid
and stays valid: distinguishing "found nothing" from "looked, then gave up" needs a
judgment about the transcript this does not attempt. Two checks narrow that gap from either
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
from collections.abc import Callable, Iterable, Iterator
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
    results = [event for event in objects(text.splitlines()) if event.get("type") == "result"]
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


@dataclass(frozen=True)
class Evidence:
    """Where a run's own account of itself lives, and how long the run took.

    Passed to `gate` instead of a finished `RunStats` so the counting happens where the
    stream has already been read. grok's answer arrives INSIDE its event stream, so
    `events_file` is then the same path as the answer file and the gate counts from the text
    it already holds rather than opening a megabyte twice.
    """

    events_file: Path
    mode: str
    duration_s: float | None


# BOTH ADAPTERS ASK ONE QUESTION: did the model reach outside itself? They answer it
# differently because the streams differ in kind, and the difference is worth stating rather
# than discovering.
#
# grok names a tool call structurally — a `tool_use` content block — so ANY of them counts
# and there is no list of tool names to go stale. codex names it by an item KIND, so the
# adapter has to carry a list, and a list can go out of date. That asymmetry is why only the
# codex side needs the drift detection below.

# A `tool_use` block, inside an ASSISTANT message. Restricting to assistant events is
# defence in depth rather than a fix for an observed shape: grok returns tool RESULTS in
# `user` events as `tool_result` blocks, which the block-type test already excludes. If a
# future build ever echoed a `tool_use` block back, this stops it being counted twice.
# Verified against grok 1.0.13: one real review carried 99 `tool_use` blocks over 31 turns.
GROK_TOOL_BLOCK = "tool_use"

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

# Kinds this wrapper knows about and deliberately does not count: the model talking to
# itself, and the plan it writes for itself. Named explicitly so that "a kind we chose to
# skip" and "a kind we have never heard of" stay different facts — which is the whole of the
# drift check in `_codex_stats`.
CODEX_QUIET_ITEMS = frozenset({"agent_message", "reasoning", "todo_list", "error"})


def _whole_number(value: JSONValue) -> int | None:
    """A count a provider reported, or None. `True` is an `int` in Python and is not one."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _grok_stats(events: Iterable[JSONObject]) -> tuple[int, int | None, int | None]:
    calls = 0
    turns: int | None = None
    output_tokens: int | None = None
    for event in events:
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
            # LAST wins, where the extractor refuses a stream carrying more than one. The
            # laxity is unreachable rather than a disagreement: `gate` extracts and validates
            # the answer BEFORE it asks for these numbers, so a multi-`result` stream has
            # already failed the gate and no caller ever sees the stats taken from it.
            turns = _whole_number(event.get("num_turns"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = _whole_number(usage.get("output_tokens"))
    return calls, turns, output_tokens


def _codex_stats(events: Iterable[JSONObject]) -> tuple[int, int | None, int | None]:
    seen: set[str] = set()
    unknown: set[str] = set()
    total = 0
    calls = 0
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
            if isinstance(ident, str):
                # Deduped on the item's own id, so a call the run started and never finished
                # still counts as evidence that something was inspected.
                if ident not in seen:
                    seen.add(ident)
                    calls += 1
            elif kind == "item.completed":
                # No id, so the two events cannot be paired. Count the terminal one only:
                # counting both would double a single call, and an unfinished id-less call
                # going uncounted is the safe direction for a number that gates a refusal.
                calls += 1

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
    if calls == 0 and unknown:
        # THE MISDIAGNOSIS THIS PREVENTS. After a provider-CLI upgrade renames its item
        # kinds, every run counts zero and would otherwise be refused as "the model never
        # opened the diff" — a falsehood, told identically on every run, about a component
        # that is working. The wrapper is the broken part and the message says so.
        raise errors.EnvError(
            "counted no tool calls, but codex's stream carries item kind(s) this wrapper "
            f"does not recognise: {', '.join(sorted(unknown))}. That is provider CLI drift, "
            "not model behaviour -- the tool-kind list in validate.CODEX_TOOL_ITEMS needs to "
            "catch up before any run through codex can be believed."
        )
    return calls, turns, output_tokens


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
        calls, turns, tokens = _grok_stats(events)
    elif mode == "codex-items":
        calls, turns, tokens = _codex_stats(events)
    else:
        # The MACHINE, not the model. An events mode this module does not implement is a
        # mis-wired build, and reporting it as a gate failure would blame the answer for a
        # defect in the wrapper — the same misdiagnosis the drift check above exists to stop.
        raise errors.EnvError(
            f"unknown event mode {mode!r}; this build cannot count what its own provider did"
        )
    return RunStats(tool_calls=calls, turns=turns, output_tokens=tokens, duration_s=duration_s)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def describe_run(stats: RunStats) -> str:
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


# The sidecar's name, derived from the findings artifact's. `cli.ARTIFACT_SUFFIXES` builds
# the same path from the other end; both sit on `<stem>` and this is the only place that has
# to turn one into the other.
PROVENANCE_SUFFIX = "-provenance.json"


def refused_run(artifact: Path) -> RunStats | None:
    """The run behind this artifact, IF its provenance records no tool calls at all.

    The reading counterpart to `write_provenance`, here so one module owns the sidecar's
    shape from both ends rather than two agreeing by memory.

    This exists because a refusal has to survive being handed on. `gate` refuses a vacuous
    run with exit 6 and keeps the artifact as evidence — and an artifact on disk is exactly
    what the retrieval command renders, cheerfully and at exit 0. The refusal was laundered
    by this package's own reader, so the reader has to be able to see it too.

    `None` means "nothing here says this run was vacuous": no sidecar, an unreadable or
    malformed one, or one recording at least one tool call. Only a POSITIVE reading refuses,
    because the sidecar is a record rather than a gate — an artifact written before this
    field existed, or one a person assembled by hand, must still render.
    """
    sidecar = artifact.with_name(artifact.stem + PROVENANCE_SUFFIX)
    try:
        record = loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stats = record.get("run_stats") if isinstance(record, dict) else None
    if not isinstance(stats, dict):
        return None
    calls = _whole_number(stats.get("tool_calls"))
    if calls != 0:
        return None
    return RunStats(
        tool_calls=calls,
        turns=_whole_number(stats.get("turns")),
        output_tokens=_whole_number(stats.get("output_tokens")),
        duration_s=_seconds(stats.get("duration_s")),
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
    evidence: Evidence,
    label: str,
    key: str = "findings",
    check: Callable[[Artifact, JSONObject], int] = validate,
    summarize: Callable[[Artifact, int], str] = severity_tally,
    noun: str = "findings",
) -> int:
    """Validate a run's answer and write its artifacts. Returns the process exit status.

    Raises `VacuousRun` when the run made no tool calls — after the artifacts are written,
    because that dud IS the evidence and a caller has to be able to look at what it refused.
    Raises `EnvError` when the stream cannot be counted at all, which is a fact about this
    build or the provider CLI rather than about the model, and must not be told as one.

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
        objects(text.splitlines())
        if evidence.events_file == answer_file
        else file_objects(evidence.events_file),
        evidence.duration_s,
    )

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
        # The PROVENANCE sidecar, not the findings file. Naming the findings file here was a
        # laundering route: it invited the caller to open the very artifact just refused, and
        # `ce-persona-findings` would render it as an ordinary review. The sidecar is the
        # record of WHY it was refused, and it is the only one of the two safe to read.
        raise errors.VacuousRun(
            f"refusing to report {count} {noun} from a run that made no tool calls "
            f"({describe_run(stats)}). The model never opened the diff, so nothing it "
            f"reported is founded; the evidence is {provenance_out}."
        )

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
