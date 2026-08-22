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

`transcript` remains for a runner that offers only plain text. Neither shipped provider uses
it. It carries two hard-won anchors and the tests that pin them.

WHY THE PROMPT IS AN ARGUMENT (transcript mode only)
----------------------------------------------------
13 of the 16 persona briefs carry a schema-valid EXAMPLE object, and a runner that echoes
its prompt replays them into the transcript. Scanning the whole output for the last object
with a `findings` key therefore validates the persona's own example whenever the model
answers with prose. So: cut at the LAST occurrence of the prompt's final line and scan only
what follows.

WHAT THIS DOES NOT COVER
------------------------
A well-formed but EMPTY findings array is schema-valid and stays valid: distinguishing
"found nothing" from "quietly gave up" needs a judgement about the transcript this does not
attempt. The grok path narrows that gap from one side by checking the run's own terminal
status before believing its answer, but `stop_reason` is an open vocabulary and the denylist
cannot be exhaustive.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import NoReturn, cast

# What may follow the findings object and still count as "at the end": whitespace and a
# closing code fence. Anything else means the object was quoted mid-transcript.
TRAILING_NOISE = re.compile(r"\s*(?:```)?\s*\Z")

# Parsed JSON, spelled out. The schema is owned by the compound-engineering plugin, read at
# run time, and free to grow fields this package has never heard of — so the value type is
# "some JSON", narrowed by isinstance where it is used, rather than a TypedDict asserting a
# shape this package has no authority to fix.
type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]
Artifact = JSONObject


class GateError(Exception):
    """The answer is not a valid findings artifact, with a legible reason."""


def fail(message: str) -> NoReturn:
    raise GateError(message)


def loads(text: str) -> JSONValue:
    return cast(JSONValue, json.loads(text))


def _as_object(value: JSONValue, what: str) -> JSONObject:
    if not isinstance(value, dict):
        fail(f"{what} is not a JSON object")
    return value


def _scan(text: str) -> list[tuple[JSONObject, bool, bool]]:
    """Every findings object in `text`, as (object, sits-at-the-end, has-any-findings)."""
    decoder = json.JSONDecoder()
    out: list[tuple[JSONObject, bool, bool]] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            decoded, consumed = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        obj = cast(JSONValue, decoded)
        if not isinstance(obj, dict) or "findings" not in obj:
            continue
        entries = obj.get("findings")
        at_end = TRAILING_NOISE.fullmatch(text[i + consumed :]) is not None
        out.append((obj, at_end, isinstance(entries, list) and len(entries) > 0))
    return out


def _boundary_lines(text: str, boundary: str) -> list[int]:
    """Indexes of the lines that ARE the boundary, ignoring surrounding whitespace.

    A whole-line match, never a substring. An unanchored search matches the boundary
    wherever it appears — including inside a finding's own quoted evidence, which happens
    the moment this tool reviews itself — and cutting there slices the real answer in half.
    """
    return [n for n, line in enumerate(text.split("\n")) if line.strip() == boundary]


def findings_object(text: str, prompt: str) -> Artifact:
    """The model's own findings object: after the echoed prompt, at the end of the output.

    Three anchors, because each of the first two alone has been bypassed.

    The boundary cut drops the runner's echo of the prompt, where the persona brief's
    EXAMPLE object lives. It cuts at the LAST whole-line boundary, so a runner that echoes
    more than once still gets past its own replay. An empty boundary must not cut at all —
    `"abc".rfind("")` is 3, which would slice the whole output away.

    The end anchor drops everything the model merely QUOTED: tool output, a replayed brief.

    The third rejects an at-end EMPTY object when a non-empty one appeared anywhere: taking
    the last would discard a real review, which is worse than a false clean because the work
    actually happened. It deliberately scans the UNCUT text. Scoping it to what follows the
    boundary lets a model answer, replay the boundary line, then paste the brief's empty
    example — pushing its own answer out of the anchor's view. The cost is a false refusal
    when a `-c` context brief embeds a non-empty findings object of its own and the review
    genuinely finds nothing; that fails loudly, with a legible reason, which is the side to
    err on for a gate.
    """
    boundary = prompt.rstrip().rsplit("\n", 1)[-1] if prompt.strip() else ""
    lines = text.split("\n")
    marks = _boundary_lines(text, boundary) if boundary else []

    # Extraction reads only what follows the last echo; the ambiguity check reads everything.
    body = "\n".join(lines[marks[-1] + 1 :]) if marks else text

    found: JSONObject | None = None
    quoted_any = False
    for obj, at_end, _ in _scan(body):
        if at_end:
            found = obj
        else:
            quoted_any = True

    if found is None:
        if quoted_any:
            fail(
                "findings JSON found, but not at the end of the output — the model quoted "
                "an object instead of returning one"
            )
        fail("no findings JSON object in the model's output")

    tail = found.get("findings")
    if isinstance(tail, list) and not tail and any(nonempty for _, _, nonempty in _scan(text)):
        fail(
            "the output ends with an EMPTY findings object while a non-empty one appears "
            "earlier — ambiguous, and taking the last would discard a real review"
        )
    return found


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
        "additionalProperties",
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
    properties = spec.get("properties")
    if isinstance(properties, dict):
        for key, sub in properties.items():
            if isinstance(sub, dict):
                check_schema_supported(sub, f"{label}.{key}")
    items = spec.get("items")
    if isinstance(items, dict):
        check_schema_supported(items, f"{label}.items")


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
    result: JSONObject | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = loads(line)
        except ValueError:
            continue
        # Last wins: the TERMINAL result event is the run's verdict, and a first-wins read
        # would let an early one certify a run that kept going and then failed.
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is None:
        fail("no `result` event in grok's output stream")

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
    if isinstance(obj, dict) and "findings" in obj:
        return obj
    raw = result.get("result")
    if isinstance(raw, str) and raw.strip():
        return from_object_file(raw)
    fail("grok's result event carried no structured output")


def _sha256(filename: str) -> str:
    """Content hash, or a recorded reason. An unreadable asset is provenance too, and must
    not abort a review that already ran."""
    try:
        return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


def write_provenance(path: Path, pairs: list[str], files: dict[str, str]) -> None:
    """Record which brief and schema this run used, by content hash.

    Two arms are only comparable if they ran the same brief, and briefs live in a plugin
    cache that updates underneath us, so without this the drift between two runs is
    invisible and a comparison silently stops meaning anything. A sidecar, so the findings
    artifact keeps exactly the shape the plugin's fold-in expects.
    """
    record: JSONObject = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        record[key] = value
    for key, filename in files.items():
        record[f"{key}_file"] = filename
        record[f"{key}_sha256"] = _sha256(filename)
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


def extract(mode: str, text: str, prompt: str) -> Artifact:
    if mode == "object":
        return from_object_file(text)
    if mode == "grok-events":
        return from_grok_events(text)
    if mode == "transcript":
        return findings_object(text, prompt)
    fail(f"unknown extraction mode {mode!r}")


def gate(
    *,
    answer_file: Path,
    schema_path: Path,
    mode: str,
    prompt: str,
    findings_out: Path,
    provenance_out: Path,
    prov_pairs: list[str],
    prov_files: dict[str, str],
    label: str,
) -> int:
    """Validate a run's answer and write its artifacts. Returns the process exit status."""
    try:
        text = answer_file.read_text(encoding="utf-8", errors="replace")
        try:
            schema = _as_object(loads(schema_path.read_text(encoding="utf-8")), "findings schema")
        except (OSError, ValueError) as exc:
            fail(f"cannot read findings schema {schema_path}: {exc}")
        found = extract(mode, text, prompt)
        count = validate(found, schema)
    except GateError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return 1

    # Findings first, provenance second: the sidecar's job is attesting THIS artifact, so it
    # must never be the only thing on disk.
    findings_out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    write_provenance(provenance_out, prov_pairs, prov_files)

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
        with contextlib.suppress(OSError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0
