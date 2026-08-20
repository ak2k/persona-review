#!/usr/bin/env python3
"""Validate a persona run's findings JSON against the plugin's findings-schema.json.

Shared by ce-grok-persona and ce-codex-persona. A review that returns prose and no
schema-shaped object, or an object missing required fields, reads exactly like a clean
review to anything scraping the tail of stdout — so the wrappers exit non-zero instead.

    ce-persona-validate.py --mode grok-events [--findings-out P] <events> <schema>
    ce-persona-validate.py --mode object      [--findings-out P] <final-msg> <schema>
    ce-persona-validate.py                    [--findings-out P] <output> <schema> <prompt>

THREE MODES, BECAUSE THE RUNNERS DIFFER
---------------------------------------
`grok-events` reads grok's NDJSON and takes `structured_output` from the terminal
`result` event: `--json-schema` composes with `--output-format streaming-messages-json`
(verified against 1.0.4, contradicting grok's own help text), so the answer arrives
parsed AND the stream still grows for liveness.

`object` reads a file that IS the answer — `codex exec -o` writes only the agent's final
message. Codex cannot take the schema itself: OpenAI's strict mode rejects it
("'additionalProperties' is required to be supplied and to be false") and would force
every optional field, changing what a finding means.

`transcript` is the fallback for plain output, and the only mode that needs the prompt.
Both native modes exist to avoid it: scanning prose for the model's own object caused
two silent false-passes, described below.

WHY THE PROMPT FILE IS AN ARGUMENT (transcript mode only)
---------------------------------------------------------
13 of the 16 compound-engineering persona briefs carry a schema-valid EXAMPLE object
(`{"reviewer": "adversarial", "findings": [], ...}`), and `codex exec` replays its prompt
into the transcript — six copies of the persona text in a real run. Scanning the whole
output for the last object carrying a `findings` key therefore validates the persona's
own example whenever the model answers with prose and no JSON, reporting a clean review
of a change nobody reviewed. A gate that fabricates assurance is worse than no gate.

So: cut at the LAST occurrence of the prompt's final line and scan only what follows. A
runner that does not echo (grok on `--output-format plain`) leaves no boundary, and the
whole output stays fair game.

WHY THE OBJECT MUST BE LAST
---------------------------
The cut alone was not enough. An answer that gives up in prose while QUOTING a
schema-shaped object — tool output, a replayed brief — put that object after the
boundary, and it validated: the same false green, displaced by one paragraph. So a
candidate counts only when nothing but whitespace or a closing fence follows it. The
prompt already tells the model to put its findings last, so this asks for nothing new.

WHAT THIS DOES NOT COVER
------------------------
A well-formed but EMPTY findings array is schema-valid and stays valid — a model that
quietly gave up still passes. Distinguishing "found nothing" from "gave up" needs a
judgement about the transcript that this deliberately does not attempt.

`grok-events` narrows that gap from one side only: schema-constrained decoding means a
truncated or refused run STILL returns a well-formed empty array, so the run's own
terminal signals (`is_error`, `subtype`, `stop_reason`) are checked before its answer is
believed. Those checks are generalised from one observed successful run, and the
stop_reason half is a denylist — a novel early-stop value would pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Any, NoReturn

# What may follow the findings object and still count as "at the end": whitespace and
# a closing code fence. Anything else means the object is quoted mid-transcript.
TRAILING_NOISE = re.compile(r"\s*(?:```)?\s*\Z")

# A finding, and the artifact that holds them. Free-form by design: the schema lives in
# the compound-engineering plugin, is read at run time, and may add fields we do not know
# about — so the value type is Any, explicitly, rather than a shape we would be inventing.
Finding = dict[str, Any]
Artifact = dict[str, Any]


def fail(message: str) -> NoReturn:
    sys.exit(f"ce-persona: {message}")


def findings_object(text: str, prompt: str) -> Artifact:
    """The model's own findings object: after the echoed prompt, at the end of the output.

    Two anchors, because each alone has been bypassed. The boundary cut drops the
    wrappers' own prompt echo, where 13 of the 16 persona briefs carry a schema-valid
    EXAMPLE object. The end anchor drops everything the model merely QUOTED — tool
    output, a replayed brief — which is how a give-up answer still validated after the
    cut was added. The prompt already demands the object come last, so requiring it is
    not a new constraint on the model.
    """
    # An empty boundary must not cut: `"abc".rfind("")` is 3, so a blank prompt would
    # slice the whole output away and every answer would read as "no findings JSON".
    # `from_object_file` relies on this to reuse the end anchor with no boundary at all.
    boundary = prompt.rstrip().rsplit("\n", 1)[-1]
    cut = text.rfind(boundary) if boundary else -1
    if cut != -1:
        text = text[cut + len(boundary) :]

    decoder = json.JSONDecoder()
    found = None
    quoted = False
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if not (isinstance(obj, dict) and "findings" in obj):
            continue
        if TRAILING_NOISE.fullmatch(text[i + consumed :]):
            found = obj
        else:
            quoted = True
    if found is None:
        if quoted:
            fail(
                "findings JSON found, but not at the end of the output — the model "
                "quoted an object instead of returning one"
            )
        fail("no findings JSON object in the model's output")
    return found


# JSON Schema type -> what Python accepts. bool is excluded from the numeric types
# because `True` is an int in Python and a model emitting `"line": true` must not pass.
JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_names(label: str, spec: dict[str, Any]) -> list[str]:
    """The declared type(s) for one field, normalised to a list of known names.

    JSON Schema lets `type` be a LIST, and the installed plugin schema uses that form:
    `"suggested_fix": {"type": ["string", "null"]}`. Reading only the `str` case skipped
    that field's check entirely, silently -- so an unrecognised declaration fails here
    instead. A gate that cannot understand a rule must not certify against it.
    """
    declared: object = spec.get("type")
    if declared is None:
        return []
    raw: list[object] = list(declared) if isinstance(declared, list) else [declared]
    names = [name for name in raw if isinstance(name, str) and name in JSON_TYPES]
    if len(names) != len(raw):
        fail(
            f"findings schema declares a type this gate cannot check for {label}: "
            f"{declared!r}. Passing a rule it does not understand is how a validator "
            f"certifies a shape nobody validated."
        )
    return names


def _check_type(label: str, value: object, spec: dict[str, Any]) -> None:
    """Enforce the declared type and the schema's simple bounds for one field.

    Presence and enums alone let a model pass `"line": "nope"`, `"pre_existing": "no"`,
    or `"evidence": []` -- shapes the plugin schema rejects and downstream merge tooling
    then trips over, having been told the review was valid.
    """
    names = _type_names(label, spec)
    if names:
        allowed = tuple(t for name in names for t in JSON_TYPES[name])
        # Only exclude bool when EVERY declared type is numeric; a union that genuinely
        # admits booleans still should.
        numeric_only = all(name in ("integer", "number") for name in names)
        ok = isinstance(value, allowed) and not (numeric_only and isinstance(value, bool))
        if not ok:
            fail(f"{label} must be {' or '.join(names)}, got {type(value).__name__}")

    minimum = spec.get("minItems")
    if isinstance(value, list) and isinstance(minimum, int) and len(value) < minimum:
        fail(f"{label} needs at least {minimum} item(s), got {len(value)}")

    # The numeric and string bounds the plugin schema actually declares: `line` carries
    # `minimum: 1` and `title` carries `maxLength: 100`. Skipping them passed a finding
    # pointing at line 0 -- no such line -- as a valid citation.
    lower = spec.get("minimum")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(lower, (int, float))
        and not isinstance(lower, bool)
        and value < lower
    ):
        fail(f"{label} must be >= {lower}, got {value}")
    cap = spec.get("maxLength")
    if isinstance(value, str) and isinstance(cap, int) and len(value) > cap:
        fail(f"{label} is {len(value)} characters, over the schema's maxLength of {cap}")

    # An array's declared element shape. `evidence` is an array of strings, so without
    # this `"evidence": [{"a": 1}]` satisfied minItems and then got rendered as the
    # quote-the-line evidence a reader is supposed to be able to check against the code.
    items = spec.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        for n, element in enumerate(value):
            _check_type(f"{label}[{n}]", element, items)


def validate(found: Artifact, schema: dict[str, Any]) -> int:
    missing = [k for k in schema.get("required", []) if k not in found]
    if missing:
        fail(f"findings JSON missing required keys: {', '.join(missing)}")
    if not isinstance(found["findings"], list):
        fail("findings must be an array")

    # Top-level shape too: `"residual_risks": "none"` is not an empty list of risks.
    for key, spec in schema.get("properties", {}).items():
        if key != "findings" and key in found and isinstance(spec, dict):
            _check_type(key, found[key], spec)

    # The schema is read from a plugin cache this repo does not control, so treat an
    # unexpected shape as a failed run with a legible reason rather than a traceback.
    item = schema.get("properties", {}).get("findings", {}).get("items")
    if not isinstance(item, dict):
        fail("findings schema has no properties.findings.items object")
    properties = item.get("properties")
    if not isinstance(properties, dict) or not properties:
        fail("findings schema has no properties.findings.items.properties object")
    required = item.get("required")
    if not required:
        fail("findings schema declares no required finding fields")

    # Every enum the schema declares, not just severity. The shape this wrapper
    # replaced taught `autofix_class: safe_auto` and `owner: review-fixer`, neither of
    # which the current schema allows, so a model repeating either must not pass.
    enums = {
        key: spec["enum"]
        for key, spec in properties.items()
        if isinstance(spec, dict) and spec.get("enum")
    }
    if not enums:
        fail("findings schema declares no finding enums")

    for n, finding in enumerate(found["findings"], 1):
        if not isinstance(finding, dict):
            fail(f"finding {n} is not an object")
        absent = [k for k in required if k not in finding]
        if absent:
            fail(f"finding {n} missing {', '.join(absent)}")
        for key, allowed in enums.items():
            if key in finding and finding[key] not in allowed:
                fail(f"finding {n} has {key}={finding[key]!r}, not one of {allowed}")
        for key, spec in properties.items():
            if key in finding:
                _check_type(f"finding {n} field {key}", finding[key], spec)
    return len(found["findings"])


def from_object_file(text: str) -> Artifact:
    """The findings object from a file holding the model's FINAL MESSAGE and nothing else.

    `codex exec -o FILE` writes only the agent's final message -- 72 bytes against a
    613-byte transcript on a trivial prompt -- so there is no transcript to scan.

    A bare object is the common case, so try that first. But the prompt asks for the
    object "at the END of your reply (after any analysis)", and a model that takes that
    literally -- a paragraph, then the object, often inside a ```json fence -- had a
    complete review thrown away with "final message is not JSON" after a full high-effort
    run. Fall back to the same end-anchored scan transcript mode uses, with no boundary to
    cut: this file is ALREADY only the final message, so "last thing in it" is the same
    guarantee, and an object the model merely quoted mid-message still loses.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            pass
        else:
            if not isinstance(obj, dict) or "findings" not in obj:
                fail("final message is JSON but carries no findings key")
            return obj
    return findings_object(text, "")


# `subtype` is a closed vocabulary, so an allowlist is safe: anything else is a run that
# did not finish normally.
GOOD_SUBTYPES = frozenset({"success"})

# `stop_reason` is an OPEN vocabulary, so this is a denylist -- an unfamiliar but healthy
# value must not fail a good review. Deliberately not exhaustive: these are the
# terminations that yield a truncated or refused answer which schema-constrained decoding
# would still render as a well-formed empty findings object.
BAD_STOP_REASONS = frozenset(
    {
        "max_tokens",
        "max_output_tokens",
        "max_turns",
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

    `--json-schema` composes with `--output-format streaming-messages-json` (verified
    against grok 1.0.4, contradicting grok's own help text), so the terminal event
    carries `structured_output` already parsed, next to is_error and stop_reason. The
    stream also gives byte-growth liveness, which the schema-constrained `json` format
    cannot.
    """
    result = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        # Last wins, deliberately: the TERMINAL result event is the run's verdict, and a
        # first-wins read would let an early one certify a run that kept going.
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is None:
        fail("no `result` event in grok's output stream")
    if result.get("is_error"):
        fail(f"grok reported an error result (stop_reason={result.get('stop_reason')!r})")

    # Beyond is_error, because schema-constrained decoding is what makes this necessary:
    # a run that hit the token ceiling or refused still hands back a well-formed
    # `{"findings": []}`, which is indistinguishable from a clean review. These two
    # checks are asymmetric on purpose, and both are generalised from a single observed
    # successful run (grok 1.0.4: subtype="success", stop_reason="end_turn").
    subtype = result.get("subtype")
    if isinstance(subtype, str) and subtype not in GOOD_SUBTYPES:
        fail(f"grok's run ended with subtype={subtype!r}, which is not a completed review")
    stop = result.get("stop_reason")
    if isinstance(stop, str) and stop in BAD_STOP_REASONS:
        fail(f"grok stopped early (stop_reason={stop!r}); the answer is truncated or refused")
    obj = result.get("structured_output")
    if isinstance(obj, dict) and "findings" in obj:
        return obj
    # Fall back to the raw final text when the runner gave no structured output.
    raw = result.get("result")
    if isinstance(raw, str) and raw.strip():
        return from_object_file(raw)
    fail("grok's result event carried no structured output")


MODES = ("transcript", "object", "grok-events")


def _sha256(filename: str) -> str:
    """Content hash, or a recorded reason. An unreadable asset is provenance too, and
    must not abort a review that already ran."""
    try:
        with open(filename, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        return f"unreadable: {exc}"


def write_provenance(path: str, pairs: list[str], files: dict[str, str]) -> None:
    """Record which brief and schema this run used, hashing their contents.

    Written here rather than by the wrappers because it is structured data: a shell
    heredoc interpolating paths into JSON produces invalid JSON the moment a path holds
    a quote or a backslash.
    """
    record: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        record[key] = value
    for key, filename in files.items():
        record[f"{key}_sha256"] = _sha256(filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)


def main(argv: list[str]) -> int:
    mode = "transcript"
    findings_out = None
    provenance_out = None
    prov_pairs: list[str] = []
    prov_files: dict[str, str] = {}
    positional = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
        elif arg == "--findings-out" and i + 1 < len(argv):
            findings_out = argv[i + 1]
            i += 2
        elif arg == "--provenance-out" and i + 1 < len(argv):
            provenance_out = argv[i + 1]
            i += 2
        elif arg == "--prov" and i + 1 < len(argv):
            prov_pairs.append(argv[i + 1])
            i += 2
        elif arg == "--prov-file" and i + 1 < len(argv):
            key, _, filename = argv[i + 1].partition("=")
            prov_files[key] = filename
            prov_pairs.append(f"{key}_file={filename}")
            i += 2
        else:
            positional.append(arg)
            i += 1

    if mode not in MODES:
        sys.exit(f"usage: --mode must be one of {', '.join(MODES)}")
    if mode == "transcript" and len(positional) != 3:
        sys.exit("usage: ce-persona-validate.py <output-file> <schema-file> <prompt-file>")
    if mode != "transcript" and len(positional) != 2:
        sys.exit(f"usage: ce-persona-validate.py --mode {mode} <output-file> <schema-file>")

    output, schema_path = positional[0], positional[1]
    with open(output, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        with open(schema_path, encoding="utf-8") as fh:
            schema = json.load(fh)
    except (OSError, ValueError) as exc:
        fail(f"cannot read findings schema {schema_path}: {exc}")

    if mode == "object":
        found = from_object_file(text)
    elif mode == "grok-events":
        found = from_grok_events(text)
    else:
        with open(positional[2], encoding="utf-8", errors="replace") as fh:
            prompt = fh.read()
        found = findings_object(text, prompt)

    count = validate(found, schema)

    if provenance_out:
        write_provenance(provenance_out, prov_pairs, prov_files)

    if findings_out:
        with open(findings_out, "w", encoding="utf-8") as fh:
            json.dump(found, fh, indent=1)

    # stdout is the caller's context: one line, never the transcript. Severity counts
    # first so a caller can triage without opening the artifact at all.
    tally = {}
    for finding in found["findings"]:
        sev = finding.get("severity", "?")
        tally[sev] = tally.get(sev, 0) + 1
    breakdown = ", ".join(f"{n} {sev}" for sev, n in sorted(tally.items()))
    where = f" -> {findings_out}" if findings_out else ""
    print(f"ce-persona: {count} findings{f' ({breakdown})' if breakdown else ''}{where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
