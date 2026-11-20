"""The decide mode: a caller's prompt and JSON Schema in, one schema-valid object out.

A review and a validation read a repository; a decision reads only its prompt, so its run
starts in an empty directory and a run with no tool calls is the expected case rather than
a refusal. What stays the same is the frame around the model: the argv builders, the
watchdogs, the artifacts and a gate that fails closed.

The schema is the CALLER's, not the plugin's, so the gate here walks it at every depth.
The findings gate enforces `required` and `properties` only at the top level and on a
finding, which suits the plugin's flat schema and would silently pass a nested decision
object whose inner fields are missing. A schema using a keyword this walk does not enforce
is refused before the model is called.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from . import errors, validate
from .validate import JSONObject, JSONValue, fail

# Every keyword the walk below enforces, plus the annotations it may ignore honestly.
# `additionalProperties` is enforced in its boolean form only; a schema-valued one is a rule
# this walk does not implement and is refused like any other.
DECIDE_KEYWORDS = validate.ENFORCED_KEYWORDS | validate.IGNORED_KEYWORDS | {"additionalProperties"}

BOUNDARY = "Return the JSON object now."

# The hard deadline of one decide call, whatever CE_PERSONA_HARD_SECS says: a caller plans
# its night around this bound, so a larger value in the environment must not extend it.
HARD_SECS_CAP = 1200.0


def hard_secs(configured: float) -> float:
    return min(configured, HARD_SECS_CAP)


def check_schema(spec: JSONValue, label: str = "schema") -> None:
    """Refuse a schema that uses a rule this gate does not enforce, at any depth."""
    if not isinstance(spec, dict):
        fail(f"{label} is not a schema object")
    unknown = sorted(set(spec) - DECIDE_KEYWORDS)
    if unknown:
        fail(
            f"{label} uses JSON Schema keyword(s) this gate does not implement: "
            f"{', '.join(unknown)}"
        )
    validate._type_names(label, spec)  # pyright: ignore[reportPrivateUsage]
    extra = spec.get("additionalProperties")
    if extra is not None and not isinstance(extra, bool):
        fail(f"{label}.additionalProperties must be true or false")
    for bound in ("minItems", "maxItems", "minLength", "maxLength"):
        value = spec.get(bound)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            fail(f"{label}.{bound} must be a whole number")
    for bound in ("minimum", "maximum"):
        value = spec.get(bound)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            fail(f"{label}.{bound} must be a number")
    enum = spec.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        fail(f"{label}.enum must be a non-empty array")
    required = spec.get("required")
    if required is not None and (
        not isinstance(required, list) or not all(isinstance(k, str) for k in required)
    ):
        fail(f"{label}.required must be an array of names")
    properties = spec.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            fail(f"{label}.properties is not an object")
        for name, sub in properties.items():
            check_schema(sub, f"{label}.{name}")
    items = spec.get("items")
    if items is not None:
        # Tuple-form items is a different rule; accepting it would check nothing.
        if not isinstance(items, dict):
            fail(f"{label}.items is not a single schema object")
        check_schema(items, f"{label}.items")


def check_root(schema: JSONObject) -> None:
    """The answer is one JSON object, so the schema must describe one."""
    check_schema(schema)
    if validate._type_names("schema", schema) != ["object"]:  # pyright: ignore[reportPrivateUsage]
        fail('schema must declare "type": "object" at the top level')


def _same(left: JSONValue, right: JSONValue) -> bool:
    # JSON keeps booleans and numbers apart; Python's `True == 1` does not.
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    return left == right


def check_value(label: str, value: JSONValue, spec: JSONObject) -> None:
    """Hold one value to its schema, recursing into objects and arrays."""
    # Type and bounds come from the shared rules; `items` is walked here instead, because
    # the shared walk checks an element's type and bounds but not its own properties.
    validate._check_type(  # pyright: ignore[reportPrivateUsage]
        label, value, {k: v for k, v in spec.items() if k != "items"}
    )
    enum = spec.get("enum")
    if isinstance(enum, list) and not any(_same(value, allowed) for allowed in enum):
        fail(f"{label} is {value!r}, not one of {enum}")
    if isinstance(value, dict):
        required = spec.get("required")
        if isinstance(required, list):
            missing = [k for k in required if isinstance(k, str) and k not in value]
            if missing:
                fail(f"{label} is missing {', '.join(missing)}")
        properties = spec.get("properties")
        props: JSONObject = properties if isinstance(properties, dict) else {}
        for key, item in value.items():
            sub = props.get(key)
            if isinstance(sub, dict):
                check_value(f"{label}.{key}", item, sub)
            elif spec.get("additionalProperties") is False:
                fail(f"{label} carries {key!r}, which the schema does not allow")
    items = spec.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        for n, element in enumerate(value):
            check_value(f"{label}[{n}]", element, items)


def check_answer(found: validate.Artifact, schema: JSONObject) -> int:
    """The decide gate's `check`: the whole answer against the caller's schema."""
    check_root(schema)
    check_value("answer", found, schema)
    return 1


def summarize(found: validate.Artifact, count: int) -> str:
    del found, count
    return ""


def read_schema(path: Path) -> tuple[str, JSONObject]:
    """The schema's text and parsed form, refused before any model is called."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise errors.UsageError(f"cannot read --schema {path}: {exc}") from exc
    try:
        parsed = validate.loads(text)
    except ValueError as exc:
        raise errors.UsageError(f"--schema {path} is not JSON: {exc}") from exc
    try:
        if not isinstance(parsed, dict):
            fail("schema is not a JSON object")
        check_root(parsed)
    except errors.GateError as exc:
        raise errors.UsageError(f"--schema {path}: {exc}") from exc
    return text, parsed


def build_prompt(*, prompt: str, schema_text: str) -> str:
    """The caller's prompt, then the answer contract.

    The schema travels in the prompt for both providers: codex is not given it any other
    way, and the gate enforces it on the answer either way.
    """
    return (
        f"{prompt.rstrip()}\n\n"
        "## Answer format\n\n"
        "Your final message must be exactly one JSON object that satisfies this JSON Schema, "
        "with no prose and no code fences around it:\n\n"
        f"```json\n{schema_text.strip()}\n```\n\n"
        f"{BOUNDARY}\n"
    )


# What a provider CLI prints when the subscription login itself is refused. Matched only in
# the CLI's own diagnostics (stderr) and in error-shaped events, never in the model's text,
# because a decision about billing can quote any of these words.
SUBSCRIPTION_REFUSED = re.compile(
    r"\b40[12]\b|payment required|usage balance exhausted|usage[ _]limit"
    r"|insufficient (?:credits|balance|quota)|exceeded your (?:current )?quota"
    r"|unauthori[sz]ed|not logged in|login is required|log in again"
    r"|authentication (?:failed|required|error)|token (?:has )?expired",
    re.IGNORECASE,
)

# The most of each file read for the match: a refusal is printed at the end of a short run,
# and an event stream is the provider's size to choose.
TAIL_BYTES = 64 * 1024


def _tail_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def error_events(events: Iterable[JSONObject]) -> list[JSONObject]:
    """The events in which the CLI, not the model, reports a failure.

    codex: `error` and `turn.failed`. grok: a `result` that reports `is_error`.
    """
    return [
        event
        for event in events
        if event.get("type") in ("error", "turn.failed")
        or (event.get("type") == "result" and event.get("is_error") is True)
    ]


def _strings(value: JSONValue) -> list[str]:
    # Strings only: an event's token counts and durations are numbers, and one that happens
    # to be 401 or 402 is not a status.
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def refusal(stderr_file: Path, events_file: Path, *, include_stderr: bool) -> str | None:
    """The first line naming a refused subscription login, or None.

    stderr is read only for a run that failed: a successful run's stderr can carry warnings
    from components unrelated to the model's login.
    """
    texts = [_tail_text(stderr_file)] if include_stderr else []
    events = validate.objects(validate.ndjson_lines(_tail_text(events_file)))
    texts.extend(line for event in error_events(events) for line in _strings(event))
    for text in texts:
        for line in text.splitlines():
            if SUBSCRIPTION_REFUSED.search(line):
                return line.strip()[: validate.FAILURE_LINE_CHARS]
    return None
