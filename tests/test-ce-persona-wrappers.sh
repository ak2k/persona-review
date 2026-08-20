#!/usr/bin/env bash
# Process-level tests for ce-grok-persona / ce-codex-persona against stub runners.
#
# The unit suite reads the wrappers as text; this runs them. It is the only thing that
# exercises what the wrappers actually promise: the runner's exit status propagates, a
# run that returns no schema-shaped findings exits non-zero rather than reading as a
# clean review, the findings artifact is written, and the TRANSCRIPT NEVER REACHES
# STDOUT -- stdout is one summary line, because a calling agent pays for every token of
# it.
#
# The stubs emit the runners' real shapes: grok's NDJSON with a terminal `result` event
# carrying `structured_output`, and codex's `-o` final-message file.
#
# No network, no credentials, no real model: the stubs on PATH are the runners.
set -euo pipefail

# The directory under test. Defaults to ../bin (the source tree); $PERSONA_REVIEW_BIN
# points it at a BUILT package, which is how the flake check runs these against what
# actually ships rather than against edited files.
SCRIPTS="${PERSONA_REVIEW_BIN:-$(cd "$(dirname "$0")/../bin" && pwd)}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A fixture assets tree standing in for the plugin's ce-code-review/references. The persona
# ends with a schema-valid EXAMPLE object, as 13 of the 16 real briefs do, and mentions the
# findings schema, which is how the wrappers tell a findings-capable brief from the three
# that specify markdown output.
mkdir -p "$WORK/assets/personas"
# shellcheck disable=SC2016  # a literal persona brief, not a shell expansion
printf '# Adversarial Reviewer\n\nBreak it. Return findings matching the findings schema.\n\n```json\n{"reviewer":"adversarial","findings":[],"residual_risks":[],"testing_gaps":[]}\n```\n' \
  >"$WORK/assets/personas/adversarial-reviewer.md"
# A brief that cannot do the job: markdown output, no findings contract. The wrappers must
# refuse it up front rather than producing a run that ends in "no findings JSON object",
# which is indistinguishable from a model giving up.
printf '# Agent Native Reviewer\n\n## Output Format\n\nA markdown capability table.\n' \
  >"$WORK/assets/personas/agent-native-reviewer.md"
# Production-shaped: types and array minimums, not just enums. A fixture weaker than the
# real schema would let these process tests pass answers the installed plugin rejects.
cat >"$WORK/assets/findings-schema.json" <<'JSON'
{
  "required": ["reviewer", "findings", "residual_risks", "testing_gaps"],
  "properties": {
    "reviewer": { "type": "string" },
    "residual_risks": { "type": "array" },
    "testing_gaps": { "type": "array" },
    "findings": {
      "items": {
        "required": ["title", "severity", "file", "line", "evidence", "pre_existing"],
        "properties": {
          "title": { "type": "string" },
          "file": { "type": "string" },
          "line": { "type": "integer" },
          "evidence": { "type": "array", "minItems": 1 },
          "pre_existing": { "type": "boolean" },
          "severity": { "enum": ["P0", "P1", "P2", "P3"] }
        }
      }
    }
  }
}
JSON

export CE_REVIEW_ASSETS="$WORK/assets"
export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin"

FINDING='{"title":"t","severity":"P1","file":"f","line":1,"evidence":["f:1 -- x"],"pre_existing":false}'
ANSWER="{\"reviewer\":\"adversarial-reviewer\",\"findings\":[$FINDING],\"residual_risks\":[],\"testing_gaps\":[]}"
TYPE_INVALID='{"reviewer":"a","findings":[{"title":"t","severity":"P1","file":"f","line":"nope","evidence":["x"],"pre_existing":false}],"residual_risks":[],"testing_gaps":[]}'
FAILURES=0

# $1 label, $2 expected exit, $3 wrapper, $4 stub body
check() {
  local label="$1" want="$2" wrapper="$3" body="$4" got=0 runner stub
  runner="${wrapper#ce-}"
  stub="$WORK/bin/${runner%-persona}"
  printf '#!/usr/bin/env bash\n%s\n' "$body" >"$stub"
  chmod +x "$stub"
  rm -rf "$WORK/run" && mkdir -p "$WORK/run"
  CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/$wrapper" adversarial-reviewer \
    >"$WORK/out.txt" 2>"$WORK/err.txt" || got=$?
  if [ "$got" = "$want" ]; then
    printf 'ok:   %s (exit %s)\n' "$label" "$got"
  else
    printf 'FAIL: %s -- wanted exit %s, got %s\n' "$label" "$want" "$got"
    sed 's/^/      /' "$WORK/err.txt" | tail -3
    FAILURES=$((FAILURES + 1))
  fi
}

fail_case() {
  printf 'FAIL: %s\n' "$1"
  FAILURES=$((FAILURES + 1))
}

# --- grok: NDJSON stream whose terminal `result` event carries structured_output ------
check "grok: structured_output passes" 0 ce-grok-persona \
  "cat <<'EOF'
{\"type\":\"system\",\"subtype\":\"init\"}
{\"type\":\"result\",\"is_error\":false,\"structured_output\":$ANSWER}
EOF"
check "grok: is_error result fails" 1 ce-grok-persona \
  "echo '{\"type\":\"result\",\"is_error\":true,\"stop_reason\":\"max_tokens\"}'"
check "grok: no result event fails" 1 ce-grok-persona \
  "echo '{\"type\":\"system\",\"subtype\":\"init\"}'"
check "grok: type-invalid structured_output fails" 1 ce-grok-persona \
  "echo '{\"type\":\"result\",\"is_error\":false,\"structured_output\":$TYPE_INVALID}'"
check "grok: runner failure propagates" 3 ce-grok-persona "echo partial; exit 3"

# --- codex: -o writes the final message; the transcript goes to the events file -------
# The stub must honour -o, so it parses argv the way the real CLI does. The single quotes are
# deliberate: this is the stub's source text, expanded when the stub runs, not here.
# shellcheck disable=SC2016
CODEX_STUB='last=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) last="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cat >/dev/null
echo "transcript line the caller must never see"
printf %s '"'"'BODY'"'"' > "$last"'

check "codex: final message passes" 0 ce-codex-persona "${CODEX_STUB/BODY/$ANSWER}"
check "codex: prose final message fails" 1 ce-codex-persona "${CODEX_STUB/BODY/I gave up.}"
check "codex: type-invalid final message fails" 1 ce-codex-persona "${CODEX_STUB/BODY/$TYPE_INVALID}"
check "codex: runner failure propagates" 4 ce-codex-persona "cat >/dev/null; exit 4"

# --- an oversized prompt is refused BEFORE the model runs, never summarized -----------
printf '\n'
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\necho "the model must never be called for an oversized prompt"; exit 0\n' >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
oversized=0
CE_PERSONA_RUN_DIR="$WORK/run" CE_PERSONA_MAX_PROMPT_TOKENS=1 \
  "$SCRIPTS/ce-grok-persona" adversarial-reviewer >"$WORK/out.txt" 2>"$WORK/err.txt" || oversized=$?
if [ "$oversized" = 78 ] && grep -q 'over the 1-token budget' "$WORK/err.txt"; then
  printf 'ok:   oversized prompt refused up front (exit 78)\n'
else
  fail_case "oversized prompt was not refused with 78 (exit $oversized)"
fi

# --- a brief that cannot return findings is refused BEFORE the model runs -------------
printf '\n'
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\necho "the model must never be called for this persona"; exit 0\n' >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
incapable=0
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" agent-native-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || incapable=$?
if [ "$incapable" = 2 ] && grep -q 'markdown output format' "$WORK/err.txt"; then
  printf 'ok:   markdown-only persona refused up front (exit 2)\n'
else
  fail_case "markdown-only persona was not refused (exit $incapable)"
fi

# --- the contract itself: stdout is a summary line, artifacts land in the run dir -----
printf '\n'
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\ncat <<EOF\n{"type":"result","is_error":false,"structured_output":%s}\nEOF\n' "$ANSWER" >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || fail_case "contract run exited non-zero"

if [ "$(wc -l <"$WORK/out.txt")" -eq 1 ]; then
  printf 'ok:   stdout is exactly one line: %s\n' "$(cat "$WORK/out.txt")"
else
  fail_case "stdout was $(wc -l <"$WORK/out.txt") lines, want 1"
fi
grep -q 'P1' "$WORK/out.txt" || fail_case "summary line carries no severity breakdown"
[ -s "$WORK/run/adversarial-reviewer-grok.json" ] ||
  fail_case "no findings artifact written"
[ -s "$WORK/run/adversarial-reviewer-grok-events.jsonl" ] ||
  fail_case "no event stream retained"
python3 -c "
import json,sys
o=json.load(open('$WORK/run/adversarial-reviewer-grok.json'))
sys.exit(0 if o.get('findings') and o['findings'][0]['severity']=='P1' else 1)
" || fail_case "findings artifact is not the validated object"
printf 'ok:   findings artifact + event stream written to the run dir\n'

if [ "$FAILURES" -ne 0 ]; then
  printf '\n%s wrapper test(s) failed\n' "$FAILURES" >&2
  exit 1
fi
printf '\nall wrapper process tests passed\n'
