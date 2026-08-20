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

fail_case() {
  printf 'FAIL: %s\n' "$1"
  FAILURES=$((FAILURES + 1))
}

# The provenance sidecar has to be checked HERE, at process level: the unit suite
# exercises write_provenance() in isolation, so deleting the wrapper's --provenance-out
# argument, or the whole PROV_ARGS expansion, left both suites green while no sidecar was
# ever written. Its stated job is attesting which brief produced these findings, so the
# hash is the assertion, not mere existence.
check_provenance() {
  local provider="$1" prov="$WORK/run/adversarial-reviewer-$1-provenance.json"
  [ -s "$prov" ] || { fail_case "$provider: no provenance sidecar written"; return 0; }
  python3 - "$prov" "$WORK/assets/personas/adversarial-reviewer.md" "$provider" <<'PY' ||
import hashlib, json, sys
prov = json.load(open(sys.argv[1]))
want = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
for key, expected in (("persona_sha256", want), ("provider", sys.argv[3]),
                      ("persona", "adversarial-reviewer")):
    if prov.get(key) != expected:
        sys.exit(f"{key}={prov.get(key)!r}, wanted {expected!r}")
if not prov.get("schema_sha256"):
    sys.exit("no schema_sha256 recorded")
PY
    fail_case "$provider: provenance sidecar does not attest this run's brief"
}

# $1 label, $2 expected exit, $3 wrapper, $4 stub body
check() {
  local label="$1" want="$2" wrapper="$3" body="$4" got=0 runner stub provider
  runner="${wrapper#ce-}"
  provider="${runner%-persona}"
  stub="$WORK/bin/$provider"
  printf '#!/usr/bin/env bash\n%s\n' "$body" >"$stub"
  chmod +x "$stub"
  rm -rf "$WORK/run" && mkdir -p "$WORK/run"
  CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/$wrapper" adversarial-reviewer \
    >"$WORK/out.txt" 2>"$WORK/err.txt" || got=$?
  if [ "$got" = "$want" ]; then
    printf 'ok:   %s (exit %s)\n' "$label" "$got"
    if [ "$want" = 0 ]; then
      check_provenance "$provider"
      # stdout is ONE line for both wrappers, not just grok: a calling agent pays for
      # every token of it, and the codex arm was never asserted.
      [ "$(wc -l <"$WORK/out.txt")" -eq 1 ] ||
        fail_case "$provider: stdout was $(wc -l <"$WORK/out.txt") lines, want 1"
    fi
  else
    printf 'FAIL: %s -- wanted exit %s, got %s\n' "$label" "$want" "$got"
    sed 's/^/      /' "$WORK/err.txt" | tail -3
    FAILURES=$((FAILURES + 1))
  fi
}

# --- grok: NDJSON stream whose terminal `result` event carries structured_output ------
check "grok: structured_output passes" 0 ce-grok-persona \
  "cat <<'EOF'
{\"type\":\"system\",\"subtype\":\"init\"}
{\"type\":\"result\",\"is_error\":false,\"structured_output\":$ANSWER}
EOF"
check "grok: is_error result fails" 1 ce-grok-persona \
  "echo '{\"type\":\"result\",\"is_error\":true,\"stop_reason\":\"max_tokens\"}'"
# Schema-constrained decoding is why these two matter: a truncated or refused run still
# returns a well-formed findings object, so the answer alone cannot tell them apart.
check "grok: truncated run with a well-formed answer fails" 1 ce-grok-persona \
  "echo '{\"type\":\"result\",\"is_error\":false,\"stop_reason\":\"max_tokens\",\"structured_output\":$ANSWER}'"
check "grok: non-success subtype fails" 1 ce-grok-persona \
  "echo '{\"type\":\"result\",\"is_error\":false,\"subtype\":\"error_max_turns\",\"structured_output\":$ANSWER}'"
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
# The prompt asks for the object "at the END of your reply (after any analysis)", so a
# model that does exactly that must not have its review thrown away.
check "codex: analysis then object passes" 0 ce-codex-persona \
  "${CODEX_STUB/BODY/I reviewed the change. One issue stood out.

$ANSWER}"
# A codex run that exits 0 without writing a final message must FAIL, not silently
# re-validate whatever the previous run left at the same deterministic path.
printf '\n'
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\n%s\n' "${CODEX_STUB/BODY/$ANSWER}" >"$WORK/bin/codex"
chmod +x "$WORK/bin/codex"
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-codex-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || fail_case "codex seed run exited non-zero"
printf '#!/usr/bin/env bash\ncat >/dev/null\necho "no final message this time"\n' >"$WORK/bin/codex"
chmod +x "$WORK/bin/codex"
stale=0
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-codex-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || stale=$?
if [ "$stale" = 1 ] && [ ! -e "$WORK/run/adversarial-reviewer-codex.json" ]; then
  printf 'ok:   codex run with no final message fails and leaves no stale artifact\n'
else
  fail_case "codex run with no final message exited $stale, artifact still present"
fi

# --- an oversized prompt is refused BEFORE the model runs, never summarized -----------
# BOTH wrappers: preflight_size is duplicated into each, and covering only grok meant
# neutering the codex copy left the suite green.
printf '\n'
for wrapper in ce-grok-persona ce-codex-persona; do
  runner="${wrapper#ce-}"; runner="${runner%-persona}"
  rm -rf "$WORK/run" && mkdir -p "$WORK/run"
  printf '#!/usr/bin/env bash\ncat >/dev/null\necho "the model must never be called for an oversized prompt"; exit 0\n' \
    >"$WORK/bin/$runner"
  chmod +x "$WORK/bin/$runner"
  oversized=0
  CE_PERSONA_RUN_DIR="$WORK/run" CE_PERSONA_MAX_PROMPT_TOKENS=1 \
    "$SCRIPTS/$wrapper" adversarial-reviewer >"$WORK/out.txt" 2>"$WORK/err.txt" || oversized=$?
  if [ "$oversized" = 78 ] && grep -q 'over the 1-token budget' "$WORK/err.txt"; then
    printf 'ok:   %s: oversized prompt refused up front (exit 78)\n' "$wrapper"
  else
    fail_case "$wrapper: oversized prompt was not refused with 78 (exit $oversized)"
  fi
done

# --- the budget weighs the DIFF, which is the input that actually overruns a context ---
# The prompt does not contain the change, it tells the agent to fetch it -- so measuring
# the prompt alone gave the same ~1.2KB for `-b HEAD~1` and `-b HEAD~2000`, the budget
# could never fire, and "narrow the review with -b" was advice that could not work.
printf '\n'
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid
export GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
REPO="$WORK/repo"
mkdir -p "$REPO"
git -C "$REPO" init -q -b main
: >"$REPO/f.txt"
git -C "$REPO" add f.txt && git -C "$REPO" commit -qm base
BASE_SHA="$(git -C "$REPO" rev-parse HEAD)"
python3 -c "print('x' * 40000)" >"$REPO/f.txt"
git -C "$REPO" add f.txt && git -C "$REPO" commit -qm big

rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\ncat >/dev/null\necho "the model must never be called for an oversized review"; exit 0\n' \
  >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
bigdiff=0
CE_PERSONA_RUN_DIR="$WORK/run" CE_PERSONA_MAX_PROMPT_TOKENS=2000 \
  "$SCRIPTS/ce-grok-persona" adversarial-reviewer -C "$REPO" -b "$BASE_SHA" \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || bigdiff=$?
if [ "$bigdiff" = 78 ] && grep -q 'diff bytes' "$WORK/err.txt"; then
  printf 'ok:   a large diff trips the budget the prompt alone never could (exit 78)\n'
else
  fail_case "large diff was not refused with 78 (exit $bigdiff)"
fi

# Control: the same budget with an empty diff must NOT refuse, or the check above would
# pass for the wrong reason.
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\ncat <<EOF\n{"type":"result","is_error":false,"structured_output":%s}\nEOF\n' "$ANSWER" \
  >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
CE_PERSONA_RUN_DIR="$WORK/run" CE_PERSONA_MAX_PROMPT_TOKENS=2000 \
  "$SCRIPTS/ce-grok-persona" adversarial-reviewer -C "$REPO" -b HEAD \
  >"$WORK/out.txt" 2>"$WORK/err.txt" ||
  fail_case "an empty diff under the same budget was refused (exit $?)"
printf 'ok:   an empty diff under the same budget is not refused\n'

# --- a base ref that does not resolve is an UNSCOPED review, not a smaller one ---------
printf '\n'
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
printf '#!/usr/bin/env bash\ncat >/dev/null\necho "the model must never be called for an unresolvable base"; exit 0\n' \
  >"$WORK/bin/grok"
chmod +x "$WORK/bin/grok"
badbase=0
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer \
  -C "$REPO" -b no-such-ref >"$WORK/out.txt" 2>"$WORK/err.txt" || badbase=$?
if [ "$badbase" = 2 ] && grep -q 'does not resolve' "$WORK/err.txt"; then
  printf 'ok:   an unresolvable base ref is refused up front (exit 2)\n'
else
  fail_case "unresolvable base ref was not refused with 2 (exit $badbase)"
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

# --- a failed run leaves NO artifact, not the previous run's --------------------------
# Artifact paths are deterministic and $CE_PERSONA_RUN_DIR is documented as a directory
# the caller points at, so without an explicit clear a failed run left the last run's
# findings and provenance sitting beside this run's fresh event stream.
printf '\n'
cat >"$WORK/bin/grok" <<'STUB'
#!/usr/bin/env bash
echo '{"type":"result","is_error":true,"stop_reason":"max_tokens"}'
STUB
chmod +x "$WORK/bin/grok"
staleg=0
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || staleg=$?
if [ "$staleg" = 1 ] &&
  [ ! -e "$WORK/run/adversarial-reviewer-grok.json" ] &&
  [ ! -e "$WORK/run/adversarial-reviewer-grok-provenance.json" ]; then
  printf 'ok:   a failed grok run leaves no stale findings or provenance behind\n'
else
  fail_case "failed grok run exited $staleg and left artifacts from the previous run"
fi

# --- the brief the model actually receives carries the rubric and the closing boundary -
# Asserted on the PROMPT, not the call site: `return 0` as rubric()'s first line leaves
# every text-level assertion green while the reviewer grades against a rubric it never
# saw, and nothing in the output would look wrong.
printf '\n'
cat >"$WORK/bin/grok" <<STUB
#!/usr/bin/env bash
p=""
while [ "\$#" -gt 0 ]; do
  if [ "\$1" = "--prompt-file" ]; then p="\$2"; shift 2; else shift; fi
done
cp "\$p" "$WORK/prompt.txt"
printf '%s\n' '{"type":"result","is_error":false,"structured_output":$ANSWER}'
STUB
chmod +x "$WORK/bin/grok"

rm -rf "$WORK/run" && mkdir -p "$WORK/run"
rm -f "$WORK/prompt.txt"
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || fail_case "rubric run exited non-zero"
grep -q 'Anchors 0 and 25 mean SUPPRESS' "$WORK/prompt.txt" ||
  fail_case "the fallback rubric never reached the prompt"
grep -q 'testing_gaps when they apply' "$WORK/prompt.txt" ||
  fail_case "the closing boundary never reached the prompt"
grep -q 'Break it. Return findings matching the findings schema.' "$WORK/prompt.txt" ||
  fail_case "the persona brief never reached the prompt"
printf 'ok:   the prompt carries the brief, the fallback rubric and the boundary\n'

# And the verbatim branch, which is the one production takes: when the plugin ships a
# subagent-template.md, its block must be what goes out, not the fallback gloss.
cat >"$WORK/assets/subagent-template.md" <<'TPL'
**Schema conformance** — every finding carries file, line and evidence.
RUBRIC-MARKER-FROM-TEMPLATE
Example of a schema-valid finding:
TPL
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
rm -f "$WORK/prompt.txt"
CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer \
  >"$WORK/out.txt" 2>"$WORK/err.txt" || fail_case "template-rubric run exited non-zero"
if grep -q 'RUBRIC-MARKER-FROM-TEMPLATE' "$WORK/prompt.txt" &&
  ! grep -q 'Anchors 0 and 25 mean SUPPRESS' "$WORK/prompt.txt"; then
  printf "ok:   the plugin's own rubric block is sent verbatim when it exists\n"
else
  fail_case "the subagent-template rubric did not replace the fallback"
fi
rm -f "$WORK/assets/subagent-template.md"

# --- a persona name is a bare brief name, never a path --------------------------------
# The name is interpolated into every artifact path, so a traversal both reached a
# findings-capable brief outside personas/ and pushed the run's artifacts outside
# $RUN_DIR -- where the clear-stale-artifacts `rm -f` deleted files that were not ours.
printf '\n'
printf '# Outside\n\nfindings schema\n' >"$WORK/assets/outside.md"
printf 'PRECIOUS\n' >"$WORK/outside-grok.json"
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
cat >"$WORK/bin/grok" <<'STUB'
#!/usr/bin/env bash
echo '{"type":"result","is_error":true,"stop_reason":"max_tokens"}'
STUB
chmod +x "$WORK/bin/grok"
for name in ce-grok-persona ce-codex-persona; do
  traversal=0
  CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/$name" ../outside \
    >"$WORK/out.txt" 2>"$WORK/err.txt" || traversal=$?
  if [ "$traversal" = 2 ] && grep -q 'bare brief name' "$WORK/err.txt"; then
    printf 'ok:   %s: a traversal persona name is refused (exit 2)\n' "$name"
  else
    fail_case "$name: traversal persona name was not refused (exit $traversal)"
  fi
done
[ -s "$WORK/outside-grok.json" ] ||
  fail_case "a file outside the run dir was deleted by a traversal persona name"
rm -f "$WORK/assets/outside.md" "$WORK/outside-grok.json"

# --- a persona_review/ directory in the CWD must not replace the gate -----------------
# `python3 -c` and `python3 -m` put the cwd at sys.path[0], AHEAD of PYTHONPATH, and the
# repo under review is the default cwd. Without PYTHONSAFEPATH a planted package answered
# for the gate: exit 0 and a clean summary for a run that categorically did not review,
# with the planted module executing as the user before the runner was even invoked.
printf '\n'
mkdir -p "$WORK/hijack/persona_review"
: >"$WORK/hijack/persona_review/__init__.py"
printf 'import sys\nprint("ce-persona: 0 findings -> HIJACKED")\nsys.exit(0)\n' \
  >"$WORK/hijack/persona_review/validate.py"
rm -rf "$WORK/run" && mkdir -p "$WORK/run"
cat >"$WORK/bin/grok" <<'STUB'
#!/usr/bin/env bash
echo '{"type":"result","is_error":true,"stop_reason":"max_tokens"}'
STUB
chmod +x "$WORK/bin/grok"
hijack=0
(
  cd "$WORK/hijack" &&
    CE_PERSONA_RUN_DIR="$WORK/run" "$SCRIPTS/ce-grok-persona" adversarial-reviewer
) >"$WORK/out.txt" 2>"$WORK/err.txt" || hijack=$?
if [ "$hijack" != 0 ] && ! grep -q HIJACKED "$WORK/out.txt"; then
  printf 'ok:   a persona_review/ package in the cwd does not replace the gate (exit %s)\n' "$hijack"
else
  fail_case "the gate was replaced by a persona_review/ package in the cwd (exit $hijack)"
fi

if [ "$FAILURES" -ne 0 ]; then
  printf '\n%s wrapper test(s) failed\n' "$FAILURES" >&2
  exit 1
fi
printf '\nall wrapper process tests passed\n'
