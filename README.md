# persona-review

Run a [compound-engineering](https://github.com/EveryInc/compound-engineering-plugin) reviewer
persona through xAI's `grok` or OpenAI's `codex`, and get **schema-valid findings** back — not a
transcript.

Built for a calling agent. One invocation costs the caller a single line of stdout:

```console
$ ce-grok-persona adversarial-reviewer -b origin/main
ce-grok-persona: 5 findings (1 P1, 3 P2, 1 P3) -> /tmp/.../adversarial-reviewer-grok.json
```

The model's reasoning, its tool calls, and the full event stream go to files. A 1.4 MB event
stream and a 10 KB findings artifact is a real measurement from a real run — that ratio is the
point.

## Why this exists

Compound-engineering already runs a cross-model adversarial pass, and if that is what you want, use
it: `/ce-code-review` does this and merges the results. This exists for the three things it cannot
do — any persona (its worker hardcodes one), any model and reasoning effort (it pins one per
provider), and **streaming on the grok route** (it constrains decoding in a way that precludes it).

## The contract

| Tier | Cost | What you get |
|------|------|--------------|
| 0 | one line + exit status | counts by severity, artifact path. A gating caller reads nothing else. |
| 1 | ~20 tokens/finding | `ce-persona-findings <artifact>` — severity, `file:line`, title, confidence, and the one quoted line that motivates it. P0/P1 only; `--all` for every severity |
| 2 | per finding | `ce-persona-findings <artifact> --show N` — why it matters, full evidence, suggested fix |
| 2 | every finding | `ce-persona-findings <artifact> --show all` — the same render for every finding, every severity, in `#` order, inside one fence, each separated from the next by a line reading `----`. The separator is a reading aid: finding text can contain the same line, so the fence, not the separator, is the boundary |
| — | whole artifact | `ce-persona-findings <artifact> --json` — the raw object, unchanged and unfenced, for a programmatic caller |
| — | whole artifact | `ce-persona-findings <artifact> --return` — the compact **return** object compound-engineering's merge helper expects, unfenced |
| — | whole artifact | `ce-persona-findings <artifact> --anchors -C <dir>` — for each finding, where its quote is in the reviewed tree and the keys a poster marks its comments with, unfenced |
| — | ~10 tokens/verdict | `ce-persona-findings <verdicts-artifact>` — one row per verdict in `#` order: `#N validated — <reason>` or `#N REJECTED — <reason>` |
| — | one verdict | `ce-persona-findings <verdicts-artifact> --show N` — the verdict addressed to finding `#N`; `--show all` is the listing above, accepted for symmetry |

The same command reads both artifact shapes, because both are model output being handed to an
agent and one reader is one place to keep the fence and the refusal. On a verdicts artifact
`--all` changes nothing (a verdict has no severity, so no row is hidden from the default
listing) and `--return` is a **usage error**: the merge helper reads findings, and a verdict
projected into that shape would arrive with no title, file or line. A file carrying both a
`findings` and a `verdicts` list is exit `1` — it is two half-written artifacts, and rendering
either half would report a complete result.

`N` in `--show N` is the number rendered as `#N` in the listing. Numbering follows the
artifact's own order; the listing is *displayed* most-severe-first, so `#1` is not
necessarily the top row. `--show all` follows the numbering rather than the display order,
and takes the same precedence as `--show N`: it wins over `--list`/`--all`, `--json`
outranks it, and it cannot be combined with `--return`. On an artifact with no entries it
prints `no findings` (or `no verdicts`) unfenced at exit `0`, as `--all` does.

**Exit status is the verdict.** For `ce-grok-persona` / `ce-codex-persona`:

| Exit | Meaning |
|------|---------|
| `0` | schema-valid findings (an empty findings array is valid) |
| `1` | the answer was not schema-valid findings — the gate refused |
| `2` | usage error: bad arguments, unknown or markdown-only persona, bad `-C`, unresolvable `-b`, malformed `CE_PERSONA_*` value |
| `3` | environment error: the runner, `git` or the plugin assets are missing, `CE_PERSONA_RUN_DIR` cannot be created, the runner's event vocabulary changed and this build can no longer count what a run did, or the model attempted local tool calls and none succeeded — stderr gives the counts and the first failure's output, and the artifacts are kept as evidence |
| `4` | the runner itself exited non-zero |
| `5` | idle or hard timeout; the run was killed and partial output kept |
| `6` | the model answered without attempting a single local tool call — it inspected nothing. Through codex a local call is a `command_execution` or `local_shell_call` item; a file change, patch, web search, MCP call or function call is not one |
| `78` | over `CE_PERSONA_MAX_PROMPT_TOKENS` — refused, never summarized |

The distinctions that matter to a caller are `1`, `4` and `6`. `1` is the model's fault — it
answered with something that is not findings. `4` is the runner's — it exited non-zero and
never got that far. **`6` is the one worth retrying**: the model answered, the answer may be
perfectly schema-valid, and it attempted zero local tool calls, so it never opened the diff and
certified nothing — empty findings or a page of them. That is a run that happened, not a
hypothetical: one turn, 151 output tokens, four and a half seconds, `{"findings": []}`, exit
`0`. Nothing is wrong with the machine or the invocation, so the same command is worth
running once; a second `6` says something about the model rather than about the code. The
wrapper does not retry for you, deliberately — another full-effort run is the caller's
budget to spend.

**`6` is never the answer when the wrapper is the broken part.** If a provider CLI upgrade
renames the event kinds this counts, every run would count zero and `6` would blame the model
on every one of them — a permanent outage wearing the costume of a bad model. So a stream
carrying kinds this build does not recognize, or no events at all, exits `3` naming the
unrecognized kinds instead; `6` is reached only when the stream was understood and there was
genuinely nothing in it.

**A local call counts only if the stream shows it succeeded.** Through codex, a command counts
when its `item.completed` carries `exit_code` 0; a file change or patch is not a local call,
because a write shows no read of the tree. Through grok, a call counts when it is a tool that
reads the tree (a command, a search, a file read, a directory listing, or a background
command's output) and its `tool_result` has `is_error` false and content that is a JSON object
in which every report, including a background command's own, carries no `status` other than
`completed` and no exit code other than 0. A status grok has never sent is refused rather than
taken for success. For a poll of several background commands, at least one of them must
report exit code 0 and no status other than `completed`. A todo, a task kill or a file edit
never counts. A call also counts only when its id pairs it with its result unambiguously: an
id that two calls share, or one reported both succeeding and failing, counts for none of them.
A run that attempted local calls
and had none succeed exits `3`, not `6`: the model tried, and a provider that cannot start a
command fails every call the same way on every run. That is a run that happened too —
codex-cli 0.156.1 could not start five commands, each completed `failed` with exit code 1, and
0.3.4 passed its empty findings at exit `0`. The one stderr line gives the counts, the status
the first failed call reported when it is not `completed`, and the first line that call
printed, and names the id of any success that did not count because its id was ambiguous; it
does not name a cause, because a command that ran and
exited non-zero, like `rg` finding nothing, reads the same as one that never started. The
artifacts are kept as evidence, as for `6`.

**A review is defined as inspecting the repository.** Passing the material in the prompt
(`-c -`, or a large `-c` file) and expecting the model to review it without touching the
working tree still exits `6`: the successful local tool-call count is what makes a finding checkable
against the code, and there is no mode in which this package certifies a review of text it
cannot tie to a file. Use the model directly for that.

`ce-persona-findings` uses the same vocabulary, narrowed to what it can hit: `0` rendered, `1`
the file is unreadable, is neither a findings nor a verdicts artifact, is both at once, or
could not be projected into a usable return, `2` usage error — including no such finding or
verdict number, `--return` with `--json` or `--show`, `--anchors` with `--json`, `--show` or
`--return`, `--return`, `--verify-quotes` or `--anchors` on a verdicts artifact,
`--verify-quotes` without `--return` or without `-C`, `--anchors` without `-C`, a `-C` given
without `--verify-quotes` or `--anchors`, and a `-C` that is missing, is not a directory, or
names an unknown user —
and `6` when the artifact's provenance records a run that made no local tool calls, or none that
succeeded. That last one matters because a refusal has to survive being handed on: the review
command keeps the dud artifact as evidence, and without the check this package would launder its
own refusal into an ordinary listing at exit `0`, one command later. A run whose every local call
failed is refused with `6` here although the review command exited `3` for it — this command has
no environment status, and the refusal says which status the review gave.

Everything `ce-persona-findings` renders is wrapped in `BEGIN/END UNTRUSTED MODEL OUTPUT` with a
per-run nonce. It is text a model wrote about a repository it read, being handed to another agent
as *its* input; the fence is what lets the consumer tell data from instructions. `--json`,
`--return` and `--anchors` are unfenced, because a programmatic caller parses them rather than
reading them.

### `--return`: an artifact as a merge input

Compound-engineering's `/ce-code-review` merges reviewer **compact returns**, not artifacts, and
its `findings-mechanics.py` demotes any finding whose `first_evidence` is missing to confidence
50 — where the gate suppresses it. A lens that filled only the `evidence` array therefore reads
as having found nothing. `--return` projects an artifact into that return shape so the merge can
be run from what the lenses wrote to disk:

```console
$ ce-persona-findings correctness.json --return > return.json
ce-persona-findings: correctness: 4 findings, 4 first_evidence backfilled from evidence[0]
```

Every top-level key except `findings` is copied verbatim (`independence_verified` included — the
helper reads it to decide cross-model promotion), and each finding keeps only the eleven
merge-tier keys the helper reads. `first_evidence` is the artifact's own value when it has one,
otherwise `evidence[0]`, which the plugin's contract makes the same string; it is never emitted
empty. The one line on stderr says how many were backfilled, so stdout stays parseable. Exit `1`
when the artifact would make a return the helper drops whole — no `reviewer`, or a
`residual_risks` / `testing_gaps` that is not a list. One artifact gives one object; assemble
several with `jq -s .`.

**Merge state is not among those keys.** `settled_conflict`, `reviewers` and
`independent_reviewers` are the orchestrator's to stamp on its own reconciled returns, and are
never copied out of a lens artifact. A truthy `settled_conflict` exempts a finding from the
helper's confidence gate, so carrying one in would walk a finding whose quote was just dropped
straight past the gate this projection exists to feed.

`--verify-quotes -C <dir>` additionally checks each quote against the file and line it cites,
read from the **working tree** under `<dir>` (in the plugin's local-aligned mode that tree is the
reviewed head, and this keeps git out of a reader). A quote the tree does not corroborate is
**dropped**, never rewritten — rewriting would manufacture evidence the lens did not give, and
dropping it lets the helper demote the finding on its own rule. Each drop prints its reason on
stderr; the artifact on disk is not touched, the exit status stays `0`, and every other byte of
the object is identical to plain `--return`.

What is checked is the quote minus the citation being checked: `f.py:12 -- code`, `f.py:12: code`
and `` `code` -- f.py:12`` all work, as do a citation wearing markdown decoration
(`**f.py:12**`, `(f.py:12)`, a backticked path, `` `f.py`:12``) and a `:col` suffix. Whitespace
is collapsed on both sides and a quote may span several lines. A quote is checked whole first,
with a repeated citation of the same location counted once; a quote citing several locations
that each carry their own text is checked location by location, every citation must resolve,
and nothing may sit outside those segments. At least one citation must name the finding's own
`file`. A backticked span is read as the quote only when the text outside the citation *is* that
span — checking a backticked aside instead certified prose as a quoted line.

A quote is dropped when no citation resolves, when its citation resolves **outside** `<dir>`
(the check is on the resolved path, so `..`, an absolute path and a symlink pointing out of the
tree are one case and the file is never read), when the cited lines are out of range, when the
text is empty, when the tree contradicts it, when no citation names the finding's own `file` —
the quote founds some other location, not this one — or when the compared text is **shorter than
12 characters** — a one-word fragment is on the line as a substring while saying nothing about the
finding, and verification that cannot fail is worse than none.

### `--anchors`: where each finding's quote is

A poster that turns findings into pull-request comments has two questions for every finding:
which line the comment goes on, and whether an earlier run already posted it. `--anchors`
answers both from the reviewed tree, so the poster never searches a diff for a quote itself:

```console
$ ce-persona-findings "$CE_PERSONA_RUN_DIR/correctness-codex.json" --anchors -C . \
    > "$CE_PERSONA_RUN_DIR/correctness-codex-anchors.json"
ce-persona-findings: anchors: 7 findings (5 verified, 1 relocated, 0 ambiguous, 1 not found, 0 unverifiable, 0 no evidence)
```

`-C` is required and names the tree that was reviewed; as with `--verify-quotes`, the working
tree under it is read and git is not asked. `--anchors` cannot be combined with `--json`,
`--show` or `--return`, and on a verdicts artifact it is a usage error. An artifact whose
provenance records no local tool calls, or none that succeeded, exits `6` with nothing on stdout,
as in every other mode.
Written beside the artifact as `<persona>-<provider>-anchors.json`, the file is cleared with the
run's other artifacts when the next run of that persona and provider starts.

stdout is one JSON document, unfenced, and stderr is the one summary line above:

```json
{
 "anchors_version": 1,
 "artifact": "<the path as given>",
 "artifact_sha256": "<sha256 of the artifact's bytes>",
 "tree": "<the -C directory, resolved>",
 "head": "<the reviewed commit, or unresolved: ...>",
 "findings": [{"#": 1, "file": "src/a.py", "path": "src/a.py", "line": 42, "state": "verified",
   "via": "line", "start": 41, "end": 42, "occurrences": 1, "candidates": [],
   "reason": "on the finding's line", "quote_key": "<64 hex>", "evidence_key": "<64 hex>"}]
}
```

- `artifact_sha256` hashes the bytes that were read and parsed, in one read, so a poster can
  check the document describes the artifact it loaded.
- `head` is the provenance sidecar's `head_sha` when that is a 40- or 64-character lowercase
  hex object id. Otherwise it is `unresolved: no provenance` or `unresolved: the provenance
  records no head commit`. A poster compares it with the head it posts to.
- `findings` has one entry per finding that is a JSON object. `#` is the finding's position in
  the artifact, the `#N` of the listing; an entry that is not an object is skipped, not
  renumbered.

Each entry:

| Field | Value |
|-------|-------|
| `#` | the finding's number |
| `file` | the finding's `file`, or `null` when it is not a string |
| `path` | the in-tree, `/`-separated path that was read; `null` when no file was read |
| `line` | the finding's `line` when it is an integer, else `null` |
| `state` | one of the six states below |
| `via` | how a placed quote was placed: `line`, `citation` or `search`; `null` for every other state |
| `start`, `end` | the 1-based, inclusive lines the quote covers; only for `verified` and `relocated`, else `null` |
| `occurrences` | on how many distinct line spans the claim occurs in the file; `null` when the file was not searched for it |
| `candidates` | for `ambiguous` only, the first 20 of those spans as `[start, end]` pairs in file order; `[]` for every other state |
| `reason` | a short diagnostic from a fixed vocabulary, for a person to read; not a contract |
| `quote_key` | 64 hex characters; `null` only for `no_evidence` |
| `evidence_key` | 64 hex characters; only for `verified` and `relocated`, else `null` |

The quote is the finding's `first_evidence`, or `evidence[0]` when it has none: the same quote
tier 1 shows. The file is split on `\n` only, as a diff numbers its lines, and matched as one
stream: its non-blank lines, whitespace collapsed, joined by single spaces. So a quote matches
however the file indents or wraps the code, and its span is the lines the match covers. The
claim searched through the whole file is the quote's claim, defined under the keys below. A
citation of the finding's own file also checks the quote less that citation, and its own
segment when the quote cites several places, but only at the lines it cites. A range
(`f.py:20-22`) cites every line from its first to its last; one written backwards cites only its
first.

A quote citing several places, each citation carrying its own snippet of at least 12 characters
and no text outside them, is `not_found` before any state below is tried when one of its
snippets is not on the lines its citation names. A snippet is on those lines when a place it
occurs in its file overlaps them, so one that starts a line before a cited range still holds; its
text is in the file either way. A snippet of another file is checked in that file, read under
the same rules as the finding's own, so one whose file is missing, outside the tree, not a
regular file, or reached through a link or a `..` is not where it says. One snippet that holds
cannot place the quote, since the comment would publish the others as code the tree holds.
`occurrences` still counts the claim. The states, first match wins:

- **`verified`**, `via` `line`: the quote occurs on lines that include the finding's `line`.
- **`relocated`**, `via` `citation`: it occurs at a line the quote cites in the finding's file.
- **`relocated`**, `via` `search`: it occurs exactly once in the file, somewhere else.
- **`ambiguous`**: it occurs more than once, and none of those places covers the finding's line
  or a line the quote cites.
- **`not_found`**: the file does not hold it, or one snippet of a quote citing several places is
  not where it cites.
- **`unverifiable`**: nothing could be checked. The finding has no `file`. Or the file is
  missing, outside the tree, not a regular file, reached through a link or a `..` (the path it
  resolves to is not the one the finding names), or unreadable. Or the quote cites only other
  files. Or every claim is shorter than 12 characters once whitespace is collapsed.
- **`no_evidence`**: there is no quote, or the quote is only a citation and claims no code.

A quote with no real newline that holds a literal `\n` or `\t` is searched again with those read
as whitespace when it is not found as written. That second reading never reaches a key.

#### The keys

A poster writes both keys into hidden markers on the comments it posts, so a later run can find
the thread it opened. Each is a SHA-256 hex digest of UTF-8 text whose parts are joined by NUL:

- `quote_key = sha256("persona-review/quote-key/1" \0 claim)`, and `null` when the claim is
  empty. It is the finding's identity across runs. The locators the steps below take off are
  where a quote carries line numbers, so inserting lines above the code does not move it; the
  limits below list the locators it keeps.
- `evidence_key = sha256("persona-review/evidence-key/1" \0 path \0 text)`, where `path` is
  the entry's `path` and `text` is the file's own lines `start` to `end`, each whitespace
  collapsed, blank ones dropped, joined by `\n`. It hashes the file rather than the lens's
  wording, so it moves only when that code changes. Two findings quoting the same lines share it.

The claim is the code the quote says the file holds, with what a lens writes around the code
taken off. A *citation* here is `path:line`, with an optional `:col`, an optional range (`-N`,
`–N` or `—N`) and markdown decoration (`**f.py:12**`, `(f.py:12)`, `` `f.py`:12 ``), whose path
is the finding's own file, that file's basename, or looks like a path: it contains `/` or `\`,
or ends in a file extension, and is not a URL. The finding's own file is read both as its
`file` spells it and with `\` read as `/` and empty, `.` and `..` segments dropped, each with or
without a leading `./`, so `./app/[id]/page.tsx` and `app/[id]/page.tsx` cite the same file
whichever side spells it which way.

1. Strip the quote's surrounding whitespace.
2. Split the quote on `\n`. A line that begins, after whitespace, with a citation loses that
   citation, then one `(verbatim)` right after it (in any case), then at most one separator —
   `:`, `--`, `—`, `–`, or `-` followed by whitespace — then its surrounding whitespace. If what
   is left of that line is exactly one backtick span, its contents replace it. A line that does
   not begin with a citation is untouched.
3. Only when no line lost a citation: a citation ending the quote comes off together with the
   separator before it, and with a `(verbatim)` between the two. A parenthesized one comes off
   with or without a separator. A bare one with no separator stays, since it may be what the
   line says (`# see docs/setup.md:40`).
4. If what is left is exactly one backtick span, take its contents.
5. Collapse every run of whitespace to one space.

A citation inside a line, ` / ` or `...` between snippets, and prose after the code are left
in, because each may be what the line says, and a claim must never be edited into text the lens
did not write.

**The keys are frozen.** A marker on a posted comment outlives any release, so these
definitions — and everything they call, including the citation pattern `--verify-quotes`
shares — are never edited. A different definition ships as a new key beside the old one, its
domain string ending in `/2` rather than `/1`, and costs one re-post of every open comment. The
document's own shape is versioned by `anchors_version`, which is `1`.

#### Limits

- **`#L12`, `L12`, `line 12` and fenced code blocks** are not read as locators. They stay in the
  claim, which the file then does not hold, so such a quote is usually `not_found`.
- **Quotes joined by `; `** (`f.py:3: a(); f.py:9: b()`) are not split, because code contains
  `; `. Only the first citation comes off, so the claim keeps the others and `quote_key` moves
  when lines are inserted above. The citations of the finding's own file are still checked at
  the lines they name, so when the code is where they cite it the entry is placed and its
  `evidence_key` is stable.
- **Lines joined by a literal `\n`** (a backslash and an `n`, not a newline:
  `f.py:108: a\nf.py:109: b`) are one line to the claim. Only the first locator comes off, so
  the claim keeps the others and `quote_key` moves when lines are inserted above. The locators
  left in also keep the claim from matching the file, so such a quote is usually `not_found`.
- **A Windows backslash citation** (`src\f.py:12`) comes off the claim, so its key is stable, but
  it names no file under the tree. When the quote cites nothing else, it cites only other files
  and is `unverifiable`.
- **A bracketed path wrapped in quote marks** (`'app/[id]/page.tsx:12'`) is read as a citation
  of `id]/page.tsx`. It stays in the claim and does not name the finding's file.
- **A citation in the middle of a line** is never taken off, since it may be what the line says.
  A quote that holds one keeps that line number in its claim.
- **A code token shaped like a citation** (`obj.attr:10`, `buf[self.pos:10]`) is read as a
  citation of another file, because `.attr` looks like a file extension. When the quote has no
  citation of the finding's file, it cites only other files and is `unverifiable`. In a quote
  citing several places, the token splits the snippet it sits in, and a true quote can be
  `not_found`.
- **An absolute path inside the tree** is read like a `..`: the path it resolves to is not the
  one it spells. A snippet of another file cited by one is not where it says, so the quote is
  `not_found`, and a `file` given as one is `unverifiable`. A citation of the finding's own file
  by an absolute path still names that file.
- **`-C` has to be the tree the review ran in.** `head` comes from the review's provenance and is
  not checked against `-C`, so lines located in another checkout are reported under the reviewed
  commit.
- **A dirty worktree's lines are reported under the reviewed commit.** The quote is located in
  the working tree, and `head` is the `HEAD` the review recorded, which holds none of the
  uncommitted changes.
- **On a case-insensitive filesystem**, the macOS default, a `file` spelled in another case than
  the tree's (`SRC/F.py` for `src/f.py`) is read, and `path` and `evidence_key` carry that
  spelling, which git does not have. A case-sensitive filesystem reports it `unverifiable`.
- **A literal `\n` in the code comes first.** A quote with no real newline that holds a literal
  `\n` is placed on a line holding that literal text when one exists, such as a string, and read
  with the `\n` as whitespace only when none does.
- **Snippets the reader cannot cut out.** A quote written as `` `a` / `b` ``, as
  `` `a` followed by `b` `` or as a backtick span followed by `.` is not matched even when it is
  true, and the entry is `not_found`. In a quote citing several places, one snippet written that
  way is enough. The rule errs toward leaving a finding unplaced rather than publishing a snippet
  the tree may not hold.
- **`--anchors` output written into a run directory must not overlap a new review there.** A
  review of the same persona and provider starting in that directory clears
  `<persona>-<provider>-anchors.json`, including one a redirect is still writing, and `--anchors`
  then exits `0` with its file gone.

## Validating a findings batch

`/ce-code-review`'s Stage 5b sends a batch of findings to a second model and asks it to confirm
or reject each one. That pass has to run as a foreground agent call, so it has no disk sentinel,
nothing to poll, and a runner at its context checkpoint cannot dispatch it at all. These two
commands run the same prompt out of process, with the watchdogs, the provenance and the
refusals the review commands already have:

```console
$ ce-grok-validate validator-input.json -b origin/main
ce-grok-validate: 5 verdicts (3 validated, 2 rejected) -> /tmp/.../validator-grok.json
$ ce-persona-findings /tmp/.../validator-grok.json
#1 validated — confirmed at resolver.py:88; the cache is read before the write lands
#2 REJECTED — the handler re-raises one line down, so nothing is swallowed
...
```

`ce-codex-validate` is the same command through `codex`. They take the same `-C`, `-b`, `-m`,
`-e` and `-c` options as the review commands, and the same `CE_PERSONA_*` environment.

**The batch** is the plugin's own `validator-input.json`: a JSON array of finding objects, each
with an integer `#` of 1 or more, unique across the array. The file's text goes into the prompt
verbatim — the caller assembled that document and it is not re-serialized. An array that is not
an array, an element that is not an object, a missing, non-integer, boolean, sub-1 or duplicate
`#`, and an empty array are each exit `2` naming the offending element. An empty batch is
refused rather than run because its only correct answer is `[]`, at the price of a full-effort
model run.

**The prompt** is the plugin's `validator-batch-template.md`, read from the same assets
directory as the persona briefs and the findings schema (`$CE_REVIEW_ASSETS`). Its first fenced
block is the prompt body; a missing file or a file with no fence exits `3`, because that is a
machine that is not set up and no different argument would fix it.

**The verdicts schema is this package's own**, shipped as package data rather than read from the
assets directory — the plugin inlines the verdict shape in that template and ships no schema
file, so there is nothing to read and the shape the gate enforces has to be ours. It requires
`verdicts` to be an array of objects with an integer `#`, a boolean `validated` and a non-empty
`reason`. It is the one place this package owns a schema, and `AGENTS.md` records why.

**Coverage is enforced**: the `#` values that come back must be exactly the batch's, each once.
Missing, extra and duplicated numbers are all exit `1`, named in the message. This is the
template's own "one verdict for every input `#`" rule, checked rather than trusted, because a
batch that comes back one verdict short otherwise reads as a completed validation and the
finding nobody judged is carried as though it had been.

**Artifacts** land on the stem `validator-<provider>` in `CE_PERSONA_RUN_DIR`: `.json` (the
verdicts object), `-provenance.json`, `-events.jsonl`, `-prompt.md`, `-stderr.log`, and a
`.lock` held for the run. The sidecar records `kind=validator` and hashes the batch, the
template, the schema and the exact prompt the model received. The batch is hashed from the
bytes that went into the prompt, not re-read afterwards, so a file replaced mid-run is not
attested as the one the validator saw.

**The exit statuses are the review commands' own**, with the nouns changed: `0` schema-valid
verdicts covering the batch, `1` the answer was not that, or also carries a findings list, `2`
usage or a malformed batch, `3` environment or local tool calls that all failed, `4` the runner
exited non-zero, `5` timeout, `6` the model attempted no local tool call, `78` over budget.
`--help` renders the table in the
validate mode's words. The second half of `1` is the reader's rule asked at the writing end: a file
carrying both lists is neither artifact, so `ce-persona-findings` refuses it, and a run that
reported success for one would be certifying verdicts nothing can render.

**`6` here is not worth retrying blind.** `validated: true` across a whole batch from a run that
opened nothing is precisely the rubber stamp this mode exists to refuse, and the artifacts are
kept as evidence — `ce-persona-findings` refuses them too, for the same reason it refuses a
review's. Another full-effort run is the caller's budget to spend; this package does not spend
it for you.

**There is no turn limit, only wall clock.** `grok` has a `--max-turns` and `codex` does not,
and a cap that exists on one provider is a contract this package cannot publish. `CE_PERSONA_IDLE_SECS`
and `CE_PERSONA_HARD_SECS` bound a validation exactly as they bound a review.

## How the findings are extracted

Not by scanning prose. That heuristic produced three silent false-passes.

- **grok** — `--json-schema` combined with `--output-format streaming-messages-json`. This composes,
  despite grok's help text saying the schema implies non-streaming JSON: the terminal `result` event
  carries `structured_output` already parsed, *and* the stream still grows for liveness. The run's
  own terminal status (`is_error`, `subtype`, `stop_reason`) is checked before its answer is
  believed, because schema-constrained decoding means a truncated run still returns a well-formed
  empty findings array.
- **codex** — `-o` writes only the agent's final message, and the prompt requires that message to be
  exactly one JSON object. Not `--output-schema`: OpenAI's strict mode rejects the plugin's schema
  outright and would force every optional field, changing what a finding means.

The strictness is load-bearing. When prose is allowed around the answer, a model that gives up can
append the brief's own schema-valid **example** object and read as a clean review — and a model that
found real defects can append the same example and have them silently replaced by an empty array.
Both were demonstrated by independent reviewers against a version that permitted it.

## Requirements

- `grok` or `codex` on PATH, **already logged in**. They are deliberately not dependencies of this
  package: it uses the subscription you already have rather than an API key of its own.
- The compound-engineering plugin installed, for the persona briefs and the findings schema. Point
  `$CE_REVIEW_ASSETS` at a `references/` directory to override.
- `python3` 3.12 or newer.
- `git`, only when `-b` is passed.

## Install

```nix
inputs.persona-review.url = "github:ak2k/persona-review";
# then add inputs.persona-review.packages.${system}.default to your profile
```

## Environment

| Variable | Effect |
|----------|--------|
| `CE_REVIEW_ASSETS` | the plugin `references/` directory to read briefs and schema from |
| `CE_PERSONA_RUN_DIR` | where artifacts land; defaults to a fresh temp dir |
| `CE_PERSONA_MAX_PROMPT_TOKENS` | budget, default 80000. Counts the prompt **plus** `git diff <base>..HEAD` |
| `CE_PERSONA_IDLE_SECS` | kill after this long with no output, default 600. Must be **> 0** |
| `CE_PERSONA_HARD_SECS` | kill after this long overall, default 2400. Must be **> 0** |

Every value is parsed and validated before any work happens, and a bad one exits `2` naming the
variable. Two rules are worth knowing because they refuse things you might expect to work:

- **A watchdog cannot be switched off.** `0` is rejected, not treated as "no timeout". An unwatched
  run is a full-effort model run that nothing will stop and nothing will read. Set a large value if
  you want a long one.
- **An unrecognized `CE_PERSONA_*` name is an error.** `CE_PERSONA_IDEL_SECS=30` would otherwise be
  ignored in silence while the real idle timeout stayed at its default — a misconfiguration that
  looks exactly like a working one.

**The budget only means something with `-b`.** Without a base ref the prompt does not contain the
change and does not name a range — the model picks its own scope — so there is genuinely nothing to
weigh, and the budget bounds the prompt alone. That is a real limit, stated rather than papered over
with a guessed base.

## Concurrency

Artifact paths are deterministic and `CE_PERSONA_RUN_DIR` is reusable, which together mean two runs
of the same persona through the same provider in the same directory address the same files. A run
takes an exclusive lock on its artifact stem; a second one **exits `2` rather than interleaving**.
Wait, or point `CE_PERSONA_RUN_DIR` somewhere else. Different personas, different providers and the
default fresh temp dir never contend.

This is a refusal rather than a queue because the alternative is not a lost race: the second run's
directory clear unlinks the first run's event stream while its provider is still writing, both
append to one events file, and the gate then validates an interleaving of two transcripts.

## Provenance

Each run writes `<persona>-<provider>-provenance.json` beside its findings, recording what produced
them: the provider, model and effort; the persona brief, the findings schema and **the exact prompt
the model received**, each by SHA-256; the repository with the resolved `head_sha` and `base_sha`;
and a `run_stats` object counting what the run *did* — `tool_calls`, `local_tool_calls`,
`local_tool_attempts`, `turns`, `output_tokens` and `duration_s`. `tool_calls` is every call the
run made; `local_tool_attempts` is the ones that act on the working directory, and
`local_tool_calls` the ones among them that succeeded. Through codex the local calls are the
`command_execution` and `local_shell_call` items, and one succeeded when it completed with
`exit_code` 0. `file_change`, `patch_apply`, `web_search`, `mcp_tool_call`, `function_call`
and `custom_tool_call` count in `tool_calls` only, because none of them proves the tree was
read. Through grok a local attempt is a call to one of `run_terminal_command`, `grep`,
`read_file`, `list_dir` or `get_command_or_subagent_output` (`--disable-web-search` removes
its web tools), and it succeeded when its `tool_result` has `is_error` false and a JSON object
as content, and every report in it carries no status but `completed` and no exit code but 0;
a poll of several background commands succeeded when at least one of them completed with exit
code 0. `todo_write`, `kill_command_or_subagent`, `search_replace` and `write` count in
`tool_calls` only. On both sides a call succeeded only when its id pairs it unambiguously: a
grok id that two `tool_use` blocks share, or a codex id that two item kinds share, or either
one reported both succeeding and failing, adds nothing to `local_tool_calls`. A grok stream with no local attempt that calls a tool name this build
does not know exits `3` naming it, as codex kind drift does. The exit status turns on the two local counts — no local attempts is exit `6`, attempts
and no successes exit `3` — so they are recorded rather than only acted on: a refusal you cannot
audit afterwards is one you have to take on trust. `ce-persona-findings` refuses a sidecar whose
`local_tool_calls` is zero.

Older sidecars are read by the rule they were written under. One written before 0.3.3 carries no
`local_tool_calls` and is refused when `tool_calls` is zero. One written by 0.3.3 or 0.3.4 carries
no `local_tool_attempts`, and its `local_tool_calls` counted attempts, failed ones included: a
zero there is refused as before, but a run whose every local call failed recorded a non-zero
count and still renders. The codex-cli 0.156.1 run above, reviewed by 0.3.4, left
`local_tool_calls: 5` and renders; the same run reviewed by 0.3.5 is refused at both ends.

The hashes are the point. Briefs live in a plugin cache that updates underneath you, so two runs are
only comparable if they ran the same brief — and `base_ref=HEAD~1` names a different commit every
day, so without the resolved SHAs a finding reading `f.py:42` cannot be tied to the code it was
about. Reviewing a directory that is not a git repository is fine; the SHA fields record
`unresolved:` rather than going missing.

## Changes in 0.3.7

A local tool call counts only when the stream shows, unambiguously, one call that read the tree
and succeeded. Runs that used to pass on a call that did not show that now exit `3` or `6`.
Every command and flag is unchanged.

- **A codex write is not a local call.** `file_change` and `patch_apply` count in `tool_calls`
  only. A run whose only calls were writes exits `6`. A run whose commands all failed beside a
  completed write exits `3`; it used to exit `0`. A stream with a write, an unrecognized kind
  and no command exits `3` as drift; it used to exit `0`.
- **A grok result must report success.** Its content has to be a JSON object, and every report
  in it may carry no `status` but `completed` and no exit code but 0. Content that is not an
  object, and a `failed` status with no exit code beside it, used to count. A status grok has
  never sent is refused. A command in a batch poll counts only with exit code 0 and no status
  but `completed`.
- **A call's id has to pair it unambiguously.** A grok id that two `tool_use` blocks share, a
  codex id that two item kinds share, or an id reported both succeeding and failing, counts for
  none of them. Such a run used to count the success. Identical repeated reports still count
  once.
- `run_stats` keeps its fields and meanings. Older sidecars are read as before.

## Changes in 0.3.6

Defaults only; every command, flag and exit status is unchanged.

- **Codex default model moved.** `ce-codex-persona` now defaults to `gpt-6.1-sol` (was
  `gpt-6-sol`). `-m` overrides it, as before.

## Changes in 0.3.5

A local tool call now counts only if it succeeded, and a run that attempted local calls and had
none succeed exits `3` instead of `0`. Every other command, flag and exit status is unchanged.

- **A failed local call no longer counts as inspection.** Through codex a command counts when it
  completes with `exit_code` 0, and a file change or patch when it completes with status
  `completed`. Through grok a call counts when it is one of the tools that read the tree and its
  `tool_result` has `is_error` false and reports no non-zero or null exit code and no command
  still running, or, for a poll of several background commands, at least one that exited 0; a
  call with no result, or a todo, task kill or file edit, does not count. A
  codex run whose commands all failed to start used to exit `0` with empty findings.
- **grok tool-name drift is detected.** A grok stream with no local attempt that calls a tool
  this build does not know exits `3` naming it. Before, any grok call counted, whatever it was.
- **Exit `3` for a run whose every local call failed.** Its stderr line gives the counts and the
  first line the first failed call printed. Nothing goes to stdout, and the artifacts are kept as
  evidence. A run that attempted no local call still exits `6`.
- **`run_stats` gains `local_tool_attempts`**, and `local_tool_calls` now counts the successes.
  `ce-persona-findings` refuses a sidecar whose `local_tool_calls` is zero, as before, and its
  refusal says whether the review exited `6` or `3`. Sidecars written by 0.3.3 and 0.3.4 are read
  by the rule they were written under, so one whose failed calls were counted still renders.
- **Codex vocabulary drift is keyed on local attempts.** A stream of recognized commands that all
  failed, beside an unrecognized kind, exits `3` as failed calls rather than as drift.
- **grok's stream is split at newlines only.** JSON carries U+2028, U+2029 and U+0085 raw inside
  a string, and a line holding one used to be cut in two and dropped: an answer holding one
  exited `1`, and a successful call whose result held one did not count.

## Changes in 0.3.4

One addition, and a fix to `--verify-quotes` that changes what the merge helper receives. Every
other command, flag and exit status is unchanged.

- **`ce-persona-findings <artifact> --anchors -C <dir>`** prints one JSON document saying, for
  each finding, where its quote is in the reviewed tree, with the two keys a poster marks its
  comments with. See [`--anchors`](#--anchors-where-each-findings-quote-is). A run now also
  clears a stale `<persona>-<provider>-anchors.json`.
- **`--verify-quotes` parses range citations.** `f.py:30-60`, with a hyphen, en dash or em dash,
  now reads as a citation of line 30. The `-60` used to stay in the compared text, so a true
  quote cited that way was dropped. More true quotes now survive, and the merge helper sees
  them.
  - Where a quote citing several locations is cut into per-location segments shifts with this,
    and so does the text of some drop reasons.
  - Code shaped like `t:30-60` is now read as a range citation too.
- **US spelling in two messages.** The `--help` text and the error for an unknown
  `CE_PERSONA_*` name now say `unrecognized`.

## Changes in 0.3.3

One addition; exit `6` now covers a codex run whose only tool calls did not act on the
repository, and exit `3` a codex run that paired such calls with an unrecognized tool kind.
Every other command, flag and exit status is unchanged.

- **`ce-persona-findings <artifact> --show all`** renders tier 2 for every finding, every
  severity, in `#` order, inside one fence, entries separated by a `----` line. It follows the
  precedence of `--show N`. On a verdicts artifact it is the default listing.
- **Exit `6` counts local tool calls.** Through codex those are `command_execution`,
  `local_shell_call`, `file_change` and `patch_apply` items. A run whose only tool calls were
  web searches, MCP calls, function calls or custom tool calls used to pass the zero-call
  refusal; it now exits `6`, from the review and validate commands alike. `run_stats` in provenance gains `local_tool_calls`, and `ce-persona-findings` refuses
  an artifact whose sidecar records zero of them. A sidecar written by an earlier version is
  refused only when `tool_calls` is zero, as before.
- **Codex vocabulary drift is detected beside calls that are not local.** A stream holding a
  tool kind this build does not recognize and no local call exits `3` as drift, even when it
  also holds web searches or MCP calls. It previously counted those and exited `0`.

## Changes in 0.3.2

Defaults only; every command, flag and exit status is unchanged.

- **Codex default model moved.** `ce-codex-persona` now defaults to `gpt-6-sol` (was
  `gpt-6-astra`). `-m` overrides it, as before.

## Changes in 0.3.1

Defaults only; every command, flag and exit status is unchanged.

- **Default models moved.** `ce-grok-persona` now defaults to `grok-4.7` (was `grok-4.6`) and
  `ce-codex-persona` to `gpt-6-astra` (was `gpt-5.6-sol`). `-m` overrides either, as before.

## Changes in 0.3.0

Additions only; every existing command, flag and exit status is unchanged.

- **`ce-grok-validate` / `ce-codex-validate`** — the validator mode above: the plugin's Stage 5b
  batch, run out of process with the watchdogs, provenance and refusals the review commands have.
  See *Validating a findings batch*.
- **`ce-persona-findings` reads a verdicts artifact**, listing one row per verdict in `#` order
  with `--show`, `--json` and the vacuous-run refusal applying to it as they do to a review's.
  `--return` on one is a usage error, and an artifact carrying both shapes is exit `1`.
- **`ce-persona-findings --return`** projects a findings artifact into the compact **return**
  object compound-engineering's merge helper reads, so a merge can be run from what the lenses
  wrote to disk. See *`--return`: an artifact as a merge input*.
- **`ce-persona-findings --return --verify-quotes -C <dir>`** checks each `first_evidence`
  against the file and line it cites in that working tree and drops the ones the tree does not
  corroborate, never rewriting one.

## Breaking changes in 0.2.0

The implementation moved from two shell scripts to one Python package. The commands and their flags
are unchanged; the exit statuses are not:

- A **missing persona argument** now exits `2` with usage on stderr. It previously printed usage to
  **stdout and exited 0** — success, for an invocation that reviewed nothing.
- A **runner failure** is now `4` rather than the runner's own status. Propagating it verbatim
  collided with the codes reserved for usage (`2`) and over-budget (`78`), so a caller could not
  tell them apart.
- New refusals: `3` for a missing runner/git/plugin, `5` for a timeout, `2` for a persona name that
  is not a bare brief name or a `-b` that does not resolve.
- `CE_PERSONA_MAX_PROMPT_TOKENS` now counts the diff as well as the prompt, so an existing tuned
  value bounds more than it used to.
- `ce-persona-findings` gained `-h`/`--help` and now uses `2` for usage errors rather than `1` for
  everything, so a caller can tell a bad invocation from an unusable artifact.
- Both review commands now enforce timeouts. A wedged provider previously hung forever and took the
  calling agent with it; `CE_PERSONA_IDLE_SECS` and `CE_PERSONA_HARD_SECS` bound that.
- **`CE_PERSONA_*` values are validated strictly.** A timeout of `0`, a non-finite or negative
  number, and an unrecognized `CE_PERSONA_*` name all exit `2`. Previously `0` disabled the watchdog
  and a misspelled name was ignored in silence.
- **Concurrent runs of the same persona and provider in one run directory are refused** with `2`
  rather than overwriting each other. See Concurrency above.

## Development

```console
nix flake check   # package build, lint, strict types, unit + process suites, mutation harness
nix develop
```

Five checks, and the fifth is the unusual one. `tests/test_mutations.py` reverts each fix in a
scratch copy of the **built** package and requires a test to die — because the recurring defect here
has never been a wrong guard, it has been a guard that *cannot* fail. Seven shipped green in a
single review cycle. A new guard without an entry in that table is not finished.

`AGENTS.md` records the house-template divergences (no pydantic, no rich, no structlog) with the
measurements behind each, and the structural invariants worth not breaking.

The unit suite exercises the packaged library and asserts which copy it imported — an earlier
version put its own source root ahead of `PYTHONPATH` and passed with every packaged module replaced
by `raise RuntimeError`. The process suite drives the **installed console scripts** against stub
runners, with a PATH that deliberately cannot reach a real `grok` or `codex`, and every case runs
for **both** providers: when the two runners were separate scripts, guards were repeatedly added to
both and tested against only one.

Type checking is `basedpyright` in **strict** mode, with no baseline and no rule suppressions. The
JSON boundary is typed (`validate.JSONValue`) rather than silenced. The `types` check ends with a
negative control that injects `return x + None` into the library and fails if the checker accepts
it — because this check once passed while analyzing exactly one file and none of the code that
ships.

`python3 -m persona_review.flags` probes the installed `grok` for CLI drift. It needs a real
authenticated binary, so no flake check runs it.

## License

Apache-2.0. Persona briefs and the findings schema are read at run time from the
compound-engineering plugin, which is MIT — see `NOTICE`.
