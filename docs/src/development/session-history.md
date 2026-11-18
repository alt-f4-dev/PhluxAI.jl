# Durable session history

Every `ChatSession` and `ModelSession` has an append-only `journal.jsonl`. The journal is authoritative; `hot.json` is a replaceable active-context cache, and `cold.jsonl` remains a compatibility log. Compression changes the active view and appends a derived summary; it does not delete original events.

## Durability and scope

Writing requires a POSIX local filesystem supporting `flock`, file `fsync`, and directory `fsync`. Appends are acknowledged only after the complete newline-terminated record has been flushed and synced. New directory entries are synced before the writer is returned. One OS lease excludes other writer handles, including handles in the same process; a `ReentrantLock` serializes tasks sharing a handle. Process termination releases the lease without a stale-lock deletion heuristic. Close every writer when finished.

A failed write or sync poisons its writer handle. Close it and verify/recover before continuing. An unacknowledged complete event can survive a crash; callers must inspect event IDs instead of assuming that an interrupted append was absent. The tests exercise process termination, not power loss or storage-controller guarantees.

Events preserve received reasoning fields and complete streaming chunks even when `persist_reasoning=false`. That option controls the active/hot view and compatibility log; it does not redact the authoritative journal. Keep the session directory private and apply the same access controls to its backups and recovery archives. Data never returned by a model service cannot be recorded. A record that exceeds the configured byte limit is rejected explicitly rather than truncated.

Sessions without `session_dir` use a temporary journal. Supply a persistent session directory when history must outlive temporary-file cleanup.

## Journal API

```julia
using PhluxAI
using JSON3

writer = open_journal("history/journal.jsonl")
try
    event = append_event!(writer, :measurement_note, (text = "α\nβ", units = "meV"))
    println(event.id)
    head = journal_head(writer)
    # Store this JSON independently when suffix-deletion detection is required.
    trusted_checkpoint = checkpoint_json(head)
    restored_checkpoint = parse_checkpoint(trusted_checkpoint)
    @assert verify_journal(writer; checkpoint = restored_checkpoint) == head
    foreach_event(writer) do item
        println(item.sequence, " ", item.kind, " ", JSON3.read(item.payload_json))
    end
finally
    close(writer)
end
```

`JournalEvent` contains the session UUID, sequence number, immutable event UUID, kind, UTC timestamp, exact `payload_json`, previous digest, and current SHA-256 digest. Event kinds are strings of at most 256 UTF-8 bytes; the core verifier uses string-key JSON parsing and does not intern user-defined kind or payload-key names as symbols. Event UUIDs are derived from the session UUID and sequence, so repeated or reordered IDs cannot pass sequence validation. Supply `payload_json` explicitly to preserve an existing JSON string's whitespace and number spelling. Other payloads are recorded in their JSON representation.

The versioned physical record stores the exact hashed body string alongside its digest. Verification checks the schema, sequence, identity, payload syntax, and hash chain. This detects accidental mutation, interior deletion, duplication, and reordering. A retained `JournalHead` also detects deletion or replacement of its committed prefix. Hashes alone cannot authenticate a maliciously rewritten journal or detect deletion of an otherwise valid suffix; those checks require an independently trusted checkpoint. Checkpoints stored beside the journal share its compromise and deletion risks.

`foreach_event(f, path)` visits a verified prefix and raises at the first invalid record; prior callbacks may already have run. Use `verify_journal` before acting on a whole history. `foreach_event(f, writer)` holds the writer's lock for a stable view. Reader memory is bounded by a record, rather than the complete history, unless the callback retains events. Appending does not reread or rewrite old records. Opening verifies the existing journal once. The default maximum serialized record is 16 MiB, including envelope escaping; the standalone journal API accepts `max_record_bytes`.

## Session inspection and resumption

```julia
using PhluxAI

session = ModelSession("fixture"; session_dir = "research-session", start_model = false, discover_capabilities = false)
set_temperature!(session, 0.25)
save(session)
anchor = journal_head(session_journal(session))
close(session)

resumed = load_model_session("fixture", "research-session"; start_model = false, discover_capabilities = false, checkpoint = anchor)
try
    report = verify_session(resumed; checkpoint = anchor)
    println(report.artifact_issues)
    println(pending_operations(resumed))
    foreach_event(session_journal(resumed)) do event
        println(event.sequence, " ", event.kind)
    end
finally
    close(resumed)
end
```

Session events cover source messages, received model responses and metrics, raw stream chunks, runtime settings, model selection changes, tool invocation starts, approval decisions, tool results, errors, and summary derivation. The replayed active view uses committed context-state events plus subsequent messages. Repeated saves and identical resumes do not duplicate state events. `last_response` is reconstructed, including returned tool executions.

Interactive defaults and queued artifact references are restored. Explicit load-time settings override saved defaults. The model name and endpoint supplied to `load_model_session` select the resumed model; changes are recorded. A saved tool mode does not restore host executables, custom handlers, source grants, or cgroup delegations. Reinstall a custom `ToolContext` with `set_tool_context!` before using those tools. Custom image backends likewise require explicit host configuration. Persisted names are provenance, not executable authority.

`verify_session` checks the journal and reports missing, changed, or inaccessible artifact bytes referenced by message events. It does not recreate missing files or authorize sources. Tool evidence and source locators remain in the original tool payloads; source availability is checked separately by the analysis API.

`pending_operations` returns invocations whose start has no committed finish, with status `:uncertain`. Loading never runs them. Tool call IDs that have already been invoked are rejected by the interactive executor, including uncertain and completed calls. Inspect external effects before explicitly creating a new invocation. A tool result is journaled before its finish marker; interruption between those commits remains uncertain even if result bytes survived.

## Summary provenance

Summary events record the preceding journal-head digest and sequence, the previous summary, the exact selected input messages, generation budgets/model, received summary text and reasoning, and completion status. Rejected and failed compression attempts are also recorded. Active-context state is committed separately. A crash after receiving a candidate but before committing the new state retains the old view and the received candidate. Preserving the source history does not establish semantic completeness of an LLM-generated summary.

## Recovery

Verification rejects incomplete final records by default. After stopping other writers, explicitly recover a torn tail:

```julia
using PhluxAI

writer = open_journal("history/journal.jsonl"; recover = true)
try
    println(verify_journal(writer))
finally
    close(writer)
end
```

Recovery durably archives the incomplete suffix in a uniquely named `.incomplete-*` file, verifies the retained prefix and optional checkpoint, and truncates only that incomplete suffix. Complete malformed records, unsupported schemas, broken chains, and oversized records require investigation; `recover=true` does not silently repair them. For a session, pass `recover=true` and an optional checkpoint to `load_model_session` or `OllamaClient.load_session`.

## Legacy sessions

Load an existing `hot.json`/`cold.jsonl` session through the normal load API. Migration saves byte-identical originals in `legacy-originals`, records source digests, imports each surviving cold line once, and commits a current context view. Interrupted imports resume only if the legacy source digests are unchanged. Unsupported versions, malformed records, and incomplete legacy tails fail explicitly. Existing legacy files are never treated as a trustworthy historical reconstruction of events already lost before migration.

After migration, `hot.json` becomes the new derived cache, while the backups retain original bytes. The journal remains authoritative on later loads even if the hot file is stale or absent. No migration can recreate reasoning previously omitted by the old retention policy or messages absent from both legacy stores.

## Validation

The CPU profile runs storage corruption tests, multithreaded appends, subprocess writer exclusion and crash-boundary tests, 4,000-event streaming verification, legacy migration, mock HTTP streaming, interactive tool approvals/results, and compression regressions on Julia 1.12 and 1.13. CI sets four Julia threads to exercise simultaneous task appends. Run `julia --project=. --startup-file=no test/runtests.jl`. The existing parser/native and actual Bubblewrap/cgroup profiles remain separate integration gates; a host without namespaces or cgroup delegation cannot establish their results locally.
