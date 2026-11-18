# Implementation decisions

## DEC-2026-001: Provision self-contained parser wheels

**Date:** 2026-11-17
**Status:** Accepted
**Scope:** Parser runtime dependencies, CI provisioning, and source-analysis documentation.

### Context

The native parser tests passed, but the isolated HDF5 handlers failed. An import probe in the worker filesystem reproduced NumPy's `ImportError: libblas.so.3` on the Ubuntu 24.04 runner. The distribution library link depends on `/etc/alternatives`, outside the permitted runtime mounts.

### Decision

Provision the versions in `deps/analysis-requirements.txt` from binary wheels in the disposable system runtime. Install all dependencies into the visible `/usr` tree, ignoring host installations, and retain an isolated import probe before sandbox integration tests.

### Alternatives

Mounting `/etc/alternatives` would enlarge the worker filesystem. Binding selected host BLAS links would add distribution-specific library discovery to every worker launch. Self-contained wheels avoid both changes and preserve the existing authorization boundary.

### Evidence and reasoning

The isolated import probe in GitHub Actions run 44221771744 failed with the missing BLAS library. The local parser environment uses the selected NumPy, h5py, and pdfplumber versions and passes all ten parser scenarios.

### Consequences

Runtime images require compatible binary wheels. Provisioning fails if wheels are unavailable; CI does not fall back to a source build or broaden mounts. Direct parser versions are pinned; transitive packages remain resolved by pip. General-purpose host Python installations should not be overwritten.

### Validation

The parser scenarios pass locally. Hosted isolated-import and end-to-end worker validation are required on the resulting commit before closure.


## DEC-2026-002: Durable journal as session authority

**Date:** 2026-11-18
**Status:** Accepted
**Scope:** Session journals, save/load, model and tool events, streaming, summaries, legacy migration, and local validation.

### Context

The hot store is a compressed context view, while the previous cold log lacks integrity and durability guarantees. Failed requests and reasoning omitted by the old retention policy cannot be reconstructed from those files.

### Decision

Use a versioned JSON-lines journal with immutable sequence-derived UUIDs, SHA-256 linkage, exact JSON payload strings, bounded string event kinds without symbol interning, one POSIX OS writer lease, and an in-process lock. Acknowledge appends after file fsync; sync new directory entries. Record original messages and received stream chunks independently of context-state events. Preserve original events through compression. Keep hot/cold files as compatibility views and preserve legacy originals before migration. Rehydrate data and settings, while requiring the host to rebind executable contexts and source authorization.

### Alternatives

Rewriting full history on every save would scale poorly. A transactional database would add a dependency and migration burden. PID-file stale-lock heuristics are unnecessary where kernel leases release automatically after process death. A hash chain without an independent checkpoint cannot detect removal of a valid suffix or authenticate hostile replacement.

### Consequences

Durable writing currently requires POSIX flock/fsync on a local filesystem. Received reasoning is retained in the journal independently of the hot-view retention flag. Record sizes are bounded, so oversized inputs fail explicitly. Opening verifies the journal once; appending and streaming iteration do not materialize complete history. Active-view replay is proportional to active context and unfinished operations. Unfinished tool calls remain uncertain and are never rerun automatically.

### Validation

Storage, corruption, task concurrency, process crashes around write/sync boundaries, writer exclusion, long-history iteration, migration, mock HTTP, tool approval/result replay, and existing compression checks are exercised locally. Hosted sandbox validation remains a separate gate requiring a namespace-capable host with delegated cgroup v2; it cannot be claimed from this workspace.
