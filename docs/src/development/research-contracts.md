# Research data contracts

PhluxAI provides minimal, allocation-independent identity and provenance metadata in `ResearchContracts` for future data ingestion, retrieval, and tool execution. These types do not yet implement a document parser, source store, authorization system, or operation journal.

`SourceRef(sha256, format, size_bytes)` records the SHA-256 digest of source bytes, a format label such as `:pdf` or `:hdf5`, and the byte length. The digest is validated syntactically, but constructing a `SourceRef` does not hash a file or verify its contents. Importers must independently compute and verify the hash and apply their own access policy.

`EvidenceRef(source, scheme, locator; extraction_sha256=nothing)` identifies a location or selection within a source and optionally records the digest of derived material. A PDF page locator and an HDF5 hyperslab locator are metadata strings under separate schemes; they are not trusted filesystem paths, executable selectors, or read permissions. The eventual parser adapters are responsible for validating their syntax and bounds.

`OperationRecord(id, action; inputs, outputs, parameters_json, status)` records an operation identifier, a tool/action name, source references, a JSON object of parameters, and an execution status. The constructor copies reference vectors to prevent later mutation through the original input vectors. The metadata record is not a durable journal or proof of execution. Its status and outputs must be recorded by a trusted controller, not by a language model's unsupported assertion.

The current `ArtifactRef`, `ToolSpec`, `ToolExecution`, and `ModelCapabilities` remain unchanged. Higher-level source import, exact token counting, and model training require additional implementation.

## Tool contracts and execution policy

`ToolHandlerDeclaration` associates a stable handler ID, model-visible `ToolSpec`, risk class, and `ExecutionPolicy`. Define `tool_declaration(handler)` and `tool_command(handler, arguments, sources)` on a host-installed subtype of `AbstractToolHandler`, then register it with `register!` on a `ToolRegistry(; policy_only = true)`. The command builder runs as trusted Julia code and receives the validated `/inputs/<id>` paths. Return an argument vector with an absolute executable path in the sandbox runtime.

Install the registry using `set_tool_context!` with a `ToolContext` that supplies `worker_cgroup` and, for source-reading tools, `authorize_sources(call, declaration)`. The latter returns a dictionary of permitted source IDs and `SourceGrant(source_ref, path)` values for that invocation. Include `source_ids` in the tool's argument schema to allow a model to request a subset of those IDs. The executor verifies the granted files against their declared byte lengths and SHA-256 digests, then stages only those requested inputs as read-only mounts.

A policy worker starts with a private scratch workspace, the Linux system runtime, and its granted inputs. The executor requires Bubblewrap, disables networking by default, and clamps timeout and output to the stricter of the handler and session settings. An explicitly network-enabled policy also needs a session that permits networking. `:trusted_local` declarations can be serialized, but policy dispatch rejects them because they cannot enforce source isolation.

On Linux, `worker_cgroup` must be a delegated cgroup v2 directory with the memory controller enabled for child cgroups. Run PhluxAI under that delegation so the launcher can move itself into a fresh worker cgroup before executing Bubblewrap. `max_memory_bytes` is rounded down to a kernel page boundary and applies to the whole worker tree. Swap is disabled for the worker cgroup, and the executor reports memory failures from the kernel counters before cleaning up the group.

The legacy workspace registry still serves `jupy` and `jupip` sessions. Policy-only registries are required for workers that read individually granted sources. The sandbox CI job exercises the declared handler path with Bubblewrap, verifies read-only mounts and network namespace isolation, and checks that two individually admissible allocations trigger the shared memory budget when run together.

`ExecutionPolicy` and `ToolHandlerDeclaration` retain their strict versioned JSON encoding. Deserializing metadata does not authorize execution or source access.
