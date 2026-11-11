# Research data contracts

PhluxAI provides minimal, allocation-independent identity and provenance metadata in `ResearchContracts` for future data ingestion, retrieval, and tool execution. These types do not yet implement a document parser, source store, authorization system, or operation journal.

`SourceRef(sha256, format, size_bytes)` records the SHA-256 digest of source bytes, a format label such as `:pdf` or `:hdf5`, and the byte length. The digest is validated syntactically, but constructing a `SourceRef` does not hash a file or verify its contents. Importers must independently compute and verify the hash and apply their own access policy.

`EvidenceRef(source, scheme, locator; extraction_sha256=nothing)` identifies a location or selection within a source and optionally records the digest of derived material. A PDF page locator and an HDF5 hyperslab locator are metadata strings under separate schemes; they are not trusted filesystem paths, executable selectors, or read permissions. The eventual parser adapters are responsible for validating their syntax and bounds.

`OperationRecord(id, action; inputs, outputs, parameters_json, status)` records an operation identifier, a tool/action name, source references, a JSON object of parameters, and an execution status. The constructor copies reference vectors to prevent later mutation through the original input vectors. The metadata record is not a durable journal or proof of execution. Its status and outputs must be recorded by a trusted controller, not by a language model's unsupported assertion.

The current `ArtifactRef`, `ToolSpec`, `ToolExecution`, and `ModelCapabilities` remain unchanged. Higher-level source import, execution authorization, exact token counting, and model training are future implementation stages, not implied capabilities of these definitions.

## Tool contracts and execution policy

The metadata-only `ToolContracts` module adds `ExecutionPolicy`, `ToolHandlerDeclaration`, and `AbstractToolHandler`. It reuses existing `ToolSpec` and `ToolRisk` rather than creating competing tool registries. A declaration associates a stable handler ID, model-visible tool schema, risk class, and execution policy.

The default policy requires Bubblewrap, explicit approval, network isolation, source access restricted to authorized sources, finite execution time, bounded memory, and bounded output. Metadata validation rejects automatic approval of network, write, and execute risk classes; a declared `:trusted_local` isolation policy requires explicit unsafe-local opt-in and explicit approval. It is **not** sufficient to enforce these restrictions at runtime; Stage 3 must implement independently validated authorization and isolation. `ToolContracts.tool_declaration` is an extension point for future handler implementations. No handler execution function is registered by Stage 2.

`ExecutionPolicy` and `ToolHandlerDeclaration` support the same strict versioned JSON encoding. Deserialization of a trusted-local policy reconstructs its explicit unsafe-local declaration, not a runtime permission grant. These types must not be treated as capabilities or as proof that a sandbox was actually applied.
