# Read-only PDF and HDF5 analysis

PhluxAI provides content-addressed source snapshots, eight registered analysis operations, and structured extraction results. Parsing runs in a Bubblewrap worker under the execution policy, per-call source authorization, shared output budget, deadline, and cgroup v2 memory limit.

## Runtime

The host application uses Julia and the package's existing dependencies. Workers use system Python with `h5py`, NumPy, `pdfplumber`, and PDFMiner. OCR uses Poppler and Tesseract with English language data. On Ubuntu 24.04, install `python3-pip poppler-utils tesseract-ocr` through apt. In a disposable runtime image, install the parser wheels with `/usr/bin/python3 -m pip install --break-system-packages --ignore-installed --upgrade --only-binary=:all: --target /usr/lib/python3/dist-packages -r deps/analysis-requirements.txt`. This provisions NumPy 2.3.5, h5py 3.16.0, and pdfplumber 0.11.9, with the wheels' bundled native libraries. Avoid Ubuntu's `python3-h5py`/NumPy runtime for this mount layout: its BLAS links can depend on `/etc/alternatives`, which is inaccessible to workers. Do not overwrite packages in a general-purpose host installation; build a dedicated runtime image. Test fixtures additionally require `python3-reportlab`, Pillow, and DejaVu Sans.

The default worker interpreter is `/usr/bin/python3`. Dependencies must be accessible through the sandbox's system runtime. The declared-worker mount layout hides `/usr/local` and host home directories. Installing Python packages only into a host virtual environment does not make them available inside the sandbox.

Configure a delegated cgroup v2 directory with the memory controller enabled for children, then run the application under that delegation. Pass the directory through `worker_cgroup`. The sandbox CI job demonstrates this configuration. Missing memory controllers, denied namespaces, and inaccessible delegations cause execution to fail.

## Import and authorize sources

`SourceStore(root; max_source_bytes=100*1024*1024)` stores private source snapshots. `import_source(store, path; format=:auto)` streams the input while enforcing its byte limit and computing SHA-256. The returned `SourceRef` describes the original bytes. Signatures identify PDF and HDF5 files, including HDF5 user blocks. Parsing validates the internal format later.

Snapshot publication is atomic. Reimporting identical content reuses its snapshot. `verify_source(store, source)` checks its digest. The worker verifies granted source bytes during staging, so modifying a stored snapshot invalidates subsequent analysis.

A host authorization callback must grant source access for each invocation. Only requested source IDs present in that callback's dictionary are mounted. The worker receives private scratch and read-only `/inputs/<sha256>` paths.

```julia
using PhluxAI

store = SourceStore("analysis-sources")
source = import_source(store, "measurement.h5")
sandbox = SandboxConfig(mktempdir())
artifacts = PhluxAI.ArtifactStore.Store(mktempdir())
authorize(call, declaration) = Dict(source.sha256 => source_grant(store, source))
context = analysis_context(sandbox, artifacts, store; worker_cgroup = ENV["PHLUXAI_WORKER_CGROUP"], authorize)
approve(call, risk) = risk == TOOL_READ

result = analyze_source(context, source, :hdf5_read; dataset = "/entry/data",
                        selection = [(start = 0, stop = 10, step = 1)], approval_callback = approve)

if result.execution.success
    println(result.data)
    println(result.evidence)
    println(result.operation)
else
    println(result.error_code, " ", result.message)
end
```

The example explicitly grants one source and approves read requests. Applications can restrict the callback by tool name, source identity, or current user authorization. Model-provided paths and evidence locators do not grant access.

For a model session, construct the context using `artifact_store(session)`, then call `set_tool_context!(session, context)` and `set_tool_mode!(session, TOOLS_ASK)`. The session exposes the same schemas and executor as `analyze_source`.

## Operations and coordinates

| Operation | Parameters | Result |
|---|---|---|
| `pdf_metadata` | Source ID | Page count and bounded document metadata |
| `pdf_text` | `pages`, default `[1]` | Text and `has_text` for each page |
| `pdf_tables` | `pages`, default `[1]` | Detected tables, cells, and bounding boxes |
| `pdf_ocr` | `pages` and `dpi`, default 150 | English text, word boxes, and Tesseract confidence values |
| `hdf5_hierarchy` | `dataset`, default `"/"` | Bounded hierarchy and blocked-link markers |
| `hdf5_metadata` | `dataset`, default `"/"` | Shape, dtype, chunks, storage flags, and bounded attributes |
| `hdf5_read` | Dataset path and optional `selection` | Selected shape, dtype, values, and element count |
| `hdf5_statistics` | Dataset path and optional `selection` | Counts, finite minimum/maximum/mean, NaN/infinity counts |

PDF pages are one-based. PDF boxes use points with a top-left origin. OCR boxes use raster pixels with a top-left origin and include the requested DPI.

HDF5 slices are zero-based, half-open, and have positive strides. Supply one `(start, stop, step)` object per axis. Omitting selection requests the full dataset only when it fits the configured bounds. Scalar datasets use zero slices. Empty and null dataspaces return zero-element results. The API reads selected hyperslabs directly and checks decoded chunk sizes before compressed reads.

Reads support fixed-width real numeric types up to 64 bits. Integers retain their JSON integer representation. Nonfinite floats appear as `null` in data arrays. Statistics omit them from finite aggregates and report their counts separately. Variable-length and compound reads are rejected. Oversized or variable-length attributes have omission markers.

Soft and external links are never followed. Reads of virtual datasets and external-storage datasets are rejected. Hierarchy inspection marks blocked links and handles hard-link cycles. Third-party HDF5 filters are blocked; plugin discovery is disabled before HDF5 initializes.

## Limits and extraction behavior

`AnalysisLimits` configures requested pages, total document pages, table cells, raster DPI/pixels, hierarchy nodes, attribute bytes, selected elements, decoded bytes, and parser-result bytes. All limits must be positive. The execution policy independently bounds worker memory, process output, and time.

Scanned pages return empty text until OCR is requested explicitly. Table detection uses PDF geometry and may require inspection for complex layouts. OCR confidence is a parser output, not a calibrated probability. Encrypted PDFs are unsupported. File links and attachments are not opened or executed.

The parser byte limit applies to its serialized response. Returned tool output additionally includes bounded provenance metadata. The policy/session process-output limit can stop the worker before its parser response completes.

## Results and provenance

`AnalysisResult` contains the `ToolExecution`, parsed `data`, `EvidenceRef` values, `OperationRecord`, `error_code`, `message`, extraction `method`, and runtime `versions`. Evidence identifies PDF pages/regions or HDF5 paths/selections. The extraction digest matches the operation's JSON output identity and is computed over the UTF-8 JSON encoding of `result.data`.

Provenance is returned per operation. Source changes, rejected access, invalid parameters, unsupported types/references, malformed files, missing dependencies, and worker failures are reported explicitly. Resource errors use `timeout`, `memory_limit`, or `output_limit`. Pre-execution rejections use `execution_rejected`.

## Validation

Run the CPU suite with `PHLUXAI_TEST_PROFILE=cpu julia --project=. --startup-file=no test/runtests.jl`. It covers snapshots, policy entry points, source identity, provenance, and existing APIs.

Run parser fixtures with `/usr/bin/python3 -m unittest discover -s test/analysis -p test_worker.py -v`. Run native worker checks with `PHLUXAI_TEST_PROFILE=analysis julia --project=. --startup-file=no test/runtests.jl`. For native local tests, `PHLUXAI_ANALYSIS_PYTHON` can name an absolute interpreter in a separate test environment.

Run `PHLUXAI_TEST_PROFILE=sandbox julia --project=. --startup-file=no test/runtests.jl` under valid delegation with `PHLUXAI_WORKER_CGROUP` set. This exercises the public analysis API under isolation, including every operation, source identity, denied references, and recovery. CI first probes parser imports within the isolated runtime, so host-only dependency availability cannot count as worker readiness. Native tests establish extraction behavior; the sandbox profile establishes isolation on its execution host.
