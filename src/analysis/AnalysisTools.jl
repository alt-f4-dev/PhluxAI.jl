module AnalysisTools

using JSON3
using SHA: sha256
using UUIDs: UUID, uuid4, uuid5

import ..JupyTools
import ..PolicyWorkers: source_ids
import ..ProtocolTypes: ToolSpec, ToolCall, ToolExecution
import ..ResearchContracts: SourceRef, EvidenceRef, OperationRecord, contract_payload, parse_contract
import ..Sandboxing: SandboxConfig, ProcessResult
import ..SourceStores: SourceStore
import ..ToolContracts: AbstractToolHandler, ExecutionPolicy, ToolHandlerDeclaration, TOOL_READ, tool_declaration, tool_command

const OPERATIONS = (:pdf_metadata, :pdf_text, :pdf_tables, :pdf_ocr, :hdf5_hierarchy, :hdf5_metadata, :hdf5_read, :hdf5_statistics)
include_dependency(joinpath(@__DIR__, "worker.py"))
const WORKER_CODE = read(joinpath(@__DIR__, "worker.py"), String)

"""Host-configured bounds. Models can select pages and hyperslabs within these limits."""
Base.@kwdef struct AnalysisLimits
    max_pages::Int = 16
    max_document_pages::Int = 10000
    max_cells::Int = 10000
    max_dpi::Int = 300
    max_pixels::Int = 16_000_000
    max_nodes::Int = 1000
    max_attribute_bytes::Int = 4096
    max_elements::Int = 100000
    max_decoded_bytes::Int = 16 * 1024 * 1024
    max_output_bytes::Int = 1_000_000
end

function _limit_payload(limits::AnalysisLimits)
    all(name -> getfield(limits, name) > 0, fieldnames(AnalysisLimits)) || throw(ArgumentError("analysis limits must be positive"))
    limits.max_dpi >= 72 || throw(ArgumentError("max_dpi must be at least 72"))
    return NamedTuple{fieldnames(AnalysisLimits)}(Tuple(getfield(limits, name) for name in fieldnames(AnalysisLimits)))
end

struct AnalysisHandler <: AbstractToolHandler
    operation::Symbol
    store::SourceStore
    limits::AnalysisLimits
    policy::ExecutionPolicy
    python::String
end

function tool_declaration(handler::AnalysisHandler)::ToolHandlerDeclaration
    properties = Dict{String,Any}("source_ids" => (type = "array", items = (type = "string",), minItems = 1, maxItems = 1))
    if handler.operation in (:pdf_text, :pdf_tables, :pdf_ocr)
        properties["pages"] = (type = "array", items = (type = "integer", minimum = 1), maxItems = handler.limits.max_pages)
    end
    if handler.operation == :pdf_ocr
        properties["dpi"] = (type = "integer", minimum = 72, maximum = handler.limits.max_dpi)
    end
    if startswith(String(handler.operation), "hdf5_")
        properties["dataset"] = (type = "string", description = "Absolute HDF5 path. Only hard links are followed.")
    end
    if handler.operation in (:hdf5_read, :hdf5_statistics)
        properties["selection"] = (type = "array", description = "One zero-based half-open {start,stop,step} slice per axis.",
                                    items = (type = "object", properties = (start = (type = "integer", minimum = 0),
                                             stop = (type = "integer", minimum = 0), step = (type = "integer", minimum = 1)),
                                             additionalProperties = false))
    end
    schema = JSON3.write((type = "object", properties, required = ["source_ids"], additionalProperties = false))
    spec = ToolSpec(String(handler.operation), "Bounded read-only source analysis with evidence and provenance.", schema)
    return ToolHandlerDeclaration("analysis.$(handler.operation).v1", spec, TOOL_READ; policy = handler.policy)
end

function _source(handler::AnalysisHandler, arguments)::SourceRef
    ids = source_ids(arguments)
    length(ids) == 1 || throw(ArgumentError("analysis requires exactly one source identifier"))
    format = startswith(String(handler.operation), "pdf_") ? :pdf : :hdf5
    id = only(ids)
    occursin(r"\A[0-9a-f]{64}\z", id) || throw(ArgumentError("analysis source ID must be its SHA-256 digest"))
    path = joinpath(handler.store.root, "$id.$format")
    isfile(path) || throw(ArgumentError("source has not been imported with the requested format"))
    return SourceRef(id, format, filesize(path))
end

function tool_command(handler::AnalysisHandler, arguments, sources)::Vector{String}
    source = _source(handler, arguments)
    limits = _limit_payload(handler.limits)
    request = JSON3.write((operation = handler.operation, arguments, limits))
    return [handler.python, "-I", "-c", WORKER_CODE, request, sources[source.sha256]]
end

function JupyTools._validate_sources(handler::AnalysisHandler, arguments, grants)::Nothing
    source = _source(handler, arguments)
    grants[source.sha256].source == source || throw(ArgumentError("granted source identity differs from the requested source"))
    return nothing
end

"""Construct a policy-only registry of PDF and HDF5 operations. Source access is authorized separately for each call."""
function analysis_registry(store::SourceStore; limits::AnalysisLimits = AnalysisLimits(), policy::ExecutionPolicy = ExecutionPolicy(),
                           python::AbstractString = "/usr/bin/python3")::JupyTools.ToolRegistry
    _limit_payload(limits)
    isabspath(python) || throw(ArgumentError("analysis python must be an absolute sandbox runtime path"))
    policy.network && throw(ArgumentError("source analysis requires network=false"))
    policy.isolation == :bubblewrap || throw(ArgumentError("source analysis requires Bubblewrap"))
    registry = JupyTools.ToolRegistry(; policy_only = true)
    for operation in OPERATIONS
        JupyTools.register!(registry, AnalysisHandler(operation, store, limits, policy, String(python)))
    end
    return registry
end

"""
    analysis_context(sandbox, artifacts, store; worker_cgroup, authorize, kwargs...)

Create the tool context used by `set_tool_context!` and `analyze_source`. `authorize(call, declaration)` must return a dictionary of source IDs to `SourceGrant` values. Source import alone does not authorize model access.
"""
function analysis_context(sandbox::SandboxConfig, artifacts, store::SourceStore; worker_cgroup::Union{AbstractString,Nothing} = nothing,
                          authorize::Union{Function,Nothing} = nothing, kwargs...)::JupyTools.ToolContext
    registry = analysis_registry(store; kwargs...)
    return JupyTools.ToolContext(sandbox, artifacts; registry, worker_cgroup, authorize_sources = authorize)
end

"""Structured analysis result, including evidence, operation provenance, and execution failure details."""
struct AnalysisResult
    execution::ToolExecution
    data::Any
    evidence::Vector{EvidenceRef}
    operation::OperationRecord
    error_code::Union{Symbol,Nothing}
    message::String
    method::Union{String,Nothing}
    versions::Dict{String,String}
end

function _envelope(call::ToolCall, payload, source::SourceRef)
    encoded_data = String(JSON3.write(get(payload, :data, nothing)))
    digest = bytes2hex(sha256(encoded_data))
    evidence = EvidenceRef[]
    if get(payload, :ok, false)
        for item in payload.evidence
            push!(evidence, EvidenceRef(source, Symbol(item.scheme), JSON3.write(item.locator); extraction_sha256 = digest))
        end
    end
    success = get(payload, :ok, false) === true
    record = OperationRecord(uuid5(UUID(UInt128(0)), call.id), call.name; inputs = [source], parameters_json = call.arguments_json,
                              outputs = success ? [SourceRef(digest, :json, ncodeunits(encoded_data))] : SourceRef[],
                              status = success ? :succeeded : :failed)
    return JSON3.write((analysis_schema_version = 1, ok = success, data = get(payload, :data, nothing),
                       error = get(payload, :error, nothing), message = get(payload, :message, ""),
                       method = get(payload, :method, nothing), versions = get(payload, :versions, nothing),
                       evidence = contract_payload.(evidence), operation = contract_payload(record)))
end

function JupyTools._worker_result(handler::AnalysisHandler, call::ToolCall, result::ProcessResult,
                                  memory_exceeded::Bool, oom_killed::Bool)::ToolExecution
    arguments = JSON3.read(call.arguments_json)
    source = _source(handler, arguments)
    code = memory_exceeded ? :memory_limit : result.timed_out ? :timeout : result.output_limited ? :output_limit : :worker_failed
    payload = (ok = false, error = String(code), message = "source analysis worker failed")
    if result.success && !memory_exceeded
        payload = try
            parsed = JSON3.read(result.stdout)
            parsed isa JSON3.Object && get(parsed, :ok, nothing) isa Bool || throw(ArgumentError("invalid worker envelope"))
            if parsed.ok
                all(key -> haskey(parsed, key), (:data, :evidence, :method, :versions)) || throw(ArgumentError("missing worker fields"))
            else
                get(parsed, :error, nothing) isa AbstractString || throw(ArgumentError("missing worker error"))
            end
            parsed
        catch
            (ok = false, error = "invalid_worker_response", message = "source analysis worker returned invalid JSON")
        end
    end
    output = _envelope(call, payload, source)
    return ToolExecution(call.id, call.name, get(payload, :ok, false) === true, output; exit_code = result.exit_code)
end

"""
    analyze_source(context, source, operation; mode=TOOLS_ASK, approval_callback=nothing, kwargs...) -> AnalysisResult

Run one bounded source operation through normal tool dispatch. PDF page numbers are one-based. HDF5 selections use zero-based half-open slices. Runtime isolation is mandatory.
"""
function analyze_source(context::JupyTools.ToolContext, source::SourceRef, operation::Symbol; mode::JupyTools.ToolMode = JupyTools.TOOLS_ASK,
                         approval_callback::Union{Function,Nothing} = nothing, kwargs...)::AnalysisResult
    operation in OPERATIONS || throw(ArgumentError("unsupported source analysis operation"))
    expected = startswith(String(operation), "pdf_") ? :pdf : :hdf5
    source.format == expected || throw(ArgumentError("source format differs from the requested operation"))
    haskey(kwargs, :source_ids) && throw(ArgumentError("source_ids is derived from the supplied SourceRef"))
    parameters = String(JSON3.write((; source_ids = [source.sha256], kwargs...)))
    call = ToolCall(string(uuid4()), String(operation), parameters)
    execution = JupyTools.execute_tool_call(context, call; mode, approval_callback)
    payload = try
        JSON3.read(execution.output)
    catch
        nothing
    end
    if payload isa JSON3.Object && get(payload, :analysis_schema_version, nothing) == 1
        evidence = EvidenceRef[parse_contract(EvidenceRef, JSON3.write(item)) for item in payload.evidence]
        record = parse_contract(OperationRecord, JSON3.write(payload.operation))
        code = payload.error === nothing ? nothing : Symbol(payload.error)
        method = payload.method === nothing ? nothing : String(payload.method)
        versions = payload.versions === nothing ? Dict{String,String}() : Dict(String(k) => String(v) for (k, v) in pairs(payload.versions))
        return AnalysisResult(execution, payload.data, evidence, record, code, String(payload.message), method, versions)
    end
    record = OperationRecord(UUID(call.id), call.name; inputs = [source], parameters_json = parameters, status = :failed)
    return AnalysisResult(execution, nothing, EvidenceRef[], record, :execution_rejected, execution.output, nothing, Dict{String,String}())
end

export AnalysisLimits, AnalysisResult, analysis_registry, analysis_context, analyze_source

end # module AnalysisTools
