module ToolContracts

using JSON3

import ..ProtocolTypes: ToolSpec
import ..JupyTools: ToolRisk, TOOL_READ, TOOL_WRITE, TOOL_EXECUTE, TOOL_NETWORK
import ..ResearchContracts
import ..ResearchContracts: contract_payload, from_contract

"""Metadata-only resource and approval requirements. Enforcement belongs to the Stage 3 executor."""
struct ExecutionPolicy
    isolation::Symbol
    approval::Symbol
    network::Bool
    source_access::Symbol
    timeout_seconds::Float64
    max_memory_bytes::Int
    max_output_bytes::Int

    function ExecutionPolicy(; isolation::Symbol = :bubblewrap, approval::Symbol = :required,
                              network::Bool = false, source_access::Symbol = :authorized_only,
                              timeout_seconds::Real = 120.0, max_memory_bytes::Integer = 2_147_483_648,
                              max_output_bytes::Integer = 2_000_000, allow_unsafe_local::Bool = false)
        isolation in (:bubblewrap, :trusted_local) || throw(ArgumentError("unsupported isolation backend: $isolation"))
        isolation == :trusted_local && !allow_unsafe_local &&
            throw(ArgumentError("trusted_local requires allow_unsafe_local=true"))
        approval in (:required, :automatic) || throw(ArgumentError("unsupported approval policy: $approval"))
        source_access == :authorized_only || throw(ArgumentError("source_access must be :authorized_only"))
        network && approval != :required && throw(ArgumentError("network access requires explicit approval"))
        isolation == :trusted_local && approval != :required &&
            throw(ArgumentError("trusted local execution requires explicit approval"))
        seconds = Float64(timeout_seconds)
        isfinite(seconds) && seconds > 0 || throw(ArgumentError("timeout_seconds must be positive and finite"))
        memory = Int(max_memory_bytes)
        output = Int(max_output_bytes)
        memory > 0 || throw(ArgumentError("max_memory_bytes must be positive"))
        output > 0 || throw(ArgumentError("max_output_bytes must be positive"))
        return new(isolation, approval, network, source_access, seconds, memory, output)
    end
end

"""Descriptive contract for a future registered handler; no executable callable is stored or invoked."""
struct ToolHandlerDeclaration
    handler_id::String
    spec::ToolSpec
    risk::ToolRisk
    policy::ExecutionPolicy

    function ToolHandlerDeclaration(handler_id::AbstractString, spec::ToolSpec, risk::ToolRisk;
                                    policy::ExecutionPolicy = ExecutionPolicy())
        identifier = String(strip(handler_id))
        occursin(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$", identifier) ||
            throw(ArgumentError("handler_id must be a valid stable identifier"))
        schema = try
            JSON3.read(spec.parameters_json)
        catch error
            throw(ArgumentError("tool schema is not valid JSON: $(sprint(showerror, error))"))
        end
        schema isa JSON3.Object || throw(ArgumentError("tool schema must be a JSON object"))
        get(schema, :type, nothing) == "object" ||
            throw(ArgumentError("tool parameter schema must declare type=object"))
        risk == TOOL_NETWORK && (!policy.network || policy.approval != :required) &&
            throw(ArgumentError("network-risk tools require network=true and explicit approval"))
        risk != TOOL_READ && policy.approval != :required &&
            throw(ArgumentError("write/execute/network tools require explicit approval"))
        return new(identifier, spec, risk, policy)
    end
end

"""Third-party implementers may provide a declaration; Stage 3 will define execution dispatch."""
abstract type AbstractToolHandler end

function tool_declaration(handler::AbstractToolHandler)::ToolHandlerDeclaration
    throw(MethodError(tool_declaration, (handler,)))
end

contract_payload(x::ExecutionPolicy) = (
    schema_version = ResearchContracts.CONTRACT_SCHEMA_VERSION, kind = "ExecutionPolicy",
    isolation = String(x.isolation), approval = String(x.approval), network = x.network,
    source_access = String(x.source_access), timeout_seconds = x.timeout_seconds,
    max_memory_bytes = x.max_memory_bytes, max_output_bytes = x.max_output_bytes,
)

contract_payload(x::ToolHandlerDeclaration) = (
    schema_version = ResearchContracts.CONTRACT_SCHEMA_VERSION, kind = "ToolHandlerDeclaration",
    handler_id = x.handler_id,
    spec = (name = x.spec.name, description = x.spec.description, parameters_json = x.spec.parameters_json),
    risk = String(Symbol(x.risk)), policy = contract_payload(x.policy),
)

function from_contract(::Type{ExecutionPolicy}, object)::ExecutionPolicy
    ResearchContracts._envelope(object, "ExecutionPolicy",
                                ("isolation", "approval", "network", "source_access", "timeout_seconds",
                                 "max_memory_bytes", "max_output_bytes"))
    network = ResearchContracts._field(object, "network")
    network isa Bool || throw(ArgumentError("network must be a boolean"))
    seconds = ResearchContracts._field(object, "timeout_seconds")
    seconds isa Real && !(seconds isa Bool) || throw(ArgumentError("timeout_seconds must be numeric"))
    isolation = Symbol(ResearchContracts._string(object, "isolation"))
    return ExecutionPolicy(; isolation, approval = Symbol(ResearchContracts._string(object, "approval")),
                           network, source_access = Symbol(ResearchContracts._string(object, "source_access")),
                           timeout_seconds = seconds,
                           max_memory_bytes = ResearchContracts._integer(object, "max_memory_bytes"),
                           max_output_bytes = ResearchContracts._integer(object, "max_output_bytes"),
                           allow_unsafe_local = isolation == :trusted_local)
end

function from_contract(::Type{ToolHandlerDeclaration}, object)::ToolHandlerDeclaration
    ResearchContracts._envelope(object, "ToolHandlerDeclaration", ("handler_id", "spec", "risk", "policy"))
    spec_obj = ResearchContracts._field(object, "spec")
    spec_obj isa AbstractDict || throw(ArgumentError("spec must be a JSON object"))
    Set(String(k) for k in keys(spec_obj)) == Set(("name", "description", "parameters_json")) ||
        throw(ArgumentError("spec contains missing or unexpected fields"))
    spec = ToolSpec(ResearchContracts._string(spec_obj, "name"), ResearchContracts._string(spec_obj, "description"),
                    ResearchContracts._string(spec_obj, "parameters_json"))
    risk_label = ResearchContracts._string(object, "risk")
    risk = if risk_label == "TOOL_READ"
        TOOL_READ
    elseif risk_label == "TOOL_WRITE"
        TOOL_WRITE
    elseif risk_label == "TOOL_EXECUTE"
        TOOL_EXECUTE
    elseif risk_label == "TOOL_NETWORK"
        TOOL_NETWORK
    else
        throw(ArgumentError("unsupported tool risk: $risk_label"))
    end
    policy = from_contract(ExecutionPolicy, ResearchContracts._field(object, "policy"))
    return ToolHandlerDeclaration(ResearchContracts._string(object, "handler_id"), spec, risk; policy)
end

export ExecutionPolicy, ToolHandlerDeclaration, AbstractToolHandler, tool_declaration

end # module ToolContracts
