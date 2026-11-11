module ResearchContracts

using JSON3
using UUIDs: UUID

function _validated_digest(value::AbstractString)::String
    digest = lowercase(String(strip(value)))
    length(digest) == 64 && all(isxdigit, digest) || throw(ArgumentError("SHA-256 must contain exactly 64 hexadecimal characters"))
    return digest
end

function _validated_label(value::Symbol, name::String)::Symbol
    isempty(String(value)) && throw(ArgumentError("$name must not be empty"))
    return value
end

"""
    SourceRef(sha256, format, size_bytes)

Content-addressed identity and descriptive metadata for an immutable input. Construction validates metadata, but does not import, read, or verify any external file.
"""
struct SourceRef
    sha256::String
    format::Symbol
    size_bytes::Int

    function SourceRef(sha256::AbstractString, format::Symbol, size_bytes::Integer)
        digest = _validated_digest(sha256)
        kind = _validated_label(format, "format")
        size = Int(size_bytes)
        size >= 0 || throw(ArgumentError("size_bytes must be nonnegative"))
        return new(digest, kind, size)
    end
end

"""
    EvidenceRef(source, scheme, locator; extraction_sha256=nothing)

A source-grounded locator, for example a PDF page or an HDF5 dataset selection. A locator is descriptive metadata, not an executable path or an authorization capability.
"""
struct EvidenceRef
    source::SourceRef
    scheme::Symbol
    locator::String
    extraction_sha256::Union{String,Nothing}

    function EvidenceRef(source::SourceRef, scheme::Symbol, locator::AbstractString; extraction_sha256::Union{AbstractString,Nothing} = nothing)
        selected_scheme = _validated_label(scheme, "scheme")
        selected_locator = String(strip(locator))
        isempty(selected_locator) && throw(ArgumentError("locator must not be empty"))
        digest = isnothing(extraction_sha256) ? nothing : _validated_digest(extraction_sha256)
        return new(source, selected_scheme, selected_locator, digest)
    end
end

"""
    OperationRecord(id, action; inputs, outputs, parameters_json, status)

Immutable metadata describing an operation. A record is not proof that the operation executed, that its outputs are correct, or that access to referenced sources was authorized.
"""
struct OperationRecord
    id::UUID
    action::String
    inputs::Vector{SourceRef}
    outputs::Vector{SourceRef}
    parameters_json::String
    status::Symbol

    function OperationRecord(
        id::UUID,
        action::AbstractString;
        inputs::AbstractVector{SourceRef} = SourceRef[],
        outputs::AbstractVector{SourceRef} = SourceRef[],
        parameters_json::AbstractString = "{}",
        status::Symbol = :planned,
    )
        selected_action = String(strip(action))
        isempty(selected_action) && throw(ArgumentError("action must not be empty"))
        params = String(strip(parameters_json))
        parsed = try
            JSON3.read(params)
        catch error
            throw(ArgumentError("parameters_json must be valid JSON: $(sprint(showerror, error))"))
        end
        parsed isa JSON3.Object || throw(ArgumentError("parameters_json must encode a JSON object"))
        status in (:planned, :approved, :running, :succeeded, :failed, :cancelled, :inconclusive) ||
            throw(ArgumentError("unsupported operation status: $status"))
        return new(id, selected_action, collect(inputs), collect(outputs), params, status)
    end
end


# Model metadata contracts. Neither identity nor capabilities imply runtime availability.
struct ModelRef
    backend::Symbol
    model_id::String
    revision::Union{String,Nothing}
    sha256::Union{String,Nothing}

    function ModelRef(backend::Symbol, model_id::AbstractString; revision::Union{AbstractString,Nothing} = nothing,
                      sha256::Union{AbstractString,Nothing} = nothing)
        selected_backend = _validated_label(backend, "backend")
        identifier = String(strip(model_id))
        isempty(identifier) && throw(ArgumentError("model_id must not be empty"))
        ncodeunits(identifier) <= 1024 || throw(ArgumentError("model_id is too long"))
        occursin(r"[\x00-\x1f\x7f]", identifier) && throw(ArgumentError("model_id contains control characters"))
        selected_revision = isnothing(revision) ? nothing : String(strip(revision))
        !isnothing(selected_revision) && isempty(selected_revision) && throw(ArgumentError("revision must not be empty"))
        digest = isnothing(sha256) ? nothing : _validated_digest(sha256)
        return new(selected_backend, identifier, selected_revision, digest)
    end
end

function _capability_name(value::Union{Symbol,AbstractString})::Symbol
    name = lowercase(String(value))
    occursin(r"^[a-z][a-z0-9_]*$", name) || throw(ArgumentError("invalid model capability: $value"))
    return Symbol(name)
end

struct ModelCapabilitySet
    features::Tuple{Vararg{Symbol}}

    function ModelCapabilitySet(values::AbstractVector)
        all(value -> value isa Symbol || value isa AbstractString, values) ||
            throw(ArgumentError("capabilities must be strings or symbols"))
        normalized = sort!(unique(Symbol[_capability_name(value) for value in values]); by = String)
        return new(Tuple(normalized))
    end
end

ModelCapabilitySet(values::Tuple)::ModelCapabilitySet = ModelCapabilitySet(collect(values))
supports_capability(caps::ModelCapabilitySet, feature::Union{Symbol,AbstractString})::Bool =
    _capability_name(feature) in caps.features

function require_model_capability(caps::ModelCapabilitySet, feature::Union{Symbol,AbstractString})::Nothing
    supports_capability(caps, feature) || throw(ArgumentError("model does not declare the '$feature' capability"))
    return nothing
end

capability_set(caps::ModelCapabilitySet)::ModelCapabilitySet = caps

struct ModelDescriptor
    ref::ModelRef
    capabilities::ModelCapabilitySet
end

Base.:(==)(a::SourceRef, b::SourceRef)::Bool =
    a.sha256 == b.sha256 && a.format == b.format && a.size_bytes == b.size_bytes
Base.isequal(a::SourceRef, b::SourceRef)::Bool = a == b
Base.hash(x::SourceRef, seed::UInt)::UInt = hash((x.sha256, x.format, x.size_bytes), seed)

Base.:(==)(a::ModelRef, b::ModelRef)::Bool =
    a.backend == b.backend && a.model_id == b.model_id && a.revision == b.revision && a.sha256 == b.sha256
Base.isequal(a::ModelRef, b::ModelRef)::Bool = a == b
Base.hash(x::ModelRef, seed::UInt)::UInt = hash((x.backend, x.model_id, x.revision, x.sha256), seed)

Base.:(==)(a::ModelCapabilitySet, b::ModelCapabilitySet)::Bool = a.features == b.features
Base.isequal(a::ModelCapabilitySet, b::ModelCapabilitySet)::Bool = a == b
Base.hash(x::ModelCapabilitySet, seed::UInt)::UInt = hash(x.features, seed)

# Explicit JSON schemas: no automatic reconstruction of arbitrary Julia types.
const CONTRACT_SCHEMA_VERSION = 1

contract_payload(x::SourceRef) = (schema_version = 1, kind = "SourceRef",
                                  sha256 = x.sha256, format = String(x.format), size_bytes = x.size_bytes)
contract_payload(x::EvidenceRef) = (schema_version = 1, kind = "EvidenceRef", source = contract_payload(x.source),
                                    scheme = String(x.scheme), locator = x.locator, extraction_sha256 = x.extraction_sha256)
contract_payload(x::OperationRecord) = (schema_version = 1, kind = "OperationRecord", id = string(x.id), action = x.action,
                                        inputs = contract_payload.(x.inputs), outputs = contract_payload.(x.outputs),
                                        parameters_json = x.parameters_json, status = String(x.status))
contract_payload(x::ModelRef) = (schema_version = 1, kind = "ModelRef", backend = String(x.backend),
                                 model_id = x.model_id, revision = x.revision, sha256 = x.sha256)
contract_payload(x::ModelCapabilitySet) = (schema_version = 1, kind = "ModelCapabilitySet", features = String.(x.features))
contract_payload(x::ModelDescriptor) = (schema_version = 1, kind = "ModelDescriptor",
                                        ref = contract_payload(x.ref), capabilities = contract_payload(x.capabilities))

"""Serialize supported contract metadata into versioned JSON with stable field ordering."""
contract_json(x)::String = String(JSON3.write(contract_payload(x)))

function _parse_object(json::AbstractString)::JSON3.Object
    result = try
        JSON3.read(String(json))
    catch error
        throw(ArgumentError("invalid contract JSON: $(sprint(showerror, error))"))
    end
    result isa JSON3.Object || throw(ArgumentError("contract JSON must be an object"))
    return result
end

function _field(obj::AbstractDict, key::String)
    haskey(obj, Symbol(key)) && return obj[Symbol(key)]
    haskey(obj, key) && return obj[key]
    throw(ArgumentError("missing contract field: $key"))
end

function _envelope(obj, kind::String, fields::Tuple)::Nothing
    obj isa AbstractDict || throw(ArgumentError("$kind must be a JSON object"))
    Set(String(k) for k in keys(obj)) == Set(("schema_version", "kind", fields...)) ||
        throw(ArgumentError("$kind contains missing or unexpected fields"))
    version = _field(obj, "schema_version")
    version isa Integer && !(version isa Bool) && version == CONTRACT_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported contract schema version"))
    _field(obj, "kind") == kind || throw(ArgumentError("unexpected contract kind: expected $kind"))
    return nothing
end

function _string(obj::AbstractDict, key::String)::String
    value = _field(obj, key)
    value isa AbstractString || throw(ArgumentError("$key must be a string"))
    return String(value)
end

function _optional_string(obj::AbstractDict, key::String)::Union{String,Nothing}
    value = _field(obj, key)
    isnothing(value) && return nothing
    value isa AbstractString || throw(ArgumentError("$key must be a string or null"))
    return String(value)
end

function _integer(obj::AbstractDict, key::String)::Int
    value = _field(obj, key)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$key must be an integer"))
    return Int(value)
end

function _array(obj::AbstractDict, key::String)::AbstractVector
    value = _field(obj, key)
    value isa AbstractVector || throw(ArgumentError("$key must be an array"))
    return value
end

function from_contract(::Type{SourceRef}, obj)::SourceRef
    _envelope(obj, "SourceRef", ("sha256", "format", "size_bytes"))
    return SourceRef(_string(obj, "sha256"), Symbol(_string(obj, "format")), _integer(obj, "size_bytes"))
end

function from_contract(::Type{EvidenceRef}, obj)::EvidenceRef
    _envelope(obj, "EvidenceRef", ("source", "scheme", "locator", "extraction_sha256"))
    return EvidenceRef(from_contract(SourceRef, _field(obj, "source")), Symbol(_string(obj, "scheme")), _string(obj, "locator");
                       extraction_sha256 = _optional_string(obj, "extraction_sha256"))
end

function from_contract(::Type{OperationRecord}, obj)::OperationRecord
    _envelope(obj, "OperationRecord", ("id", "action", "inputs", "outputs", "parameters_json", "status"))
    identifier = try
        UUID(_string(obj, "id"))
    catch error
        throw(ArgumentError("invalid operation UUID: $(sprint(showerror, error))"))
    end
    inputs = SourceRef[from_contract(SourceRef, source) for source in _array(obj, "inputs")]
    outputs = SourceRef[from_contract(SourceRef, source) for source in _array(obj, "outputs")]
    return OperationRecord(identifier, _string(obj, "action"); inputs, outputs,
                           parameters_json = _string(obj, "parameters_json"), status = Symbol(_string(obj, "status")))
end

function from_contract(::Type{ModelRef}, obj)::ModelRef
    _envelope(obj, "ModelRef", ("backend", "model_id", "revision", "sha256"))
    return ModelRef(Symbol(_string(obj, "backend")), _string(obj, "model_id");
                    revision = _optional_string(obj, "revision"), sha256 = _optional_string(obj, "sha256"))
end

function from_contract(::Type{ModelCapabilitySet}, obj)::ModelCapabilitySet
    _envelope(obj, "ModelCapabilitySet", ("features",))
    values = _array(obj, "features")
    all(x -> x isa AbstractString, values) || throw(ArgumentError("capability features must be strings"))
    return ModelCapabilitySet(collect(values))
end

function from_contract(::Type{ModelDescriptor}, obj)::ModelDescriptor
    _envelope(obj, "ModelDescriptor", ("ref", "capabilities"))
    return ModelDescriptor(from_contract(ModelRef, _field(obj, "ref")),
                           from_contract(ModelCapabilitySet, _field(obj, "capabilities")))
end

"""Decode a supported versioned metadata contract; unknown fields and versions are rejected."""
function parse_contract(::Type{T}, json::AbstractString) where {T}
    return from_contract(T, _parse_object(json))
end

export SourceRef, EvidenceRef, OperationRecord, ModelRef, ModelCapabilitySet, ModelDescriptor
export contract_payload, contract_json, parse_contract, from_contract, supports_capability, require_model_capability, capability_set

end # module ResearchContracts
