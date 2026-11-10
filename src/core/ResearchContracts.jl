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

export SourceRef, EvidenceRef, OperationRecord

end # module ResearchContracts
