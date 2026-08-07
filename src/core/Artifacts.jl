module ArtifactStore

using Base64
using SHA
using UUIDs

import ..ProtocolTypes: ArtifactRef, is_image

const DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024

"""Filesystem-backed artifact store with path and size confinement."""
struct Store
    root::String
    max_file_bytes::Int
    allowed_mime_types::Set{String}

    function Store(
        root::AbstractString;
        max_file_bytes::Integer = DEFAULT_MAX_FILE_BYTES,
        allowed_mime_types = String[],
    )
        directory = abspath(normpath(String(root)))
        isempty(strip(directory)) && throw(ArgumentError(
            "artifact-store root must not be empty",
        ))
        mkpath(directory)
        directory = realpath(directory)
        limit = Int(max_file_bytes)
        limit > 0 || throw(ArgumentError(
            "max_file_bytes must be positive",
        ))
        allowed = Set{String}(
            _normalize_mime_type(value) for value in allowed_mime_types
        )
        return new(directory, limit, allowed)
    end
end

function _normalize_mime_type(value::AbstractString)::String
    mime = lowercase(strip(String(value)))
    isempty(mime) && throw(ArgumentError("MIME type must not be empty"))
    occursin('/', mime) || throw(ArgumentError(
        "invalid MIME type '$mime'",
    ))
    return mime
end

function _check_size(store::Store, size_bytes::Integer)::Int
    size = Int(size_bytes)
    size >= 0 || throw(ArgumentError(
        "artifact size must be nonnegative",
    ))
    size <= store.max_file_bytes || throw(ArgumentError(
        "artifact size $size exceeds limit $(store.max_file_bytes)",
    ))
    return size
end

function _allowed(store::Store, mime_type::String)::Bool
    return isempty(store.allowed_mime_types) ||
           mime_type in store.allowed_mime_types
end

const _EXTENSION_MIME = Dict(
    ".png" => "image/png",
    ".jpg" => "image/jpeg",
    ".jpeg" => "image/jpeg",
    ".gif" => "image/gif",
    ".webp" => "image/webp",
    ".svg" => "image/svg+xml",
    ".pdf" => "application/pdf",
    ".json" => "application/json",
    ".jl" => "text/x-julia",
    ".py" => "text/x-python",
    ".txt" => "text/plain",
    ".csv" => "text/csv",
)

const _MIME_EXTENSION = Dict(
    "image/png" => ".png",
    "image/jpeg" => ".jpg",
    "image/gif" => ".gif",
    "image/webp" => ".webp",
    "image/svg+xml" => ".svg",
    "application/pdf" => ".pdf",
    "application/json" => ".json",
    "text/x-julia" => ".jl",
    "text/x-python" => ".py",
    "text/plain" => ".txt",
    "text/csv" => ".csv",
)

function _starts_with(
    bytes::AbstractVector{UInt8},
    signature::AbstractVector{UInt8},
)::Bool
    length(bytes) >= length(signature) || return false
    for index in eachindex(signature)
        bytes[index] == signature[index] || return false
    end
    return true
end

"""Detect common image, PDF, JSON, source, and text MIME types."""
function detect_mime_type(
    bytes::AbstractVector{UInt8};
    filename::Union{AbstractString,Nothing} = nothing,
)::String
    _starts_with(
        bytes,
        UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    ) && return "image/png"
    _starts_with(bytes, UInt8[0xff, 0xd8, 0xff]) &&
        return "image/jpeg"
    _starts_with(bytes, collect(codeunits("GIF87a"))) &&
        return "image/gif"
    _starts_with(bytes, collect(codeunits("GIF89a"))) &&
        return "image/gif"
    _starts_with(bytes, collect(codeunits("%PDF-"))) &&
        return "application/pdf"

    if length(bytes) >= 12
        riff = all(
            bytes[index] == codeunit("RIFF", index)
            for index in 1:4
        )
        webp = all(
            bytes[index + 8] == codeunit("WEBP", index)
            for index in 1:4
        )
        riff && webp && return "image/webp"
    end

    sample_length = min(length(bytes), 4096)
    if sample_length > 0
        sample = try
            lowercase(String(copy(bytes[1:sample_length])))
        catch
            ""
        end
        occursin("<svg", sample) && return "image/svg+xml"
        stripped = strip(sample)
        if startswith(stripped, '{') || startswith(stripped, '[')
            return "application/json"
        end
    end

    if !isnothing(filename)
        extension = lowercase(splitext(String(filename))[2])
        inferred = get(_EXTENSION_MIME, extension, nothing)
        !isnothing(inferred) && return inferred
    end

    return "application/octet-stream"
end

function _safe_filename(filename::AbstractString)::String
    value = String(strip(filename))
    isempty(value) && throw(ArgumentError(
        "artifact filename must not be empty",
    ))
    basename(value) == value || throw(ArgumentError(
        "artifact filename must not contain directory components",
    ))
    value in (".", "..") && throw(ArgumentError(
        "invalid artifact filename '$value'",
    ))
    occursin('\0', value) && throw(ArgumentError(
        "artifact filename contains a NUL byte",
    ))
    return value
end

function _unique_filename(
    store::Store,
    requested::Union{AbstractString,Nothing},
    mime_type::String,
    prefix::AbstractString,
)::String
    extension = get(_MIME_EXTENSION, mime_type, ".bin")
    filename = if isnothing(requested)
        "$(String(prefix))-$(uuid4())$extension"
    else
        value = _safe_filename(requested)
        isempty(splitext(value)[2]) ? value * extension : value
    end
    !ispath(joinpath(store.root, filename)) && return filename
    stem, suffix = splitext(filename)
    return "$stem-$(uuid4())$suffix"
end

function _relative_path(store::Store, path::AbstractString)::String
    candidate = abspath(normpath(String(path)))
    relative = relpath(candidate, store.root)
    relative == ".." && throw(ArgumentError(
        "artifact path escapes the artifact-store root",
    ))
    startswith(relative, "../") && throw(ArgumentError(
        "artifact path escapes the artifact-store root",
    ))
    startswith(relative, "..\\") && throw(ArgumentError(
        "artifact path escapes the artifact-store root",
    ))
    return relative
end

function resolve_artifact(
    store::Store,
    artifact::ArtifactRef,
)::String
    isabspath(artifact.path) && throw(ArgumentError(
        "artifact paths must be relative to the store root",
    ))
    candidate = joinpath(store.root, artifact.path)
    relative = _relative_path(store, candidate)
    resolved = joinpath(store.root, relative)

    # Existing symlinks must not provide an escape from the store root.
    if ispath(resolved)
        canonical = realpath(resolved)
        _relative_path(store, canonical)
        return canonical
    end
    return resolved
end

_digest(bytes::AbstractVector{UInt8})::String =
    bytes2hex(SHA.sha256(bytes))

function _atomic_write(
    destination::String,
    bytes::AbstractVector{UInt8},
)::Nothing
    directory = dirname(destination)
    mkpath(directory)
    temporary = tempname(directory)
    try
        open(temporary, "w") do io
            write(io, bytes)
            flush(io)
        end
        mv(temporary, destination; force = true)
    catch
        isfile(temporary) && rm(temporary; force = true)
        rethrow()
    end
    return nothing
end

function write_artifact(
    store::Store,
    bytes::AbstractVector{UInt8};
    filename::Union{AbstractString,Nothing} = nothing,
    mime_type::Union{AbstractString,Nothing} = nothing,
    prefix::AbstractString = "artifact",
)::ArtifactRef
    size = _check_size(store, length(bytes))
    selected_mime = isnothing(mime_type) ?
        detect_mime_type(bytes; filename) :
        _normalize_mime_type(mime_type)
    _allowed(store, selected_mime) || throw(ArgumentError(
        "MIME type '$selected_mime' is not allowed",
    ))
    stored_name = _unique_filename(
        store,
        filename,
        selected_mime,
        prefix,
    )
    destination = joinpath(store.root, stored_name)
    _atomic_write(destination, bytes)
    return ArtifactRef(
        stored_name,
        selected_mime,
        _digest(bytes),
        size,
    )
end

function import_artifact(
    store::Store,
    path::AbstractString;
    filename::Union{AbstractString,Nothing} = nothing,
    mime_type::Union{AbstractString,Nothing} = nothing,
    prefix::AbstractString = "import",
)::ArtifactRef
    source = abspath(normpath(String(path)))
    isfile(source) || throw(ArgumentError(
        "artifact source does not exist: $source",
    ))
    _check_size(store, filesize(source))
    requested = isnothing(filename) ? basename(source) : filename
    return write_artifact(
        store,
        read(source);
        filename = requested,
        mime_type,
        prefix,
    )
end

function _split_data_uri(
    encoded::AbstractString,
)::Tuple{Union{String,Nothing},String}
    value = strip(String(encoded))
    startswith(lowercase(value), "data:") || return nothing, value
    separator = findfirst(==(','), value)
    isnothing(separator) && throw(ArgumentError("malformed data URI"))
    header = value[6:(separator - 1)]
    payload = value[(separator + 1):end]
    fields = split(header, ';')
    mime = isempty(fields[1]) ? nothing :
        _normalize_mime_type(fields[1])
    any(lowercase(field) == "base64" for field in fields[2:end]) ||
        throw(ArgumentError("data URI is not base64 encoded"))
    return mime, payload
end

function decode_base64_artifact(
    store::Store,
    encoded::AbstractString;
    filename::Union{AbstractString,Nothing} = nothing,
    mime_type::Union{AbstractString,Nothing} = nothing,
    prefix::AbstractString = "generated",
)::ArtifactRef
    uri_mime, payload = _split_data_uri(encoded)
    compact = filter(character -> !isspace(character), payload)
    approximate_bytes = cld(3 * ncodeunits(compact), 4)
    _check_size(store, approximate_bytes)
    bytes = try
        Base64.base64decode(compact)
    catch error
        throw(ArgumentError(
            "invalid base64 artifact payload: " * sprint(showerror, error),
        ))
    end
    selected_mime = isnothing(mime_type) ? uri_mime : mime_type
    return write_artifact(
        store,
        bytes;
        filename,
        mime_type = selected_mime,
        prefix,
    )
end

function read_artifact(
    store::Store,
    artifact::ArtifactRef,
)::Vector{UInt8}
    path = resolve_artifact(store, artifact)
    isfile(path) || throw(ArgumentError(
        "artifact file does not exist: $(artifact.path)",
    ))
    _check_size(store, filesize(path))
    return read(path)
end

function encode_artifact_base64(
    store::Store,
    artifact::ArtifactRef,
)::String
    return Base64.base64encode(read_artifact(store, artifact))
end

function verify_artifact(
    store::Store,
    artifact::ArtifactRef,
)::Bool
    path = try
        resolve_artifact(store, artifact)
    catch
        return false
    end
    isfile(path) || return false
    filesize(path) == artifact.size_bytes || return false
    return _digest(read(path)) == artifact.sha256
end

function list_artifacts(store::Store)::Vector{String}
    files = String[]
    for (directory, _, names) in walkdir(store.root)
        for name in names
            push!(files, relpath(joinpath(directory, name), store.root))
        end
    end
    sort!(files)
    return files
end

function valid_image_artifact(
    store::Store,
    artifact::ArtifactRef,
)::Bool
    return is_image(artifact) && verify_artifact(store, artifact)
end

export Store,
       DEFAULT_MAX_FILE_BYTES,
       detect_mime_type,
       resolve_artifact,
       write_artifact,
       import_artifact,
       decode_base64_artifact,
       read_artifact,
       encode_artifact_base64,
       verify_artifact,
       list_artifacts,
       valid_image_artifact

end # module ArtifactStore
