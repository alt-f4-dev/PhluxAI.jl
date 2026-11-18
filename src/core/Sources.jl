module SourceStores

using SHA: SHA2_256_CTX, update!, digest!

import ..ResearchContracts: SourceRef
import ..PolicyWorkers: SourceGrant

"""A private store of source snapshots, bounded independently of worker output."""
struct SourceStore
    root::String
    max_source_bytes::Int

    function SourceStore(root::AbstractString; max_source_bytes::Integer = 100 * 1024 * 1024)
        max_source_bytes > 0 || throw(ArgumentError("max_source_bytes must be positive"))
        mkpath(root)
        return new(realpath(root), Int(max_source_bytes))
    end
end

function _format(path::String)::Symbol
    open(path) do io
        size = filesize(path)
        prefix = read(io, min(1024, size))
        occursin("%PDF-", String(copy(prefix))) && return :pdf
        signature = UInt8[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]
        offset = 0
        while offset <= size - 8
            seek(io, offset)
            read(io, 8) == signature && return :hdf5
            offset = offset == 0 ? 512 : Base.checked_mul(offset, 2)
        end
        throw(ArgumentError("source bytes do not identify a PDF or HDF5 file"))
    end
end

function _source_path(store::SourceStore, source::SourceRef)::String
    source.format in (:pdf, :hdf5) || throw(ArgumentError("unsupported source format"))
    return joinpath(store.root, "$(source.sha256).$(source.format)")
end

"""
    import_source(store::SourceStore, path; format=:auto) -> SourceRef

Copy a regular file into a private snapshot, computing its identity while enforcing the store's byte limit. Parsing occurs only in a worker. A claimed format must match the file signature.
"""
function import_source(store::SourceStore, path::AbstractString; format::Symbol = :auto)::SourceRef
    format in (:auto, :pdf, :hdf5) || throw(ArgumentError("unsupported source format"))
    isfile(path) || throw(ArgumentError("source must be a regular file"))
    temporary, output = mktemp(store.root)
    digest = SHA2_256_CTX()
    total = 0
    try
        open(path, "r") do input
            isfile(stat(input)) || throw(ArgumentError("source must be a regular file"))
            filesize(input) <= store.max_source_bytes || throw(ArgumentError("source exceeds max_source_bytes"))
            buffer = Vector{UInt8}(undef, 65536)
            while !eof(input)
                remaining = store.max_source_bytes - total
                remaining > 0 || throw(ArgumentError("source exceeds max_source_bytes"))
                count = readbytes!(input, buffer, min(length(buffer), remaining))
                total += count
                total <= store.max_source_bytes || throw(ArgumentError("source exceeds max_source_bytes"))
                write(output, view(buffer, 1:count))
                update!(digest, buffer, count)
            end
        end
        close(output)
        detected = _format(temporary)
        format in (:auto, detected) || throw(ArgumentError("source signature differs from its claimed format"))
        source = SourceRef(bytes2hex(digest!(digest)), detected, total)
        destination = _source_path(store, source)
        chmod(temporary, 0o400)
        # Hard-link publication is atomic and never overwrites a concurrently imported snapshot.
        try
            hardlink(temporary, destination)
        catch error
            isfile(destination) && !islink(destination) || rethrow()
            filesize(destination) == total || throw(ArgumentError("stored source size mismatch"))
            verify_source(store, source)
        end
        return source
    finally
        close(output)
        rm(temporary; force = true)
    end
end

"""Return a grant for an imported snapshot. The worker verifies its digest when staging it."""
function source_grant(store::SourceStore, source::SourceRef)::SourceGrant
    path = _source_path(store, source)
    isfile(path) && !islink(path) || throw(ArgumentError("source has not been imported"))
    filesize(path) == source.size_bytes || throw(ArgumentError("stored source size mismatch"))
    return SourceGrant(source, path)
end

function verify_source(store::SourceStore, source::SourceRef)::Nothing
    grant = source_grant(store, source)
    digest = SHA2_256_CTX()
    open(grant.path) do input
        buffer = Vector{UInt8}(undef, 65536)
        while !eof(input)
            count = readbytes!(input, buffer)
            update!(digest, buffer, count)
        end
    end
    bytes2hex(digest!(digest)) == source.sha256 || throw(ArgumentError("stored source digest mismatch"))
    return nothing
end

export SourceStore, import_source, source_grant, verify_source

end # module SourceStores
