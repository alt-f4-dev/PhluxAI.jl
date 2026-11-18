module SessionJournals

using Dates
using JSON3
using SHA
using UUIDs

const SCHEMA_VERSION = 1
const ZERO_DIGEST = repeat("0", 64)
const DEFAULT_MAX_RECORD_BYTES = 16 * 1024 * 1024

"""A journal integrity or incomplete-tail failure. `offset` identifies the first invalid record."""
struct JournalError <: Exception
    message::String
    offset::Int64
end
Base.showerror(io::IO, error::JournalError) = print(io, error.message, " at byte ", error.offset)

"""Immutable journal event. `payload_json` preserves the accepted JSON bytes exactly."""
struct JournalEvent
    session_id::UUID
    sequence::Int64
    id::UUID
    kind::String
    timestamp::String
    payload_json::String
    previous_sha256::String
    sha256::String
end

"""Verified journal prefix. Retain independently to detect removal of a committed suffix."""
struct JournalHead
    session_id::UUID
    sequence::Int64
    sha256::String
    offset::Int64
end

"""Exclusive POSIX journal writer. Use `open_journal`, `append_event!`, and `close`."""
mutable struct SessionJournal
    path::String
    io::IOStream
    lease::IOStream
    head::JournalHead
    max_record_bytes::Int
    lock::ReentrantLock
    poisoned::Bool
end

function _sync(io::IOStream)::Nothing
    flush(io)
    ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || systemerror("journal fsync", true)
    return nothing
end

function _sync_directory(path::String)::Nothing
    descriptor = ccall(:open, Cint, (Cstring, Cint), path, 0)
    descriptor >= 0 || systemerror("open journal directory", true)
    try
        ccall(:fsync, Cint, (Cint,), descriptor) == 0 || systemerror("directory fsync", true)
    finally
        ccall(:close, Cint, (Cint,), descriptor)
    end
    return nothing
end

function _durable_mkpath(directory::String)::Nothing
    pending = String[]
    probe = directory
    while !isdir(probe)
        push!(pending, probe)
        probe = dirname(probe)
    end
    for path in reverse(pending)
        mkdir(path; mode = 0o700)
        _sync_directory(dirname(path))
    end
    return nothing
end

function _record(io::IO, limit::Int)::Tuple{Vector{UInt8},Bool}
    bytes = UInt8[]
    while !eof(io)
        byte = read(io, UInt8)
        byte == 0x0a && return bytes, true
        length(bytes) < limit || throw(JournalError("record exceeds byte limit", position(io) - length(bytes) - 1))
        push!(bytes, byte)
    end
    return bytes, false
end

function _decode(bytes::Vector{UInt8}, head::JournalHead, offset::Int64)::JournalEvent
    try
        envelope = JSON3.read(String(bytes), Dict{String,Any})
        body = String(envelope["body_json"])
        digest = String(envelope["sha256"])
        bytes2hex(sha256(body)) == digest || throw(ArgumentError("digest mismatch"))
        data = JSON3.read(body, Dict{String,Any})
        data["schema_version"] == SCHEMA_VERSION || throw(ArgumentError("unsupported journal schema"))
        session_id = UUID(String(data["session_id"]))
        sequence = Int64(data["sequence"])
        sequence == head.sequence + 1 || throw(ArgumentError("invalid sequence"))
        head.sequence == 0 || session_id == head.session_id || throw(ArgumentError("session identity changed"))
        String(data["previous_sha256"]) == head.sha256 || throw(ArgumentError("broken hash chain"))
        id = UUID(String(data["id"]))
        id == uuid5(session_id, string(sequence)) || throw(ArgumentError("invalid event identity"))
        payload = String(data["payload_json"])
        JSON3.read(payload, Any)
        kind = String(data["kind"])
        !isempty(strip(kind)) && ncodeunits(kind) <= 256 || throw(ArgumentError("invalid event kind"))
        return JournalEvent(session_id, sequence, id, kind, String(data["timestamp"]), payload,
                            String(data["previous_sha256"]), digest)
    catch error
        error isa InterruptException && rethrow()
        throw(JournalError("invalid journal record: " * sprint(showerror, error), offset))
    end
end

function _scan(f::Function, io::IO; max_record_bytes::Int, checkpoint::Union{JournalHead,Nothing} = nothing,
               end_offset::Int64 = typemax(Int64))
    head = JournalHead(UUID(UInt128(0)), 0, ZERO_DIGEST, 0)
    anchored = checkpoint === nothing || checkpoint.sequence == 0
    while position(io) < end_offset && !eof(io)
        offset = Int64(position(io))
        bytes, complete = _record(io, max_record_bytes)
        complete || throw(JournalError("incomplete final record; explicit recovery required", offset))
        event = _decode(bytes, head, offset)
        head = JournalHead(event.session_id, event.sequence, event.sha256, position(io))
        if checkpoint !== nothing && head.sequence == checkpoint.sequence
            head == checkpoint || throw(JournalError("checkpoint mismatch", offset))
            anchored = true
        end
        f(event)
    end
    anchored || throw(JournalError("journal is shorter than the retained checkpoint", head.offset))
    return head
end

"""
    foreach_event(f, path; checkpoint=nothing, max_record_bytes=16*1024*1024)

Visit the verified prefix in order without materializing history. Each record is bounded; a later corrupt record raises `JournalError`. Use a closed writer or the writer overload for a stable view.
"""
function foreach_event(f::Function, path::AbstractString; checkpoint::Union{JournalHead,Nothing} = nothing,
                       max_record_bytes::Integer = DEFAULT_MAX_RECORD_BYTES)::JournalHead
    max_record_bytes > 0 || throw(ArgumentError("max_record_bytes must be positive"))
    return open(String(path), "r") do io
        _scan(f, io; checkpoint, max_record_bytes = Int(max_record_bytes))
    end
end

function foreach_event(f::Function, journal::SessionJournal; kwargs...)::JournalHead
    return lock(journal.lock) do
        foreach_event(f, journal.path; max_record_bytes = journal.max_record_bytes, kwargs...)
    end
end

verify_journal(path::AbstractString; kwargs...) = foreach_event(_ -> nothing, path; kwargs...)
verify_journal(journal::SessionJournal; kwargs...) = foreach_event(_ -> nothing, journal; kwargs...)
journal_head(journal::SessionJournal)::JournalHead = lock(() -> journal.head, journal.lock)

"""Serialize a verified head for storage in an independently trusted location."""
checkpoint_json(head::JournalHead)::String = JSON3.write((schema_version = SCHEMA_VERSION, session_id = string(head.session_id),
                                                        sequence = head.sequence, sha256 = head.sha256, offset = head.offset))

"""Parse a versioned checkpoint. Trust in its storage is the caller's responsibility."""
function parse_checkpoint(text::AbstractString)::JournalHead
    value = JSON3.read(text, Dict{String,Any})
    value["schema_version"] == SCHEMA_VERSION || throw(ArgumentError("unsupported checkpoint schema"))
    head = JournalHead(UUID(String(value["session_id"])), Int64(value["sequence"]), String(value["sha256"]), Int64(value["offset"]))
    head.sequence >= 0 && head.offset >= 0 && occursin(r"\A[0-9a-f]{64}\z", head.sha256) ||
        throw(ArgumentError("invalid journal checkpoint"))
    return head
end

"""
    open_journal(path; recover=false, checkpoint=nothing, max_record_bytes=16*1024*1024)

Open one exclusive POSIX writer. A second writer is rejected; OS locks release on process exit. Explicit recovery archives an incomplete final record durably before truncating it. Complete malformed records are never repaired automatically.
"""
function open_journal(path::AbstractString; recover::Bool = false, checkpoint::Union{JournalHead,Nothing} = nothing,
                      max_record_bytes::Integer = DEFAULT_MAX_RECORD_BYTES)::SessionJournal
    Sys.isunix() || throw(ArgumentError("durable journal writing currently requires POSIX fsync and flock"))
    max_record_bytes > 0 || throw(ArgumentError("max_record_bytes must be positive"))
    filename = abspath(String(path))
    _durable_mkpath(dirname(filename))
    filename = joinpath(realpath(dirname(filename)), basename(filename))
    islink(filename) && throw(ArgumentError("journal must not be a symbolic link"))
    lease = open(filename * ".lock", "a+")
    chmod(filename * ".lock", 0o600)
    if ccall(:flock, Cint, (Cint, Cint), fd(lease), 2 | 4) != 0
        close(lease)
        throw(ArgumentError("journal already has an exclusive writer"))
    end
    io = nothing
    try
        io = open(filename, "a+")
        chmod(filename, 0o600)
        seekstart(io)
        head = try
            _scan(_ -> nothing, io; checkpoint, max_record_bytes = Int(max_record_bytes))
        catch error
            recover && error isa JournalError && startswith(error.message, "incomplete final record") || rethrow()
            seek(io, error.offset)
            tail = read(io)
            archive = filename * ".incomplete-" * string(uuid4())
            open(archive, "w") do output
                chmod(archive, 0o600)
                write(output, tail)
                _sync(output)
            end
            _sync_directory(dirname(filename))
            # Validate the retained prefix and checkpoint before touching the original bytes.
            seekstart(io)
            retained = _scan(_ -> nothing, io; checkpoint, max_record_bytes = Int(max_record_bytes), end_offset = error.offset)
            truncate(io, error.offset)
            _sync(io)
            retained
        end
        _sync(io)
        _sync_directory(dirname(filename))
        seekend(io)
        session_id = head.sequence == 0 ? uuid4() : head.session_id
        head = JournalHead(session_id, head.sequence, head.sha256, head.offset)
        return SessionJournal(filename, io, lease, head, Int(max_record_bytes), ReentrantLock(), false)
    catch
        io === nothing || close(io)
        close(lease)
        rethrow()
    end
end

"""
    append_event!(journal, kind, payload) -> JournalEvent

Append an immutable event and fsync before acknowledging it. A write/sync failure poisons the handle; close and inspect/recover before further appends. JSON strings may be supplied verbatim through `payload_json`.
"""
function append_event!(journal::SessionJournal, kind::Union{Symbol,AbstractString}, payload = nothing;
                       payload_json::Union{AbstractString,Nothing} = nothing)::JournalEvent
    name = String(kind)
    !isempty(strip(name)) && ncodeunits(name) <= 256 || throw(ArgumentError("event kind must contain 1–256 UTF-8 bytes"))
    encoded = payload_json === nothing ? String(JSON3.write(payload)) : String(payload_json)
    ncodeunits(encoded) <= journal.max_record_bytes || throw(ArgumentError("journal payload exceeds byte limit"))
    JSON3.read(encoded, Any)
    return lock(journal.lock) do
        isopen(journal.io) || throw(ArgumentError("journal is closed"))
        journal.poisoned && throw(ArgumentError("journal requires inspection after a failed write"))
        head = journal.head
        filesize(journal.path) == head.offset || throw(JournalError("journal size changed outside writer", head.offset))
        sequence = head.sequence + 1
        id = uuid5(head.session_id, string(sequence))
        timestamp = string(now(UTC))
        body = String(JSON3.write((schema_version = SCHEMA_VERSION, session_id = string(head.session_id), sequence,
                                  id = string(id), kind = name, timestamp, payload_json = encoded, previous_sha256 = head.sha256)))
        digest = bytes2hex(sha256(body))
        record = String(JSON3.write((body_json = body, sha256 = digest)))
        ncodeunits(record) <= journal.max_record_bytes || throw(ArgumentError("journal record exceeds byte limit"))
        try
            write(journal.io, record, '\n')
            _sync(journal.io)
        catch
            journal.poisoned = true
            rethrow()
        end
        journal.head = JournalHead(head.session_id, sequence, digest, position(journal.io))
        return JournalEvent(head.session_id, sequence, id, name, timestamp, encoded, head.sha256, digest)
    end
end

function Base.close(journal::SessionJournal)::Nothing
    lock(journal.lock) do
        try
            close(journal.io)
        finally
            close(journal.lease)
        end
    end
    return nothing
end

Base.isopen(journal::SessionJournal)::Bool = isopen(journal.io)
Base.:(==)(a::JournalHead, b::JournalHead) = a.session_id == b.session_id && a.sequence == b.sequence && a.sha256 == b.sha256 && a.offset == b.offset

export SessionJournal, JournalEvent, JournalHead, JournalError, open_journal, append_event!, foreach_event, verify_journal,
       journal_head, checkpoint_json, parse_checkpoint

end # module SessionJournals
