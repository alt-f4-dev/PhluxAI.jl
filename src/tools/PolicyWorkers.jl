module PolicyWorkers

using SHA: SHA2_256_CTX, update!, digest!

import ..ResearchContracts: SourceRef
import ..Sandboxing
import ..ToolContracts: AbstractToolHandler, ExecutionPolicy, tool_command

"""Host grant binding a verified source identity to a local regular file."""
struct SourceGrant
    source::SourceRef
    path::String

    SourceGrant(source::SourceRef, path::AbstractString) = new(source, abspath(path))
end

function source_ids(arguments)::Vector{String}
    values = get(arguments, :source_ids, String[])
    values isa AbstractVector || throw(ArgumentError("source_ids must be an array of source identifiers"))
    ids = String[]
    seen = Set{String}()
    for value in values
        value isa AbstractString && occursin(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,127}\z", value) ||
            throw(ArgumentError("invalid source identifier"))
        value in seen && throw(ArgumentError("duplicate source identifier"))
        push!(seen, value)
        push!(ids, value)
    end
    return ids
end

function _copy_grant(destination::String, grant::SourceGrant)::Nothing
    isfile(grant.path) || throw(ArgumentError("granted source must be a regular file"))
    digest = SHA2_256_CTX()
    open(grant.path, "r") do input
        info = stat(input)
        isfile(info) && info.size == grant.source.size_bytes || throw(ArgumentError("granted source size mismatch"))
        open(destination, "w") do output
            buffer = Vector{UInt8}(undef, 65536)
            remaining = grant.source.size_bytes
            while remaining > 0
                expected = min(remaining, length(buffer))
                count = readbytes!(input, buffer, expected)
                count == expected || throw(ArgumentError("granted source size mismatch"))
                write(output, view(buffer, 1:count))
                update!(digest, buffer, count)
                remaining -= count
            end
            eof(input) || throw(ArgumentError("granted source size mismatch"))
        end
    end
    bytes2hex(digest!(digest)) == grant.source.sha256 || throw(ArgumentError("granted source digest mismatch"))
    return nothing
end

function _write_control(group::String, file::String, value)::Nothing
    open(joinpath(group, file), "w") do control
        print(control, value, '\n')
    end
    return nothing
end

function _counter(group::String, file::String, key::String)::Int
    for line in eachline(joinpath(group, file))
        name, value = split(line)
        name == key && return parse(Int, value)
    end
    error("worker cgroup did not report $key")
end

function _with_memory_limit(f, parent::String, bytes::Int)
    Sys.islinux() || throw(ArgumentError("worker memory limits require Linux cgroup v2"))
    root = realpath(parent)
    startswith(root, "/sys/fs/cgroup/") && isfile(joinpath(root, "cgroup.controllers")) ||
        throw(ArgumentError("worker_cgroup must name a delegated cgroup v2 directory"))
    group = mktempdir(root; prefix = "phluxai-", cleanup = false)
    armed = false
    try
        isfile(joinpath(group, "cgroup.kill")) || throw(ArgumentError("worker cgroup does not support process-tree cleanup"))
        page_size = Int(ccall(:getpagesize, Cint, ()))
        limit = bytes - mod(bytes, page_size)
        _write_control(group, "memory.max", limit)
        _write_control(group, "memory.swap.max", 0)
        _write_control(group, "memory.oom.group", 1)
        strip(read(joinpath(group, "memory.max"), String)) == string(limit) ||
            throw(ArgumentError("worker memory limit could not be applied"))
        armed = true
        return f(group)
    finally
        if armed
            _write_control(group, "cgroup.kill", 1)
            empty_group() = _counter(group, "cgroup.events", "populated") == 0
            timedwait(empty_group, 5.0; pollint = 0.01) == :ok || error("worker cgroup did not empty after cleanup")
        end
        rm(group)
    end
end

const ENTER_CGROUP = raw"""
    printf '%s\n' "$$" > "$1/cgroup.procs" || exit 125
    shift
    exec "$@"
    """

function run_worker(handler::AbstractToolHandler, arguments, grants::Dict{String,SourceGrant}, ids::Vector{String}, policy::ExecutionPolicy,
                    session::Sandboxing.SandboxConfig, cgroup_parent::Union{Nothing,String})
    policy.isolation == :bubblewrap && session.backend != Sandboxing.SANDBOX_LOCAL ||
        throw(ArgumentError("declared tools require Bubblewrap for per-source isolation"))
    !policy.network || session.network || throw(ArgumentError("sandbox network access is disabled"))
    cgroup_parent === nothing && throw(ArgumentError("declared tools require a delegated worker cgroup"))
    Sys.which("bwrap") !== nothing || throw(ArgumentError("Bubblewrap is unavailable"))
    return mktempdir() do directory
        _with_memory_limit(cgroup_parent, policy.max_memory_bytes) do group
            workspace = joinpath(directory, "workspace")
            inputs = mkdir(joinpath(directory, "inputs"))
            config = Sandboxing.SandboxConfig(workspace; backend = Sandboxing.SANDBOX_BWRAP, network = policy.network,
                                              timeout_seconds = min(policy.timeout_seconds, session.timeout_seconds),
                                              max_output_bytes = min(policy.max_output_bytes, session.max_output_bytes))
            mounts = Dict{String,String}()
            paths = Dict{String,String}()
            for id in ids
                path = joinpath(inputs, id)
                _copy_grant(path, grants[id])
                mounts[id] = path
                paths[id] = "/inputs/$id"
            end
            raw_argv = tool_command(handler, arguments, paths)
            raw_argv isa AbstractVector{<:AbstractString} || throw(ArgumentError("worker command must be an argv vector"))
            argv = String.(raw_argv)
            !isempty(argv) && isabspath(first(argv)) || throw(ArgumentError("worker executable must be an absolute sandbox path"))
            sandbox = Sandboxing._bwrap_command(config, argv, workspace; source_mounts = mounts)
            launcher = Cmd(["/bin/sh", "-c", ENTER_CGROUP, "phluxai-worker", group, sandbox.exec...])
            result = Sandboxing._run_command(config, argv, launcher)
            oom_killed = _counter(group, "memory.events", "oom_kill") > 0
            memory_exceeded = oom_killed || _counter(group, "memory.events", "oom") > 0
            return result, memory_exceeded, oom_killed
        end
    end
end

end # module PolicyWorkers
