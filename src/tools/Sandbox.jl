module Sandboxing

using Dates

@enum SandboxBackend begin
    SANDBOX_AUTO
    SANDBOX_BWRAP
    SANDBOX_LOCAL
end

"""Configuration for constrained `jupy` and `jupip` process execution."""
struct SandboxConfig
    workspace::String
    backend::SandboxBackend
    network::Bool
    timeout_seconds::Float64
    max_output_bytes::Int
    allow_unsafe_local::Bool
    readonly_paths::Vector{String}
    environment::Dict{String,String}

    function SandboxConfig(
        workspace::AbstractString;
        backend::SandboxBackend = SANDBOX_AUTO,
        network::Bool = false,
        timeout_seconds::Real = 120.0,
        max_output_bytes::Integer = 2_000_000,
        allow_unsafe_local::Bool = false,
        readonly_paths::AbstractVector{<:AbstractString} = String[],
        environment::AbstractDict{<:AbstractString,<:AbstractString} =
            Dict{String,String}(),
    )
        directory = abspath(normpath(String(workspace)))
        isempty(strip(directory)) && throw(ArgumentError(
            "sandbox workspace must not be empty",
        ))
        mkpath(directory)
        directory = realpath(directory)
        timeout = Float64(timeout_seconds)
        isfinite(timeout) && timeout > 0.0 || throw(ArgumentError(
            "timeout_seconds must be positive and finite",
        ))
        output_limit = Int(max_output_bytes)
        output_limit > 0 || throw(ArgumentError(
            "max_output_bytes must be positive",
        ))
        paths = unique(abspath.(normpath.(String.(readonly_paths))))
        env = Dict{String,String}(
            String(key) => String(value) for (key, value) in environment
        )
        mkpath(joinpath(directory, "home"))
        mkpath(joinpath(directory, ".julia_depot"))
        return new(
            directory,
            backend,
            network,
            timeout,
            output_limit,
            allow_unsafe_local,
            paths,
            env,
        )
    end
end

struct ProcessResult
    command::Vector{String}
    success::Bool
    timed_out::Bool
    exit_code::Union{Int,Nothing}
    stdout::String
    stderr::String
    started_at::DateTime
    elapsed_seconds::Float64
    output_limited::Bool
end

ProcessResult(command, success, timed_out, exit_code, stdout, stderr, started_at, elapsed_seconds) =
    ProcessResult(command, success, timed_out, exit_code, stdout, stderr, started_at, elapsed_seconds, false)

function _within(root::String, path::String)::Bool
    relative = relpath(path, root)
    relative == "." && return true
    relative == ".." && return false
    startswith(relative, "../") && return false
    startswith(relative, "..\\") && return false
    return true
end

"""Resolve a relative path and reject paths outside the sandbox workspace."""
function resolve_workspace_path(
    config::SandboxConfig,
    relative_path::AbstractString;
    must_exist::Bool = false,
)::String
    value = String(strip(relative_path))
    isempty(value) && throw(ArgumentError("path must not be empty"))
    isabspath(value) && throw(ArgumentError(
        "sandbox paths must be relative to the workspace",
    ))
    path = abspath(normpath(joinpath(config.workspace, value)))
    _within(config.workspace, path) || throw(ArgumentError(
        "path escapes the sandbox workspace: $relative_path",
    ))
    must_exist && !ispath(path) && throw(ArgumentError(
        "sandbox path does not exist: $relative_path",
    ))

    # Reject existing symlinks, or symlinked parents of a new path, that
    # resolve outside the workspace.
    probe = path
    while !ispath(probe) && probe != config.workspace
        probe = dirname(probe)
    end
    canonical_probe = realpath(probe)
    _within(config.workspace, canonical_probe) || throw(ArgumentError(
        "path resolves outside the sandbox workspace: $relative_path",
    ))
    if ispath(path)
        canonical = realpath(path)
        _within(config.workspace, canonical) || throw(ArgumentError(
            "path resolves outside the sandbox workspace: $relative_path",
        ))
        return canonical
    end
    return path
end

function _selected_backend(config::SandboxConfig)::SandboxBackend
    config.backend == SANDBOX_BWRAP && return SANDBOX_BWRAP
    config.backend == SANDBOX_LOCAL && return SANDBOX_LOCAL
    Sys.islinux() && !isnothing(Sys.which("bwrap")) &&
        return SANDBOX_BWRAP
    config.allow_unsafe_local && return SANDBOX_LOCAL
    throw(ArgumentError(
        "no supported sandbox backend is available; install bubblewrap " *
        "or explicitly set allow_unsafe_local=true",
    ))
end

function _home_mounts(config::SandboxConfig)::Vector{String}
    paths = copy(config.readonly_paths)
    depot = joinpath(homedir(), ".julia")
    isdir(depot) && push!(paths, depot)

    for executable_name in ("jupy", "jupip", "julia")
        executable = Sys.which(executable_name)
        isnothing(executable) || push!(paths, dirname(String(executable)))
    end

    for directory in (
        joinpath(homedir(), ".juliaup"),
        joinpath(homedir(), ".julia", "juliaup"),
        joinpath(homedir(), ".local", "share", "juliaup"),
    )
        isdir(directory) && push!(paths, directory)
    end

    install_root = get(
        ENV,
        "JUPY_INSTALL_ROOT",
        joinpath(homedir(), ".local", "share", "jupy"),
    )
    isdir(install_root) && push!(paths, install_root)
    return unique(filter(ispath, abspath.(normpath.(paths))))
end

function _parent_directories(path::String)::Vector{String}
    values = String[]
    current = dirname(path)
    while current != "/" && current != "."
        push!(values, current)
        current = dirname(current)
    end
    reverse!(values)
    return values
end

function _bwrap_command(
    config::SandboxConfig,
    arguments::Vector{String},
    working_directory::String;
    source_mounts::Union{Nothing,Dict{String,String}} = nothing,
)::Cmd
    executable = Sys.which("bwrap")
    isnothing(executable) && throw(ArgumentError(
        "bubblewrap executable `bwrap` was not found",
    ))

    command = String[
        String(executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/bin", "/bin",
        "--ro-bind-try", "/sbin", "/sbin",
        "--ro-bind-try", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--dir",
        "/workspace",
        "--bind",
        config.workspace,
        "/workspace",
        "--chdir",
        "/workspace/$(relpath(working_directory, config.workspace))",
        "--setenv",
        "HOME",
        "/workspace/home",
        "--setenv",
        "PATH",
        get(ENV, "PATH", "/usr/local/bin:/usr/bin:/bin"),
        "--setenv",
        "LANG",
        get(ENV, "LANG", "C.UTF-8"),
        "--setenv",
        "LC_ALL",
        get(ENV, "LC_ALL", "C.UTF-8"),
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "XDG_CACHE_HOME",
        "/workspace/home/.cache",
        "--setenv",
        "JULIA_DEPOT_PATH",
        "/workspace/.julia_depot:$(joinpath(homedir(), ".julia"))",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
    ]

    config.network || append!(command, ["--unshare-net"])

    if source_mounts !== nothing
        # Policy workers receive private scratch, the system runtime, and the exact staged inputs.
        append!(command, ["--unshare-user", "--disable-userns", "--cap-drop", "ALL", "--tmpfs", "/usr/local",
                          "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "JULIA_DEPOT_PATH", "/workspace/.julia_depot",
                          "--dir", "/inputs"])
        for (id, path) in source_mounts
            append!(command, ["--ro-bind", path, "/inputs/$id"])
        end
        append!(command, ["--", arguments...])
        return Cmd(command)
    end

    created = Set{String}(["/home"])
    for path in _home_mounts(config)
        startswith(path, "/home/") || continue
        for parent in _parent_directories(path)
            parent in created && continue
            push!(command, "--dir", parent)
            push!(created, parent)
        end
        push!(command, "--ro-bind", path, path)
    end

    for (key, value) in config.environment
        push!(command, "--setenv", key, value)
    end
    push!(command, "--")
    append!(command, arguments)
    return Cmd(command)
end

function _local_command(
    config::SandboxConfig,
    arguments::Vector{String},
    working_directory::String,
)::Cmd
    config.allow_unsafe_local || throw(ArgumentError(
        "unsafe local execution is disabled",
    ))
    config.network || throw(ArgumentError(
        "SANDBOX_LOCAL cannot enforce network=false; " *
        "use SANDBOX_BWRAP for network isolation",
    ))
    env = copy(ENV)
    env["HOME"] = joinpath(config.workspace, "home")
    env["JULIA_DEPOT_PATH"] = joinpath(config.workspace, ".julia_depot")
    for (key, value) in config.environment
        env[key] = value
    end
    return Cmd(Cmd(arguments); dir = working_directory, env)
end

function _capture_output(stream::IO, output::IOBuffer, remaining::Ref{Int}, limited::Ref{Bool})::Nothing
    chunk = Vector{UInt8}(undef, 8192)
    while !limited[] && !eof(stream)
        # Read buffered bytes so both capture tasks share the budget without yielding inside the update.
        count = readbytes!(stream, chunk, min(length(chunk), bytesavailable(stream)))
        retained = min(count, remaining[])
        write(output, view(chunk, 1:retained))
        remaining[] -= retained
        limited[] |= retained < count
        yield()
    end
    return nothing
end

function _output_text(buffer::IOBuffer)::String
    text = String(take!(buffer))
    # An output limit can split a multibyte character. Replacement preserves the byte budget.
    return isvalid(text) ? text : map(c -> isvalid(c) ? c : '?', text)
end

"""Run an argv vector inside the configured sandbox."""
function run_sandboxed(
    config::SandboxConfig,
    arguments::AbstractVector{<:AbstractString};
    cwd::AbstractString = ".",
)::ProcessResult
    argv = String[String(value) for value in arguments]
    isempty(argv) && throw(ArgumentError("command must not be empty"))
    working_directory = resolve_workspace_path(config, cwd; must_exist = true)
    backend = _selected_backend(config)
    command = backend == SANDBOX_BWRAP ?
        _bwrap_command(config, argv, working_directory) :
        _local_command(config, argv, working_directory)

    return _run_command(config, argv, command)
end

function _run_command(config::SandboxConfig, argv::Vector{String}, command::Cmd)::ProcessResult
    stdout_pipe, stderr_pipe = Pipe(), Pipe()
    stdout_buffer, stderr_buffer = IOBuffer(), IOBuffer()
    remaining = Ref(config.max_output_bytes)
    limited = Ref(false)
    started_at = now(UTC)
    started_ns = time_ns()
    process = nothing
    readers = ()
    status = :ok
    try
        process = run(pipeline(command; stdin = devnull, stdout = stdout_pipe, stderr = stderr_pipe); wait = false)
        close(stdout_pipe.in)
        close(stderr_pipe.in)
        stdout_reader = @async _capture_output(stdout_pipe.out, stdout_buffer, remaining, limited)
        stderr_reader = @async _capture_output(stderr_pipe.out, stderr_buffer, remaining, limited)
        readers = (stdout_reader, stderr_reader)
        # EOF is part of completion because descendants may hold pipes open after the launcher exits.
        done() = limited[] || any(istaskfailed, readers) || (process_exited(process) && all(istaskdone, readers))
        time_left = max(0.0, config.timeout_seconds - (time_ns() - started_ns) / 1.0e9)
        status = timedwait(done, time_left; pollint = 0.01)
    finally
        try
            if !isnothing(process) && !process_exited(process)
                kill(process, Base.SIGKILL)
            end
        finally
            close(stdout_pipe)
            close(stderr_pipe)
            isnothing(process) || wait(process)
        end
    end
    foreach(fetch, readers)
    timed_out = status == :timed_out
    exit_code = process.exitcode < 0 || process.termsignal != 0 ? nothing : process.exitcode
    succeeded = !timed_out && !limited[] && success(process)
    elapsed = (time_ns() - started_ns) / 1.0e9
    return ProcessResult(argv, succeeded, timed_out, exit_code, _output_text(stdout_buffer), _output_text(stderr_buffer),
                         started_at, elapsed, limited[])
end

export SandboxBackend,
       SANDBOX_AUTO,
       SANDBOX_BWRAP,
       SANDBOX_LOCAL,
       SandboxConfig,
       ProcessResult,
       resolve_workspace_path,
       run_sandboxed

end # module Sandboxing
