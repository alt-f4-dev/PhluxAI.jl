module ComfyLifecycle

using HTTP
using JSON3

import ..ComfyClient: COMFY_BASE, ComfyBackend

const _LOCK = ReentrantLock()
const _OWNED = Ref{Union{Nothing,NamedTuple{(:pid, :stamp),Tuple{Int,String}}}}(nothing)
const _LEASE = Ref{Union{Nothing,String}}(nothing)
const _EXIT_REGISTERED = Ref(false)
const _ACTIVE_INTERACTIVE_REPLS = Ref(0)

function _enabled()::Bool
    value = lowercase(strip(get(ENV, "PHLUXAI_COMFY_AUTOSTART", "1")))
    return !(value in ("0", "false", "off", "no", "never"))
end

"""Probe the expected ComfyUI endpoint; a generic HTTP 200 is insufficient."""
function _available(url::AbstractString)::Bool
    try
        response = HTTP.get(
            String(url) * "/system_stats";
            status_exception = false,
            redirect = false,
            proxy = HTTP.ProxyConfig(),
            request_timeout = 1.5,
        )
        response.status == 200 || return false
        stats = JSON3.read(String(copy(response.body)))
        return haskey(stats, :system) && haskey(stats, :devices)
    catch
        return false
    end
end

"""Parse the last successful comfy-cli envelope, ignoring ordinary banner lines."""
function _cli_envelope(raw::AbstractString)
    for line in Iterators.reverse(split(String(raw), '\n'))
        candidate = try
            JSON3.read(line)
        catch
            continue
        end
        get(candidate, :type, nothing) == "envelope" || continue
        get(candidate, :ok, false) === true || throw(ErrorException(
            "comfy-cli returned an unsuccessful envelope: " * String(line),
        ))
        return candidate
    end
    throw(ErrorException("comfy-cli did not return a successful JSON envelope"))
end

function _cli(args::Vector{String}; deadline::Int = 120)
    executable = Sys.which("comfy")
    isnothing(executable) && throw(ArgumentError(
        "comfy executable is missing from PATH; install/configure comfy-cli before enabling ComfyUI autostart",
    ))
    timeout = Sys.which("timeout")
    isnothing(timeout) && throw(ArgumentError(
        "managed comfy-cli launch requires GNU timeout on this Linux installation",
    ))
    command = Cmd([String(timeout), "--signal=TERM", "--kill-after=5s", "$(deadline)s", String(executable), "--json", args...])
    command = addenv(
        command,
        "COMFY_WHERE" => "local",
        "COMFY_LOCAL_URL" => COMFY_BASE,
        "COMFY_NO_TELEMETRY" => "1",
        "DO_NOT_TRACK" => "1",
    )
    output = read(pipeline(command; stderr = devnull), String)
    return _cli_envelope(output)
end

"""Linux PID + kernel start time; protects against recycled process identifiers."""
function _process_stamp(pid::Integer)::Union{String,Nothing}
    Sys.islinux() || return nothing
    pid > 0 || return nothing
    try
        record = read("/proc/$(pid)/stat", String)
        closing = findlast(')', record)
        isnothing(closing) && return nothing
        fields = split(strip(record[(closing + 1):end]))
        length(fields) >= 20 || return nothing
        return String(fields[20])
    catch
        return nothing
    end
end

function _lease_directory()::String
    base = get(ENV, "XDG_RUNTIME_DIR", "")
    if isempty(strip(base)) || !isdir(base)
        base = joinpath(homedir(), ".local", "state")
    end
    directory = joinpath(base, "PhluxAI", "comfy-8188")
    mkpath(directory)
    chmod(directory, 0o700)
    return directory
end

function _register_lease!()::Nothing
    !isnothing(_LEASE[]) && return nothing
    stamp = _process_stamp(getpid())
    isnothing(stamp) && throw(ErrorException("cannot establish the current process identity"))
    path = joinpath(_lease_directory(), "lease.$(getpid()).$(stamp)")
    open(path, "w") do io
        println(io, COMFY_BASE)
    end
    _LEASE[] = path
    return nothing
end

"""Best-effort detection of live PhluxAI borrowers in other Julia processes."""
function _has_other_borrowers()::Bool
    directory = _lease_directory()
    for path in readdir(directory; join = true)
        name = basename(path)
        startswith(name, "lease.") || continue
        path == _LEASE[] && continue
        parts = split(name, '.')
        length(parts) == 3 || continue
        pid = tryparse(Int, parts[2])
        isnothing(pid) && continue
        if _process_stamp(pid) == parts[3]
            return true
        elseif isfile(path)
            rm(path; force = true)
        end
    end
    return false
end

function _release_lease!()::Nothing
    path = _LEASE[]
    _LEASE[] = nothing
    if !isnothing(path) && isfile(path)
        rm(path; force = true)
    end
    return nothing
end

function _background_data(envelope)
    data = get(envelope, :data, nothing)
    isnothing(data) && return nothing
    config = get(data, :config, nothing)
    isnothing(config) && return nothing
    return get(config, :background, nothing)
end

function _background_matches(owner)::Bool
    environment = _cli(["env"]; deadline = 20)
    background = _background_data(environment)
    isnothing(background) && return false
    return Int(get(background, :pid, 0)) == owner.pid &&
           Int(get(background, :port, 0)) == 8188 &&
           String(get(background, :host, "")) in ("127.0.0.1", "localhost")
end

function _stop_matches(owner)::Bool
    dry_run = _cli(["stop", "--dry-run"]; deadline = 20)
    data = get(dry_run, :data, nothing)
    isnothing(data) && return false
    return Int(get(data, :pid, 0)) == owner.pid && get(data, :dry_run, false) === true
end

"""Ensure the default local ComfyUI is running. Never claim a pre-existing server."""
function ensure_default_running!()::Symbol
    return lock(_LOCK) do
        if _available(COMFY_BASE)
            _register_lease!()
            owner = _OWNED[]
            if !isnothing(owner) && _process_stamp(owner.pid) == owner.stamp
                return :owned
            end
            _OWNED[] = nothing
            return :borrowed
        end
        _enabled() || throw(ArgumentError(
            "ComfyUI is not running and PHLUXAI_COMFY_AUTOSTART=0; start it with `comfy launch --background`",
        ))
        result = _cli(["launch", "--background"])
        data = get(result, :data, nothing)
        isnothing(data) && throw(ErrorException("ComfyUI launch returned no process metadata"))
        pid = Int(get(data, :pid, 0))
        url = String(get(data, :url, ""))
        pid > 0 && url == COMFY_BASE || throw(ErrorException(
            "ComfyUI launch reported an unexpected PID or endpoint; refusing to claim ownership",
        ))
        _available(COMFY_BASE) || throw(ErrorException("ComfyUI launch returned successfully, but its API is unavailable"))
        stamp = _process_stamp(pid)
        isnothing(stamp) && throw(ErrorException(
            "ComfyUI started, but its process identity cannot be verified; it will not be stopped automatically",
        ))
        _OWNED[] = (pid = pid, stamp = stamp)
        _register_lease!()
        return :owned
    end
end

"""Ensure a configured backend is ready; only the default localhost endpoint is auto-managed."""
function ensure_backend!(backend::ComfyBackend)::Symbol
    if backend.base_url == COMFY_BASE
        return ensure_default_running!()
    end
    _available(backend.base_url) || throw(ArgumentError(
        "ComfyUI at $(backend.base_url) is unavailable; automatic startup is limited to $(COMFY_BASE)",
    ))
    return :external
end

ensure_backend!(::Any)::Symbol = :not_comfy

"""Stop only the CLI-tracked PID started by this Julia process, if not shared."""
function shutdown_owned!()::Nothing
    lock(_LOCK) do
        owner = _OWNED[]
        try
            if !isnothing(owner) && !isnothing(_LEASE[]) && !_has_other_borrowers() &&
               _process_stamp(owner.pid) == owner.stamp && _background_matches(owner) && _stop_matches(owner)
                result = _cli(["stop"]; deadline = 25)
                data = get(result, :data, nothing)
                if isnothing(data) || Int(get(data, :pid, 0)) != owner.pid || get(data, :stopped, false) !== true
                    @warn "comfy-cli returned an unverified stop result; inspect the managed service manually"
                end
            elseif !isnothing(owner) && _has_other_borrowers()
                @info "ComfyUI left running because another PhluxAI process still uses it"
            end
        catch error
            @warn "Unable to verify or stop PhluxAI-owned ComfyUI; leaving it for manual inspection" exception = (error, catch_backtrace())
        finally
            _OWNED[] = nothing
            try
                _release_lease!()
            catch error
                @warn "Unable to remove ComfyUI lifecycle lease" exception = (error, catch_backtrace())
            end
        end
    end
    return nothing
end

"""State visible to interactive macros; does not change service ownership."""
function lifecycle_status()
    return lock(_LOCK) do
        owner = _OWNED[]
        return (
            endpoint = COMFY_BASE,
            running = _available(COMFY_BASE),
            owned = !isnothing(owner) && _process_stamp(owner.pid) == owner.stamp,
            other_phluxai_borrowers = _has_other_borrowers(),
        )
    end
end

"""Record an interactive session without starting or stopping any service."""
function enter_interactive_repl!()::Nothing
    # A prior interactive session may have stopped an owned server at \exit.
    # Re-entry should restore autostart, but a text-only session remains usable
    # if the optional image service cannot start.
    if isinteractive() && _enabled() && Sys.islinux()
        try
            ensure_default_running!()
        catch error
            @warn "ComfyUI automatic restart unavailable; text-only PhluxAI remains usable" exception = (error, catch_backtrace())
        end
    end
    lock(_LOCK) do
        _ACTIVE_INTERACTIVE_REPLS[] += 1
    end
    return nothing
end

"""
Release an interactive session. Stop an owned ComfyUI when the final active
interactive session closes. Externally started services are never stopped.
"""
function leave_interactive_repl!(; close_on_exit::Bool = true)::Nothing
    should_stop = lock(_LOCK) do
        _ACTIVE_INTERACTIVE_REPLS[] > 0 || throw(ArgumentError("unbalanced ComfyUI interactive REPL lifecycle"))
        _ACTIVE_INTERACTIVE_REPLS[] -= 1
        return close_on_exit && _ACTIVE_INTERACTIVE_REPLS[] == 0
    end
    should_stop && shutdown_owned!()
    return nothing
end

function register_exit_hook!()::Nothing
    lock(_LOCK) do
        if !_EXIT_REGISTERED[]
            atexit(shutdown_owned!)
            _EXIT_REGISTERED[] = true
        end
    end
    return nothing
end

function start_on_import!()::Nothing
    register_exit_hook!()
    if isinteractive() && _enabled() && Sys.islinux()
        try
            ensure_default_running!()
        catch error
            @warn "ComfyUI automatic startup unavailable; text-only PhluxAI remains usable" exception = (error, catch_backtrace())
        end
    end
    return nothing
end

export ensure_default_running!, ensure_backend!, shutdown_owned!, register_exit_hook!, start_on_import!, lifecycle_status, enter_interactive_repl!, leave_interactive_repl!

end # module ComfyLifecycle
