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

"""Serialize ownership/lease changes across Julia processes on Linux.
The registry lock is never deleted: unlinking a lock file can split lock domains.
"""
function _with_registry_lock(f::Function)
    Sys.islinux() || return f()
    path = joinpath(_lease_directory(), "registry.lock")
    return open(path, "a+") do io
        fd = Base.fd(io)
        ccall(:flock, Cint, (Cint, Cint), fd, 2) == 0 || throw(SystemError("flock registry lock", Base.Libc.errno()))
        try
            return f()
        finally
            ccall(:flock, Cint, (Cint, Cint), fd, 8) == 0 || @warn "Unable to unlock ComfyUI registry"
        end
    end
end

_owner_path()::String = joinpath(_lease_directory(), "managed-owner")

"""Remember the exact server PID/start time so the last borrower can stop it."""
function _write_managed_owner!(owner)::Nothing
    path = _owner_path()
    temporary = tempname(dirname(path))
    try
        open(temporary, "w") do io
            println(io, owner.pid, " ", owner.stamp)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return nothing
end

function _read_managed_owner()::Union{Nothing,NamedTuple{(:pid, :stamp),Tuple{Int,String}}}
    path = _owner_path()
    isfile(path) || return nothing
    fields = split(strip(read(path, String)))
    length(fields) == 2 || return nothing
    pid = tryparse(Int, fields[1])
    isnothing(pid) && return nothing
    pid > 0 || return nothing
    isempty(fields[2]) && return nothing
    return (pid = pid, stamp = String(fields[2]))
end

"""Wait for the HTTP API after CLI launch without relinquishing ownership identity."""
function _wait_for_api(url::AbstractString; timeout::Real = 30.0, poll_interval::Real = 0.25)::Bool
    deadline = time() + timeout
    while true
        _available(url) && return true
        time() >= deadline && return false
        sleep(min(poll_interval, max(0.0, deadline - time())))
    end
end

"""Stop only a process whose PID, start time, and comfy-cli metadata all match."""
function _stop_verified_owner!(owner)::Bool
    _process_stamp(owner.pid) == owner.stamp || return false
    _background_matches(owner) || return false
    _stop_matches(owner) || return false
    result = _cli(["stop"]; deadline = 25)
    data = get(result, :data, nothing)
    return !isnothing(data) && Int(get(data, :pid, 0)) == owner.pid && get(data, :stopped, false) === true
end

"""Complete a managed server lifetime only after the final registered borrower exits."""
function _finish_managed_owner!(owner; borrowers::Function = _has_other_borrowers, stop::Function = _stop_verified_owner!)::Symbol
    borrowers() && return :borrowed
    if _process_stamp(owner.pid) != owner.stamp
        isfile(_owner_path()) && rm(_owner_path(); force = true)
        return :stale
    end
    stop(owner) || return :unverified
    isfile(_owner_path()) && rm(_owner_path(); force = true)
    return :stopped
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
        _with_registry_lock() do
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
            stamp = _process_stamp(pid)
            isnothing(stamp) && throw(ErrorException(
                "ComfyUI started, but its process identity cannot be verified; inspect the service manually",
            ))
            owner = (pid = pid, stamp = stamp)
            # Persist identity before readiness polling. If cleanup fails,
            # the exit hook can still safely retry the exact owned PID.
            _write_managed_owner!(owner)
            _OWNED[] = owner
            _register_lease!()
            if !_wait_for_api(COMFY_BASE)
                stopped = try
                    _stop_verified_owner!(owner)
                catch error
                    @warn "ComfyUI startup cleanup failed; inspect the server manually" exception = (error, catch_backtrace())
                    false
                end
                if stopped
                    isfile(_owner_path()) && rm(_owner_path(); force = true)
                    _OWNED[] = nothing
                    _release_lease!()
                else
                    @warn "ComfyUI API did not become ready; ownership retained for safe exit cleanup" pid
                end
                throw(ErrorException("ComfyUI API did not become ready within 30 seconds of successful CLI launch"))
            end
            return :owned
        end
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
        try
            _with_registry_lock() do
                owner = isnothing(_OWNED[]) ? _read_managed_owner() : _OWNED[]
                # Drop our own lease while holding the cross-process registry lock.
                # Another process cannot register itself between this check and stop.
                _release_lease!()
                if !isnothing(owner)
                    result = _finish_managed_owner!(owner)
                    if result == :borrowed
                        @info "ComfyUI left running for another PhluxAI process; last borrower will stop it"
                    elseif result == :unverified
                        @warn "Unable to verify or stop PhluxAI-managed ComfyUI; inspect the service manually" pid = owner.pid
                    end
                end
            end
        catch error
            @warn "Unable to finalize PhluxAI ComfyUI lifecycle" exception = (error, catch_backtrace())
            try
                _release_lease!()
            catch cleanup_error
                @warn "Unable to remove ComfyUI lifecycle lease" exception = (cleanup_error, catch_backtrace())
            end
        finally
            _OWNED[] = nothing
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
    lock(_LOCK) do
        # Startup and increment must be atomic against final-session shutdown.
        if isinteractive() && _enabled() && Sys.islinux()
            try
                ensure_default_running!()
            catch error
                @warn "ComfyUI automatic restart unavailable; text-only PhluxAI remains usable" exception = (error, catch_backtrace())
            end
        end
        _ACTIVE_INTERACTIVE_REPLS[] += 1
    end
    return nothing
end

"""
Release an interactive session. Stop managed ComfyUI when the final session
closes; if another process still borrows it, that borrower will stop it later.
"""
function _leave_interactive_repl!(close_on_exit::Bool, stop_owned::Function)::Nothing
    lock(_LOCK) do
        _ACTIVE_INTERACTIVE_REPLS[] > 0 || throw(ArgumentError("unbalanced ComfyUI interactive REPL lifecycle"))
        _ACTIVE_INTERACTIVE_REPLS[] -= 1
        if close_on_exit && _ACTIVE_INTERACTIVE_REPLS[] == 0
            stop_owned()
        end
    end
    return nothing
end

function leave_interactive_repl!(; close_on_exit::Bool = true)::Nothing
    return _leave_interactive_repl!(close_on_exit, shutdown_owned!)
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
