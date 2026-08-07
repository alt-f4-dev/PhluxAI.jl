module PhluxAIJupyInstaller

const DEFAULT_REPOSITORY =
    "https://github.com/alt-f4-dev/jupy-cli.git"
const DEFAULT_REF =
    "ac6e2339fa49c2b5d390d83d5fbc213fe614a74c"

function _command_path(name::String)::Union{String,Nothing}
    resolved = Sys.which(name)
    !isnothing(resolved) && return String(resolved)

    if Sys.iswindows()
        local_app_data = get(
            ENV,
            "LOCALAPPDATA",
            joinpath(homedir(), "AppData", "Local"),
        )
        candidate = joinpath(
            local_app_data,
            "Jupy",
            "bin",
            "$name.cmd",
        )
    else
        bin_directory = get(
            ENV,
            "JUPY_BIN_DIR",
            joinpath(homedir(), ".local", "bin"),
        )
        candidate = joinpath(bin_directory, name)
    end

    return isfile(candidate) ? candidate : nothing
end

function _windows_command_line(values::Vector{String})::String
    quoted = String[]
    for value in values
        escaped = replace(value, '"' => "\"\"")
        push!(quoted, "\"$escaped\"")
    end
    return join(quoted, ' ')
end

function _command(
    executable::String,
    arguments::Vector{String};
    dir::Union{String,Nothing} = nothing,
)::Cmd
    values = String[executable; arguments]
    if Sys.iswindows() && endswith(lowercase(executable), ".cmd")
        command_processor = get(ENV, "COMSPEC", "cmd.exe")
        values = String[
            command_processor,
            "/D",
            "/S",
            "/C",
            _windows_command_line(values),
        ]
    end
    command = Cmd(values)
    return isnothing(dir) ? command : Cmd(command; dir = dir)
end

function _run(command::Cmd; quiet::Bool = false)::Bool
    selected = quiet ? pipeline(
        command;
        stdout = devnull,
        stderr = devnull,
    ) : command

    try
        run(selected)
        return true
    catch error
        @debug "Jupy command failed" command exception = (
            error,
            catch_backtrace(),
        )
        return false
    end
end

function _read(command::Cmd)::Union{String,Nothing}
    try
        return strip(read(command, String))
    catch error
        @debug "Jupy command output could not be read" command exception = (
            error,
            catch_backtrace(),
        )
        return nothing
    end
end

function _tool_version(command::String)::Union{String,Nothing}
    return _read(_command(command, ["--jupy-tool-version"]))
end

function _installed_core_path()::String
    configured = get(ENV, "JUPY_CORE", nothing)
    !isnothing(configured) && return abspath(
        normpath(expanduser(String(configured))),
    )

    if Sys.iswindows()
        local_app_data = get(
            ENV,
            "LOCALAPPDATA",
            joinpath(homedir(), "AppData", "Local"),
        )
        return joinpath(
            local_app_data,
            "Jupy",
            "core",
            "jupy_core.py",
        )
    end

    data_home = get(
        ENV,
        "XDG_DATA_HOME",
        joinpath(homedir(), ".local", "share"),
    )
    install_root = get(
        ENV,
        "JUPY_INSTALL_ROOT",
        joinpath(data_home, "jupy"),
    )
    return joinpath(install_root, "core", "jupy_core.py")
end

function _supports_build_safe_jupip()::Bool
    core = _installed_core_path()
    isfile(core) || return false
    try
        return occursin(
            "JUPY_SKIP_JULIA_SETUP",
            read(core, String),
        )
    catch
        return false
    end
end

function _installed_commands()::Union{NamedTuple,Nothing}
    jupy = _command_path("jupy")
    jupip = _command_path("jupip")
    isnothing(jupy) && return nothing
    isnothing(jupip) && return nothing

    jupy_version = _tool_version(jupy)
    jupip_version = _tool_version(jupip)
    isnothing(jupy_version) && return nothing
    isnothing(jupip_version) && return nothing
    jupy_version == jupip_version || return nothing
    _supports_build_safe_jupip() || return nothing

    return (
        jupy = jupy,
        jupip = jupip,
        version = jupy_version,
    )
end

function _git_executable()::String
    executable = Sys.which("git")
    isnothing(executable) && error(
        "PhluxAI requires Git to install jupy-cli and ComfyUI.",
    )
    return String(executable)
end

function _checkout_jupy(
    destination::String,
    repository::String,
    reference::String,
)::Nothing
    git = _git_executable()
    mkpath(destination)

    commands = (
        Cmd([git, "-C", destination, "init", "--quiet"]),
        Cmd([
            git,
            "-C",
            destination,
            "remote",
            "add",
            "origin",
            repository,
        ]),
        Cmd([
            git,
            "-C",
            destination,
            "fetch",
            "--depth=1",
            "origin",
            reference,
        ]),
        Cmd([
            git,
            "-C",
            destination,
            "checkout",
            "--detach",
            "FETCH_HEAD",
        ]),
    )

    for command in commands
        _run(command) || error(
            "Failed to retrieve jupy-cli from $repository " *
            "at $reference.",
        )
    end
    return nothing
end

function _install_from_checkout(checkout::String)::Nothing
    if Sys.iswindows()
        powershell = Sys.which("powershell.exe")
        isnothing(powershell) &&
            (powershell = Sys.which("powershell"))
        isnothing(powershell) && error(
            "PowerShell is required to install jupy-cli on Windows.",
        )
        script = joinpath(checkout, "install.ps1")
        isfile(script) || error(
            "jupy-cli install.ps1 was not found.",
        )
        command = _command(
            String(powershell),
            [
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script,
            ];
            dir = checkout,
        )
    else
        bash = Sys.which("bash")
        isnothing(bash) && error(
            "Bash is required to install jupy-cli.",
        )
        script = joinpath(checkout, "install.sh")
        isfile(script) || error(
            "jupy-cli install.sh was not found.",
        )
        command = _command(
            String(bash),
            [script];
            dir = checkout,
        )
    end

    _run(command) || error("The jupy-cli installer failed.")
    return nothing
end

"""Install or update user-local `jupy` and `jupip` commands."""
function install_jupy!(project_root::AbstractString)::NamedTuple
    root = abspath(normpath(String(project_root)))
    isfile(joinpath(root, "Project.toml")) || error(
        "No Project.toml was found at the PhluxAI project root: $root",
    )

    installed = _installed_commands()
    if !isnothing(installed)
        @info(
            "PhluxAI: jupy-cli is ready",
            version = installed.version,
            jupy = installed.jupy,
            jupip = installed.jupip,
        )
        return installed
    end

    repository = get(
        ENV,
        "PHLUXAI_JUPY_REPOSITORY",
        DEFAULT_REPOSITORY,
    )
    reference = get(ENV, "PHLUXAI_JUPY_REF", DEFAULT_REF)
    @info "PhluxAI: installing jupy-cli" repository reference

    mktempdir(prefix = "phluxai-jupy-install-") do temporary
        checkout = joinpath(temporary, "jupy-cli")
        _checkout_jupy(checkout, repository, reference)
        _install_from_checkout(checkout)
    end

    installed = _installed_commands()
    isnothing(installed) && error(
        "jupy-cli installed, but jupy/jupip did not pass " *
        "validation. Confirm that the installed jupy_core.py " *
        "includes JUPY_SKIP_JULIA_SETUP support.",
    )

    @info(
        "PhluxAI: jupy-cli installed",
        version = installed.version,
        jupy = installed.jupy,
        jupip = installed.jupip,
    )
    return installed
end

export install_jupy!

end # module PhluxAIJupyInstaller
