# PhluxAI.jl/deps/build.jl
#
# Runs automatically on `Pkg.add("PhluxAI")` and `Pkg.build("PhluxAI")`.
# Installs or verifies Ollama, jupy-cli, comfy-cli, and local ComfyUI.
# Model weights remain user-selected and are not downloaded here.

include(joinpath(@__DIR__, "install_jupy.jl"))
include(joinpath(@__DIR__, "install_comfy.jl"))

const OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"
const OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
const PHLUXAI_PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

_check_ollama()::Bool = !isnothing(Sys.which("ollama"))
_has_curl()::Bool = !isnothing(Sys.which("curl"))
_has_brew()::Bool = !isnothing(Sys.which("brew"))
_has_winget()::Bool = !isnothing(Sys.which("winget"))

function _env_flag(name::String, default::Bool = false)::Bool
    value = get(ENV, name, nothing)
    isnothing(value) && return default
    normalized = lowercase(strip(String(value)))
    return !(normalized in ("", "0", "false", "no", "off"))
end

function _is_wsl()::Bool
    proc_version = "/proc/version"
    isfile(proc_version) || return false
    content = lowercase(read(proc_version, String))
    return occursin("microsoft", content) || occursin("wsl", content)
end

function _run_install(
    command::Cmd;
    description::String,
)::Bool
    try
        run(command)
        return true
    catch error
        @warn "$description failed" exception = (
            error,
            catch_backtrace(),
        )
        return false
    end
end

function _install_linux_ollama()::Nothing
    @info "PhluxAI: installing Ollama with the official Linux script"
    if !_has_curl()
        @warn(
            "PhluxAI: curl is unavailable; install Ollama manually",
            download_url = OLLAMA_DOWNLOAD_URL,
        )
        return nothing
    end

    script = tempname() * ".sh"
    try
        downloaded = _run_install(
            `curl -fsSL $OLLAMA_INSTALL_URL -o $script`;
            description = "Ollama installer download",
        )
        downloaded || return nothing
        installed = _run_install(
            `sh $script`;
            description = "Ollama installation",
        )
        installed || return nothing
    finally
        isfile(script) && rm(script; force = true)
    end

    if _check_ollama()
        @info "PhluxAI: Ollama installed" path = Sys.which("ollama")
    else
        @warn(
            "Ollama installer completed, but ollama is not on PATH",
            expected_path = "/usr/local/bin/ollama",
        )
    end
    return nothing
end

function _install_wsl_ollama()::Nothing
    @info "PhluxAI: WSL environment detected"
    _install_linux_ollama()
    if _check_ollama()
        @info(
            "Start Ollama manually when the WSL service is inactive",
            command = "ollama serve",
        )
    end
    return nothing
end

function _install_macos_ollama()::Nothing
    @info "PhluxAI: installing Ollama on macOS"
    if !_has_brew()
        @warn(
            "Homebrew was not found; install Ollama manually",
            download_url = OLLAMA_DOWNLOAD_URL,
        )
        return nothing
    end

    success = _run_install(
        `brew install --cask ollama`;
        description = "Homebrew Ollama installation",
    )
    success || return nothing
    @info "Launch Ollama once from Applications before using PhluxAI"
    return nothing
end

function _install_windows_ollama()::Nothing
    @info "PhluxAI: installing Ollama on Windows"
    if !_has_winget()
        @warn(
            "WinGet was not found; install Ollama manually",
            download_url = OLLAMA_DOWNLOAD_URL,
        )
        return nothing
    end

    _run_install(
        `winget install --id Ollama.Ollama -e --silent`;
        description = "WinGet Ollama installation",
    )
    return nothing
end

function _install_ollama()::Nothing
    if _check_ollama()
        @info(
            "PhluxAI: Ollama is already installed",
            path = Sys.which("ollama"),
        )
    elseif Sys.islinux() && _is_wsl()
        _install_wsl_ollama()
    elseif Sys.islinux()
        _install_linux_ollama()
    elseif Sys.isapple()
        _install_macos_ollama()
    elseif Sys.iswindows()
        _install_windows_ollama()
    else
        @warn(
            "Unrecognized operating system; install Ollama manually",
            download_url = OLLAMA_DOWNLOAD_URL,
        )
    end
    return nothing
end

function _install_hybrid_runtime()::Nothing
    jupy = PhluxAIJupyInstaller.install_jupy!(PHLUXAI_PROJECT_ROOT)
    PhluxAIComfyInstaller.install_comfy!(
        PHLUXAI_PROJECT_ROOT;
        jupip_path = jupy.jupip,
    )
    return nothing
end

_env_flag("PHLUXAI_SKIP_OLLAMA_INSTALL") || _install_ollama()
_env_flag("PHLUXAI_SKIP_COMFY_INSTALL") || _install_hybrid_runtime()
