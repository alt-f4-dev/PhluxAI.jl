module ModelSessions

import ..ArtifactStore
import ..CapabilityDiscovery
import ..ImageInterface
import ..JupyTools
import ..OllamaClient
import ..ProtocolTypes
import ..Sandboxing

import ..OllamaClient: ask!, await_compression!, compression_running
import ..ProtocolTypes: ArtifactRef,
                        ModelInput,
                        ModelResponse,
                        ThinkingRequest,
                        ToolExecution,
                        has_tool_calls,
                        normalize_thinking,
                        thinking_label,
                        with_artifacts,
                        with_tool_executions

const DEFAULT_SYSTEM_PROMPT = ""
const DEFAULT_TEMPERATURE = 0.7
const DEFAULT_TIMEOUT = 120
const DEFAULT_POLL_INTERVAL = 2.0
const DEFAULT_MAX_TOOL_STEPS = 8
const _SELF_MODULE = @__MODULE__
const _PARENT_MODULE = parentmodule(_SELF_MODULE)

mutable struct ModelSession
    session::OllamaClient.ChatSession
    temperature::Float64
    thinking::ThinkingRequest
    show_reasoning::Bool
    keep_alive::Union{Nothing,Int,String}
    request_timeout::Float64
    read_idle_timeout::Float64
    tool_mode::JupyTools.ToolMode
    sandbox::Sandboxing.SandboxConfig
    tool_context::Union{JupyTools.ToolContext,Nothing}
    capabilities::Union{CapabilityDiscovery.ModelCapabilities,Nothing}
    image_backend::ImageInterface.AbstractImageBackend
    attachments::Vector{ArtifactRef}
    stop_model_on_close::Bool
    lock::ReentrantLock
    closed::Bool
end

function _temperature(value::Real)::Float64
    converted = Float64(value)
    isfinite(converted) ||
        throw(ArgumentError("temperature must be finite"))
    converted >= 0.0 || throw(ArgumentError(
        "temperature must be nonnegative",
    ))
    return converted
end

function _positive_int(value::Integer, name::String)::Int
    converted = Int(value)
    converted > 0 || throw(ArgumentError("$name must be positive"))
    return converted
end

function _positive_float(value::Real, name::String)::Float64
    converted = Float64(value)
    isfinite(converted) && converted > 0.0 || throw(ArgumentError(
        "$name must be positive and finite",
    ))
    return converted
end

function _nonnegative_timeout(value::Real, name::String)::Float64
    converted = Float64(value)
    isfinite(converted) && converted >= 0.0 || throw(ArgumentError(
        "$name must be nonnegative and finite",
    ))
    return converted
end

function _keep_alive(value)::Union{Nothing,Int,String}
    isnothing(value) && return nothing
    value isa Integer && return Int(value)
    value isa AbstractString || throw(ArgumentError(
        "keep_alive must be nothing, an integer, or a duration string",
    ))
    normalized = String(strip(value))
    isempty(normalized) && throw(ArgumentError(
        "keep_alive string must not be empty",
    ))
    return normalized
end

function _require_open(ms::ModelSession)::Nothing
    ms.closed && throw(ArgumentError("ModelSession is closed"))
    return nothing
end

function _session_dir(
    value::Union{AbstractString,Nothing},
)::Union{String,Nothing}
    isnothing(value) && return nothing
    directory = String(value)
    isempty(strip(directory)) && throw(ArgumentError(
        "session_dir must not be empty",
    ))
    return directory
end

function _artifact_store(
    directory::Union{String,Nothing},
    artifact_dir::Union{AbstractString,Nothing},
)::ArtifactStore.Store
    root = if !isnothing(artifact_dir)
        String(artifact_dir)
    elseif !isnothing(directory)
        joinpath(directory, "artifacts")
    else
        mktempdir(prefix = "phluxai-artifacts-")
    end
    return ArtifactStore.Store(root)
end

function _sandbox_config(
    directory::Union{String,Nothing},
    sandbox_dir::Union{AbstractString,Nothing};
    backend::Sandboxing.SandboxBackend,
    network::Bool,
    timeout_seconds::Real,
    allow_unsafe_local::Bool,
    readonly_paths::AbstractVector{<:AbstractString},
    environment::AbstractDict{<:AbstractString,<:AbstractString},
)::Sandboxing.SandboxConfig
    root = if !isnothing(sandbox_dir)
        String(sandbox_dir)
    elseif !isnothing(directory)
        joinpath(directory, "sandbox")
    else
        mktempdir(prefix = "phluxai-sandbox-")
    end
    return Sandboxing.SandboxConfig(
        root;
        backend,
        network,
        timeout_seconds,
        allow_unsafe_local,
        readonly_paths,
        environment,
    )
end

function _discover_capabilities(
    model::OllamaClient.OllamaModel,
    enabled::Bool,
)::Union{CapabilityDiscovery.ModelCapabilities,Nothing}
    enabled || return nothing
    try
        return CapabilityDiscovery.model_capabilities(
            model.name;
            base_url = model.base_url,
            request_timeout = 10.0,
        )
    catch error
        @debug "Model capability discovery failed" exception = (
            error,
            catch_backtrace(),
        )
        return nothing
    end
end

function _stop_policy(
    stop_model_on_close::Union{Bool,Nothing},
    start_model::Bool,
)::Bool
    return isnothing(stop_model_on_close) ? start_model : stop_model_on_close
end

function ModelSession(
    session::OllamaClient.ChatSession;
    temperature::Real = DEFAULT_TEMPERATURE,
    thinking = nothing,
    show_reasoning::Bool = false,
    keep_alive = nothing,
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    tool_mode::JupyTools.ToolMode = JupyTools.TOOLS_OFF,
    sandbox::Sandboxing.SandboxConfig = Sandboxing.SandboxConfig(
        mktempdir(prefix = "phluxai-sandbox-"),
    ),
    capabilities::Union{
        CapabilityDiscovery.ModelCapabilities,
        Nothing,
    } = nothing,
    image_backend::Union{
        ImageInterface.AbstractImageBackend,
        Nothing,
    } = nothing,
    stop_model_on_close::Bool = false,
)::ModelSession
    selected_image_backend = isnothing(image_backend) ?
        OllamaClient.OllamaImageBackend(session.model) : image_backend
    model_session = ModelSession(
        session,
        _temperature(temperature),
        normalize_thinking(thinking),
        show_reasoning,
        _keep_alive(keep_alive),
        _nonnegative_timeout(request_timeout, "request_timeout"),
        _nonnegative_timeout(read_idle_timeout, "read_idle_timeout"),
        tool_mode,
        sandbox,
        nothing,
        capabilities,
        selected_image_backend,
        ArtifactRef[],
        stop_model_on_close,
        ReentrantLock(),
        false,
    )
    tool_mode == JupyTools.TOOLS_OFF || _ensure_tool_context!(model_session)
    return model_session
end

function ModelSession(
    model_name::AbstractString;
    cfg::OllamaClient.BudgetConfig = OllamaClient.BudgetConfig(
        num_ctx = 16384,
        num_thread = 8,
    ),
    system_prompt::AbstractString = DEFAULT_SYSTEM_PROMPT,
    session_dir::Union{AbstractString,Nothing} = nothing,
    artifact_dir::Union{AbstractString,Nothing} = nothing,
    sandbox_dir::Union{AbstractString,Nothing} = nothing,
    base_url::AbstractString = OllamaClient.OLLAMA_BASE,
    temperature::Real = DEFAULT_TEMPERATURE,
    thinking = nothing,
    show_reasoning::Bool = false,
    persist_reasoning::Bool = false,
    keep_alive = nothing,
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    tool_mode::JupyTools.ToolMode = JupyTools.TOOLS_OFF,
    sandbox_backend::Sandboxing.SandboxBackend = Sandboxing.SANDBOX_AUTO,
    sandbox_network::Bool = false,
    sandbox_timeout::Real = 120.0,
    allow_unsafe_local::Bool = false,
    sandbox_readonly_paths::AbstractVector{<:AbstractString} = String[],
    sandbox_environment::AbstractDict{<:AbstractString,<:AbstractString} =
        Dict{String,String}(),
    discover_capabilities::Bool = true,
    image_backend::Union{
        ImageInterface.AbstractImageBackend,
        Nothing,
    } = nothing,
    timeout::Integer = DEFAULT_TIMEOUT,
    poll_interval::Real = DEFAULT_POLL_INTERVAL,
    start_model::Bool = true,
    stop_model_on_close::Union{Bool,Nothing} = nothing,
    session_type::AbstractString = "model",
)::ModelSession
    model = OllamaClient.OllamaModel(model_name, base_url)
    directory = _session_dir(session_dir)
    store = _artifact_store(directory, artifact_dir)
    sandbox = _sandbox_config(
        directory,
        sandbox_dir;
        backend = sandbox_backend,
        network = sandbox_network,
        timeout_seconds = sandbox_timeout,
        allow_unsafe_local,
        readonly_paths = sandbox_readonly_paths,
        environment = sandbox_environment,
    )
    startup_timeout = _positive_int(timeout, "timeout")
    startup_poll = _positive_float(poll_interval, "poll_interval")
    stop_on_close = _stop_policy(stop_model_on_close, start_model)
    started = false
    try
        if start_model
            OllamaClient.start_model!(
                model.name,
                model.base_url;
                timeout = startup_timeout,
                poll_interval = startup_poll,
            )
            started = true
        end
        chat_session = OllamaClient.ChatSession(
            model,
            String(system_prompt);
            cfg,
            session_dir = directory,
            session_type,
            artifact_store = store,
            persist_reasoning,
        )
        capabilities = _discover_capabilities(
            model,
            discover_capabilities,
        )
        return ModelSession(
            chat_session;
            temperature,
            thinking,
            show_reasoning,
            keep_alive,
            request_timeout,
            read_idle_timeout,
            tool_mode,
            sandbox,
            capabilities,
            image_backend,
            stop_model_on_close = stop_on_close,
        )
    catch
        if started && stop_on_close
            try
                OllamaClient.stop_model!(model)
            catch cleanup_error
                @warn "Model cleanup failed" exception = (
                    cleanup_error,
                    catch_backtrace(),
                )
            end
        end
        rethrow()
    end
end

function load_model_session(
    model_name::AbstractString,
    session_dir::AbstractString;
    cfg::Union{OllamaClient.BudgetConfig,Nothing} = nothing,
    base_url::AbstractString = OllamaClient.OLLAMA_BASE,
    artifact_dir::Union{AbstractString,Nothing} = nothing,
    temperature::Real = DEFAULT_TEMPERATURE,
    thinking = nothing,
    show_reasoning::Bool = false,
    keep_alive = nothing,
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    tool_mode::JupyTools.ToolMode = JupyTools.TOOLS_OFF,
    sandbox_dir::Union{AbstractString,Nothing} = nothing,
    sandbox_backend::Sandboxing.SandboxBackend = Sandboxing.SANDBOX_AUTO,
    sandbox_network::Bool = false,
    sandbox_timeout::Real = 120.0,
    allow_unsafe_local::Bool = false,
    sandbox_readonly_paths::AbstractVector{<:AbstractString} = String[],
    sandbox_environment::AbstractDict{<:AbstractString,<:AbstractString} =
        Dict{String,String}(),
    discover_capabilities::Bool = true,
    image_backend::Union{
        ImageInterface.AbstractImageBackend,
        Nothing,
    } = nothing,
    timeout::Integer = DEFAULT_TIMEOUT,
    poll_interval::Real = DEFAULT_POLL_INTERVAL,
    start_model::Bool = true,
    stop_model_on_close::Union{Bool,Nothing} = nothing,
)::ModelSession
    directory = _session_dir(session_dir)::String
    model = OllamaClient.OllamaModel(model_name, base_url)
    store = _artifact_store(directory, artifact_dir)
    sandbox = _sandbox_config(
        directory,
        sandbox_dir;
        backend = sandbox_backend,
        network = sandbox_network,
        timeout_seconds = sandbox_timeout,
        allow_unsafe_local,
        readonly_paths = sandbox_readonly_paths,
        environment = sandbox_environment,
    )
    stop_on_close = _stop_policy(stop_model_on_close, start_model)
    started = false
    try
        if start_model
            OllamaClient.start_model!(
                model.name,
                model.base_url;
                timeout = _positive_int(timeout, "timeout"),
                poll_interval = _positive_float(
                    poll_interval,
                    "poll_interval",
                ),
            )
            started = true
        end
        session = OllamaClient.load_session(
            model,
            directory;
            artifact_store = store,
        )
        isnothing(cfg) || (session.cfg = cfg)
        capabilities = _discover_capabilities(
            model,
            discover_capabilities,
        )
        return ModelSession(
            session;
            temperature,
            thinking,
            show_reasoning,
            keep_alive,
            request_timeout,
            read_idle_timeout,
            tool_mode,
            sandbox,
            capabilities,
            image_backend,
            stop_model_on_close = stop_on_close,
        )
    catch
        if started && stop_on_close
            try
                OllamaClient.stop_model!(model)
            catch
            end
        end
        rethrow()
    end
end

function _ensure_tool_context!(ms::ModelSession)::JupyTools.ToolContext
    if isnothing(ms.tool_context)
        ms.tool_context = JupyTools.ToolContext(
            ms.sandbox,
            ms.session.artifact_store,
        )
    end
    return ms.tool_context
end

function _approval_callback(
    call::ProtocolTypes.ToolCall,
    risk::JupyTools.ToolRisk,
)::Bool
    println()
    println("Tool request: ", call.name)
    println("Risk: ", risk)
    println("Arguments: ", call.arguments_json)
    print("Approve this tool call? [y/N] ")
    flush(stdout)
    answer = try
        lowercase(strip(readline()))
    catch
        ""
    end
    return answer in ("y", "yes")
end

function respond!(
    ms::ModelSession,
    input::Union{AbstractString,ModelInput};
    temperature::Real = ms.temperature,
    max_tokens::Integer = ms.session.cfg.max_tokens,
    thinking = ms.thinking,
    show_reasoning::Bool = ms.show_reasoning,
    keep_alive = ms.keep_alive,
    request_timeout::Real = ms.request_timeout,
    read_idle_timeout::Real = ms.read_idle_timeout,
    tool_mode::JupyTools.ToolMode = ms.tool_mode,
    approval_callback::Union{Nothing,Function} = nothing,
    max_tool_steps::Integer = DEFAULT_MAX_TOOL_STEPS,
    stream::Bool = true,
    print_tokens::Bool = stream,
    io::IO = stdout,
)::ModelResponse
    model_input = input isa ModelInput ? input : ModelInput(input)
    selected_temperature = _temperature(temperature)
    selected_tokens = _positive_int(max_tokens, "max_tokens")
    selected_thinking = normalize_thinking(thinking)
    selected_keep_alive = _keep_alive(keep_alive)
    selected_request_timeout = _nonnegative_timeout(
        request_timeout,
        "request_timeout",
    )
    selected_read_idle_timeout = _nonnegative_timeout(
        read_idle_timeout,
        "read_idle_timeout",
    )
    steps = _positive_int(max_tool_steps, "max_tool_steps")

    return lock(ms.lock) do
        _require_open(ms)
        registry_specs = ProtocolTypes.ToolSpec[]
        context = nothing
        if tool_mode != JupyTools.TOOLS_OFF
            context = _ensure_tool_context!(ms)
            registry_specs = JupyTools.tool_specs(context.registry)
        end
        response = OllamaClient.respond!(
            ms.session,
            model_input;
            temperature = selected_temperature,
            max_tokens = selected_tokens,
            keep_alive = selected_keep_alive,
            thinking = selected_thinking,
            tools = registry_specs,
            request_timeout = selected_request_timeout,
            read_idle_timeout = selected_read_idle_timeout,
            stream,
            print_tokens,
            print_reasoning = show_reasoning,
            io,
            finalize = false,
        )
        executions = ToolExecution[]
        tool_step = 0
        while has_tool_calls(response)
            tool_step += 1
            tool_step <= steps || begin
                OllamaClient.finalize_interaction!(ms.session)
                throw(ErrorException(
                    "tool loop exceeded max_tool_steps=$steps",
                ))
            end
            context === nothing && break
            for call in response.tool_calls
                execution = JupyTools.execute_tool_call(
                    context,
                    call;
                    mode = tool_mode,
                    approval_callback,
                )
                push!(executions, execution)
                OllamaClient.append_tool_execution!(
                    ms.session,
                    execution,
                )
            end
            response = OllamaClient.continue_response!(
                ms.session;
                temperature = selected_temperature,
                max_tokens = selected_tokens,
                keep_alive = selected_keep_alive,
                thinking = selected_thinking,
                tools = registry_specs,
                request_timeout = selected_request_timeout,
                read_idle_timeout = selected_read_idle_timeout,
                stream,
                print_tokens,
                print_reasoning = show_reasoning,
                io,
                finalize = false,
            )
        end
        OllamaClient.finalize_interaction!(ms.session)
        combined = with_tool_executions(response, executions)
        lock(ms.session.lock) do
            ms.session.last_response = combined
        end
        return combined
    end
end

function ask!(
    ms::ModelSession,
    input::AbstractString;
    kwargs...,
)::String
    return respond!(ms, input; kwargs...).text
end

function _image_dimensions(
    size::AbstractString,
)::Tuple{Int,Int}
    normalized = lowercase(strip(String(size)))
    match_value = match(r"^([1-9][0-9]*)x([1-9][0-9]*)$", normalized)
    isnothing(match_value) && throw(ArgumentError(
        "image size must use WIDTHxHEIGHT notation",
    ))
    return parse(Int, match_value.captures[1]),
           parse(Int, match_value.captures[2])
end

function _select_image_backend(
    ms::ModelSession,
    override::Union{ImageInterface.AbstractImageBackend,Nothing},
)::ImageInterface.AbstractImageBackend
    return isnothing(override) ? ms.image_backend : override
end

function _prepare_image_generation!(
    ms::ModelSession,
    backend::ImageInterface.AbstractImageBackend,
    unload_ollama::Union{Bool,Nothing},
)::Nothing
    local_ollama = OllamaClient.is_loopback_url(
        ms.session.model.base_url,
    )
    should_unload = isnothing(unload_ollama) ?
        ImageInterface.should_unload_ollama(backend, local_ollama) :
        unload_ollama
    should_unload || return nothing
    OllamaClient.stop_model!(
        ms.session.model;
        request_timeout = 10.0,
    )
    return nothing
end

"""
    generate_images!(session, request; image_backend, unload_ollama)

Generate image artifacts through the configured image backend and persist the
turn in the session history.
"""
function generate_images!(
    ms::ModelSession,
    request::ImageInterface.ImageGenerationRequest;
    image_backend::Union{
        ImageInterface.AbstractImageBackend,
        Nothing,
    } = nothing,
    unload_ollama::Union{Bool,Nothing} = nothing,
)::ModelResponse
    return lock(ms.lock) do
        _require_open(ms)
        selected = _select_image_backend(ms, image_backend)
        _prepare_image_generation!(ms, selected, unload_ollama)
        return OllamaClient.generate_images!(
            ms.session,
            selected,
            request,
        )
    end
end

"""
    generate_images!(session, prompt; kwargs...) -> ModelResponse

Convenience image-generation entry point. The default backend is stored on the
session. `backend=:auto|:http|:cli` remains as an Ollama compatibility option;
use `image_backend=` for a concrete ComfyUI or custom backend.
"""
function generate_images!(
    ms::ModelSession,
    prompt::AbstractString;
    image_backend::Union{
        ImageInterface.AbstractImageBackend,
        Nothing,
    } = nothing,
    backend::Union{Symbol,Nothing} = nothing,
    endpoint::AbstractString = "/v1/images/generations",
    size::Union{AbstractString,Nothing} = nothing,
    n::Union{Integer,Nothing} = nothing,
    negative_prompt::Union{AbstractString,Nothing} = nothing,
    width::Union{Integer,Nothing} = nothing,
    height::Union{Integer,Nothing} = nothing,
    count::Integer = 1,
    seed::Union{Integer,Nothing} = nothing,
    steps::Union{Integer,Nothing} = nothing,
    cfg_scale::Union{Real,Nothing} = nothing,
    sampler_name::Union{AbstractString,Nothing} = nothing,
    scheduler::Union{AbstractString,Nothing} = nothing,
    filename_prefix::AbstractString = "phluxai",
    quality::Union{AbstractString,Nothing} = nothing,
    style::Union{AbstractString,Nothing} = nothing,
    user::Union{AbstractString,Nothing} = nothing,
    request_timeout::Real = ms.request_timeout,
    cli_timeout::Real = 300.0,
    unload_ollama::Union{Bool,Nothing} = nothing,
)::ModelResponse
    selected_width = isnothing(width) ? nothing : Int(width)
    selected_height = isnothing(height) ? nothing : Int(height)
    if !isnothing(size)
        isnothing(width) && isnothing(height) || throw(ArgumentError(
            "use either size or width/height, not both",
        ))
        selected_width, selected_height = _image_dimensions(size)
    end
    selected_count = isnothing(n) ? Int(count) : Int(n)
    isnothing(n) || count == 1 || throw(ArgumentError(
        "use either n or count, not both",
    ))

    selected_backend = _select_image_backend(ms, image_backend)
    ollama_keywords = !isnothing(backend) ||
                      endpoint != "/v1/images/generations" ||
                      !isnothing(quality) ||
                      !isnothing(style) ||
                      !isnothing(user) ||
                      request_timeout != ms.request_timeout ||
                      cli_timeout != 300.0
    if ollama_keywords
        isnothing(image_backend) || throw(ArgumentError(
            "backend and image_backend cannot be supplied together",
        ))
        selected_backend = OllamaClient.OllamaImageBackend(
            ms.session.model;
            transport = something(backend, :auto),
            endpoint,
            quality,
            style,
            user,
            request_timeout = _nonnegative_timeout(
                request_timeout,
                "request_timeout",
            ),
            cli_timeout,
        )
    end

    request = ImageInterface.ImageGenerationRequest(
        prompt;
        negative_prompt,
        width = selected_width,
        height = selected_height,
        count = selected_count,
        seed,
        steps,
        cfg_scale,
        sampler_name,
        scheduler,
        filename_prefix,
    )
    return generate_images!(
        ms,
        request;
        image_backend = selected_backend,
        unload_ollama,
    )
end

function set_image_backend!(
    ms::ModelSession,
    backend::ImageInterface.AbstractImageBackend,
)::ModelSession
    lock(ms.lock) do
        _require_open(ms)
        ms.image_backend = backend
    end
    return ms
end


function set_temperature!(
    ms::ModelSession,
    temperature::Real,
)::ModelSession
    value = _temperature(temperature)
    lock(ms.lock) do
        _require_open(ms)
        ms.temperature = value
    end
    return ms
end

function set_thinking!(ms::ModelSession, value)::ModelSession
    normalized = normalize_thinking(value)
    lock(ms.lock) do
        _require_open(ms)
        ms.thinking = normalized
    end
    return ms
end

function set_reasoning_visible!(
    ms::ModelSession,
    visible::Bool,
)::ModelSession
    lock(ms.lock) do
        _require_open(ms)
        ms.show_reasoning = visible
    end
    return ms
end

function set_keep_alive!(ms::ModelSession, value)::ModelSession
    normalized = _keep_alive(value)
    lock(ms.lock) do
        _require_open(ms)
        ms.keep_alive = normalized
    end
    return ms
end

function set_timeouts!(
    ms::ModelSession;
    request_timeout::Union{Real,Nothing} = nothing,
    read_idle_timeout::Union{Real,Nothing} = nothing,
)::ModelSession
    lock(ms.lock) do
        _require_open(ms)
        if !isnothing(request_timeout)
            ms.request_timeout = _nonnegative_timeout(
                request_timeout,
                "request_timeout",
            )
        end
        if !isnothing(read_idle_timeout)
            ms.read_idle_timeout = _nonnegative_timeout(
                read_idle_timeout,
                "read_idle_timeout",
            )
        end
    end
    return ms
end

function set_tool_mode!(
    ms::ModelSession,
    mode::JupyTools.ToolMode,
)::ModelSession
    lock(ms.lock) do
        _require_open(ms)
        mode == JupyTools.TOOLS_OFF || _ensure_tool_context!(ms)
        ms.tool_mode = mode
    end
    return ms
end


"""Install a programmatically configured tool context for this session."""
function set_tool_context!(
    ms::ModelSession,
    context::JupyTools.ToolContext,
)::ModelSession
    lock(ms.lock) do
        _require_open(ms)
        context.artifacts.root == ms.session.artifact_store.root ||
            throw(ArgumentError(
                "tool context must use the session artifact store",
            ))
        ms.sandbox = context.sandbox
        ms.tool_context = context
    end
    return ms
end

function attach!(ms::ModelSession, path::AbstractString)::ArtifactRef
    artifact = lock(ms.lock) do
        _require_open(ms)
        imported = ArtifactStore.import_artifact(
            ms.session.artifact_store,
            path,
        )
        push!(ms.attachments, imported)
        return imported
    end
    return artifact
end

function clear_attachments!(ms::ModelSession)::Nothing
    lock(ms.lock) do
        _require_open(ms)
        empty!(ms.attachments)
    end
    return nothing
end

function pending_attachments(ms::ModelSession)::Vector{ArtifactRef}
    return lock(ms.lock) do
        _require_open(ms)
        return copy(ms.attachments)
    end
end

function save(ms::ModelSession)::Nothing
    lock(ms.lock) do
        _require_open(ms)
        OllamaClient.save_session(ms.session)
    end
    return nothing
end

compression_running(ms::ModelSession)::Bool =
    OllamaClient.compression_running(ms.session)

await_compression!(ms::ModelSession)::Nothing =
    OllamaClient.await_compression!(ms.session)

last_response(ms::ModelSession) = OllamaClient.last_response(ms.session)
last_response_metrics(ms::ModelSession) =
    OllamaClient.last_response_metrics(ms.session)

model_name(ms::ModelSession)::String = ms.session.model.name
base_url(ms::ModelSession)::String = ms.session.model.base_url
session_directory(ms::ModelSession)::Union{String,Nothing} =
    ms.session.session_dir
system_prompt(ms::ModelSession)::String = ms.session.prompt
default_temperature(ms::ModelSession)::Float64 = ms.temperature
default_thinking(ms::ModelSession)::ThinkingRequest = ms.thinking
default_keep_alive(ms::ModelSession) = ms.keep_alive
default_tool_mode(ms::ModelSession)::JupyTools.ToolMode = ms.tool_mode
default_image_backend(
    ms::ModelSession,
)::ImageInterface.AbstractImageBackend = ms.image_backend
artifact_store(ms::ModelSession)::ArtifactStore.Store =
    ms.session.artifact_store
sandbox_config(ms::ModelSession)::Sandboxing.SandboxConfig = ms.sandbox
model_capabilities(ms::ModelSession) = ms.capabilities

function Base.isopen(ms::ModelSession)::Bool
    return lock(ms.lock) do
        return !ms.closed
    end
end

function Base.close(ms::ModelSession)::Nothing
    lock(ms.lock) do
        ms.closed && return nothing
        session_error = nothing
        stop_error = nothing
        try
            OllamaClient.close_session(ms.session)
        catch error
            session_error = error
        end
        if ms.stop_model_on_close
            try
                OllamaClient.stop_model!(ms.session.model)
            catch error
                stop_error = error
            end
        end
        ms.closed = true
        session_error === nothing || throw(session_error)
        stop_error === nothing || throw(stop_error)
    end
    return nothing
end

function _print_artifacts(response::ModelResponse)::Nothing
    isempty(response.artifacts) && isempty(response.tool_executions) &&
        return nothing
    artifacts = copy(response.artifacts)
    for execution in response.tool_executions
        append!(artifacts, execution.artifacts)
    end
    isempty(artifacts) && return nothing
    println("Artifacts:")
    for artifact in artifacts
        println(
            "  ",
            artifact.path,
            " (",
            artifact.mime_type,
            ", ",
            artifact.size_bytes,
            " bytes)",
        )
    end
    return nothing
end

function _tool_mode(value::AbstractString)::JupyTools.ToolMode
    normalized = lowercase(strip(String(value)))
    normalized == "off" && return JupyTools.TOOLS_OFF
    normalized == "ask" && return JupyTools.TOOLS_ASK
    normalized == "auto" && return JupyTools.TOOLS_AUTO
    throw(ArgumentError("tool mode must be off, ask, or auto"))
end

function _show_help()::Nothing
    println("Commands:")
    println("  \\temp [value|default]")
    println("  \\think [off|on|low|medium|high|max|default]")
    println("  \\reasoning [show|hide]")
    println("  \\tools [off|ask|auto]")
    println("  \\imagebackend")
    println("  \\image PROMPT")
    println("  \\attach PATH")
    println("  \\attachments | \\clearattachments")
    println("  \\artifacts | \\sandbox | \\save")
    println("  \\exit | \\quit | \\bye")
    return nothing
end

function _handle_common_command!(
    ms::ModelSession,
    input::AbstractString,
)::Bool
    stripped = String(strip(input))
    startswith(stripped, "\\") || return false
    parts = split(stripped; limit = 2)
    command = lowercase(parts[1])
    argument = length(parts) == 2 ? String(strip(parts[2])) : ""

    if command == "\\help"
        _show_help()
    elseif command == "\\temp"
        if isempty(argument)
            println("Temperature: ", default_temperature(ms))
        elseif lowercase(argument) == "default"
            set_temperature!(ms, DEFAULT_TEMPERATURE)
            println("Temperature reset to ", DEFAULT_TEMPERATURE)
        else
            set_temperature!(ms, parse(Float64, argument))
            println("Temperature set to ", default_temperature(ms))
        end
    elseif command == "\\think"
        if isempty(argument)
            println("Thinking: ", thinking_label(default_thinking(ms)))
        elseif lowercase(argument) == "default"
            set_thinking!(ms, nothing)
            println("Thinking reset to model default")
        else
            set_thinking!(ms, argument)
            println("Thinking set to ", thinking_label(default_thinking(ms)))
        end
    elseif command == "\\reasoning"
        normalized = lowercase(argument)
        normalized in ("show", "on") &&
            set_reasoning_visible!(ms, true)
        normalized in ("hide", "off") &&
            set_reasoning_visible!(ms, false)
        normalized in ("show", "on", "hide", "off") ||
            throw(ArgumentError("use \\reasoning show or hide"))
        println("Reasoning display: ", ms.show_reasoning ? "show" : "hide")
    elseif command == "\\tools"
        if isempty(argument)
            println("Tool mode: ", default_tool_mode(ms))
        else
            set_tool_mode!(ms, _tool_mode(argument))
            println("Tool mode set to ", default_tool_mode(ms))
        end
    elseif command == "\\imagebackend"
        isempty(argument) || throw(ArgumentError(
            "\\imagebackend does not accept an argument",
        ))
        backend = default_image_backend(ms)
        println("Image backend: ", ImageInterface.backend_name(backend))
        println("Local only: ", ImageInterface.is_local_backend(backend))
        println("Resource policy: ", ImageInterface.resource_policy(backend))
    elseif command == "\\image"
        isempty(argument) && throw(ArgumentError(
            "\\image requires a generation prompt",
        ))
        response = generate_images!(ms, argument)
        _print_artifacts(response)
    elseif command == "\\attach"
        isempty(argument) &&
            throw(ArgumentError("\\attach requires a file path"))
        artifact = attach!(ms, argument)
        println("Attached: ", artifact.path)
    elseif command == "\\attachments"
        attachments = pending_attachments(ms)
        isempty(attachments) && println("No pending attachments.")
        for artifact in attachments
            println("  ", artifact.path, " (", artifact.mime_type, ")")
        end
    elseif command == "\\clearattachments"
        clear_attachments!(ms)
        println("Pending attachments cleared.")
    elseif command == "\\artifacts"
        files = ArtifactStore.list_artifacts(ms.session.artifact_store)
        isempty(files) && println("No stored artifacts.")
        foreach(file -> println("  ", file), files)
    elseif command == "\\sandbox"
        println("Workspace: ", ms.sandbox.workspace)
        println("Backend: ", ms.sandbox.backend)
        println("Network: ", ms.sandbox.network)
    elseif command == "\\save"
        save(ms)
        println("Session saved.")
    else
        return false
    end
    return true
end

function _is_exit_command(input::AbstractString)::Bool
    command = lowercase(strip(input))
    return command in ("\\exit", "\\quit", "\\bye")
end

function repl(
    ms::ModelSession;
    close_on_exit::Bool = true,
)::Nothing
    _require_open(ms)
    println("Model ready. Type '\\help' for commands.")
    println("─" ^ 60)
    try
        while true
            print("\nyou> ")
            flush(stdout)
            raw = try
                readline()
            catch error
                if error isa EOFError || error isa InterruptException
                    println()
                    break
                end
                rethrow()
            end
            stripped = String(strip(raw))
            isempty(stripped) && continue
            _is_exit_command(stripped) && break
            try
                _handle_common_command!(ms, stripped) && continue
            catch error
                @warn sprint(showerror, error)
                continue
            end
            attachments = pending_attachments(ms)
            input = isempty(attachments) ?
                ModelInput(stripped) : ModelInput(stripped, attachments)
            print("\nassistant> ")
            flush(stdout)
            try
                response = respond!(
                    ms,
                    input;
                    approval_callback = _approval_callback,
                )
                clear_attachments!(ms)
                _print_artifacts(response)
            catch error
                if error isa InterruptException
                    println("\nRequest interrupted.")
                else
                    @warn "Request failed" exception = (
                        error,
                        catch_backtrace(),
                    )
                end
            end
        end
    finally
        if close_on_exit && isopen(ms)
            println("\nClosing session...")
            close(ms)
            println("Session saved. Goodbye.")
        end
    end
    return nothing
end

macro model(model_name, session_dir = nothing)
    budget_ref = GlobalRef(_PARENT_MODULE, :_take_budget!)
    constructor_ref = GlobalRef(_SELF_MODULE, :ModelSession)
    repl_ref = GlobalRef(_SELF_MODULE, :repl)
    return quote
        local _cfg, _ = $budget_ref()
        local _session = $constructor_ref(
            $(esc(model_name));
            cfg = _cfg,
            session_dir = $(esc(session_dir)),
        )
        $repl_ref(_session)
    end
end

export ModelSession,
       load_model_session,
       respond!,
       ask!,
       generate_images!,
       set_image_backend!,
       set_temperature!,
       set_thinking!,
       set_reasoning_visible!,
       set_keep_alive!,
       set_timeouts!,
       set_tool_mode!,
       set_tool_context!,
       attach!,
       clear_attachments!,
       pending_attachments,
       save,
       repl,
       compression_running,
       await_compression!,
       last_response,
       last_response_metrics,
       model_name,
       base_url,
       session_directory,
       system_prompt,
       default_temperature,
       default_thinking,
       default_keep_alive,
       default_tool_mode,
       default_image_backend,
       artifact_store,
       sandbox_config,
       model_capabilities,
       @model

end # module ModelSessions
