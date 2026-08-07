"""Local Ollama framework for text, reasoning, multimodal, and tool sessions."""
module PhluxAI

using JSON3

include("core/ProtocolTypes.jl")
include("core/Artifacts.jl")
include("core/ModelCapabilities.jl")
include("core/ImageInterface.jl")
include("tools/Sandbox.jl")
include("tools/JupyTools.jl")
include("core/OllamaClient.jl")
include("core/ComfyClient.jl")
include("core/ModelSession.jl")
include("agents/PhysicalAssistant.jl")

import .ArtifactStore
import .CapabilityDiscovery
import .ComfyClient
import .ImageInterface
import .JupyTools
import .ModelSessions
import .OllamaClient
import .PhysicalAssistant
import .ProtocolTypes
import .Sandboxing

import .OllamaClient: ask!, await_compression!, compression_running

using .ProtocolTypes: ArtifactPart,
                      ArtifactRef,
                      ChatMessage,
                      ModelInput,
                      ModelResponse,
                      ResponseMetrics,
                      TextPart,
                      ThinkingLevel,
                      ToolCall,
                      ToolExecution,
                      ToolSpec,
                      THINK_HIGH,
                      THINK_LOW,
                      THINK_MAX,
                      THINK_MEDIUM,
                      generation_rate,
                      has_reasoning,
                      has_tool_calls,
                      prompt_rate,
                      response_images

using .OllamaClient: BudgetConfig,
                     OllamaImageBackend,
                     OllamaModel,
                     effective_response_budget,
                     history_budget,
                     compression_trigger

using .ModelSessions: ModelSession
using .ImageInterface: AbstractImageBackend,
                       ImageGenerationRequest,
                       ImageResourcePolicy,
                       NoImageBackend,
                       IMAGE_AUTO,
                       IMAGE_KEEP_LOADED,
                       IMAGE_UNLOAD_OLLAMA,
                       backend_name,
                       is_local_backend,
                       resource_policy,
                       should_unload_ollama
using .ComfyClient: COMFY_BASE,
                    ComfyBackend,
                    ComfyBindings,
                    WorkflowBinding

using .JupyTools: ToolContext,
                  ToolMode,
                  ToolRegistry,
                  ToolRisk,
                  TOOLS_AUTO,
                  TOOLS_ASK,
                  TOOLS_OFF,
                  TOOL_EXECUTE,
                  TOOL_NETWORK,
                  TOOL_READ,
                  TOOL_WRITE,
                  default_registry,
                  register!
using .Sandboxing: ProcessResult,
                   SandboxBackend,
                   SandboxConfig,
                   SANDBOX_AUTO,
                   SANDBOX_BWRAP,
                   SANDBOX_LOCAL
using .CapabilityDiscovery: ModelCapabilities,
                            ModelDetails
using .PhysicalAssistant: AssistantMode,
                          CODE,
                          DERIVE,
                          DRAFT,
                          GENERAL,
                          PhysicalAgent,
                          PhysicsAssistant

const PHLUXAI_VERSION = v"0.2.0"
const _MODEL_SESSION_TYPE = "model"
const _PHYSICS_SESSION_TYPE = "physics"
const _GENERIC_SESSION_TYPE = "generic"

# -----------------------------------------------------------------------------
# Unified high-level API
# -----------------------------------------------------------------------------

respond!(session::ModelSession, input; kwargs...) =
    ModelSessions.respond!(session, input; kwargs...)

respond!(assistant::PhysicsAssistant, input; kwargs...) =
    PhysicalAssistant.respond!(assistant, input; kwargs...)

generate_images(
    backend::AbstractImageBackend,
    request::ImageGenerationRequest,
    store::ArtifactStore.Store,
)::ModelResponse = ImageInterface.generate_images(
    backend,
    request,
    store,
)

generate_images(
    backend::AbstractImageBackend,
    prompt::AbstractString;
    artifact_store::ArtifactStore.Store,
    kwargs...,
)::ModelResponse = ImageInterface.generate_images(
    backend,
    prompt,
    artifact_store;
    kwargs...,
)

generate_images(
    model::OllamaModel,
    prompt::AbstractString;
    kwargs...,
)::ModelResponse = OllamaClient.generate_images(model, prompt; kwargs...)

generate_images!(
    session::ModelSession,
    request::ImageGenerationRequest;
    kwargs...,
)::ModelResponse = ModelSessions.generate_images!(
    session,
    request;
    kwargs...,
)

generate_images!(
    session::ModelSession,
    prompt::AbstractString;
    kwargs...,
)::ModelResponse = ModelSessions.generate_images!(
    session,
    prompt;
    kwargs...,
)

generate_images!(
    assistant::PhysicsAssistant,
    request::ImageGenerationRequest;
    kwargs...,
)::ModelResponse = PhysicalAssistant.generate_images!(
    assistant,
    request;
    kwargs...,
)

generate_images!(
    assistant::PhysicsAssistant,
    prompt::AbstractString;
    kwargs...,
)::ModelResponse = PhysicalAssistant.generate_images!(
    assistant,
    prompt;
    kwargs...,
)

load_comfy_workflow(path::AbstractString)::Dict{String,Any} =
    ComfyClient.load_workflow(path)

comfy_available(backend::ComfyBackend)::Bool =
    ComfyClient.available(backend)

save(session::ModelSession)::Nothing = ModelSessions.save(session)
save(assistant::PhysicsAssistant)::Nothing =
    PhysicalAssistant.save(assistant)

repl(session::ModelSession; kwargs...)::Nothing =
    ModelSessions.repl(session; kwargs...)

repl(assistant::PhysicsAssistant; kwargs...)::Nothing =
    PhysicalAssistant.repl(assistant; kwargs...)

load_model_session(args...; kwargs...) =
    ModelSessions.load_model_session(args...; kwargs...)

load_assistant(args...; kwargs...) =
    PhysicalAssistant.load_assistant(args...; kwargs...)

load_physical_agent(args...; kwargs...) =
    PhysicalAssistant.load_physical_agent(args...; kwargs...)

model_name(value::ModelSession)::String =
    ModelSessions.model_name(value)
model_name(value::PhysicsAssistant)::String =
    PhysicalAssistant.model_name(value)

base_url(value::ModelSession)::String = ModelSessions.base_url(value)
base_url(value::PhysicsAssistant)::String =
    PhysicalAssistant.base_url(value)

session_directory(value::ModelSession) =
    ModelSessions.session_directory(value)
session_directory(value::PhysicsAssistant) =
    PhysicalAssistant.session_directory(value)

system_prompt(value::ModelSession)::String =
    ModelSessions.system_prompt(value)
system_prompt(value::PhysicsAssistant)::String =
    PhysicalAssistant.system_prompt(value)

default_temperature(value::ModelSession)::Float64 =
    ModelSessions.default_temperature(value)
default_temperature(value::PhysicsAssistant)::Float64 =
    PhysicalAssistant.default_temperature(value)

default_thinking(value::ModelSession) =
    ModelSessions.default_thinking(value)
default_thinking(value::PhysicsAssistant) =
    PhysicalAssistant.default_thinking(value)

default_keep_alive(value::ModelSession) =
    ModelSessions.default_keep_alive(value)
default_keep_alive(value::PhysicsAssistant) =
    PhysicalAssistant.default_keep_alive(value)

default_tool_mode(value::ModelSession) =
    ModelSessions.default_tool_mode(value)
default_tool_mode(value::PhysicsAssistant) =
    PhysicalAssistant.default_tool_mode(value)

last_response(value::ModelSession) =
    ModelSessions.last_response(value)
last_response(value::PhysicsAssistant) =
    PhysicalAssistant.last_response(value)

last_response_metrics(value::ModelSession) =
    ModelSessions.last_response_metrics(value)
last_response_metrics(value::PhysicsAssistant) =
    PhysicalAssistant.last_response_metrics(value)

set_temperature!(value::ModelSession, temperature)::ModelSession =
    ModelSessions.set_temperature!(value, temperature)
set_temperature!(value::PhysicsAssistant, temperature)::PhysicsAssistant =
    PhysicalAssistant.set_temperature!(value, temperature)

set_thinking!(value::ModelSession, thinking)::ModelSession =
    ModelSessions.set_thinking!(value, thinking)
set_thinking!(value::PhysicsAssistant, thinking)::PhysicsAssistant =
    PhysicalAssistant.set_thinking!(value, thinking)

set_reasoning_visible!(
    value::ModelSession,
    visible::Bool,
)::ModelSession = ModelSessions.set_reasoning_visible!(value, visible)

set_reasoning_visible!(
    value::PhysicsAssistant,
    visible::Bool,
)::PhysicsAssistant = PhysicalAssistant.set_reasoning_visible!(
    value,
    visible,
)

set_keep_alive!(value::ModelSession, policy)::ModelSession =
    ModelSessions.set_keep_alive!(value, policy)
set_keep_alive!(value::PhysicsAssistant, policy)::PhysicsAssistant =
    PhysicalAssistant.set_keep_alive!(value, policy)

set_timeouts!(value::ModelSession; kwargs...)::ModelSession =
    ModelSessions.set_timeouts!(value; kwargs...)
set_timeouts!(value::PhysicsAssistant; kwargs...)::PhysicsAssistant =
    PhysicalAssistant.set_timeouts!(value; kwargs...)

set_tool_mode!(
    value::ModelSession,
    mode::JupyTools.ToolMode,
)::ModelSession = ModelSessions.set_tool_mode!(value, mode)

set_tool_mode!(
    value::PhysicsAssistant,
    mode::JupyTools.ToolMode,
)::PhysicsAssistant = PhysicalAssistant.set_tool_mode!(value, mode)

set_tool_context!(
    value::ModelSession,
    context::JupyTools.ToolContext,
)::ModelSession = ModelSessions.set_tool_context!(value, context)

set_tool_context!(
    value::PhysicsAssistant,
    context::JupyTools.ToolContext,
)::PhysicsAssistant = PhysicalAssistant.set_tool_context!(value, context)

set_image_backend!(
    value::ModelSession,
    backend::AbstractImageBackend,
)::ModelSession = ModelSessions.set_image_backend!(value, backend)

set_image_backend!(
    value::PhysicsAssistant,
    backend::AbstractImageBackend,
)::PhysicsAssistant = PhysicalAssistant.set_image_backend!(value, backend)

default_image_backend(value::ModelSession) =
    ModelSessions.default_image_backend(value)
default_image_backend(value::PhysicsAssistant) =
    PhysicalAssistant.default_image_backend(value)

set_mode!(value::PhysicsAssistant, mode::AssistantMode)::PhysicsAssistant =
    PhysicalAssistant.set_mode!(value, mode)

default_mode(value::PhysicsAssistant)::AssistantMode =
    PhysicalAssistant.default_mode(value)

attach!(value::ModelSession, path::AbstractString) =
    ModelSessions.attach!(value, path)
attach!(value::PhysicsAssistant, path::AbstractString) =
    PhysicalAssistant.attach!(value, path)

clear_attachments!(value::ModelSession)::Nothing =
    ModelSessions.clear_attachments!(value)
clear_attachments!(value::PhysicsAssistant)::Nothing =
    PhysicalAssistant.clear_attachments!(value)

pending_attachments(value::ModelSession) =
    ModelSessions.pending_attachments(value)
pending_attachments(value::PhysicsAssistant) =
    PhysicalAssistant.pending_attachments(value)

artifact_store(value::ModelSession) = ModelSessions.artifact_store(value)
artifact_store(value::PhysicsAssistant) =
    PhysicalAssistant.artifact_store(value)

sandbox_config(value::ModelSession) = ModelSessions.sandbox_config(value)
sandbox_config(value::PhysicsAssistant) =
    PhysicalAssistant.sandbox_config(value)

model_capabilities(value::ModelSession) =
    ModelSessions.model_capabilities(value)
model_capabilities(value::PhysicsAssistant) =
    PhysicalAssistant.model_capabilities(value)
model_capabilities(model::AbstractString; kwargs...) =
    CapabilityDiscovery.model_capabilities(model; kwargs...)
show_model_details(model::AbstractString; kwargs...) =
    CapabilityDiscovery.show_model_details(model; kwargs...)
supports(capabilities::ModelCapabilities, capability::Symbol)::Bool =
    CapabilityDiscovery.supports(capabilities, capability)
require_capability(
    capabilities::ModelCapabilities,
    capability::Symbol,
)::Nothing = CapabilityDiscovery.require_capability(
    capabilities,
    capability,
)
capability_names(capabilities::ModelCapabilities)::Vector{String} =
    CapabilityDiscovery.capability_names(capabilities)

# -----------------------------------------------------------------------------
# One-shot budget configuration
# -----------------------------------------------------------------------------

const _BUDGET_DEFAULT = BudgetConfig(num_ctx = 16384, num_thread = 8)
const _BUDGET = Ref{BudgetConfig}(_BUDGET_DEFAULT)
const _BUDGET_EXPLICIT = Ref{Bool}(false)
const _BUDGET_LOCK = ReentrantLock()

function _replace_budget(
    cfg::BudgetConfig;
    num_ctx::Integer = cfg.num_ctx,
    system_budget::Integer = cfg.system_budget,
    summary_budget::Integer = cfg.summary_budget,
    response_budget::Integer = cfg.response_budget,
    safety_margin::Integer = cfg.safety_margin,
    compression_frac::Real = cfg.compression_frac,
    m_verbatim::Integer = cfg.m_verbatim,
    exact_threshold::Integer = cfg.exact_threshold,
    num_thread::Integer = cfg.num_thread,
    max_tokens::Integer = cfg.max_tokens,
)::BudgetConfig
    return BudgetConfig(
        num_ctx = num_ctx,
        system_budget = system_budget,
        summary_budget = summary_budget,
        response_budget = response_budget,
        safety_margin = safety_margin,
        compression_frac = compression_frac,
        m_verbatim = m_verbatim,
        exact_threshold = exact_threshold,
        num_thread = num_thread,
        max_tokens = max_tokens,
    )
end

function _set_budget!(; kwargs...)::BudgetConfig
    return lock(_BUDGET_LOCK) do
        cfg = _replace_budget(_BUDGET[]; kwargs...)
        _BUDGET[] = cfg
        _BUDGET_EXPLICIT[] = true
        return cfg
    end
end

function _take_budget!()::Tuple{BudgetConfig,Bool}
    return lock(_BUDGET_LOCK) do
        cfg = _BUDGET[]
        explicit = _BUDGET_EXPLICIT[]
        _BUDGET[] = _BUDGET_DEFAULT
        _BUDGET_EXPLICIT[] = false
        return cfg, explicit
    end
end

macro budget(assignments...)
    isempty(assignments) && error(
        "@budget requires at least one field=value assignment",
    )
    allowed = Set(fieldnames(BudgetConfig))
    keywords = Expr[]
    for assignment in assignments
        assignment isa Expr || error(
            "@budget expects field=value assignments",
        )
        assignment.head in (:(=), :kw) || error(
            "@budget expects field=value assignments",
        )
        field = assignment.args[1]
        field isa Symbol || error(
            "@budget field names must be symbols",
        )
        field in allowed || error("unknown BudgetConfig field '$field'")
        push!(keywords, Expr(:kw, field, esc(assignment.args[2])))
    end
    setter = GlobalRef(@__MODULE__, :_set_budget!)
    return Expr(:call, setter, Expr(:parameters, keywords...))
end

# -----------------------------------------------------------------------------
# Image-model configuration
# -----------------------------------------------------------------------------

const _DEFAULT_IMAGE_WORKFLOW = Ref{Union{ComfyBackend,Nothing}}(nothing)
const _PENDING_IMAGE_BACKEND = Ref{
    Union{AbstractImageBackend,Nothing}
}(nothing)
const _IMAGE_CONFIG_LOCK = ReentrantLock()

"""
    set_default_image_workflow!(workflow, bindings; kwargs...) -> ComfyBackend

Configure the model-neutral ComfyUI workflow used when an image model is
selected by name through `@imagemodel` or `image="..."` on `@model` and
`@agent`.

`bindings.model` must identify the workflow input that receives the explicit
ComfyUI model filename, for example `WorkflowBinding("4", "ckpt_name")`.
The workflow is validated immediately but no ComfyUI request is made.
"""
function set_default_image_workflow!(
    workflow::Union{AbstractString,AbstractDict},
    bindings::ComfyBindings;
    base_url::AbstractString = COMFY_BASE,
    poll_interval::Real = 0.25,
    timeout::Real = 300.0,
    request_timeout::Real = 30.0,
    loopback_only::Bool = true,
    resource_policy::ImageResourcePolicy = IMAGE_AUTO,
)::ComfyBackend
    isnothing(bindings.model) && throw(ArgumentError(
        "default image workflow requires bindings.model",
    ))
    backend = ComfyBackend(
        workflow,
        bindings;
        base_url,
        poll_interval,
        timeout,
        request_timeout,
        loopback_only,
        resource_policy,
    )
    lock(_IMAGE_CONFIG_LOCK) do
        _DEFAULT_IMAGE_WORKFLOW[] = backend
    end
    return backend
end

"""Return a copy of the configured default image workflow, or `nothing`."""
function default_image_workflow()::Union{ComfyBackend,Nothing}
    return lock(_IMAGE_CONFIG_LOCK) do
        backend = _DEFAULT_IMAGE_WORKFLOW[]
        return isnothing(backend) ? nothing :
               ComfyClient.with_model(backend, nothing)
    end
end

"""Remove the configured default image workflow."""
function clear_default_image_workflow!()::Nothing
    lock(_IMAGE_CONFIG_LOCK) do
        _DEFAULT_IMAGE_WORKFLOW[] = nothing
    end
    return nothing
end

_resolve_image_backend(
    backend::AbstractImageBackend,
)::AbstractImageBackend = backend

_resolve_image_backend(::Nothing)::Nothing = nothing

function _resolve_image_backend(model::AbstractString)::ComfyBackend
    selected = String(strip(model))
    isempty(selected) && throw(ArgumentError(
        "image model name must not be empty",
    ))
    return lock(_IMAGE_CONFIG_LOCK) do
        backend = _DEFAULT_IMAGE_WORKFLOW[]
        isnothing(backend) && throw(ArgumentError(
            "no default image workflow is configured; call " *
            "set_default_image_workflow!(workflow, bindings) first",
        ))
        return ComfyClient.with_model(backend, selected)
    end
end

function _set_image_model!(
    value::Union{AbstractString,AbstractImageBackend},
)::AbstractImageBackend
    backend = _resolve_image_backend(value)
    lock(_IMAGE_CONFIG_LOCK) do
        _PENDING_IMAGE_BACKEND[] = backend
    end
    return backend
end

function _take_image_backend!()::Union{AbstractImageBackend,Nothing}
    return lock(_IMAGE_CONFIG_LOCK) do
        backend = _PENDING_IMAGE_BACKEND[]
        _PENDING_IMAGE_BACKEND[] = nothing
        return backend
    end
end

"""
    @imagemodel image

Select the image backend consumed by the next `@model` or `@agent` call.
A string is interpreted as an explicit ComfyUI model filename and requires a
default workflow configured by `set_default_image_workflow!`. A concrete
`AbstractImageBackend` may also be supplied directly.
"""
macro imagemodel(image)
    setter = GlobalRef(@__MODULE__, :_set_image_model!)
    return :($setter($(esc(image))))
end

function _parse_session_image_args(
    macro_name::String,
    args::Tuple,
)::Tuple{Any,Any,Bool}
    positional = Any[]
    image_expr = nothing
    has_image = false
    for argument in args
        if argument isa Expr && argument.head in (:(=), :kw)
            length(argument.args) == 2 || error(
                "$macro_name received an invalid keyword assignment",
            )
            name = argument.args[1]
            name isa Symbol || error(
                "$macro_name keyword names must be symbols",
            )
            name == :image || error(
                "$macro_name supports only the image= keyword",
            )
            has_image && error(
                "$macro_name image= may be specified only once",
            )
            image_expr = argument.args[2]
            has_image = true
        else
            push!(positional, argument)
        end
    end
    length(positional) <= 1 || error(
        "$macro_name accepts at most one session_dir positional argument",
    )
    session_dir = isempty(positional) ? :(nothing) : positional[1]
    return session_dir, image_expr, has_image
end

# -----------------------------------------------------------------------------
# Construction and resume
# -----------------------------------------------------------------------------

function _build_agent(
    ::Val{:physics},
    model_name::AbstractString,
    cfg::BudgetConfig,
    session_dir::Union{AbstractString,Nothing};
    image_backend::Union{AbstractImageBackend,Nothing} = nothing,
)::PhysicsAssistant
    return PhysicsAssistant(
        model_name;
        cfg,
        session_dir,
        image_backend,
    )
end

function _build_agent(
    ::Val{profile},
    ::AbstractString,
    ::BudgetConfig,
    ::Union{AbstractString,Nothing};
    image_backend::Union{AbstractImageBackend,Nothing} = nothing,
) where {profile}
    throw(ArgumentError(
        "unknown agent profile :$profile; available profiles: :physics",
    ))
end

function _build_model_session(
    model_name::AbstractString,
    cfg::BudgetConfig,
    session_dir::Union{AbstractString,Nothing};
    image_backend::Union{AbstractImageBackend,Nothing} = nothing,
)::ModelSession
    return ModelSession(
        model_name;
        cfg,
        session_dir,
        image_backend,
    )
end

function _session_metadata(
    session_dir::AbstractString,
)::NamedTuple{(:session_type, :prompt),Tuple{String,String}}
    path = joinpath(String(session_dir), "hot.json")
    isfile(path) || error("No hot store found at $path")
    data = JSON3.read(read(path, String))
    return (
        session_type = String(get(data, "session_type", "generic")),
        prompt = String(get(data, "prompt", "")),
    )
end

function _resume_session(
    model_name::AbstractString,
    session_dir::AbstractString;
    cfg::Union{BudgetConfig,Nothing} = nothing,
)
    metadata = _session_metadata(session_dir)
    kind = metadata.session_type
    legacy_physics = kind == _GENERIC_SESSION_TYPE &&
        PhysicalAssistant._same_physics_prompt(metadata.prompt)
    if kind == _PHYSICS_SESSION_TYPE || legacy_physics
        return load_assistant(model_name, session_dir; cfg)
    elseif kind in (_MODEL_SESSION_TYPE, _GENERIC_SESSION_TYPE)
        session = load_model_session(model_name, session_dir; cfg)
        session.session.session_type = _MODEL_SESSION_TYPE
        return session
    end
    throw(ArgumentError(
        "unsupported persisted session type '$kind' at $session_dir",
    ))
end

macro model(model_name, args...)
    session_dir, image_expr, has_image = _parse_session_image_args(
        "@model",
        args,
    )
    take_budget = GlobalRef(@__MODULE__, :_take_budget!)
    take_image = GlobalRef(@__MODULE__, :_take_image_backend!)
    resolve_image = GlobalRef(@__MODULE__, :_resolve_image_backend)
    build = GlobalRef(@__MODULE__, :_build_model_session)
    run_repl = GlobalRef(@__MODULE__, :repl)
    if has_image
        return quote
            local _cfg, _ = $take_budget()
            local _image_backend = $resolve_image($(esc(image_expr)))
            $take_image()
            local _session = $build(
                $(esc(model_name)),
                _cfg,
                $(esc(session_dir));
                image_backend = _image_backend,
            )
            $run_repl(_session)
        end
    end
    return quote
        local _cfg, _ = $take_budget()
        local _image_backend = $take_image()
        local _session = $build(
            $(esc(model_name)),
            _cfg,
            $(esc(session_dir));
            image_backend = _image_backend,
        )
        $run_repl(_session)
    end
end

macro agent(profile, model_name, args...)
    session_dir, image_expr, has_image = _parse_session_image_args(
        "@agent",
        args,
    )
    take_budget = GlobalRef(@__MODULE__, :_take_budget!)
    take_image = GlobalRef(@__MODULE__, :_take_image_backend!)
    resolve_image = GlobalRef(@__MODULE__, :_resolve_image_backend)
    build = GlobalRef(@__MODULE__, :_build_agent)
    run_repl = GlobalRef(@__MODULE__, :repl)
    if has_image
        return quote
            local _cfg, _ = $take_budget()
            local _image_backend = $resolve_image($(esc(image_expr)))
            $take_image()
            local _assistant = $build(
                Val($(esc(profile))),
                $(esc(model_name)),
                _cfg,
                $(esc(session_dir));
                image_backend = _image_backend,
            )
            $run_repl(_assistant)
        end
    end
    return quote
        local _cfg, _ = $take_budget()
        local _image_backend = $take_image()
        local _assistant = $build(
            Val($(esc(profile))),
            $(esc(model_name)),
            _cfg,
            $(esc(session_dir));
            image_backend = _image_backend,
        )
        $run_repl(_assistant)
    end
end

macro resume(model_name, session_dir)
    take_budget = GlobalRef(@__MODULE__, :_take_budget!)
    resume_session = GlobalRef(@__MODULE__, :_resume_session)
    run_repl = GlobalRef(@__MODULE__, :repl)
    return quote
        local _cfg, _explicit = $take_budget()
        local _session = $resume_session(
            $(esc(model_name)),
            $(esc(session_dir));
            cfg = _explicit ? _cfg : nothing,
        )
        $run_repl(_session)
    end
end

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------

function _banner_enabled()::Bool
    setting = lowercase(strip(get(ENV, "PHLUXAI_BANNER", "auto")))
    setting in ("0", "false", "no", "off", "never") && return false
    setting in ("1", "true", "yes", "on", "always") && return true
    return isinteractive()
end

function welcome(
    io::IO = stdout;
    check_ollama::Bool = true,
)::Nothing
    C3 = "\033[38;2;90;0;160m"
    C4 = "\033[38;2;120;20;190m"
    C5 = "\033[38;2;150;50;220m"
    C6 = "\033[38;2;180;90;240m"
    C7 = "\033[38;2;200;130;255m"
    C8 = "\033[38;2;0;180;0m"
    C9 = "\033[38;2;40;210;10m"
    C10 = "\033[38;2;57;255;20m"
    JG = "\033[38;2;100;221;23m"
    JR = "\033[38;2;220;50;47m"
    JP = "\033[38;2;149;88;178m"
    R = "\033[0m"
    B = "\033[1m"
    println(io, "$(B)$(C3)▄▖$(C4)▌ $(C5)▜ $(C6)    " *
                "$(C8)▄▖$(C8)▄▖$(R)    $(JG)_$(R)")
    println(io, "$(B)$(C4)▙▌$(C5)▛▌$(C6)▐ $(C6)▌▌$(C6)▚▘" *
                "$(C9)▌▌$(C9)▐$(R)   " *
                "$(JR)_$(JG)($(JG)_$(R)$(JG))$(JP)_$(R)")
    println(io, "$(B)$(C5)▌ $(C6)▌▌$(C7)▐▖$(C7)▙▌$(C7)▞▖" *
                "$(C10)▛▌$(C10)▟▖$(R) " *
                "$(JR)(_$(R)$(JR))$(R) $(JP)(_$(R)$(JP))$(R)")
    println(io)
    println(io, "$(B)$(C10)PhluxAI v$(PHLUXAI_VERSION)$(R) — " *
                "Local multimodal LLM agent framework")
    println(io, "Docs: https://github.com/alt-f4-dev/PhluxAI.jl")
    println(io)
    println(io, "Quick start:")
    println(io, "  $(C10)@model \"gpt-oss:20b\"$(R)")
    println(io, "  $(C10)@imagemodel \"sd_xl_base_1.0.safetensors\"$(R)")
    println(io, "  $(C10)@agent :physics \"gpt-oss:20b\"$(R)")
    println(io, "  Interactive commands: \\help, \\think, \\tools, " *
                "\\attach, \\temp")
    println(io)
    if check_ollama && isnothing(Sys.which("ollama"))
        @warn "Ollama was not found on PATH; automatic local startup " *
              "is unavailable."
    end
    return nothing
end

function __init__()::Nothing
    _banner_enabled() && welcome()
    return nothing
end

include("core/precompile.jl")

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

export @model, @agent, @resume, @budget, @imagemodel
export PHLUXAI_VERSION, welcome

export ArtifactRef,
       ArtifactPart,
       TextPart,
       ModelInput,
       ChatMessage,
       ModelResponse,
       ResponseMetrics,
       ToolSpec,
       ToolCall,
       ToolExecution,
       ThinkingLevel,
       THINK_LOW,
       THINK_MEDIUM,
       THINK_HIGH,
       THINK_MAX,
       has_reasoning,
       has_tool_calls,
       response_images,
       generation_rate,
       prompt_rate

export OllamaModel,
       OllamaImageBackend,
       BudgetConfig,
       effective_response_budget,
       history_budget,
       compression_trigger

export AbstractImageBackend,
       NoImageBackend,
       ImageGenerationRequest,
       ImageResourcePolicy,
       IMAGE_KEEP_LOADED,
       IMAGE_UNLOAD_OLLAMA,
       IMAGE_AUTO,
       ComfyBackend,
       ComfyBindings,
       WorkflowBinding,
       COMFY_BASE,
       ModelSession,
       PhysicsAssistant,
       PhysicalAgent,
       AssistantMode,
       GENERAL,
       DERIVE,
       CODE,
       DRAFT

export respond!,
       ask!,
       generate_images,
       generate_images!,
       load_comfy_workflow,
       comfy_available,
       set_default_image_workflow!,
       default_image_workflow,
       clear_default_image_workflow!,
       save,
       repl,
       load_model_session,
       load_assistant,
       load_physical_agent,
       compression_running,
       await_compression!,
       last_response,
       last_response_metrics

export set_temperature!,
       default_temperature,
       set_thinking!,
       default_thinking,
       set_reasoning_visible!,
       set_mode!,
       default_mode,
       set_keep_alive!,
       set_timeouts!,
       default_keep_alive,
       set_tool_mode!,
       set_tool_context!,
       default_tool_mode,
       set_image_backend!,
       default_image_backend,
       backend_name,
       is_local_backend,
       resource_policy,
       should_unload_ollama,
       attach!,
       clear_attachments!,
       pending_attachments

export model_name,
       base_url,
       session_directory,
       system_prompt,
       artifact_store,
       sandbox_config,
       model_capabilities,
       show_model_details,
       supports,
       require_capability,
       capability_names

export ArtifactStore,
       CapabilityDiscovery,
       ImageInterface,
       ComfyClient,
       Sandboxing,
       JupyTools,
       OllamaClient,
       ProtocolTypes

export ToolMode,
       TOOLS_OFF,
       TOOLS_ASK,
       TOOLS_AUTO,
       ToolRisk,
       TOOL_READ,
       TOOL_WRITE,
       TOOL_EXECUTE,
       TOOL_NETWORK,
       ToolRegistry,
       ToolContext,
       default_registry,
       register!,
       SandboxBackend,
       SandboxConfig,
       ProcessResult,
       SANDBOX_AUTO,
       SANDBOX_BWRAP,
       SANDBOX_LOCAL,
       ModelCapabilities,
       ModelDetails

end # module PhluxAI
