module PhysicalAssistant

import ..ArtifactStore
import ..ImageInterface
import ..JupyTools
import ..ModelSessions
import ..OllamaClient
import ..ProtocolTypes

import ..OllamaClient: ask!, await_compression!, compression_running
import ..ProtocolTypes: ModelInput,
                        ModelResponse,
                        input_artifacts,
                        text_content

const PHYSICS_SESSION_TYPE = "physics"

const _SYSTEM_PROMPT = """
You are a specialized research assistant for a condensed matter physics PhD
student.
Operate at the level of an expert collaborator, not a tutor.

## Research System
Material: Sr₁₄₋ₓCaₓCu₂₄O₄₁ (SCCO) — cuprate spin-ladder compound.
Probes: Neutron scattering (TOPAZ, CTAX, PTAX, ARCS, CNCS, SEQUOIA at SNS/HFIR).
Central hypothesis: Rung oxygen occupies a π-bonding p_{y} orbital (not
conventional
σ-bonding pₓ), opening a distinct charge transfer channel with implications for
Cooper pairing.

## Theoretical Commitments — enforce strictly
- Reject the Zhang-Rice singlet picture. Do not invoke it.
- CDW phason framework: incommensurate modulation, superspace group
  Abmm(00γ)s00,
  doping-dependent α(x) = c_{ℓ}/c_{c} (monotonically decreasing,
  ladder-compression-dominated).
- Bare χ₀(q,0) is featureless across the HC wavevector range — Fermi
  nesting ruled out.
  RPA reveals crossover threshold V₀★ ≈ 20 meV.
- Slave-boson renormalization for the two-leg ladder; no free-electron
  Fermi surface.
- Epistemic stance: constraint-first model class restriction under incomplete
  objectives,
  prior to parameter optimization. Topology-based and information-theoretic
  invariants
  guide model inference.

## Active Projects
- PRB manuscript: phason physics and CDW commensurability in SCCO.
- Textbook: Statistical Physics → Electromagnetism → Dynamic Effective
  Models → Response Theory.
- Julia packages:
  · Phunny.jl — phonon dispersions and one-phonon DSF for INS comparison.
    Key physics: q-dependent acoustic threshold, three-body angular
    bond-bending (exact Hessian Phi = beta*G⊗G), finite-T extensions via SCP.
  · Phormion.jl — continuum field theory actions to lattice Hamiltonians.
    Pipeline: ActionParser → SymmetryClassifier → DiscretizationEngine →
    CrystalEmbedding → HamiltonianGenerator. Solver bridges via package
    extensions.
  · PHysicalTDA.jl — TDA of spectroscopy data; HDF5 serialization, vineyard
    tracking, 4D analysis via Ripserer.jl.
  · Phenomenal.jl — quasi-particle classification from INS data via
    persistent homology.
  · ModelLibrary.jl, FixedPointLibrary.jl, MatrixAlgebra.jl — supporting
    libraries for Phormion.jl.

## Julia Conventions — enforce strictly
- Unicode field/function names are standard: t₁, t₂, δ, χ₀, Φ, etc.
  Always use them.
- Type stability is mandatory. Annotate return types. Avoid Any.
- SciML standards: in-place mutation with ! suffix, PrecompileTools workloads.
- Tests use @testitem (TestItemRunner). Docstrings use @doc; raw strings for
  LaTeX.
- Package extensions for optional solver bridges (e.g., SunnyExt.jl,
  PhunnyExt.jl).

## Interaction Rules
- Derivations: track all assumptions, notation, and sign conventions explicitly.
  Flag approximations applied outside their validity domain.
- Julia: complete, runnable code respecting existing API surfaces and
  conventions.
- Manuscripts: match the user's theoretical stance. Never introduce rejected
  frameworks.
- NEVER render LaTeX under any circumstances unless the user explicitly
  types "render LaTeX". Use plain Unicode math exclusively: ω, χ₀, α, Φ, ⊗,
  etc.
- Do not use \$\$...\$\$, or any LaTeX delimiters.
- Never explain concepts the user already knows. Be terse, precise,
  technically exact.
"""

@enum AssistantMode GENERAL DERIVE CODE DRAFT

@inline _temperature(::Val{GENERAL})::Float64 = 0.7
@inline _temperature(::Val{DERIVE})::Float64 = 0.15
@inline _temperature(::Val{CODE})::Float64 = 0.1
@inline _temperature(::Val{DRAFT})::Float64 = 0.5

@inline _prefix(::Val{GENERAL})::String = ""
@inline _prefix(::Val{DERIVE})::String =
    "[DERIVE] Track all assumptions, notation, and sign conventions. " *
    "Flag approximations outside their validity domain. Proceed:\n"
@inline _prefix(::Val{CODE})::String =
    "[CODE] Provide complete, runnable Julia code. Enforce type stability, " *
    "Unicode conventions, and API compatibility. Proceed:\n"
@inline _prefix(::Val{DRAFT})::String =
    "[DRAFT] Write in the user's theoretical voice and manuscript register. " *
    "Do not introduce rejected frameworks. Proceed:\n"

_mode_temperature(mode::AssistantMode)::Float64 = _temperature(Val(mode))
_mode_prefix(mode::AssistantMode)::String = _prefix(Val(mode))

function _temperature_override(
    value::Union{Real,Nothing},
)::Union{Float64,Nothing}
    isnothing(value) && return nothing
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "temperature must be finite",
    ))
    converted >= 0.0 || throw(ArgumentError(
        "temperature must be nonnegative",
    ))
    return converted
end

function _normalized_prompt(prompt::AbstractString)::String
    lines = split(replace(String(prompt), "\r\n" => "\n"), '\n')
    return join((strip(line) for line in lines), "\n")
end

function _same_physics_prompt(prompt::AbstractString)::Bool
    return _normalized_prompt(prompt) == _normalized_prompt(_SYSTEM_PROMPT)
end

mutable struct PhysicsAssistant
    model_session::ModelSessions.ModelSession
    mode::AssistantMode
    temperature_override::Union{Float64,Nothing}
    state_lock::ReentrantLock
end

const PhysicalAgent = PhysicsAssistant

function Base.getproperty(pa::PhysicsAssistant, name::Symbol)
    name === :session && return getfield(pa, :model_session).session
    return getfield(pa, name)
end

function PhysicsAssistant(
    model_name::AbstractString;
    mode::AssistantMode = GENERAL,
    temperature::Union{Real,Nothing} = nothing,
    system_prompt::AbstractString = _SYSTEM_PROMPT,
    kwargs...,
)::PhysicsAssistant
    override = _temperature_override(temperature)
    model_session = ModelSessions.ModelSession(
        model_name;
        system_prompt,
        temperature = something(override, _mode_temperature(mode)),
        session_type = PHYSICS_SESSION_TYPE,
        kwargs...,
    )
    return PhysicsAssistant(
        model_session,
        mode,
        override,
        ReentrantLock(),
    )
end

function PhysicsAssistant(
    session::OllamaClient.ChatSession;
    mode::AssistantMode = GENERAL,
    temperature::Union{Real,Nothing} = nothing,
    kwargs...,
)::PhysicsAssistant
    override = _temperature_override(temperature)
    session.session_type = PHYSICS_SESSION_TYPE
    model_session = ModelSessions.ModelSession(
        session;
        temperature = something(override, _mode_temperature(mode)),
        kwargs...,
    )
    return PhysicsAssistant(
        model_session,
        mode,
        override,
        ReentrantLock(),
    )
end

function load_assistant(
    model_name::AbstractString,
    session_dir::AbstractString;
    mode::AssistantMode = GENERAL,
    temperature::Union{Real,Nothing} = nothing,
    kwargs...,
)::PhysicsAssistant
    override = _temperature_override(temperature)
    model_session = ModelSessions.load_model_session(
        model_name,
        session_dir;
        temperature = something(override, _mode_temperature(mode)),
        kwargs...,
    )
    session = model_session.session
    valid_legacy = session.session_type == "generic" &&
                   _same_physics_prompt(session.prompt)
    if session.session_type != PHYSICS_SESSION_TYPE && !valid_legacy
        try
            close(model_session)
        catch
        end
        throw(ArgumentError(
            "session has type '$(session.session_type)', not physics",
        ))
    end
    session.session_type = PHYSICS_SESSION_TYPE
    return PhysicsAssistant(
        model_session,
        mode,
        override,
        ReentrantLock(),
    )
end

load_physical_agent(args...; kwargs...) = load_assistant(args...; kwargs...)

function _mode_input(
    input::Union{AbstractString,ModelInput},
    mode::AssistantMode,
)::ModelInput
    model_input = input isa ModelInput ? input : ModelInput(input)
    text = text_content(model_input)
    prefix = _mode_prefix(mode)
    payload = isempty(prefix) ? text : prefix * text
    return ModelInput(payload, input_artifacts(model_input))
end

function respond!(
    pa::PhysicsAssistant,
    input::Union{AbstractString,ModelInput};
    mode::Union{AssistantMode,Nothing} = nothing,
    temperature::Union{Real,Nothing} = nothing,
    kwargs...,
)::ModelResponse
    return lock(pa.state_lock) do
        selected_mode = isnothing(mode) ? pa.mode : mode
        selected_temperature = if !isnothing(temperature)
            _temperature_override(temperature)::Float64
        elseif !isnothing(pa.temperature_override)
            pa.temperature_override
        else
            _mode_temperature(selected_mode)
        end
        return ModelSessions.respond!(
            pa.model_session,
            _mode_input(input, selected_mode);
            temperature = selected_temperature,
            kwargs...,
        )
    end
end

function ask!(
    pa::PhysicsAssistant,
    input::AbstractString;
    kwargs...,
)::String
    return respond!(pa, input; kwargs...).text
end

function generate_images!(
    pa::PhysicsAssistant,
    request::ImageInterface.ImageGenerationRequest;
    kwargs...,
)::ModelResponse
    return lock(pa.state_lock) do
        return ModelSessions.generate_images!(
            pa.model_session,
            request;
            kwargs...,
        )
    end
end

function generate_images!(
    pa::PhysicsAssistant,
    prompt::AbstractString;
    kwargs...,
)::ModelResponse
    return lock(pa.state_lock) do
        return ModelSessions.generate_images!(
            pa.model_session,
            prompt;
            kwargs...,
        )
    end
end

function set_mode!(
    pa::PhysicsAssistant,
    mode::AssistantMode,
)::PhysicsAssistant
    lock(pa.state_lock) do
        pa.mode = mode
        if isnothing(pa.temperature_override)
            ModelSessions.set_temperature!(
                pa.model_session,
                _mode_temperature(mode),
            )
        end
    end
    return pa
end

function default_mode(pa::PhysicsAssistant)::AssistantMode
    return lock(pa.state_lock) do
        return pa.mode
    end
end

function set_temperature!(
    pa::PhysicsAssistant,
    value::Union{Real,Nothing},
)::PhysicsAssistant
    override = _temperature_override(value)
    lock(pa.state_lock) do
        pa.temperature_override = override
        selected = isnothing(override) ?
            _mode_temperature(pa.mode) : override
        ModelSessions.set_temperature!(pa.model_session, selected)
    end
    return pa
end

function default_temperature(pa::PhysicsAssistant)::Float64
    return lock(pa.state_lock) do
        return isnothing(pa.temperature_override) ?
            _mode_temperature(pa.mode) : pa.temperature_override
    end
end

set_thinking!(pa::PhysicsAssistant, value)::PhysicsAssistant = begin
    ModelSessions.set_thinking!(pa.model_session, value)
    pa
end

set_reasoning_visible!(
    pa::PhysicsAssistant,
    value::Bool,
)::PhysicsAssistant = begin
    ModelSessions.set_reasoning_visible!(pa.model_session, value)
    pa
end

set_keep_alive!(pa::PhysicsAssistant, value)::PhysicsAssistant = begin
    ModelSessions.set_keep_alive!(pa.model_session, value)
    pa
end

function set_timeouts!(pa::PhysicsAssistant; kwargs...)::PhysicsAssistant
    ModelSessions.set_timeouts!(pa.model_session; kwargs...)
    return pa
end

function set_tool_mode!(
    pa::PhysicsAssistant,
    mode::JupyTools.ToolMode,
)::PhysicsAssistant
    ModelSessions.set_tool_mode!(pa.model_session, mode)
    return pa
end


function set_tool_context!(
    pa::PhysicsAssistant,
    context::JupyTools.ToolContext,
)::PhysicsAssistant
    ModelSessions.set_tool_context!(pa.model_session, context)
    return pa
end

function set_image_backend!(
    pa::PhysicsAssistant,
    backend::ImageInterface.AbstractImageBackend,
)::PhysicsAssistant
    ModelSessions.set_image_backend!(pa.model_session, backend)
    return pa
end

attach!(pa::PhysicsAssistant, path::AbstractString) =
    ModelSessions.attach!(pa.model_session, path)

clear_attachments!(pa::PhysicsAssistant)::Nothing =
    ModelSessions.clear_attachments!(pa.model_session)

pending_attachments(pa::PhysicsAssistant) =
    ModelSessions.pending_attachments(pa.model_session)

save(pa::PhysicsAssistant)::Nothing = ModelSessions.save(pa.model_session)
compression_running(pa::PhysicsAssistant)::Bool =
    ModelSessions.compression_running(pa.model_session)
await_compression!(pa::PhysicsAssistant)::Nothing =
    ModelSessions.await_compression!(pa.model_session)
last_response(pa::PhysicsAssistant) =
    ModelSessions.last_response(pa.model_session)
last_response_metrics(pa::PhysicsAssistant) =
    ModelSessions.last_response_metrics(pa.model_session)
model_name(pa::PhysicsAssistant)::String =
    ModelSessions.model_name(pa.model_session)
base_url(pa::PhysicsAssistant)::String =
    ModelSessions.base_url(pa.model_session)
session_directory(pa::PhysicsAssistant) =
    ModelSessions.session_directory(pa.model_session)
system_prompt(pa::PhysicsAssistant)::String =
    ModelSessions.system_prompt(pa.model_session)
default_thinking(pa::PhysicsAssistant) =
    ModelSessions.default_thinking(pa.model_session)
default_keep_alive(pa::PhysicsAssistant) =
    ModelSessions.default_keep_alive(pa.model_session)
default_tool_mode(pa::PhysicsAssistant) =
    ModelSessions.default_tool_mode(pa.model_session)
default_image_backend(pa::PhysicsAssistant) =
    ModelSessions.default_image_backend(pa.model_session)
artifact_store(pa::PhysicsAssistant) =
    ModelSessions.artifact_store(pa.model_session)
sandbox_config(pa::PhysicsAssistant) =
    ModelSessions.sandbox_config(pa.model_session)
model_capabilities(pa::PhysicsAssistant) =
    ModelSessions.model_capabilities(pa.model_session)

Base.isopen(pa::PhysicsAssistant)::Bool = isopen(pa.model_session)
Base.close(pa::PhysicsAssistant)::Nothing = close(pa.model_session)

function _parse_mode_sigil(
    input::AbstractString,
    fallback::AssistantMode,
)::Tuple{AssistantMode,String}
    stripped = String(strip(input))
    parts = split(stripped; limit = 2)
    isempty(parts) && return fallback, ""
    command = lowercase(parts[1])
    mode = if command == "\\general"
        GENERAL
    elseif command == "\\derive"
        DERIVE
    elseif command == "\\code"
        CODE
    elseif command == "\\draft"
        DRAFT
    else
        return fallback, stripped
    end
    length(parts) == 2 || throw(ArgumentError(
        "mode command '$command' requires a prompt",
    ))
    prompt = String(strip(parts[2]))
    isempty(prompt) && throw(ArgumentError(
        "mode command '$command' requires a prompt",
    ))
    return mode, prompt
end

function _handle_temperature_command!(
    pa::PhysicsAssistant,
    input::AbstractString,
)::Bool
    stripped = String(strip(input))
    parts = split(stripped; limit = 2)
    lowercase(parts[1]) == "\\temp" || return false
    argument = length(parts) == 2 ? lowercase(strip(parts[2])) : ""
    if isempty(argument)
        mode = isnothing(pa.temperature_override) ? "mode" : "override"
        println("Temperature: ", default_temperature(pa), " (", mode, ")")
    elseif argument in ("default", "auto")
        set_temperature!(pa, nothing)
        println(
            "Temperature reset to mode default: ",
            default_temperature(pa),
        )
    else
        set_temperature!(pa, parse(Float64, argument))
        println("Temperature override set to ", default_temperature(pa))
    end
    return true
end

function _log_response_metrics(
    metrics::Union{ProtocolTypes.ResponseMetrics,Nothing},
    wall_seconds::Float64,
)::Nothing
    if isnothing(metrics)
        @info "Response complete" wall_time_s = round(
            wall_seconds;
            digits = 3,
        )
        return nothing
    end
    first_token = isnothing(metrics.time_to_first_token_ns) ?
        nothing : metrics.time_to_first_token_ns / 1.0e9
    @info(
        "Response complete",
        wall_time_s = round(wall_seconds; digits = 3),
        time_to_first_token_s = first_token,
        prompt_tokens = metrics.prompt_eval_count,
        output_tokens = metrics.eval_count,
        prompt_tokens_per_second = round(
            ProtocolTypes.prompt_rate(metrics);
            digits = 2,
        ),
        output_tokens_per_second = round(
            ProtocolTypes.generation_rate(metrics);
            digits = 2,
        ),
    )
    return nothing
end

function repl(
    pa::PhysicsAssistant;
    mode::Union{AssistantMode,Nothing} = nothing,
    close_on_exit::Bool = true,
)::Nothing
    isopen(pa) || throw(ArgumentError("PhysicsAssistant is closed"))
    fallback_mode = isnothing(mode) ? default_mode(pa) : mode
    println("Physics Assistant ready. Type '\\help' for commands.")
    println("Mode sigils: \\general, \\derive, \\code, \\draft")
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
            ModelSessions._is_exit_command(stripped) && break
            try
                _handle_temperature_command!(pa, stripped) && continue
                ModelSessions._handle_common_command!(
                    pa.model_session,
                    stripped,
                ) && continue
            catch error
                @warn sprint(showerror, error)
                continue
            end
            active_mode, prompt = try
                _parse_mode_sigil(stripped, fallback_mode)
            catch error
                @warn sprint(showerror, error)
                continue
            end
            attachments = pending_attachments(pa)
            input = isempty(attachments) ?
                ModelInput(prompt) : ModelInput(prompt, attachments)
            print("\nassistant> ")
            flush(stdout)
            started = time_ns()
            try
                response = respond!(
                    pa,
                    input;
                    mode = active_mode,
                    approval_callback = ModelSessions._approval_callback,
                )
                clear_attachments!(pa)
                ModelSessions._print_artifacts(response)
                elapsed = (time_ns() - started) / 1.0e9
                _log_response_metrics(last_response_metrics(pa), elapsed)
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
        if close_on_exit && isopen(pa)
            println("\nClosing session...")
            close(pa)
            println("Session saved. Goodbye.")
        end
    end
    return nothing
end

export PhysicsAssistant,
       PhysicalAgent,
       AssistantMode,
       GENERAL,
       DERIVE,
       CODE,
       DRAFT,
       respond!,
       ask!,
       generate_images!,
       set_image_backend!,
       save,
       load_assistant,
       load_physical_agent,
       repl,
       set_temperature!,
       default_temperature,
       set_mode!,
       default_mode,
       set_thinking!,
       set_reasoning_visible!,
       set_keep_alive!,
       set_timeouts!,
       set_tool_mode!,
       set_tool_context!,
       attach!,
       clear_attachments!,
       pending_attachments,
       default_thinking,
       default_keep_alive,
       default_tool_mode,
       default_image_backend,
       compression_running,
       await_compression!,
       last_response,
       last_response_metrics,
       model_name,
       base_url,
       session_directory,
       system_prompt,
       artifact_store,
       sandbox_config,
       model_capabilities

end # module PhysicalAssistant
