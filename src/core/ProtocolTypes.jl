module ProtocolTypes

# -----------------------------------------------------------------------------
# Thinking configuration
# -----------------------------------------------------------------------------

"""Named reasoning levels supported by Ollama thinking models."""
@enum ThinkingLevel begin
    THINK_LOW
    THINK_MEDIUM
    THINK_HIGH
    THINK_MAX
end

const ThinkingRequest = Union{Nothing,Bool,ThinkingLevel}

normalize_thinking(::Nothing)::ThinkingRequest = nothing
normalize_thinking(value::Bool)::ThinkingRequest = value
normalize_thinking(value::ThinkingLevel)::ThinkingRequest = value
normalize_thinking(value::Symbol)::ThinkingRequest =
    normalize_thinking(String(value))

function normalize_thinking(value::AbstractString)::ThinkingRequest
    normalized = lowercase(strip(String(value)))
    normalized in ("off", "false", "none", "disabled") && return false
    normalized in ("on", "true", "enabled") && return true
    normalized == "low" && return THINK_LOW
    normalized == "medium" && return THINK_MEDIUM
    normalized == "high" && return THINK_HIGH
    normalized == "max" && return THINK_MAX
    throw(ArgumentError(
        "thinking must be nothing, a boolean, or low/medium/high/max",
    ))
end

thinking_wire_value(::Nothing)::Nothing = nothing
thinking_wire_value(value::Bool)::Bool = value
thinking_wire_value(::Val{THINK_LOW})::String = "low"
thinking_wire_value(::Val{THINK_MEDIUM})::String = "medium"
thinking_wire_value(::Val{THINK_HIGH})::String = "high"
thinking_wire_value(::Val{THINK_MAX})::String = "max"
thinking_wire_value(value::ThinkingLevel)::String =
    thinking_wire_value(Val(value))

function thinking_label(value::ThinkingRequest)::String
    isnothing(value) && return "default"
    value isa Bool && return value ? "on" : "off"
    value == THINK_LOW && return "low"
    value == THINK_MEDIUM && return "medium"
    value == THINK_HIGH && return "high"
    return "max"
end

# -----------------------------------------------------------------------------
# Artifacts and multimodal input
# -----------------------------------------------------------------------------

"""Portable reference to a file stored by an `ArtifactStore.Store`."""
struct ArtifactRef
    path::String
    mime_type::String
    sha256::String
    size_bytes::Int

    function ArtifactRef(
        path::AbstractString,
        mime_type::AbstractString,
        sha256::AbstractString,
        size_bytes::Integer,
    )
        stored_path = String(strip(path))
        isempty(stored_path) &&
            throw(ArgumentError("artifact path must not be empty"))

        mime = lowercase(String(strip(mime_type)))
        isempty(mime) &&
            throw(ArgumentError("artifact MIME type must not be empty"))

        digest = lowercase(String(strip(sha256)))
        length(digest) == 64 || throw(ArgumentError(
            "artifact SHA-256 digest must contain 64 hexadecimal characters",
        ))
        all(isxdigit, digest) || throw(ArgumentError(
            "artifact SHA-256 digest contains a non-hexadecimal character",
        ))

        bytes = Int(size_bytes)
        bytes >= 0 ||
            throw(ArgumentError("artifact size must be nonnegative"))
        return new(stored_path, mime, digest, bytes)
    end
end

is_image(artifact::ArtifactRef)::Bool =
    startswith(artifact.mime_type, "image/")

struct TextPart
    text::String

    function TextPart(text::AbstractString)
        value = String(text)
        isempty(strip(value)) &&
            throw(ArgumentError("text part must not be empty"))
        return new(value)
    end
end

struct ArtifactPart
    artifact::ArtifactRef
end

const InputPart = Union{TextPart,ArtifactPart}

"""Ordered multimodal input containing text and artifact parts."""
struct ModelInput
    parts::Vector{InputPart}

    function ModelInput(parts::AbstractVector{<:InputPart})
        isempty(parts) && throw(ArgumentError(
            "model input must contain at least one part",
        ))
        return new(InputPart[part for part in parts])
    end
end

ModelInput(text::AbstractString)::ModelInput =
    ModelInput(InputPart[TextPart(text)])

function ModelInput(
    text::AbstractString,
    artifacts::AbstractVector{ArtifactRef},
)::ModelInput
    parts = sizehint!(InputPart[TextPart(text)], 1 + length(artifacts))
    append!(parts, ArtifactPart.(artifacts))
    return ModelInput(parts)
end

function text_content(input::ModelInput)::String
    text = String[]
    for part in input.parts
        part isa TextPart && push!(text, part.text)
    end
    return join(text, '\n')
end

function input_artifacts(input::ModelInput)::Vector{ArtifactRef}
    artifacts = ArtifactRef[]
    for part in input.parts
        part isa ArtifactPart && push!(artifacts, part.artifact)
    end
    return artifacts
end

function with_artifacts(
    input::ModelInput,
    artifacts::AbstractVector{ArtifactRef},
)::ModelInput
    isempty(artifacts) && return input
    parts = copy(input.parts)
    append!(parts, ArtifactPart.(artifacts))
    return ModelInput(parts)
end

# -----------------------------------------------------------------------------
# Tool protocol
# -----------------------------------------------------------------------------

"""Model-visible function-tool definition with a JSON Schema payload."""
struct ToolSpec
    name::String
    description::String
    parameters_json::String

    function ToolSpec(
        name::AbstractString,
        description::AbstractString,
        parameters_json::AbstractString,
    )
        tool_name = String(strip(name))
        isempty(tool_name) &&
            throw(ArgumentError("tool name must not be empty"))
        schema = String(strip(parameters_json))
        isempty(schema) && throw(ArgumentError(
            "tool parameter schema must not be empty",
        ))
        return new(tool_name, String(description), schema)
    end
end

"""Normalized function call emitted by an Ollama model."""
struct ToolCall
    id::String
    name::String
    arguments_json::String

    function ToolCall(
        id::AbstractString,
        name::AbstractString,
        arguments_json::AbstractString,
    )
        call_id = String(strip(id))
        isempty(call_id) &&
            throw(ArgumentError("tool-call ID must not be empty"))
        tool_name = String(strip(name))
        isempty(tool_name) &&
            throw(ArgumentError("tool-call name must not be empty"))
        arguments = String(strip(arguments_json))
        isempty(arguments) && (arguments = "{}")
        return new(call_id, tool_name, arguments)
    end
end

"""Execution result for one approved, denied, or failed tool call."""
struct ToolExecution
    call_id::String
    name::String
    success::Bool
    output::String
    exit_code::Union{Int,Nothing}
    artifacts::Vector{ArtifactRef}

    function ToolExecution(
        call_id::AbstractString,
        name::AbstractString,
        success::Bool,
        output::AbstractString;
        exit_code::Union{Integer,Nothing} = nothing,
        artifacts::AbstractVector{ArtifactRef} = ArtifactRef[],
    )
        identifier = String(strip(call_id))
        isempty(identifier) && throw(ArgumentError(
            "tool execution call ID must not be empty",
        ))
        tool_name = String(strip(name))
        isempty(tool_name) && throw(ArgumentError(
            "tool execution name must not be empty",
        ))
        status = isnothing(exit_code) ? nothing : Int(exit_code)
        return new(
            identifier,
            tool_name,
            success,
            String(output),
            status,
            collect(artifacts),
        )
    end
end

# -----------------------------------------------------------------------------
# Conversation messages
# -----------------------------------------------------------------------------

@enum MessageRole begin
    ROLE_SYSTEM
    ROLE_USER
    ROLE_ASSISTANT
    ROLE_TOOL
end

function role_name(role::MessageRole)::String
    role == ROLE_SYSTEM && return "system"
    role == ROLE_USER && return "user"
    role == ROLE_ASSISTANT && return "assistant"
    return "tool"
end

function parse_role(value::AbstractString)::MessageRole
    role = lowercase(strip(String(value)))
    role == "system" && return ROLE_SYSTEM
    role == "user" && return ROLE_USER
    role == "assistant" && return ROLE_ASSISTANT
    role == "tool" && return ROLE_TOOL
    throw(ArgumentError("unsupported message role '$role'"))
end

"""Typed conversation message for text, images, and tool interactions."""
struct ChatMessage
    role::MessageRole
    content::String
    reasoning::Union{String,Nothing}
    artifacts::Vector{ArtifactRef}
    tool_calls::Vector{ToolCall}
    tool_call_id::Union{String,Nothing}
    tool_name::Union{String,Nothing}

    function ChatMessage(
        role::MessageRole,
        content::AbstractString = "";
        reasoning::Union{AbstractString,Nothing} = nothing,
        artifacts::AbstractVector{ArtifactRef} = ArtifactRef[],
        tool_calls::AbstractVector{ToolCall} = ToolCall[],
        tool_call_id::Union{AbstractString,Nothing} = nothing,
        tool_name::Union{AbstractString,Nothing} = nothing,
    )
        trace = isnothing(reasoning) ? nothing : String(reasoning)
        call_id = isnothing(tool_call_id) ?
            nothing : String(strip(tool_call_id))
        name = isnothing(tool_name) ? nothing : String(strip(tool_name))
        role == ROLE_TOOL && isnothing(name) && throw(ArgumentError(
            "tool messages require a tool_name",
        ))
        return new(
            role,
            String(content),
            trace,
            collect(artifacts),
            collect(tool_calls),
            call_id,
            name,
        )
    end
end

# -----------------------------------------------------------------------------
# Response metadata
# -----------------------------------------------------------------------------

Base.@kwdef struct ResponseMetrics
    total_duration_ns::Int = 0
    load_duration_ns::Int = 0
    prompt_eval_count::Int = 0
    prompt_eval_duration_ns::Int = 0
    eval_count::Int = 0
    eval_duration_ns::Int = 0
    client_total_duration_ns::Int = 0
    time_to_first_token_ns::Union{Int,Nothing} = nothing
end

function generation_rate(metrics::ResponseMetrics)::Float64
    metrics.eval_duration_ns > 0 || return 0.0
    return metrics.eval_count / (metrics.eval_duration_ns / 1.0e9)
end

function prompt_rate(metrics::ResponseMetrics)::Float64
    metrics.prompt_eval_duration_ns > 0 || return 0.0
    return metrics.prompt_eval_count /
           (metrics.prompt_eval_duration_ns / 1.0e9)
end

"""Structured response returned by programmatic and interactive APIs."""
struct ModelResponse
    text::String
    reasoning::Union{String,Nothing}
    artifacts::Vector{ArtifactRef}
    tool_calls::Vector{ToolCall}
    tool_executions::Vector{ToolExecution}
    done::Bool
    done_reason::Union{String,Nothing}
    metrics::ResponseMetrics

    function ModelResponse(
        text::AbstractString;
        reasoning::Union{AbstractString,Nothing} = nothing,
        artifacts::AbstractVector{ArtifactRef} = ArtifactRef[],
        tool_calls::AbstractVector{ToolCall} = ToolCall[],
        tool_executions::AbstractVector{ToolExecution} = ToolExecution[],
        done::Bool = true,
        done_reason::Union{AbstractString,Nothing} = nothing,
        metrics::ResponseMetrics = ResponseMetrics(),
    )
        trace = isnothing(reasoning) ? nothing : String(reasoning)
        reason = isnothing(done_reason) ? nothing : String(done_reason)
        return new(
            String(text),
            trace,
            collect(artifacts),
            collect(tool_calls),
            collect(tool_executions),
            done,
            reason,
            metrics,
        )
    end
end

has_reasoning(response::ModelResponse)::Bool =
    !isnothing(response.reasoning) && !isempty(response.reasoning)

has_tool_calls(response::ModelResponse)::Bool =
    !isempty(response.tool_calls)

function response_images(response::ModelResponse)::Vector{ArtifactRef}
    return filter(is_image, response.artifacts)
end

function with_tool_executions(
    response::ModelResponse,
    executions::AbstractVector{ToolExecution},
)::ModelResponse
    return ModelResponse(
        response.text;
        reasoning = response.reasoning,
        artifacts = response.artifacts,
        tool_calls = response.tool_calls,
        tool_executions = executions,
        done = response.done,
        done_reason = response.done_reason,
        metrics = response.metrics,
    )
end

export ThinkingLevel,
       THINK_LOW,
       THINK_MEDIUM,
       THINK_HIGH,
       THINK_MAX,
       ThinkingRequest,
       normalize_thinking,
       thinking_wire_value,
       thinking_label,
       ArtifactRef,
       is_image,
       TextPart,
       ArtifactPart,
       InputPart,
       ModelInput,
       text_content,
       input_artifacts,
       with_artifacts,
       ToolSpec,
       ToolCall,
       ToolExecution,
       MessageRole,
       ROLE_SYSTEM,
       ROLE_USER,
       ROLE_ASSISTANT,
       ROLE_TOOL,
       role_name,
       parse_role,
       ChatMessage,
       ResponseMetrics,
       generation_rate,
       prompt_rate,
       ModelResponse,
       has_reasoning,
       has_tool_calls,
       response_images,
       with_tool_executions

end # module ProtocolTypes
