module OllamaClient

using Dates
using HTTP
using JSON3
using SHA
using UUIDs
import ..SessionJournals

import ..ArtifactStore
import ..ImageInterface
import ..ProtocolTypes
import ..ProtocolTypes: ArtifactRef,
                        ChatMessage,
                        ModelInput,
                        ModelResponse,
                        ResponseMetrics,
                        ToolCall,
                        ToolExecution,
                        ToolSpec,
                        ThinkingRequest,
                        ROLE_ASSISTANT,
                        ROLE_SYSTEM,
                        ROLE_TOOL,
                        ROLE_USER,
                        generation_rate,
                        input_artifacts,
                        is_image,
                        normalize_thinking,
                        prompt_rate,
                        role_name,
                        text_content,
                        thinking_wire_value

const OLLAMA_BASE = "http://localhost:11434"
const _JSON_HEADERS = ["Content-Type" => "application/json"]
const _HOT_SCHEMA_VERSION = 2

# -----------------------------------------------------------------------------
# Model and budget configuration
# -----------------------------------------------------------------------------

struct OllamaModel
    name::String
    base_url::String

    function OllamaModel(name::AbstractString, base_url::AbstractString = OLLAMA_BASE)
        model_name = String(strip(name))
        isempty(model_name) && throw(ArgumentError("model name must not be empty"))
        endpoint = String(rstrip(strip(base_url), '/'))
        isempty(endpoint) && throw(ArgumentError("base URL must not be empty"))
        return new(model_name, endpoint)
    end
end

"""Configuration for the existing Ollama image-generation adapters."""
struct OllamaImageBackend <: ImageInterface.AbstractImageBackend
    model::OllamaModel
    transport::Symbol
    endpoint::String
    quality::Union{String,Nothing}
    style::Union{String,Nothing}
    user::Union{String,Nothing}
    request_timeout::Float64
    cli_timeout::Float64

    function OllamaImageBackend(
        model::OllamaModel;
        transport::Symbol = :auto,
        endpoint::AbstractString = "/v1/images/generations",
        quality::Union{AbstractString,Nothing} = nothing,
        style::Union{AbstractString,Nothing} = nothing,
        user::Union{AbstractString,Nothing} = nothing,
        request_timeout::Real = 0.0,
        cli_timeout::Real = 300.0,
    )
        transport in (:auto, :http, :cli) || throw(ArgumentError("Ollama image transport must be :auto, :http, or :cli"))
        path = String(strip(endpoint))
        startswith(path, '/') || throw(ArgumentError("image endpoint must begin with '/'"))
        request_limit = Float64(request_timeout)
        isfinite(request_limit) && request_limit >= 0.0 || throw(ArgumentError("request_timeout must be nonnegative and finite"))
        cli_limit = Float64(cli_timeout)
        isfinite(cli_limit) && cli_limit > 0.0 || throw(ArgumentError("cli_timeout must be positive and finite"))
        return new(model, transport, path, _optional_backend_string(quality, "quality"), _optional_backend_string(style, "style"), _optional_backend_string(user, "user"), request_limit, cli_limit)
    end
end

function _optional_backend_string(value::Union{AbstractString,Nothing}, name::String)::Union{String,Nothing}
    isnothing(value) && return nothing
    normalized = String(strip(value))
    isempty(normalized) && throw(ArgumentError("$name must not be empty"))
    return normalized
end

ImageInterface.backend_name(::OllamaImageBackend)::String = "ollama"
ImageInterface.is_local_backend(backend::OllamaImageBackend)::Bool = _is_loopback_url(backend.model.base_url)
ImageInterface.resource_policy(::OllamaImageBackend)::ImageInterface.ImageResourcePolicy = ImageInterface.IMAGE_KEEP_LOADED

Base.@kwdef struct BudgetConfig
    num_ctx::Int = 4096
    system_budget::Int = 600
    summary_budget::Int = 600
    response_budget::Int = 400
    safety_margin::Int = 200
    compression_frac::Float64 = 0.6
    m_verbatim::Int = 6
    exact_threshold::Int = 200
    num_thread::Int = 0
    max_tokens::Int = 512

    function BudgetConfig(
        num_ctx::Integer,
        system_budget::Integer,
        summary_budget::Integer,
        response_budget::Integer,
        safety_margin::Integer,
        compression_frac::Real,
        m_verbatim::Integer,
        exact_threshold::Integer,
        num_thread::Integer,
        max_tokens::Integer,
    )
        nctx = Int(num_ctx)
        system = Int(system_budget)
        summary = Int(summary_budget)
        response = Int(response_budget)
        margin = Int(safety_margin)
        fraction = Float64(compression_frac)
        verbatim = Int(m_verbatim)
        threshold = Int(exact_threshold)
        threads = Int(num_thread)
        generation = Int(max_tokens)

        nctx > 0 || throw(ArgumentError("num_ctx must be positive"))
        system >= 0 || throw(ArgumentError("system_budget must be nonnegative"))
        summary > 0 || throw(ArgumentError("summary_budget must be positive"))
        response >= 0 || throw(ArgumentError("response_budget must be nonnegative"))
        margin >= 0 || throw(ArgumentError("safety_margin must be nonnegative"))
        isfinite(fraction) || throw(ArgumentError("compression_frac must be finite"))
        0.0 < fraction <= 1.0 || throw(ArgumentError("compression_frac must lie in (0, 1]"))
        verbatim >= 0 || throw(ArgumentError("m_verbatim must be nonnegative"))
        iseven(verbatim) || throw(ArgumentError("m_verbatim must be even"))
        threshold >= 0 || throw(ArgumentError("exact_threshold must be nonnegative"))
        threads >= 0 || throw(ArgumentError("num_thread must be nonnegative"))
        generation > 0 || throw(ArgumentError("max_tokens must be positive"))
        reserved = system + summary + max(response, generation) + margin
        nctx > reserved || throw(ArgumentError("num_ctx=$nctx must exceed reserved token budget $reserved"))
        return new(nctx, system, summary, response, margin, fraction, verbatim, threshold, threads, generation)
    end
end

effective_response_budget(cfg::BudgetConfig)::Int = max(cfg.response_budget, cfg.max_tokens)

function history_budget(cfg::BudgetConfig)::Int
    return cfg.num_ctx - cfg.system_budget - cfg.summary_budget - effective_response_budget(cfg) - cfg.safety_margin
end

function compression_trigger(cfg::BudgetConfig)::Int
    return round(Int, cfg.compression_frac * history_budget(cfg))
end

approx_tokens(text::AbstractString)::Int = max(1, cld(ncodeunits(text), 4))

function approx_tokens(message::ChatMessage)::Int
    artifact_cost = 12 * length(message.artifacts)
    tool_cost = 12 * length(message.tool_calls)
    return approx_tokens(message.content) + artifact_cost + tool_cost + 4
end

function approx_tokens(messages::AbstractVector{ChatMessage})::Int
    return sum(approx_tokens, messages; init = 0)
end

function approx_tokens(message::AbstractDict{String,String})::Int
    return approx_tokens(get(message, "content", "")) + 4
end

function approx_tokens(messages::AbstractVector{<:AbstractDict{String,String}})::Int
    return sum(approx_tokens, messages; init = 0)
end

# -----------------------------------------------------------------------------
# Response compatibility
# -----------------------------------------------------------------------------

struct GenerateResponse
    model::String
    response::String
    thinking::Union{String,Nothing}
    artifacts::Vector{ArtifactRef}
    done::Bool
    done_reason::Union{String,Nothing}
    eval_count::Union{Int,Nothing}
    prompt_eval_count::Union{Int,Nothing}
    metrics::ResponseMetrics
end

function GenerateResponse(model::String, response::String, done::Bool, eval_count::Union{Int,Nothing}, prompt_eval_count::Union{Int,Nothing})::GenerateResponse
    metrics = ResponseMetrics(eval_count = something(eval_count, 0), prompt_eval_count = something(prompt_eval_count, 0))
    return GenerateResponse(model, response, nothing, ArtifactRef[], done, nothing, eval_count, prompt_eval_count, metrics)
end

const ChatResponse = ModelResponse

# -----------------------------------------------------------------------------
# Validation and JSON helpers
# -----------------------------------------------------------------------------

function _nonnegative_int(value::Integer, name::String)::Int
    converted = Int(value)
    converted >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return converted
end

function _positive_int(value::Integer, name::String)::Int
    converted = Int(value)
    converted > 0 || throw(ArgumentError("$name must be positive"))
    return converted
end

function _nonnegative_float(value::Real, name::String)::Float64
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$name must be finite"))
    converted >= 0.0 || throw(ArgumentError("$name must be nonnegative"))
    return converted
end

function _json_get(object, key::Symbol, default = nothing)
    haskey(object, key) && return object[key]
    string_key = String(key)
    haskey(object, string_key) && return object[string_key]
    return default
end

function _json_int(object, key)::Int
    value = _json_get(object, key, nothing)
    return isnothing(value) ? 0 : Int(value)
end

function _json_bool(object, key, default::Bool = false)::Bool
    value = _json_get(object, key, nothing)
    return isnothing(value) ? default : Bool(value)
end

function _json_string(object, key)::Union{String,Nothing}
    value = _json_get(object, key, nothing)
    return isnothing(value) ? nothing : String(value)
end

function _metrics_from_json(object; client_total_duration_ns::Int = 0, time_to_first_token_ns::Union{Int,Nothing} = nothing)::ResponseMetrics
    return ResponseMetrics(
        total_duration_ns = _json_int(object, :total_duration),
        load_duration_ns = _json_int(object, :load_duration),
        prompt_eval_count = _json_int(object, :prompt_eval_count),
        prompt_eval_duration_ns = _json_int(object, :prompt_eval_duration),
        eval_count = _json_int(object, :eval_count),
        eval_duration_ns = _json_int(object, :eval_duration),
        client_total_duration_ns = client_total_duration_ns,
        time_to_first_token_ns = time_to_first_token_ns,
    )
end

function _timeout_keywords(request_timeout::Real, read_idle_timeout::Real)::NamedTuple
    request = _nonnegative_float(request_timeout, "request_timeout")
    idle = _nonnegative_float(read_idle_timeout, "read_idle_timeout")
    request > 0.0 && idle > 0.0 && return (request_timeout = request, read_idle_timeout = idle,)
    request > 0.0 && return (request_timeout = request,)
    idle > 0.0 && return (read_idle_timeout = idle,)
    return (;)
end

function _post_json(url::String, payload; request_timeout::Real = 0.0)::HTTP.Response
    timeout = _nonnegative_float(request_timeout, "request_timeout")
    body = JSON3.write(payload)
    if timeout > 0.0
        return HTTP.post(url, _JSON_HEADERS, body; request_timeout = timeout)
    end
    return HTTP.post(url, _JSON_HEADERS, body)
end

function _build_options(temperature::Real, max_tokens::Integer; num_ctx::Integer = 0, num_thread::Integer = 0)::Dict{String,Any}
    temp = Float64(temperature)
    isfinite(temp) || throw(ArgumentError("temperature must be finite"))
    temp >= 0.0 || throw(ArgumentError("temperature must be nonnegative"))
    predict = _positive_int(max_tokens, "max_tokens")
    context = _nonnegative_int(num_ctx, "num_ctx")
    threads = _nonnegative_int(num_thread, "num_thread")
    options = Dict{String,Any}("temperature" => temp, "num_predict" => predict)
    context > 0 && (options["num_ctx"] = context)
    threads > 0 && (options["num_thread"] = threads)
    return options
end

function _set_keep_alive!(payload::Dict{String,Any}, keep_alive::Union{Nothing,Integer,AbstractString})::Nothing
    isnothing(keep_alive) && return nothing
    if keep_alive isa Integer
        payload["keep_alive"] = Int(keep_alive)
    else
        value = String(strip(keep_alive))
        isempty(value) && throw(ArgumentError("keep_alive must not be empty"))
        payload["keep_alive"] = value
    end
    return nothing
end

function _set_thinking!(payload::Dict{String,Any}, value)::Nothing
    normalized = normalize_thinking(value)
    wire_value = thinking_wire_value(normalized)
    isnothing(wire_value) || (payload["think"] = wire_value)
    return nothing
end

# -----------------------------------------------------------------------------
# Wire serialization
# -----------------------------------------------------------------------------

function _artifact_dict(artifact::ArtifactRef)::Dict{String,Any}
    return Dict{String,Any}("path" => artifact.path, "mime_type" => artifact.mime_type, "sha256" => artifact.sha256, "size_bytes" => artifact.size_bytes)
end

function _artifact_from_json(value)::ArtifactRef
    return ArtifactRef(String(value["path"]), String(value["mime_type"]), String(value["sha256"]), Int(value["size_bytes"]))
end

function _tool_call_dict(call::ToolCall)::Dict{String,Any}
    arguments = try
        JSON3.read(call.arguments_json)
    catch
        Dict{String,Any}()
    end
    return Dict{String,Any}("id" => call.id, "type" => "function", "function" => Dict{String,Any}("name" => call.name, "arguments" => arguments))
end

function _tool_spec_dict(spec::ToolSpec)::Dict{String,Any}
    parameters = try
        JSON3.read(spec.parameters_json)
    catch error
        throw(ArgumentError("invalid JSON Schema for tool $(spec.name): " * sprint(showerror, error)))
    end
    return Dict{String,Any}("type" => "function", "function" => Dict{String,Any}("name" => spec.name, "description" => spec.description, "parameters" => parameters))
end

function _message_wire(message::ChatMessage, store::ArtifactStore.Store)::Dict{String,Any}
    output = Dict{String,Any}("role" => role_name(message.role), "content" => message.content)
    image_artifacts = filter(is_image, message.artifacts)
    if !isempty(image_artifacts)
        output["images"] = String[ArtifactStore.encode_artifact_base64(store, artifact) for artifact in image_artifacts]
    end
    nonimage_artifacts = filter(artifact -> !is_image(artifact), message.artifacts)
    if !isempty(nonimage_artifacts)
        references = join(("[artifact: $(artifact.path), $(artifact.mime_type), " * "sha256=$(artifact.sha256)]" for artifact in nonimage_artifacts), '\n')
        output["content"] = isempty(message.content) ? references : message.content * "\n\n" * references
    end
    if !isnothing(message.reasoning) && message.role == ROLE_ASSISTANT
        output["thinking"] = message.reasoning
    end
    if !isempty(message.tool_calls)
        output["tool_calls"] = _tool_call_dict.(message.tool_calls)
    end
    if message.role == ROLE_TOOL
        isnothing(message.tool_name) || (output["tool_name"] = message.tool_name)
    end
    return output
end

function _message_storage(message::ChatMessage; include_reasoning::Bool = true)::Dict{String,Any}
    return Dict{String,Any}(
        "role" => role_name(message.role),
        "content" => message.content,
        "reasoning" => include_reasoning ? message.reasoning : nothing,
        "artifacts" => _artifact_dict.(message.artifacts),
        "tool_calls" => [Dict{String,Any}("id" => call.id, "name" => call.name, "arguments_json" => call.arguments_json) for call in message.tool_calls],
        "tool_call_id" => message.tool_call_id,
        "tool_name" => message.tool_name,
    )
end

function _message_from_storage(value)::ChatMessage
    artifacts = ArtifactRef[_artifact_from_json(item) for item in get(value, "artifacts", Any[])]
    calls = ToolCall[ToolCall(String(item["id"]),String(item["name"]),String(item["arguments_json"])) for item in get(value, "tool_calls", Any[])]
    reasoning_value = get(value, "reasoning", nothing)
    call_id_value = get(value, "tool_call_id", nothing)
    tool_name_value = get(value, "tool_name", nothing)
    return ChatMessage(
        ProtocolTypes.parse_role(String(value["role"])),
        String(get(value, "content", ""));
        reasoning = isnothing(reasoning_value) ? nothing : String(reasoning_value),
        artifacts,
        tool_calls = calls,
        tool_call_id = isnothing(call_id_value) ? nothing : String(call_id_value),
        tool_name = isnothing(tool_name_value) ? nothing : String(tool_name_value),
    )
end

function _input_message(input::ModelInput)::ChatMessage
    return ChatMessage(ROLE_USER, text_content(input); artifacts = input_artifacts(input))
end

function _parse_tool_calls(message, prefix::String)::Vector{ToolCall}
    raw_calls = get(message, :tool_calls, nothing)
    isnothing(raw_calls) && return ToolCall[]
    calls = ToolCall[]
    for (index, raw_call) in enumerate(raw_calls)
        function_data = get(raw_call, :function, nothing)
        isnothing(function_data) && continue
        name = String(get(function_data, :name, ""))
        isempty(name) && continue
        arguments = get(function_data, :arguments, Dict{String,Any}())
        arguments_json = arguments isa AbstractString ? String(arguments) : String(JSON3.write(arguments))
        id_value = get(raw_call, :id, nothing)
        id = isnothing(id_value) ? "$prefix-$index" : String(id_value)
        push!(calls, ToolCall(id, name, arguments_json))
    end
    return calls
end

function _decode_images(store::ArtifactStore.Store, encoded_images, prefix::String)::Vector{ArtifactRef}
    artifacts = ArtifactRef[]
    for (index, encoded) in enumerate(encoded_images)
        push!(artifacts, ArtifactStore.decode_base64_artifact(store,String(encoded);prefix = "$prefix-$index"))
    end
    return artifacts
end

# -----------------------------------------------------------------------------
# Low-level generate API
# -----------------------------------------------------------------------------

function exact_tokens(model::OllamaModel, text::AbstractString; num_ctx::Integer = 0, request_timeout::Real = 0.0)::Int
    options = Dict{String,Any}("num_predict" => 0)
    context = _nonnegative_int(num_ctx, "num_ctx")
    context > 0 && (options["num_ctx"] = context)
    payload = Dict{String,Any}("model" => model.name, "prompt" => String(text), "stream" => false, "options" => options)
    try
        response = _post_json("$(model.base_url)/api/generate", payload; request_timeout)
        body = JSON3.read(response.body)
        count = get(body, :prompt_eval_count, nothing)
        return isnothing(count) ? approx_tokens(text) : Int(count)
    catch error
        @debug "Exact token count failed; using approximation" exception = (error, catch_backtrace(),)
        return approx_tokens(text)
    end
end

function hybrid_tokens(model::OllamaModel, messages, trigger::Integer; exact_threshold::Integer = 200)::Int
    threshold = _nonnegative_int(exact_threshold, "exact_threshold")
    trigger_value = _nonnegative_int(trigger, "trigger")
    approximate = approx_tokens(messages)
    abs(approximate - trigger_value) > threshold && return approximate
    text = join((message isa ChatMessage ? message.content : get(message, "content", "") for message in messages), '\n')
    return exact_tokens(model, text)
end

function generate(
    model::OllamaModel,
    prompt::AbstractString;
    system::Union{AbstractString,Nothing} = nothing,
    temperature::Real = 0.7,
    max_tokens::Integer = 2048,
    stream::Bool = false,
    num_ctx::Integer = 0,
    num_thread::Integer = 0,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    print_tokens::Bool = true,
    print_reasoning::Bool = false,
    io::IO = stdout,
)::GenerateResponse
    if stream
        return generate_stream_response(model, prompt; system, temperature, max_tokens, num_ctx, num_thread, 
                                        keep_alive, thinking, request_timeout, read_idle_timeout, print_tokens, print_reasoning, io)
    end
    payload = Dict{String,Any}("model" => model.name, "prompt" => String(prompt), "stream" => false, "options" => _build_options(temperature, max_tokens; num_ctx, num_thread))
    isnothing(system) || (payload["system"] = String(system))
    _set_keep_alive!(payload, keep_alive)
    _set_thinking!(payload, thinking)
    started = time_ns()
    response = _post_json("$(model.base_url)/api/generate", payload; request_timeout)
    body = JSON3.read(response.body)
    metrics = _metrics_from_json(body; client_total_duration_ns = Int(time_ns() - started))
    eval_count = get(body, :eval_count, nothing)
    prompt_count = get(body, :prompt_eval_count, nothing)
    trace = _json_string(body, :thinking)
    reason = _json_string(body, :done_reason)
    return GenerateResponse(
        String(get(body, :model, model.name)),
        String(get(body, :response, "")),
        trace,
        ArtifactRef[],
        _json_bool(body, :done),
        reason,
        isnothing(eval_count) ? nothing : Int(eval_count),
        isnothing(prompt_count) ? nothing : Int(prompt_count),
        metrics,
    )
end

function generate_stream_response(
    model::OllamaModel,
    prompt::AbstractString;
    system::Union{AbstractString,Nothing} = nothing,
    temperature::Real = 0.7,
    max_tokens::Integer = 2048,
    num_ctx::Integer = 0,
    num_thread::Integer = 0,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    print_tokens::Bool = true,
    print_reasoning::Bool = false,
    io::IO = stdout,
)::GenerateResponse
    payload = Dict{String,Any}("model" => model.name, "prompt" => String(prompt), "stream" => true, "options" => _build_options(temperature, max_tokens; num_ctx, num_thread))
    isnothing(system) || (payload["system"] = String(system))
    _set_keep_alive!(payload, keep_alive)
    _set_thinking!(payload, thinking)
    timeout_keywords = _timeout_keywords(request_timeout, read_idle_timeout)
    text_buffer = IOBuffer()
    reasoning_buffer = IOBuffer()
    started = time_ns()
    first_token_ns = nothing
    final_chunk = nothing
    reasoning_started = false
    answer_started = false
    HTTP.open("POST", "$(model.base_url)/api/generate", _JSON_HEADERS; timeout_keywords...) do stream_io
        write(stream_io, JSON3.write(payload))
        HTTP.closewrite(stream_io)
        while !eof(stream_io)
            line = readline(stream_io)
            isempty(line) && continue
            chunk = JSON3.read(line)
            trace = String(get(chunk, :thinking, ""))
            text = String(get(chunk, :response, ""))
            if !isempty(trace)
                isnothing(first_token_ns) &&
                    (first_token_ns = Int(time_ns() - started))
                if print_reasoning && !reasoning_started
                    print(io, "\n[thinking]\n")
                    reasoning_started = true
                end
                print_reasoning && print(io, trace)
                write(reasoning_buffer, trace)
            end
            if !isempty(text)
                isnothing(first_token_ns) &&
                    (first_token_ns = Int(time_ns() - started))
                if print_tokens && reasoning_started && !answer_started
                    print(io, "\n[answer]\n")
                end
                answer_started = true
                print_tokens && print(io, text)
                write(text_buffer, text)
            end
            if _json_bool(chunk, :done)
                final_chunk = chunk
                break
            end
        end
    end
    (print_tokens || print_reasoning) && println(io)
    final_chunk === nothing && error("Ollama generation stream ended without a final chunk")
    metrics = _metrics_from_json(final_chunk; client_total_duration_ns = Int(time_ns() - started), time_to_first_token_ns = first_token_ns)
    eval_count = get(final_chunk, :eval_count, nothing)
    prompt_count = get(final_chunk, :prompt_eval_count, nothing)
    trace = String(take!(reasoning_buffer))
    return GenerateResponse(
        String(get(final_chunk, :model, model.name)),
        String(take!(text_buffer)),
        isempty(trace) ? nothing : trace,
        ArtifactRef[],
        _json_bool(final_chunk, :done),
        _json_string(final_chunk, :done_reason),
        isnothing(eval_count) ? nothing : Int(eval_count),
        isnothing(prompt_count) ? nothing : Int(prompt_count),
        metrics,
    )
end

function generate_stream(model::OllamaModel, prompt::AbstractString; kwargs...)::String
    return generate_stream_response(model, prompt; kwargs...).response
end

# -----------------------------------------------------------------------------
# Low-level chat API
# -----------------------------------------------------------------------------

function _chat_payload(
    model::OllamaModel,
    messages::Vector{ChatMessage},
    store::ArtifactStore.Store;
    temperature::Real,
    max_tokens::Integer,
    num_ctx::Integer,
    num_thread::Integer,
    keep_alive,
    thinking,
    tools::AbstractVector{ToolSpec},
    stream::Bool,
)::Dict{String,Any}
    payload = Dict{String,Any}("model" => model.name, "messages" => [_message_wire(message, store) for message in messages],
                               "stream" => stream, "options" => _build_options(temperature, max_tokens; num_ctx, num_thread))
    isempty(tools) || (payload["tools"] = _tool_spec_dict.(tools))
    _set_keep_alive!(payload, keep_alive)
    _set_thinking!(payload, thinking)
    return payload
end

function _response_from_message(object, message, store::ArtifactStore.Store; prefix::String, client_total_duration_ns::Int, time_to_first_token_ns::Union{Int,Nothing} = nothing)::ModelResponse
    text = String(get(message, :content, ""))
    trace_value = get(message, :thinking, nothing)
    trace = isnothing(trace_value) ? nothing : String(trace_value)
    raw_images = get(message, :images, Any[])
    artifacts = _decode_images(store, raw_images, prefix)
    calls = _parse_tool_calls(message, prefix)
    return ModelResponse(text; reasoning = trace, artifacts, tool_calls = calls, done = _json_bool(object, :done), done_reason = _json_string(object, :done_reason), metrics = _metrics_from_json(object; client_total_duration_ns, time_to_first_token_ns))
end

function chat_response(
    model::OllamaModel,
    messages::Vector{ChatMessage};
    artifact_store::ArtifactStore.Store,
    temperature::Real = 0.7,
    max_tokens::Integer = 512,
    num_ctx::Integer = 0,
    num_thread::Integer = 0,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    tools::AbstractVector{ToolSpec} = ToolSpec[],
    request_timeout::Real = 0.0,
)::ModelResponse
    payload = _chat_payload(model, messages, artifact_store; temperature, max_tokens, num_ctx, num_thread, keep_alive, thinking, tools, stream = false)
    started = time_ns()
    response = _post_json("$(model.base_url)/api/chat", payload; request_timeout)
    body = JSON3.read(response.body)
    message = get(body, :message, nothing)
    message === nothing && error("Ollama response contains no message")
    prefix = "response-$(time_ns())"
    return _response_from_message(body, message, artifact_store; prefix, client_total_duration_ns = Int(time_ns() - started))
end

function chat_stream_response(
    model::OllamaModel,
    messages::Vector{ChatMessage};
    artifact_store::ArtifactStore.Store,
    temperature::Real = 0.7,
    max_tokens::Integer = 512,
    num_ctx::Integer = 0,
    num_thread::Integer = 0,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    tools::AbstractVector{ToolSpec} = ToolSpec[],
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    print_tokens::Bool = true,
    print_reasoning::Bool = false,
    io::IO = stdout,
    on_chunk::Union{Function,Nothing} = nothing,
)::ModelResponse
    payload = _chat_payload(model, messages, artifact_store; temperature, max_tokens, num_ctx, num_thread, keep_alive, thinking, tools, stream = true)
    timeout_keywords = _timeout_keywords(request_timeout, read_idle_timeout)
    text_buffer = IOBuffer()
    reasoning_buffer = IOBuffer()
    encoded_images = String[]
    calls = ToolCall[]
    seen_calls = Set{Tuple{String,String,String}}()
    started = time_ns()
    first_token_ns = nothing
    final_chunk = nothing
    reasoning_started = false
    answer_started = false
    call_prefix = "call-$(time_ns())"

    HTTP.open("POST", "$(model.base_url)/api/chat", _JSON_HEADERS; timeout_keywords...) do stream_io
        write(stream_io, JSON3.write(payload))
        HTTP.closewrite(stream_io)
        while !eof(stream_io)
            line = readline(stream_io)
            isempty(line) && continue
            on_chunk === nothing || on_chunk(line)
            chunk = JSON3.read(line)
            message = get(chunk, :message, nothing)
            if message !== nothing
                trace = String(get(message, :thinking, ""))
                text = String(get(message, :content, ""))
                if !isempty(trace)
                    isnothing(first_token_ns) && (first_token_ns = Int(time_ns() - started))
                    if print_reasoning && !reasoning_started
                        print(io, "\n[thinking]\n")
                        reasoning_started = true
                    end
                    print_reasoning && print(io, trace)
                    write(reasoning_buffer, trace)
                end
                if !isempty(text)
                    isnothing(first_token_ns) && (first_token_ns = Int(time_ns() - started))
                    if print_tokens && reasoning_started && !answer_started
                        print(io, "\n[answer]\n")
                    end
                    answer_started = true
                    print_tokens && print(io, text)
                    write(text_buffer, text)
                end
                for image in get(message, :images, Any[])
                    push!(encoded_images, String(image))
                end
                for call in _parse_tool_calls(message, call_prefix)
                    key = (call.id, call.name, call.arguments_json)
                    if !(key in seen_calls)
                        push!(seen_calls, key)
                        push!(calls, call)
                    end
                end
            end
            if _json_bool(chunk, :done)
                final_chunk = chunk
                break
            end
        end
    end

    (print_tokens || print_reasoning) && println(io)
    final_chunk === nothing && error("Ollama chat stream ended without a final chunk")
    unique!(encoded_images)
    artifacts = _decode_images(artifact_store, encoded_images, "response-image-$(time_ns())")
    trace = String(take!(reasoning_buffer))
    return ModelResponse(
        String(take!(text_buffer));
        reasoning = isempty(trace) ? nothing : trace,
        artifacts,
        tool_calls = calls,
        done = _json_bool(final_chunk, :done),
        done_reason = _json_string(final_chunk, :done_reason),
        metrics = _metrics_from_json(final_chunk; client_total_duration_ns = Int(time_ns() - started), time_to_first_token_ns = first_token_ns),
    )
end

function chat(model::OllamaModel, messages::Vector{ChatMessage}; kwargs...)::String
    return chat_response(model, messages; kwargs...).text
end

function chat_stream(model::OllamaModel, messages::Vector{ChatMessage}; kwargs...)::String
    return chat_stream_response(model, messages; kwargs...).text
end

function _legacy_messages(messages::Vector{Dict{String,String}})::Vector{ChatMessage}
    return ChatMessage[ChatMessage(ProtocolTypes.parse_role(get(message, "role", "user")), get(message, "content", "")) for message in messages]
end

function chat_response(model::OllamaModel, messages::Vector{Dict{String,String}}; artifact_store::ArtifactStore.Store = ArtifactStore.Store(mktempdir()), kwargs...)::ModelResponse
    return chat_response(model, _legacy_messages(messages); artifact_store, kwargs...)
end

function chat_stream_response(model::OllamaModel, messages::Vector{Dict{String,String}}; artifact_store::ArtifactStore.Store = ArtifactStore.Store(mktempdir()), kwargs...)::ModelResponse
    return chat_stream_response(model, _legacy_messages(messages); artifact_store, kwargs...)
end

# -----------------------------------------------------------------------------
# Experimental image-generation adapters
# -----------------------------------------------------------------------------

function _image_size(value::AbstractString)::String
    size = lowercase(strip(String(value)))
    occursin(r"^[1-9][0-9]*x[1-9][0-9]*$", size) || throw(ArgumentError("image size must use WIDTHxHEIGHT notation, for example 1024x1024"))
    return size
end

function _optional_nonempty(value::Union{AbstractString,Nothing}, name::String)::Union{String,Nothing}
    isnothing(value) && return nothing
    normalized = String(strip(value))
    isempty(normalized) && throw(ArgumentError("$name must not be empty"))
    return normalized
end

function _image_generation_response(body, store::ArtifactStore.Store; prefix::String, client_total_duration_ns::Int)::ModelResponse
    data = _json_get(body, :data, nothing)
    isnothing(data) && error("image-generation response contains no data array")
    artifacts = ArtifactRef[]
    for (index, item) in enumerate(data)
        encoded = _json_get(item, :b64_json, nothing)
        isnothing(encoded) && error("image-generation result $index contains no b64_json field")
        push!(artifacts, ArtifactStore.decode_base64_artifact(store, String(encoded); prefix = "$prefix-$index"))
    end
    isempty(artifacts) && error("image-generation response contains no images")
    return ModelResponse(""; artifacts, done = true, metrics = ResponseMetrics(client_total_duration_ns = client_total_duration_ns))
end

function _generate_images_http(
    model::OllamaModel,
    prompt::String,
    store::ArtifactStore.Store;
    endpoint::AbstractString,
    size::AbstractString,
    n::Integer,
    quality::Union{AbstractString,Nothing},
    style::Union{AbstractString,Nothing},
    user::Union{AbstractString,Nothing},
    request_timeout::Real,
)::ModelResponse
    path = String(strip(endpoint))
    startswith(path, '/') || throw(ArgumentError("image endpoint must begin with '/'"))
    payload = Dict{String,Any}("model" => model.name, "prompt" => prompt, "size" => _image_size(size), "n" => _positive_int(n, "n"), "response_format" => "b64_json")
    selected_quality = _optional_nonempty(quality, "quality")
    selected_style = _optional_nonempty(style, "style")
    selected_user = _optional_nonempty(user, "user")
    isnothing(selected_quality) ||
        (payload["quality"] = selected_quality)
    isnothing(selected_style) || (payload["style"] = selected_style)
    isnothing(selected_user) || (payload["user"] = selected_user)

    started = time_ns()
    response = _post_json("$(model.base_url)$path", payload; request_timeout)
    body = JSON3.read(response.body)
    return _image_generation_response(body, store; prefix = "generated-image-$(time_ns())", client_total_duration_ns = Int(time_ns() - started))
end

function _capture_process(command::Cmd, timeout_seconds::Float64)::Tuple{Bool,Union{Int,Nothing},String,String}
    stdout_path = tempname()
    stderr_path = tempname()
    process = nothing
    timed_out = false
    try
        open(stdout_path, "w") do stdout_io
            open(stderr_path, "w") do stderr_io
                process = run(pipeline(command; stdout = stdout_io, stderr = stderr_io); wait = false)
                status = timedwait(() -> process_exited(process), timeout_seconds; pollint = 0.05)
                timed_out = status == :timed_out
                if timed_out
                    try
                        kill(process)
                    catch
                    end
                end
                wait(process)
            end
        end
        exit_code = process.exitcode < 0 ? nothing : process.exitcode
        stdout = read(stdout_path, String)
        stderr = read(stderr_path, String)
        return !timed_out && success(process), exit_code, stdout, stderr
    finally
        isfile(stdout_path) && rm(stdout_path; force = true)
        isfile(stderr_path) && rm(stderr_path; force = true)
    end
end

function _generated_image_paths(directory::String)::Vector{String}
    paths = String[]
    for (root, directories, files) in walkdir(directory)
        sort!(directories)
        sort!(files)
        for file in files
            path = joinpath(root, file)
            bytes = open(path, "r") do io
                read(io, min(filesize(path), 4096))
            end
            mime = ArtifactStore.detect_mime_type(bytes; filename = file)
            startswith(mime, "image/") && push!(paths, path)
        end
    end
    return paths
end

function _generate_images_cli(
    model::OllamaModel,
    prompt::String,
    store::ArtifactStore.Store;
    n::Integer,
    size::AbstractString,
    quality::Union{AbstractString,Nothing},
    style::Union{AbstractString,Nothing},
    user::Union{AbstractString,Nothing},
    timeout::Real,
)::ModelResponse
    _positive_int(n, "n") == 1 || throw(ArgumentError("the Ollama CLI image adapter currently supports n=1 only"))
    _image_size(size) == "1024x1024" || throw(ArgumentError("the Ollama CLI image adapter does not expose a size option"))
    isnothing(quality) || throw(ArgumentError("the Ollama CLI image adapter does not expose quality"))
    isnothing(style) || throw(ArgumentError("the Ollama CLI image adapter does not expose style"))
    isnothing(user) || throw(ArgumentError("the Ollama CLI image adapter does not expose user metadata"))
    executable = Sys.which("ollama")
    isnothing(executable) && throw(ArgumentError("Ollama CLI image generation requires `ollama` on PATH"))
    timeout_value = Float64(timeout)
    isfinite(timeout_value) && timeout_value > 0.0 || throw(ArgumentError("image CLI timeout must be positive and finite"))

    staging = mktempdir(prefix = "phluxai-image-generation-")
    started = time_ns()
    try
        environment = copy(ENV)
        model.base_url == OLLAMA_BASE || (environment["OLLAMA_HOST"] = model.base_url)
        command = Cmd(Cmd([String(executable), "run", model.name, prompt]); dir = staging, env = environment)
        succeeded, exit_code, stdout, stderr = _capture_process(command, timeout_value)
        succeeded || throw(ErrorException("Ollama image-generation command failed " * "(exit=$(something(exit_code, "unknown"))): " * (isempty(stderr) ? stdout : stderr)))
        paths = _generated_image_paths(staging)
        isempty(paths) && throw(ErrorException("Ollama completed without writing an image in $staging"))
        artifacts = ArtifactRef[ArtifactStore.import_artifact(store, path) for path in paths]
        return ModelResponse(strip(stdout); artifacts, done = true, metrics = ResponseMetrics(client_total_duration_ns = Int(time_ns() - started)))
    finally
        isdir(staging) && rm(staging; recursive = true, force = true)
    end
end

"""
    ImageInterface.generate_images(backend, request, artifact_store)

Execute an image request through the existing Ollama CLI or optional
OpenAI-compatible HTTP adapter.
"""
function ImageInterface.generate_images(backend::OllamaImageBackend, request::ImageInterface.ImageGenerationRequest, artifact_store::ArtifactStore.Store)::ModelResponse
    isnothing(request.negative_prompt) || throw(ArgumentError("the Ollama image adapter does not expose negative_prompt"))
    isnothing(request.seed) || throw(ArgumentError("the Ollama image adapter does not expose seed"))
    isnothing(request.steps) || throw(ArgumentError("the Ollama image adapter does not expose steps"))
    isnothing(request.cfg_scale) || throw(ArgumentError("the Ollama image adapter does not expose cfg_scale"))
    isnothing(request.sampler_name) || throw(ArgumentError("the Ollama image adapter does not expose sampler_name"))
    isnothing(request.scheduler) || throw(ArgumentError("the Ollama image adapter does not expose scheduler"))
    width = something(request.width, 1024)
    height = something(request.height, 1024)
    size = "$(width)x$(height)"

    selected_request_timeout = if backend.transport == :auto && backend.request_timeout == 0.0
        10.0
    else
        backend.request_timeout
    end
    http_call = () -> _generate_images_http(
        backend.model,
        request.prompt,
        artifact_store;
        endpoint = backend.endpoint,
        size,
        n = request.count,
        quality = backend.quality,
        style = backend.style,
        user = backend.user,
        request_timeout = selected_request_timeout,
    )
    cli_call = () -> _generate_images_cli(
        backend.model,
        request.prompt,
        artifact_store;
        n = request.count,
        size,
        quality = backend.quality,
        style = backend.style,
        user = backend.user,
        timeout = backend.cli_timeout,
    )

    backend.transport == :http && return http_call()
    backend.transport == :cli && return cli_call()

    prefer_cli = _is_loopback_url(backend.model.base_url) &&
                 !isnothing(Sys.which("ollama"))
    first_call = prefer_cli ? cli_call : http_call
    second_call = prefer_cli ? http_call : cli_call
    first_name = prefer_cli ? "CLI" : "HTTP"
    second_name = prefer_cli ? "HTTP" : "CLI"

    try
        return first_call()
    catch first_error
        try
            return second_call()
        catch second_error
            throw(ErrorException("both Ollama image adapters failed; $first_name: " * sprint(showerror, first_error) * "; $second_name: " * sprint(showerror, second_error)))
        end
    end
end

"""
    generate_images(model, prompt; artifact_store, backend=:auto, kwargs...)

Backward-compatible Ollama image-generation entry point. New code may
construct `OllamaImageBackend` and call `ImageInterface.generate_images`.
"""
function generate_images(
    model::OllamaModel,
    prompt::AbstractString;
    artifact_store::ArtifactStore.Store,
    backend::Symbol = :auto,
    endpoint::AbstractString = "/v1/images/generations",
    size::AbstractString = "1024x1024",
    n::Integer = 1,
    quality::Union{AbstractString,Nothing} = nothing,
    style::Union{AbstractString,Nothing} = nothing,
    user::Union{AbstractString,Nothing} = nothing,
    request_timeout::Real = 0.0,
    cli_timeout::Real = 300.0,
)::ModelResponse
    normalized_size = _image_size(size)
    dimensions = split(normalized_size, 'x'; limit = 2)
    request = ImageInterface.ImageGenerationRequest(prompt; width = parse(Int, dimensions[1]), height = parse(Int, dimensions[2]), count = n)
    selected = OllamaImageBackend(model; transport = backend, endpoint, quality, style, user, request_timeout, cli_timeout)
    return ImageInterface.generate_images(selected, request, artifact_store)
end

# -----------------------------------------------------------------------------
# Stateful sessions and compression
# -----------------------------------------------------------------------------

const _SUMMARY_SYSTEM = """
You are summarizing a technical conversation for re-injection as context.
Preserve equations, code, assumptions, decisions, filenames, artifact
references,
and tool results. Do not include private model reasoning traces. Be concise in
prose and exhaustive in technical content.

Respond using exactly these XML sections:
<assumptions></assumptions>
<results></results>
<notation></notation>
<deferred></deferred>
<rejected></rejected>
<narrative></narrative>
"""

mutable struct ChatSession
    model::OllamaModel
    prompt::String
    summary::Union{String,Nothing}
    protected_facts::Vector{String}
    messages::Vector{ChatMessage}
    cfg::BudgetConfig
    compression_task::Union{Task,Nothing}
    session_dir::Union{String,Nothing}
    _cold_io::Union{IOStream,Nothing}
    session_type::String
    artifact_store::ArtifactStore.Store
    persist_reasoning::Bool
    lock::ReentrantLock
    revision::UInt64
    last_response::Union{ModelResponse,Nothing}
    closed::Bool
    journal::SessionJournals.SessionJournal
    journal_state_sha256::String
    runtime_settings::Dict{String,Any}
end

function _default_artifact_store(session_dir::Union{String,Nothing})::ArtifactStore.Store
    root = isnothing(session_dir) ? mktempdir(prefix = "phluxai-artifacts-") : joinpath(session_dir, "artifacts")
    return ArtifactStore.Store(root)
end

function ChatSession(
    model::OllamaModel,
    prompt::AbstractString;
    cfg::BudgetConfig = BudgetConfig(),
    session_dir::Union{AbstractString,Nothing} = nothing,
    session_type::AbstractString = "generic",
    artifact_store::Union{ArtifactStore.Store,Nothing} = nothing,
    persist_reasoning::Bool = false,
)::ChatSession
    directory = isnothing(session_dir) ? nothing : String(session_dir)
    kind = String(strip(session_type))
    isempty(kind) && throw(ArgumentError("session_type must not be empty"))
    if !isnothing(directory)
        isempty(strip(directory)) && throw(ArgumentError("session_dir must not be empty"))
        mkpath(directory)
        isfile(joinpath(directory, "journal.jsonl")) && throw(ArgumentError("session already has a journal; use load_session"))
        isfile(joinpath(directory, "hot.json")) && throw(ArgumentError("legacy session exists; use load_session to migrate"))
    end
    journal_root = isnothing(directory) ? mktempdir(prefix = "phluxai-journal-") : directory
    journal = SessionJournals.open_journal(joinpath(journal_root, "journal.jsonl"))
    cold_io = nothing
    try
        cold_io = isnothing(directory) ? nothing : open(joinpath(directory, "cold.jsonl"), "a")
        store = isnothing(artifact_store) ? _default_artifact_store(directory) : artifact_store
        session = ChatSession(model, String(prompt), nothing, String[], ChatMessage[], cfg, nothing, directory, cold_io, kind,
                              store, persist_reasoning, ReentrantLock(), UInt64(0), nothing, false, journal, "", Dict{String,Any}())
        SessionJournals.append_event!(journal, :session_started, (model = model.name, base_url = model.base_url, session_type = kind))
        _save_hot_unlocked(session)
        return session
    catch
        cold_io === nothing || close(cold_io)
        close(journal)
        rethrow()
    end
end

function _require_open_unlocked(session::ChatSession)::Nothing
    session.closed && throw(ArgumentError("ChatSession is closed"))
    return nothing
end

# Pin an explicitly chosen, verbatim fact in model-visible session context. [Internal Method]
function pin_context!(session::ChatSession, fact::AbstractString)::Nothing
    value = String(strip(fact))
    isempty(value) && throw(ArgumentError("protected context must not be empty"))

    lock(session.lock) do
        _require_open_unlocked(session)
        value in session.protected_facts && return nothing

        estimate = approx_tokens(session.prompt) + sum(approx_tokens, session.protected_facts; init = 0) + approx_tokens(value) + 32
        estimate <= session.cfg.system_budget || throw(ArgumentError("protected context exceeds the estimated system_budget=$(session.cfg.system_budget)"))

        previous_revision = session.revision
        push!(session.protected_facts, value)
        session.revision += UInt64(1)
        try
            _save_hot_unlocked(session)
        catch
            pop!(session.protected_facts)
            session.revision = previous_revision
            rethrow()
        end
    end
    return nothing
end

# Heuristic capacity check for the *next* Ollama request; caller holds session.lock.
function _context_capacity_ok_unlocked(session::ChatSession, incoming::Union{Nothing,ChatMessage}, max_tokens::Integer)::Bool
    response_reserve = max(session.cfg.response_budget, Int(max_tokens))
    input_limit = session.cfg.num_ctx - response_reserve - session.cfg.safety_margin
    input_limit > 0 || return false

    estimate = approx_tokens(_assemble_messages_unlocked(session))
    isnothing(incoming) || (estimate += approx_tokens(incoming))
    return estimate <= input_limit
end

function _lock_when_idle!(session::ChatSession; incoming::Union{Nothing,ChatMessage} = nothing, max_tokens::Integer = session.cfg.max_tokens, enforce_capacity::Bool = false)::Nothing
    check_capacity = enforce_capacity || !isnothing(incoming)
    attempted_compression = false

    while true
        lock(session.lock)
        task = nothing
        try
            # Do not reject closed sessions here: close_session is idempotent
            # and checks session.closed after acquiring this lock.
            task = session.compression_task
            if !isnothing(task) && istaskdone(task)
                session.compression_task = nothing
                task = nothing
            end

            fits = !check_capacity || _context_capacity_ok_unlocked(session, incoming, max_tokens)

            # Legacy callers (save/close) still wait for pending work.
            # A response can bypass a pending task only if estimated capacity
            # suffices; its revision change will invalidate that task's result.
            if fits && (check_capacity || isnothing(task))
                return nothing  # Intentionally return WITH session.lock held.
            end

            if !fits
                attempted_compression && throw(ArgumentError("Estimated context exceeds the available window after " *
                                                             "compression; refusing to submit a possibly truncated prompt"))
                attempted_compression = true
                if isnothing(task)
                    # Emergency path: launch one compression attempt if there
                    # is a compressible history prefix; otherwise reject.
                    _maybe_compress_unlocked!(session; force = true)
                    task = session.compression_task
                    isnothing(task) && throw(ArgumentError("Estimated context exceeds the available window and " * "no compressible history prefix is available"))
                end
            end
        catch
            unlock(session.lock)
            rethrow()
        end

        unlock(session.lock)
        wait(task)  # Never hold the session lock while waiting on the worker.
    end
end
function _assemble_messages_unlocked(session::ChatSession)::Vector{ChatMessage}
    extra = isnothing(session.summary) ? 1 : 3
    messages = sizehint!(ChatMessage[], length(session.messages) + extra)
    system_text = session.prompt
    if !isempty(session.protected_facts)
        system_text = string(system_text, "\n\n[PROTECTED CONTEXT — user-approved facts, preserved verbatim]\n", join(session.protected_facts, "\n"), "\n[/PROTECTED CONTEXT]")
    end
    push!(messages, ChatMessage(ROLE_SYSTEM, system_text))

    if !isnothing(session.summary)
        push!(messages, ChatMessage(ROLE_USER, "[SESSION MEMORY — verified summary of prior turns, not a new " * "query]\n\n$(session.summary)"))
        push!(messages, ChatMessage(ROLE_ASSISTANT, "Understood. I have the prior discussion context."))
    end
    append!(messages, session.messages)
    return messages
end

function assemble_messages(session::ChatSession)::Vector{ChatMessage}
    return lock(session.lock) do
        _require_open_unlocked(session)
        return _assemble_messages_unlocked(session)
    end
end

function _summary_line(message::ChatMessage)::String
    buffer = IOBuffer()
    print(buffer, role_name(message.role), ": ", message.content)
    for artifact in message.artifacts
        print(buffer, "\n[artifact: ",artifact.path,", ", artifact.mime_type,", sha256=",artifact.sha256,"]")
    end
    for call in message.tool_calls
        print(buffer, "\n[tool call: ", call.name, " ", call.arguments_json, "]")
    end
    return String(take!(buffer))
end

function _compress(model::OllamaModel, turns::Vector{ChatMessage},
        existing_summary::Union{String,Nothing}, cfg::BudgetConfig)::GenerateResponse
    turns_text = join((_summary_line(turn) for turn in turns), "\n\n---\n\n")
    prompt = if isnothing(existing_summary)
        "Summarize the following conversation turns:\n\n" * turns_text
    else
        "Existing summary:\n$(existing_summary)\n\n" *
        "New turns:\n$turns_text\n\nMerge them into one summary."
    end
    response = generate(model, prompt; system = _SUMMARY_SYSTEM, temperature = 0.2, max_tokens = cfg.summary_budget, num_ctx = cfg.num_ctx, num_thread = cfg.num_thread, stream = false)
    return response
end

const _SUMMARY_XML_PATTERN = r"(?s)^\s*<assumptions>.*?</assumptions>\s*<results>.*?</results>\s*<notation>.*?</notation>\s*<deferred>.*?</deferred>\s*<rejected>.*?</rejected>\s*<narrative>.*?</narrative>\s*$"

function _validated_summary(candidate::GenerateResponse)::Union{String,Nothing}
    candidate.done && candidate.done_reason == "stop" || return nothing
    summary = strip(candidate.response)
    isempty(summary) && return nothing
    occursin(_SUMMARY_XML_PATTERN, summary) || return nothing
    return summary
end

function _compression_cutoff(messages::Vector{ChatMessage}, keep::Int)::Int
    length(messages) > keep || return 0
    cutoff = length(messages) - keep

    # The retained tail must start with a user message. This prevents an
    # assistant tool request from being separated from its tool results or
    # final answer during compression.
    while cutoff > 0 && messages[cutoff + 1].role != ROLE_USER
        cutoff -= 1
    end
    return cutoff
end

function _generation_storage(candidate::GenerateResponse)
    return (model = candidate.model, response = candidate.response, thinking = candidate.thinking,
            artifacts = _artifact_dict.(candidate.artifacts), done = candidate.done, done_reason = candidate.done_reason,
            eval_count = candidate.eval_count, prompt_eval_count = candidate.prompt_eval_count,
            metrics = NamedTuple{fieldnames(ResponseMetrics)}(Tuple(getfield(candidate.metrics, name) for name in fieldnames(ResponseMetrics))))
end

function _commit_compression_candidate_unlocked!(session::ChatSession, candidate::GenerateResponse, tail::Vector{ChatMessage}, snapshot_revision::UInt64)::Bool
    session.closed && return false
    summary = _validated_summary(candidate)
    if session.revision != snapshot_revision || summary === nothing
        record_session_event!(session, :summary_rejected, (snapshot_revision, received = _generation_storage(candidate)))
        return false
    end

    source_head = SessionJournals.journal_head(session.journal)
    SessionJournals.append_event!(session.journal, :summary_created,
        (source_sequence = source_head.sequence, source_sha256 = source_head.sha256, previous_summary = session.summary,
         input_messages = [_message_storage(message; include_reasoning = false) for message in session.messages[1:max(0, length(session.messages) - length(tail))]],
         model = session.model.name, settings = _budget_dict(session.cfg), summary, received = _generation_storage(candidate), system_prompt = _SUMMARY_SYSTEM, temperature = 0.2))
    previous_summary = session.summary
    previous_messages = session.messages
    previous_revision = session.revision
    session.summary = summary
    session.messages = tail
    session.revision += UInt64(1)
    try
        _save_hot_unlocked(session)
    catch
        session.summary = previous_summary
        session.messages = previous_messages
        session.revision = previous_revision
        rethrow()
    end
    return true
end

function _maybe_compress_unlocked!(session::ChatSession; force::Bool=false)::Nothing
    task = session.compression_task
    !isnothing(task) && !istaskdone(task) && return nothing
    !force && approx_tokens(session.messages) <= compression_trigger(session.cfg) && return nothing
    n_compress = _compression_cutoff(session.messages, session.cfg.m_verbatim)
    n_compress > 0 || return nothing
    turns = copy(session.messages[1:n_compress])
    tail = copy(session.messages[(n_compress + 1):end])
    existing_summary = session.summary
    snapshot_revision = session.revision
    cfg = session.cfg
    model = session.model
    worker = Threads.@spawn begin
        try
            candidate = _compress(model, turns, existing_summary, cfg)
            lock(session.lock) do
                committed = _commit_compression_candidate_unlocked!(session, candidate, tail, snapshot_revision)
                if !committed && session.revision == snapshot_revision && !session.closed
                    @warn "Compression candidate rejected; original conversation history retained" done = candidate.done done_reason = candidate.done_reason
                end
            end
        catch error
            lock(session.lock) do
                session.closed || record_session_event!(session, :compression_failed, (message = sprint(showerror, error), snapshot_revision))
            end
            @warn "Async compression failed; original conversation history retained" exception = (error, catch_backtrace(),)
        end
        return nothing
    end
    session.compression_task = worker
    return nothing
end

function _drop_transient_reasoning_unlocked!(session::ChatSession)::Nothing
    session.persist_reasoning && return nothing
    for index in eachindex(session.messages)
        message = session.messages[index]
        isnothing(message.reasoning) && continue
        session.messages[index] = ChatMessage(message.role, message.content; reasoning = nothing, artifacts = message.artifacts, tool_calls = message.tool_calls, tool_call_id = message.tool_call_id, tool_name = message.tool_name)
    end
    return nothing
end

function finalize_interaction!(session::ChatSession)::Nothing
    lock(session.lock) do
        _require_open_unlocked(session)
        _drop_transient_reasoning_unlocked!(session)
        _save_hot_unlocked(session)
        _maybe_compress_unlocked!(session)
    end
    return nothing
end

function _run_model_turn_unlocked!(
    session::ChatSession;
    temperature::Real,
    max_tokens::Integer,
    keep_alive,
    thinking,
    tools::AbstractVector{ToolSpec},
    request_timeout::Real,
    read_idle_timeout::Real,
    stream::Bool,
    print_tokens::Bool,
    print_reasoning::Bool,
    io::IO,
)::ModelResponse
    request_id = string(uuid4())
    record_session_event!(session, :operation_started, (id = request_id, operation = "model_request", model = session.model.name,
                          temperature, max_tokens, keep_alive, thinking = thinking === nothing ? nothing : string(thinking)))
    response = try
        if stream
            chat_stream_response(session.model, _assemble_messages_unlocked(session); artifact_store = session.artifact_store,
                                 temperature, max_tokens, num_ctx = session.cfg.num_ctx, num_thread = session.cfg.num_thread,
                                 keep_alive, thinking, tools, request_timeout, read_idle_timeout, print_tokens, print_reasoning, io,
                                 on_chunk = line -> record_session_event!(session, :model_chunk, (request_id, raw_json = line)))
        else
            chat_response(session.model, _assemble_messages_unlocked(session); artifact_store = session.artifact_store,
                          temperature, max_tokens, num_ctx = session.cfg.num_ctx, num_thread = session.cfg.num_thread,
                          keep_alive, thinking, tools, request_timeout)
        end
    catch error
        record_session_event!(session, :operation_finished, (id = request_id, success = false, message = sprint(showerror, error)))
        rethrow()
    end
    record_session_event!(session, :response_received, _response_storage(response))
    message = ChatMessage(ROLE_ASSISTANT, response.text; reasoning = response.reasoning, artifacts = response.artifacts, tool_calls = response.tool_calls)
    _log_cold_unlocked(session, message)
    record_session_event!(session, :operation_finished, (id = request_id, success = true))
    push!(session.messages, message)
    session.last_response = response
    session.revision += UInt64(1)
    _save_hot_unlocked(session)
    return response
end

function respond!(
    session::ChatSession,
    input::Union{AbstractString,ModelInput};
    temperature::Real = 0.7,
    max_tokens::Integer = session.cfg.max_tokens,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    tools::AbstractVector{ToolSpec} = ToolSpec[],
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    stream::Bool = true,
    print_tokens::Bool = stream,
    print_reasoning::Bool = false,
    io::IO = stdout,
    finalize::Bool = true,
)::ModelResponse
    model_input = input isa ModelInput ? input : ModelInput(input)
    user_message = _input_message(model_input)
    _lock_when_idle!(session; incoming=user_message, max_tokens=max_tokens)
    try
        _require_open_unlocked(session)
        original_length = length(session.messages)
        original_revision = session.revision
        original_response = session.last_response
        _log_cold_unlocked(session, user_message)
        push!(session.messages, user_message)
        session.revision += UInt64(1)
        response = try
            _run_model_turn_unlocked!(session; temperature, max_tokens, keep_alive, thinking, tools, request_timeout, read_idle_timeout, stream, print_tokens, print_reasoning, io)
        catch error
            resize!(session.messages, original_length)
            session.revision = original_revision
            session.last_response = original_response
            SessionJournals.append_event!(session.journal, :operation_failed, (operation = "model_request", message = sprint(showerror, error)))
            _save_hot_unlocked(session)
            rethrow()
        end
        if finalize
            _drop_transient_reasoning_unlocked!(session)
            _save_hot_unlocked(session)
            _maybe_compress_unlocked!(session)
        end
        return response
    finally
        unlock(session.lock)
    end
end

function continue_response!(
    session::ChatSession;
    temperature::Real = 0.7,
    max_tokens::Integer = session.cfg.max_tokens,
    keep_alive::Union{Nothing,Integer,AbstractString} = nothing,
    thinking = nothing,
    tools::AbstractVector{ToolSpec} = ToolSpec[],
    request_timeout::Real = 0.0,
    read_idle_timeout::Real = 0.0,
    stream::Bool = true,
    print_tokens::Bool = stream,
    print_reasoning::Bool = false,
    io::IO = stdout,
    finalize::Bool = true,
)::ModelResponse
    _lock_when_idle!(session; max_tokens=max_tokens, enforce_capacity=true)
    try
        _require_open_unlocked(session)
        original_length = length(session.messages)
        original_revision = session.revision
        original_response = session.last_response
        response = try
            _run_model_turn_unlocked!(session;  temperature, max_tokens, keep_alive, thinking, tools, request_timeout, read_idle_timeout, stream, print_tokens, print_reasoning, io)
        catch error
            resize!(session.messages, original_length)
            session.revision = original_revision
            session.last_response = original_response
            SessionJournals.append_event!(session.journal, :operation_failed, (operation = "model_request", message = sprint(showerror, error)))
            _save_hot_unlocked(session)
            rethrow()
        end
        if finalize
            _drop_transient_reasoning_unlocked!(session)
            _save_hot_unlocked(session)
            _maybe_compress_unlocked!(session)
        end
        return response
    finally
        unlock(session.lock)
    end
end

function append_tool_execution!(session::ChatSession, execution::ToolExecution)::Nothing
    lock(session.lock) do
        _require_open_unlocked(session)
        message = ChatMessage(ROLE_TOOL, execution.output; artifacts = execution.artifacts, tool_call_id = execution.call_id, tool_name = execution.name)
        push!(session.messages, message)
        session.revision += UInt64(1)
        _log_cold_unlocked(session, message)
        _save_hot_unlocked(session)
    end
    return nothing
end

function generate_images!(session::ChatSession, backend::ImageInterface.AbstractImageBackend, request::ImageInterface.ImageGenerationRequest; finalize::Bool = true)::ModelResponse
    _lock_when_idle!(session)
    try
        _require_open_unlocked(session)
        original_length = length(session.messages)
        original_revision = session.revision
        original_response = session.last_response
        user_message = ChatMessage(ROLE_USER, request.prompt)
        _log_cold_unlocked(session, user_message)
        push!(session.messages, user_message)
        session.revision += UInt64(1)
        request_id = string(uuid4())
        record_session_event!(session, :operation_started, (id = request_id, operation = "image_generation",
                              request = NamedTuple{fieldnames(typeof(request))}(Tuple(getfield(request, name) for name in fieldnames(typeof(request))))))
        response = try
            ImageInterface.generate_images(backend, request, session.artifact_store)
        catch error
            resize!(session.messages, original_length)
            session.revision = original_revision
            session.last_response = original_response
            record_session_event!(session, :operation_finished, (id = request_id, success = false, message = sprint(showerror, error)))
            _save_hot_unlocked(session)
            rethrow()
        end
        record_session_event!(session, :response_received, _response_storage(response))
        assistant_message = ChatMessage(ROLE_ASSISTANT,
                                        isempty(response.text) ? "Generated $(length(response.artifacts)) image artifact(s)." : response.text; artifacts = response.artifacts)
        push!(session.messages, assistant_message)
        session.last_response = response
        session.revision += UInt64(1)
        _log_cold_unlocked(session, assistant_message)
        record_session_event!(session, :operation_finished, (id = request_id, success = true))
        _save_hot_unlocked(session)
        finalize && _maybe_compress_unlocked!(session)
        return response
    finally
        unlock(session.lock)
    end
end

function generate_images!(
    session::ChatSession,
    prompt::AbstractString;
    backend::Symbol = :auto,
    endpoint::AbstractString = "/v1/images/generations",
    size::AbstractString = "1024x1024",
    n::Integer = 1,
    quality::Union{AbstractString,Nothing} = nothing,
    style::Union{AbstractString,Nothing} = nothing,
    user::Union{AbstractString,Nothing} = nothing,
    request_timeout::Real = 0.0,
    cli_timeout::Real = 300.0,
    finalize::Bool = true,
)::ModelResponse
    normalized_size = _image_size(size)
    dimensions = split(normalized_size, 'x'; limit = 2)
    request = ImageInterface.ImageGenerationRequest(prompt; width = parse(Int, dimensions[1]), height = parse(Int, dimensions[2]), count = n)
    selected = OllamaImageBackend(session.model; transport = backend, endpoint, quality, style, user, request_timeout, cli_timeout)
    return generate_images!(session, selected, request; finalize)
end

function ask!(session::ChatSession, input::AbstractString; kwargs...)::String
    return respond!(session, input; kwargs...).text
end

function last_response(session::ChatSession)::Union{ModelResponse,Nothing}
    return lock(session.lock) do
        return session.last_response
    end
end

function last_response_metrics(session::ChatSession)::Union{ResponseMetrics,Nothing}
    response = last_response(session)
    return isnothing(response) ? nothing : response.metrics
end

function compression_running(session::ChatSession)::Bool
    return lock(session.lock) do
        task = session.compression_task
        return !isnothing(task) && !istaskdone(task)
    end
end

function await_compression!(session::ChatSession)::Nothing
    while true
        task = lock(session.lock) do
            return session.compression_task
        end
        isnothing(task) && return nothing
        !istaskdone(task) && wait(task)
        lock(session.lock) do
            if session.compression_task === task && istaskdone(task)
                session.compression_task = nothing
            end
        end
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

function _response_storage(response::ModelResponse)
    return (message = _message_storage(ChatMessage(ROLE_ASSISTANT, response.text; reasoning = response.reasoning,
                        artifacts = response.artifacts, tool_calls = response.tool_calls)), done = response.done,
            done_reason = response.done_reason,
            metrics = NamedTuple{fieldnames(ResponseMetrics)}(Tuple(getfield(response.metrics, name) for name in fieldnames(ResponseMetrics))),
            executions = [(call_id = value.call_id, name = value.name, success = value.success, output = value.output,
                           exit_code = value.exit_code, artifacts = _artifact_dict.(value.artifacts)) for value in response.tool_executions])
end

function _response_from_storage(value)::ModelResponse
    message = _message_from_storage(value.message)
    metrics = ResponseMetrics(; (Symbol(name) => (number === nothing ? nothing : Int(number)) for (name, number) in pairs(value.metrics))...)
    executions = ToolExecution[ToolExecution(String(item.call_id), String(item.name), Bool(item.success), String(item.output);
                                exit_code = item.exit_code, artifacts = ArtifactRef[_artifact_from_json(ref) for ref in item.artifacts])
                               for item in value.executions]
    return ModelResponse(message.content; reasoning = message.reasoning, artifacts = message.artifacts, tool_calls = message.tool_calls,
                         tool_executions = executions, done = Bool(value.done), done_reason = value.done_reason, metrics)
end

function _cold_entry(message::ChatMessage; include_reasoning::Bool = true)::Dict{String,Any}
    return Dict{String,Any}("timestamp" => string(now(UTC)), "message" => _message_storage(message; include_reasoning))
end

function _log_cold_unlocked(session::ChatSession, message::ChatMessage)::Nothing
    SessionJournals.append_event!(session.journal, :message, _message_storage(message; include_reasoning = true))
    isnothing(session._cold_io) && return nothing
    entry = _cold_entry(message; include_reasoning = session.persist_reasoning)
    println(session._cold_io, JSON3.write(entry))
    flush(session._cold_io)
    return nothing
end

function _log_cold(session::ChatSession, message::ChatMessage)::Nothing
    lock(session.lock) do
        _require_open_unlocked(session)
        _log_cold_unlocked(session, message)
    end
    return nothing
end

function _budget_dict(cfg::BudgetConfig)::Dict{String,Any}
    return Dict{String,Any}("num_ctx" => cfg.num_ctx, "system_budget" => cfg.system_budget, 
                            "summary_budget" => cfg.summary_budget, "response_budget" => cfg.response_budget,
                            "safety_margin" => cfg.safety_margin, "compression_frac" => cfg.compression_frac, 
                            "m_verbatim" => cfg.m_verbatim, "exact_threshold" => cfg.exact_threshold,
                            "num_thread" => cfg.num_thread, "max_tokens" => cfg.max_tokens)
end

function _atomic_write_json(path::String, value)::Nothing
    directory = dirname(path)
    mkpath(directory)
    temporary = tempname(directory)
    try
        open(temporary, "w") do io
            JSON3.write(io, value)
            flush(io)
        end
        mv(temporary, path; force = true)
    catch
        isfile(temporary) && rm(temporary; force = true)
        rethrow()
    end
    return nothing
end

function _ordered_json(value)
    if value isa AbstractDict || value isa JSON3.Object
        names = sort!(String.(collect(keys(value))))
        return NamedTuple{Tuple(Symbol.(names))}(Tuple(_ordered_json(value[name]) for name in names))
    elseif value isa AbstractVector
        return _ordered_json.(value)
    end
    return value
end

function record_session_event!(session::ChatSession, kind::Symbol, payload)
    return lock(session.lock) do
        _require_open_unlocked(session)
        SessionJournals.append_event!(session.journal, kind, payload)
    end
end

session_journal(session::ChatSession) = session.journal

"""Return unfinished invocations as uncertain operations. Resumption never executes them."""
function pending_operations(session::ChatSession)
    pending = Dict{String,Any}()
    SessionJournals.foreach_event(session.journal) do event
        event.kind in ("operation_started", "operation_finished") || return
        payload = JSON3.read(event.payload_json)
        if event.kind == "operation_started"
            pending[String(payload.id)] = (event_id = event.id, payload = payload, status = :uncertain)
        elseif event.kind == "operation_finished"
            delete!(pending, String(payload.id))
        end
    end
    return sort!(collect(values(pending)); by = item -> string(item.event_id))
end

function _tool_seen(session::ChatSession, id::String)::Bool
    seen = false
    SessionJournals.foreach_event(session.journal) do event
        event.kind == "operation_started" || return
        payload = JSON3.read(event.payload_json)
        get(payload, :operation, "") == "tool" && String(payload.id) == id && (seen = true)
    end
    return seen
end

"""Verify journal integrity and report missing or changed artifact bytes without altering history."""
function verify_session(session::ChatSession; checkpoint::Union{SessionJournals.JournalHead,Nothing} = nothing)
    issues = NamedTuple[]
    head = SessionJournals.foreach_event(session.journal; checkpoint) do event
        event.kind == "message" || return
        payload = JSON3.read(event.payload_json)
        for value in get(payload, :artifacts, [])
            artifact = _artifact_from_json(value)
            status = try
                path = ArtifactStore.resolve_artifact(session.artifact_store, artifact)
                !isfile(path) ? :missing : (filesize(path) == artifact.size_bytes && bytes2hex(open(sha256, path)) == artifact.sha256 ? :ok : :changed)
            catch
                :inaccessible
            end
            status == :ok || push!(issues, (event_id = event.id, artifact_path = artifact.path, status))
        end
    end
    return (head, artifact_issues = issues, pending = pending_operations(session))
end

function _state_payload(session::ChatSession)
    return Dict{String,Any}("schema_version" => _HOT_SCHEMA_VERSION, "session_type" => session.session_type,
                           "model_name" => session.model.name, "base_url" => session.model.base_url,
                           "prompt" => session.prompt, "summary" => session.summary, "protected_facts" => copy(session.protected_facts),
                           "messages" => [_message_storage(message; include_reasoning = session.persist_reasoning) for message in session.messages],
                           "cfg" => _budget_dict(session.cfg), "persist_reasoning" => session.persist_reasoning)
end

function _save_hot_unlocked(session::ChatSession)::Nothing
    hot = _state_payload(session)
    encoded = JSON3.write(_ordered_json(hot))
    digest = bytes2hex(sha256(encoded))
    if digest != session.journal_state_sha256
        SessionJournals.append_event!(session.journal, :context_state; payload_json = encoded)
        session.journal_state_sha256 = digest
    end
    if session.session_dir !== nothing
        try
            _atomic_write_json(joinpath(session.session_dir, "hot.json"), hot)
        catch error
            @warn "Journal committed; hot-view cache refresh failed" exception = (error, catch_backtrace())
        end
    end
    return nothing
end

function save_session(session::ChatSession)::Nothing
    _lock_when_idle!(session)
    try
        _require_open_unlocked(session)
        _save_hot_unlocked(session)
    finally
        unlock(session.lock)
    end
    return nothing
end

function _config_from_json(config)::BudgetConfig
    return BudgetConfig(num_ctx = Int(config["num_ctx"]), system_budget = Int(config["system_budget"]), 
                        summary_budget = Int(config["summary_budget"]), response_budget = Int(config["response_budget"]), 
                        safety_margin = Int(config["safety_margin"]), compression_frac = Float64(config["compression_frac"]),
                        m_verbatim = Int(config["m_verbatim"]), exact_threshold = Int(config["exact_threshold"]), 
                        num_thread = Int(get(config, "num_thread", 0)), max_tokens = Int(get(config, "max_tokens", 512)))
end

function _migrate_legacy_messages(values)::Vector{ChatMessage}
    return ChatMessage[ChatMessage(ProtocolTypes.parse_role(String(get(value, "role", "user"))), String(get(value, "content", ""))) for value in values]
end

function load_session(model::OllamaModel, session_dir::AbstractString; artifact_store::Union{ArtifactStore.Store,Nothing} = nothing,
                      recover::Bool = false, checkpoint::Union{SessionJournals.JournalHead,Nothing} = nothing)::ChatSession
    directory = String(session_dir)
    path = joinpath(directory, "journal.jsonl")
    journal = SessionJournals.open_journal(path; recover, checkpoint)
    cold_io = nothing
    try
        data = nothing
        state_digest = ""
        extra_messages = ChatMessage[]
        settings = Dict{String,Any}()
        legacy_import = nothing
        legacy_count = 0
        restored_response = nothing
        SessionJournals.foreach_event(journal) do event
            event.kind in ("context_state", "message", "response_received", "interaction_finished", "legacy_import", "legacy_record", "runtime_settings") || return
            payload = JSON3.read(event.payload_json)
            if event.kind == "context_state"
                data = payload
                state_digest = bytes2hex(sha256(event.payload_json))
                empty!(extra_messages)
            elseif event.kind == "message"
                push!(extra_messages, _message_from_storage(payload))
            elseif event.kind in ("response_received", "interaction_finished")
                restored_response = _response_from_storage(payload)
            elseif event.kind == "legacy_import"
                legacy_import = payload
            elseif event.kind == "legacy_record"
                legacy_count += 1
            elseif event.kind == "runtime_settings"
                settings = Dict{String,Any}(String(key) => value for (key, value) in pairs(payload))
            end
        end
        if data === nothing
            # Legacy bytes are left untouched. Cold records are imported once; unavailable originals cannot be reconstructed.
            hot_path = joinpath(directory, "hot.json")
            isfile(hot_path) || throw(ArgumentError("session has no committed state or legacy hot store"))
            filesize(hot_path) <= journal.max_record_bytes || throw(ArgumentError("legacy hot store exceeds the journal record limit"))
            data = JSON3.read(read(hot_path, String))
            version = Int(get(data, "schema_version", 0))
            0 <= version <= _HOT_SCHEMA_VERSION || throw(ArgumentError("unsupported legacy hot-store schema"))
            cold_path = joinpath(directory, "cold.jsonl")
            hot_digest = bytes2hex(open(sha256, hot_path))
            cold_digest = isfile(cold_path) ? bytes2hex(open(sha256, cold_path)) : nothing
            if legacy_import === nothing
                # Backups preserve every legacy byte before the new cache is refreshed.
                backup = joinpath(directory, "legacy-originals")
                SessionJournals._durable_mkpath(abspath(backup))
                for source in (hot_path, cold_path)
                    isfile(source) || continue
                    target = joinpath(backup, basename(source))
                    if isfile(target)
                        bytes2hex(open(sha256, target)) == bytes2hex(open(sha256, source)) ||
                            throw(ArgumentError("legacy backup differs from the source"))
                    else
                        temporary = tempname(backup)
                        try
                            cp(source, temporary)
                            open(temporary, "a") do io
                                SessionJournals._sync(io)
                            end
                            mv(temporary, target)
                        finally
                            isfile(temporary) && rm(temporary)
                        end
                    end
                end
                bytes2hex(open(sha256, hot_path)) == hot_digest || throw(ArgumentError("legacy hot source changed during migration"))
                (!isfile(cold_path) ? nothing : bytes2hex(open(sha256, cold_path))) == cold_digest ||
                    throw(ArgumentError("legacy cold source changed during migration"))
                SessionJournals._sync_directory(backup)
                SessionJournals._sync_directory(directory)
                SessionJournals.append_event!(journal, :legacy_import, (hot_sha256 = hot_digest, cold_sha256 = cold_digest,
                    limitation = "Only surviving legacy records can be recovered; backups retain original bytes."))
            else
                legacy_import.hot_sha256 == hot_digest && legacy_import.cold_sha256 == cold_digest ||
                    throw(ArgumentError("legacy source changed during interrupted migration"))
            end
            if isfile(cold_path)
                open(cold_path, "r") do io
                    line_number = 0
                    while !eof(io)
                        bytes, complete = SessionJournals._record(io, journal.max_record_bytes)
                        complete || throw(ArgumentError("legacy cold log has an incomplete final record"))
                        line = String(bytes)
                        JSON3.read(line)
                        line_number += 1
                        line_number <= legacy_count && continue
                        SessionJournals.append_event!(journal, :legacy_record, (source_line = line_number, raw_json = line))
                    end
                end
            end
            legacy_messages = version <= 1 ? _migrate_legacy_messages(data.messages) : ChatMessage[_message_from_storage(value) for value in data.messages]
            data = Dict{String,Any}("schema_version" => _HOT_SCHEMA_VERSION, "session_type" => String(get(data, "session_type", "generic")),
                                   "model_name" => String(data.model_name), "base_url" => String(get(data, "base_url", model.base_url)),
                                   "prompt" => String(data.prompt), "summary" => get(data, "summary", nothing),
                                   "protected_facts" => String[String(x) for x in get(data, "protected_facts", [])],
                                   "messages" => [_message_storage(message; include_reasoning = Bool(get(data, "persist_reasoning", false))) for message in legacy_messages],
                                   "cfg" => _budget_dict(_config_from_json(data.cfg)), "persist_reasoning" => Bool(get(data, "persist_reasoning", false)))
            # Canonical state bytes ensure subsequent save/resume does not duplicate this event.
            canonical = JSON3.write(_ordered_json(data))
            SessionJournals.append_event!(journal, :context_state; payload_json = canonical)
            state_digest = bytes2hex(sha256(canonical))
            data = JSON3.read(canonical)
            empty!(extra_messages)
        end
        schema_version = Int(get(data, "schema_version", 0))
        0 <= schema_version <= _HOT_SCHEMA_VERSION || throw(ArgumentError("unsupported session state schema"))
        messages = schema_version <= 1 ? _migrate_legacy_messages(data.messages) : ChatMessage[_message_from_storage(value) for value in data.messages]
        append!(messages, extra_messages)
        summary = data.summary === nothing ? nothing : String(data.summary)
        store = isnothing(artifact_store) ? _default_artifact_store(directory) : artifact_store
        cold_io = open(joinpath(directory, "cold.jsonl"), "a")
        session = ChatSession(model, String(data.prompt), summary, String[String(x) for x in get(data, "protected_facts", String[])],
                              messages, _config_from_json(data.cfg), nothing, directory, cold_io, String(get(data, "session_type", "generic")),
                              store, Bool(get(data, "persist_reasoning", false)), ReentrantLock(), UInt64(0), nothing, false,
                              journal, state_digest, settings)
        session.last_response = restored_response
        if String(data.model_name) != model.name || String(get(data, "base_url", model.base_url)) != model.base_url
            SessionJournals.append_event!(journal, :model_changed, (previous = String(data.model_name), model = model.name, base_url = model.base_url))
        end
        return session
    catch
        cold_io === nothing || close(cold_io)
        close(journal)
        rethrow()
    end
end

function close_session(session::ChatSession)::Nothing
    _lock_when_idle!(session)
    try
        session.closed && return nothing
        try
            _save_hot_unlocked(session)
        finally
            session._cold_io === nothing || close(session._cold_io)
            session._cold_io = nothing
            close(session.journal)
            session.closed = true
        end
    finally
        unlock(session.lock)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Model lifecycle
# -----------------------------------------------------------------------------

function _is_loopback_url(base_url::String)::Bool
    endpoint = lowercase(base_url)
    return startswith(endpoint, "http://localhost") ||
           startswith(endpoint, "https://localhost") ||
           startswith(endpoint, "http://127.0.0.1") ||
           startswith(endpoint, "https://127.0.0.1") ||
           startswith(endpoint, "http://[::1]") ||
           startswith(endpoint, "https://[::1]")
end

is_loopback_url(base_url::AbstractString)::Bool = _is_loopback_url(String(strip(base_url)))

function _preload_model!(model_name::String, base_url::String; keep_alive::Union{Integer,AbstractString} = -1, request_timeout::Real = 0.0)::Nothing
    payload = Dict{String,Any}("model" => model_name, "prompt" => "", "stream" => false, "keep_alive" => keep_alive, "options" => Dict("num_predict" => 0))
    response = _post_json("$base_url/api/generate", payload; request_timeout)
    JSON3.read(response.body)
    return nothing
end

function _image_model_available(model_name::String, base_url::String; request_timeout::Real = 10.0)::Bool
    try
        response = _post_json("$base_url/api/show", Dict{String,Any}("model" => model_name); request_timeout)
        body = JSON3.read(response.body)
        capabilities = Set(lowercase(replace(String(value), '-' => '_')) for value in get(body, :capabilities, String[]))
        return any(capability in capabilities for capability in ("image_generation", "image_output", "text_to_image",))
    catch
        return false
    end
end

function _launch_local_preload(model_name::String)::Nothing
    executable = Sys.which("ollama")
    isnothing(executable) && return nothing
    command = Cmd([String(executable), "run", model_name, ""])
    run(pipeline(command; stdout = devnull, stderr = devnull); wait = false)
    return nothing
end

# Emit a model lifecycle Info record only when user-facing announcement is enabled.
function _model_lifecycle_info(announce::Bool, message::AbstractString)::Nothing
    if announce; @info message; end
    return nothing
end

function start_model!(model_name::AbstractString, base_url::AbstractString = OLLAMA_BASE; timeout::Integer = 120, poll_interval::Real = 2.0, announce::Bool=true)::Nothing
    model = OllamaModel(model_name, base_url)
    timeout_value = _positive_int(timeout, "timeout")
    poll = Float64(poll_interval)
    isfinite(poll) && poll > 0.0 || throw(ArgumentError("poll_interval must be positive and finite"))
    _model_lifecycle_info(announce, "Starting model $(model.name)...")
    deadline = time() + timeout_value
    last_error = nothing
    try
        _preload_model!(model.name, model.base_url; keep_alive = -1, request_timeout = min(timeout_value, 30))
        _model_lifecycle_info(announce, "Model $(model.name) is ready.")
        return nothing
    catch error
        last_error = error
    end
    if _image_model_available(model.name, model.base_url; request_timeout = min(timeout_value, 10))
        _model_lifecycle_info(announce, "Image-generation model $(model.name) is available.")
        return nothing
    end
    if _is_loopback_url(model.base_url)
        try
            _launch_local_preload(model.name)
        catch error
            last_error = error
        end
    end
    while time() < deadline
        remaining = max(0.1, deadline - time())
        try
            _preload_model!(model.name, model.base_url; keep_alive = -1, request_timeout = min(remaining, 10.0))
            _model_lifecycle_info(announce, "Model $(model.name) is ready.")
            return nothing
        catch error
            last_error = error
            if _image_model_available(model.name, model.base_url; request_timeout = min(remaining, 5.0))
                _model_lifecycle_info(announce, "Image-generation model $(model.name) is available.")
                return nothing
            end
            sleep(min(poll, max(0.0, deadline - time())))
        end
    end
    message = "Model $(model.name) did not become ready within $timeout_value seconds at $(model.base_url)."
    isnothing(last_error) && error(message)
    throw(ErrorException("$message Last error: $(sprint(showerror, last_error))"))
end

function stop_model!(model_name::AbstractString, base_url::AbstractString = OLLAMA_BASE; request_timeout::Real = 10.0, announce::Bool=true)::Nothing
    model = OllamaModel(model_name, base_url)
    _model_lifecycle_info(announce, "Stopping model $(model.name)...")
    try
        _preload_model!(model.name, model.base_url; keep_alive = 0, request_timeout)
    catch api_error
        !_is_loopback_url(model.base_url) && throw(api_error)
        executable = Sys.which("ollama")
        isnothing(executable) && throw(api_error)
        run(Cmd([String(executable), "stop", model.name]))
    end
    _model_lifecycle_info(announce, "Model $(model.name) stopped.")
    return nothing
end

stop_model!(model::OllamaModel; kwargs...) = stop_model!(model.name, model.base_url; kwargs...)

function load_session!(model_name::AbstractString, session_dir::AbstractString; base_url::AbstractString = OLLAMA_BASE, timeout::Integer = 120, poll_interval::Real = 2.0)::ChatSession
    start_model!(model_name, base_url; timeout, poll_interval)
    return load_session(OllamaModel(model_name, base_url), session_dir)
end

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

export OLLAMA_BASE,
       OllamaModel,
       OllamaImageBackend,
       BudgetConfig,
       effective_response_budget,
       history_budget,
       compression_trigger,
       approx_tokens,
       exact_tokens,
       hybrid_tokens,
       ResponseMetrics,
       generation_rate,
       prompt_rate,
       GenerateResponse,
       ChatResponse,
       generate,
       generate_stream,
       generate_stream_response,
       chat,
       chat_response,
       chat_stream,
       chat_stream_response,
       generate_images,
       ChatSession,
       respond!,
       continue_response!,
       append_tool_execution!,
       finalize_interaction!,
       generate_images!,
       ask!,
       assemble_messages,
       last_response,
       last_response_metrics,
       compression_running,
       await_compression!,
       is_loopback_url,
       start_model!,
       stop_model!,
       save_session,
       load_session,
       load_session!,
       close_session

end # module OllamaClient
