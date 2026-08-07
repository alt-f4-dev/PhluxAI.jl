module CapabilityDiscovery

using HTTP
using JSON3

const DEFAULT_BASE_URL = "http://localhost:11434"
const _JSON_HEADERS = ["Content-Type" => "application/json"]

"""Normalized features reported by Ollama's `/api/show` endpoint."""
struct ModelCapabilities
    completion::Bool
    vision::Bool
    thinking::Bool
    tools::Bool
    embedding::Bool
    image_generation::Bool
    raw::Set{String}
end

function _canonical(value::AbstractString)::String
    return replace(lowercase(strip(String(value))), '-' => '_')
end

function _contains_any(values::Set{String}, names::Tuple)::Bool
    return any(name in values for name in names)
end

function ModelCapabilities(values)::ModelCapabilities
    raw = Set{String}(_canonical(String(value)) for value in values)
    return ModelCapabilities(
        _contains_any(raw, ("completion", "completions")),
        "vision" in raw,
        _contains_any(raw, ("thinking", "reasoning")),
        _contains_any(raw, ("tools", "tool", "tool_calling")),
        _contains_any(raw, ("embedding", "embeddings")),
        _contains_any(
            raw,
            ("image_generation", "image_output", "text_to_image"),
        ),
        raw,
    )
end

function supports(
    capabilities::ModelCapabilities,
    capability::Symbol,
)::Bool
    capability === :completion && return capabilities.completion
    capability === :vision && return capabilities.vision
    capability === :thinking && return capabilities.thinking
    capability === :tools && return capabilities.tools
    capability === :embedding && return capabilities.embedding
    capability === :image_generation &&
        return capabilities.image_generation
    return _canonical(String(capability)) in capabilities.raw
end

function require_capability(
    capabilities::ModelCapabilities,
    capability::Symbol,
)::Nothing
    supports(capabilities, capability) && return nothing
    throw(ArgumentError(
        "model does not report the '$capability' capability",
    ))
end

capability_names(capabilities::ModelCapabilities)::Vector{String} =
    sort!(collect(capabilities.raw))

"""Selected model metadata plus preserved JSON detail fields."""
struct ModelDetails
    model::String
    modified_at::Union{String,Nothing}
    parameters::Union{String,Nothing}
    template::Union{String,Nothing}
    family::Union{String,Nothing}
    format::Union{String,Nothing}
    parameter_size::Union{String,Nothing}
    quantization_level::Union{String,Nothing}
    capabilities::ModelCapabilities
    details_json::String
    model_info_json::String
end

function _optional_string(object, key)::Union{String,Nothing}
    value = get(object, key, nothing)
    return isnothing(value) ? nothing : String(value)
end

function _nested_optional_string(
    object,
    outer,
    inner,
)::Union{String,Nothing}
    nested = get(object, outer, nothing)
    isnothing(nested) && return nothing
    return _optional_string(nested, inner)
end

function _json_fragment(object, key)::String
    value = get(object, key, nothing)
    return isnothing(value) ? "{}" : String(JSON3.write(value))
end

function parse_model_details(
    model::AbstractString,
    response,
)::ModelDetails
    raw_capabilities = get(response, :capabilities, String[])
    return ModelDetails(
        String(model),
        _optional_string(response, :modified_at),
        _optional_string(response, :parameters),
        _optional_string(response, :template),
        _nested_optional_string(response, :details, :family),
        _nested_optional_string(response, :details, :format),
        _nested_optional_string(response, :details, :parameter_size),
        _nested_optional_string(response, :details, :quantization_level),
        ModelCapabilities(raw_capabilities),
        _json_fragment(response, :details),
        _json_fragment(response, :model_info),
    )
end

function _base_url(value::AbstractString)::String
    endpoint = String(rstrip(strip(value), '/'))
    isempty(endpoint) &&
        throw(ArgumentError("base URL must not be empty"))
    return endpoint
end

function _timeout(value::Real)::Float64
    timeout = Float64(value)
    isfinite(timeout) ||
        throw(ArgumentError("request_timeout must be finite"))
    timeout >= 0.0 ||
        throw(ArgumentError("request_timeout must be nonnegative"))
    return timeout
end

function _response_text(response::HTTP.Response)::String
    return String(copy(response.body))
end

function show_model_details(
    model::AbstractString;
    base_url::AbstractString = DEFAULT_BASE_URL,
    verbose::Bool = false,
    request_timeout::Real = 30.0,
)::ModelDetails
    model_name = String(strip(model))
    isempty(model_name) &&
        throw(ArgumentError("model name must not be empty"))
    endpoint = _base_url(base_url)
    timeout = _timeout(request_timeout)
    body = JSON3.write(Dict(
        "model" => model_name,
        "verbose" => verbose,
    ))
    response = if timeout > 0.0
        HTTP.post(
            "$endpoint/api/show",
            _JSON_HEADERS,
            body;
            request_timeout = timeout,
        )
    else
        HTTP.post("$endpoint/api/show", _JSON_HEADERS, body)
    end
    200 <= response.status < 300 || throw(ErrorException(
        "Ollama /api/show returned HTTP $(response.status): " *
        first(_response_text(response), 1000),
    ))
    decoded = JSON3.read(_response_text(response))
    return parse_model_details(model_name, decoded)
end

function model_capabilities(
    model::AbstractString;
    kwargs...,
)::ModelCapabilities
    return show_model_details(model; kwargs...).capabilities
end

export ModelCapabilities,
       ModelDetails,
       supports,
       require_capability,
       capability_names,
       parse_model_details,
       show_model_details,
       model_capabilities

end # module CapabilityDiscovery
