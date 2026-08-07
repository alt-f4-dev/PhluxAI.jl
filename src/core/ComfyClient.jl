module ComfyClient

using HTTP
using JSON3
using UUIDs

import ..ArtifactStore
import ..ImageInterface
import ..ProtocolTypes: ArtifactRef,
                        ModelResponse,
                        ResponseMetrics

const COMFY_BASE = "http://127.0.0.1:8188"
const _JSON_HEADERS = ["Content-Type" => "application/json"]

"""Reference to one mutable input on a ComfyUI API-format workflow node."""
struct WorkflowBinding
    node_id::String
    input_name::String

    function WorkflowBinding(
        node_id::AbstractString,
        input_name::AbstractString,
    )
        node = String(strip(node_id))
        input = String(strip(input_name))
        isempty(node) && throw(ArgumentError(
            "workflow binding node_id must not be empty",
        ))
        isempty(input) && throw(ArgumentError(
            "workflow binding input_name must not be empty",
        ))
        return new(node, input)
    end
end

"""
Bindings between backend-neutral image parameters and a specific ComfyUI
workflow. `positive_prompt` is required; every other binding is optional.
`model` identifies the loader input that receives an explicit model filename.
"""
Base.@kwdef struct ComfyBindings
    positive_prompt::WorkflowBinding
    model::Union{WorkflowBinding,Nothing} = nothing
    negative_prompt::Union{WorkflowBinding,Nothing} = nothing
    seed::Union{WorkflowBinding,Nothing} = nothing
    width::Union{WorkflowBinding,Nothing} = nothing
    height::Union{WorkflowBinding,Nothing} = nothing
    batch_size::Union{WorkflowBinding,Nothing} = nothing
    steps::Union{WorkflowBinding,Nothing} = nothing
    cfg_scale::Union{WorkflowBinding,Nothing} = nothing
    sampler_name::Union{WorkflowBinding,Nothing} = nothing
    scheduler::Union{WorkflowBinding,Nothing} = nothing
    filename_prefix::Union{WorkflowBinding,Nothing} = nothing
end

"""
Local ComfyUI image-generation backend.

When `model` is supplied, `bindings.model` must identify the workflow loader
input that receives that exact model filename. Workflows that already hard-code
a model remain valid when `model=nothing`.

The client uses HTTP polling only: `/prompt`, `/history/{prompt_id}`, and
`/view`. `loopback_only=true` rejects every non-loopback server URL.
"""
struct ComfyBackend <: ImageInterface.AbstractImageBackend
    base_url::String
    workflow::Dict{String,Any}
    bindings::ComfyBindings
    model::Union{String,Nothing}
    poll_interval::Float64
    timeout::Float64
    request_timeout::Float64
    loopback_only::Bool
    resource_policy::ImageInterface.ImageResourcePolicy
    client_id::String
end

function ComfyBackend(
    workflow::AbstractDict,
    bindings::ComfyBindings;
    model::Union{AbstractString,Nothing} = nothing,
    base_url::AbstractString = COMFY_BASE,
    poll_interval::Real = 0.25,
    timeout::Real = 300.0,
    request_timeout::Real = 30.0,
    loopback_only::Bool = true,
    resource_policy::ImageInterface.ImageResourcePolicy =
        ImageInterface.IMAGE_AUTO,
    client_id::AbstractString = string(uuid4()),
)::ComfyBackend
    endpoint = _base_url(base_url, loopback_only)
    poll = _positive_float(poll_interval, "poll_interval")
    completion_timeout = _positive_float(timeout, "timeout")
    request_limit = _positive_float(request_timeout, "request_timeout")
    identifier = String(strip(client_id))
    isempty(identifier) && throw(ArgumentError(
        "client_id must not be empty",
    ))
    plain_workflow = _plain_workflow(workflow)
    _validate_workflow(plain_workflow)
    _validate_bindings(plain_workflow, bindings)
    selected_model = _optional_model(model)
    if !isnothing(selected_model) && isnothing(bindings.model)
        throw(ArgumentError(
            "ComfyBackend model selection requires bindings.model",
        ))
    end
    return ComfyBackend(
        endpoint,
        plain_workflow,
        bindings,
        selected_model,
        poll,
        completion_timeout,
        request_limit,
        loopback_only,
        resource_policy,
        identifier,
    )
end

function ComfyBackend(
    workflow_path::AbstractString,
    bindings::ComfyBindings;
    kwargs...,
)::ComfyBackend
    return ComfyBackend(
        load_workflow(workflow_path),
        bindings;
        kwargs...,
    )
end

function _optional_model(
    value::Union{AbstractString,Nothing},
)::Union{String,Nothing}
    isnothing(value) && return nothing
    model = String(strip(value))
    isempty(model) && throw(ArgumentError(
        "ComfyUI model name must not be empty",
    ))
    return model
end

"""Return a copy of `backend` configured with a different image model."""
function with_model(
    backend::ComfyBackend,
    model::Union{AbstractString,Nothing},
)::ComfyBackend
    return ComfyBackend(
        backend.workflow,
        backend.bindings;
        model = model,
        base_url = backend.base_url,
        poll_interval = backend.poll_interval,
        timeout = backend.timeout,
        request_timeout = backend.request_timeout,
        loopback_only = backend.loopback_only,
        resource_policy = backend.resource_policy,
        client_id = backend.client_id,
    )
end

function _positive_float(value::Real, name::String)::Float64
    converted = Float64(value)
    isfinite(converted) && converted > 0.0 || throw(ArgumentError(
        "$name must be positive and finite",
    ))
    return converted
end

function _base_url(
    value::AbstractString,
    loopback_only::Bool,
)::String
    endpoint = String(rstrip(strip(value), '/'))
    isempty(endpoint) && throw(ArgumentError(
        "ComfyUI base URL must not be empty",
    ))
    uri = try
        HTTP.URI(endpoint)
    catch error
        throw(ArgumentError(
            "invalid ComfyUI base URL: " * sprint(showerror, error),
        ))
    end
    scheme = lowercase(String(uri.scheme))
    scheme in ("http", "https") || throw(ArgumentError(
        "ComfyUI base URL must use http or https",
    ))
    host = lowercase(String(uri.host))
    isempty(host) && throw(ArgumentError(
        "ComfyUI base URL must include a host",
    ))
    loopback_only && !_loopback_host(host) && throw(ArgumentError(
        "loopback_only=true requires localhost, 127.0.0.1, or ::1",
    ))
    path = String(uri.path)
    path in ("", "/") || throw(ArgumentError(
        "ComfyUI base URL must not include an API path",
    ))
    query = isnothing(uri.query) ? "" : String(uri.query)
    isempty(query) || throw(ArgumentError(
        "ComfyUI base URL must not include a query string",
    ))
    fragment = isnothing(uri.fragment) ? "" : String(uri.fragment)
    isempty(fragment) || throw(ArgumentError(
        "ComfyUI base URL must not include a fragment",
    ))
    userinfo = isnothing(uri.userinfo) ? "" : String(uri.userinfo)
    isempty(userinfo) || throw(ArgumentError(
        "ComfyUI base URL must not include credentials",
    ))
    return endpoint
end

function _loopback_host(host::String)::Bool
    normalized = lowercase(replace(host, "[" => "", "]" => ""))
    return normalized == "localhost" ||
           normalized == "127.0.0.1" ||
           normalized == "::1"
end

function _plain_json(value)
    if value isa AbstractDict || value isa JSON3.Object
        return Dict{String,Any}(
            String(key) => _plain_json(item)
            for (key, item) in pairs(value)
        )
    elseif value isa AbstractVector || value isa JSON3.Array
        return Any[_plain_json(item) for item in value]
    end
    return value
end

function _plain_workflow(workflow)::Dict{String,Any}
    converted = _plain_json(workflow)
    converted isa Dict{String,Any} || throw(ArgumentError(
        "ComfyUI API workflow must be a JSON object",
    ))
    return converted
end

"""Load a workflow exported from ComfyUI in API format."""
function load_workflow(path::AbstractString)::Dict{String,Any}
    source = abspath(normpath(String(path)))
    isfile(source) || throw(ArgumentError(
        "ComfyUI workflow does not exist: $source",
    ))
    parsed = try
        JSON3.read(read(source, String))
    catch error
        throw(ArgumentError(
            "invalid ComfyUI workflow JSON: " * sprint(showerror, error),
        ))
    end
    workflow = _plain_workflow(parsed)
    _validate_workflow(workflow)
    return workflow
end

function _validate_workflow(workflow::Dict{String,Any})::Nothing
    isempty(workflow) && throw(ArgumentError(
        "ComfyUI workflow must contain at least one node",
    ))
    for (node_id, value) in workflow
        value isa AbstractDict || throw(ArgumentError(
            "workflow node '$node_id' must be a JSON object",
        ))
        haskey(value, "class_type") || throw(ArgumentError(
            "workflow node '$node_id' has no class_type",
        ))
        inputs = get(value, "inputs", nothing)
        inputs isa AbstractDict || throw(ArgumentError(
            "workflow node '$node_id' has no inputs object",
        ))
    end
    return nothing
end

function _validate_binding(
    workflow::Dict{String,Any},
    binding::WorkflowBinding,
)::Nothing
    node = get(workflow, binding.node_id, nothing)
    node isa AbstractDict || throw(ArgumentError(
        "workflow contains no node '$(binding.node_id)'",
    ))
    inputs = get(node, "inputs", nothing)
    inputs isa AbstractDict || throw(ArgumentError(
        "workflow node '$(binding.node_id)' has no inputs object",
    ))
    haskey(inputs, binding.input_name) || throw(ArgumentError(
        "workflow node '$(binding.node_id)' has no input " *
        "'$(binding.input_name)'",
    ))
    return nothing
end

function _configured_bindings(
    bindings::ComfyBindings,
)::Vector{Pair{String,WorkflowBinding}}
    values = Pair{String,WorkflowBinding}[
        "positive_prompt" => bindings.positive_prompt,
    ]
    optional = (
        "model" => bindings.model,
        "negative_prompt" => bindings.negative_prompt,
        "seed" => bindings.seed,
        "width" => bindings.width,
        "height" => bindings.height,
        "batch_size" => bindings.batch_size,
        "steps" => bindings.steps,
        "cfg_scale" => bindings.cfg_scale,
        "sampler_name" => bindings.sampler_name,
        "scheduler" => bindings.scheduler,
        "filename_prefix" => bindings.filename_prefix,
    )
    for (name, binding) in optional
        isnothing(binding) || push!(values, name => binding)
    end
    return values
end

function _validate_bindings(
    workflow::Dict{String,Any},
    bindings::ComfyBindings,
)::Nothing
    occupied = Dict{Tuple{String,String},String}()
    for (name, binding) in _configured_bindings(bindings)
        _validate_binding(workflow, binding)
        key = (binding.node_id, binding.input_name)
        previous = get(occupied, key, nothing)
        isnothing(previous) || throw(ArgumentError(
            "ComfyUI bindings '$previous' and '$name' target the same input",
        ))
        occupied[key] = name
    end
    return nothing
end

function _set_binding!(
    workflow::Dict{String,Any},
    binding::WorkflowBinding,
    value,
)::Nothing
    _validate_binding(workflow, binding)
    node = workflow[binding.node_id]::AbstractDict
    inputs = node["inputs"]::AbstractDict
    inputs[binding.input_name] = value
    return nothing
end

function _set_optional_binding!(
    workflow::Dict{String,Any},
    binding::Union{WorkflowBinding,Nothing},
    value,
    name::String,
)::Nothing
    isnothing(value) && return nothing
    isnothing(binding) && throw(ArgumentError(
        "the selected ComfyUI workflow has no $name binding",
    ))
    _set_binding!(workflow, binding, value)
    return nothing
end

function _workflow_for_request(
    backend::ComfyBackend,
    request::ImageInterface.ImageGenerationRequest;
    ordinal::Int,
    batch_size::Int,
)::Dict{String,Any}
    workflow = deepcopy(backend.workflow)
    bindings = backend.bindings
    _set_binding!(
        workflow,
        bindings.positive_prompt,
        request.prompt,
    )
    _set_optional_binding!(
        workflow,
        bindings.model,
        backend.model,
        "model",
    )
    _set_optional_binding!(
        workflow,
        bindings.negative_prompt,
        request.negative_prompt,
        "negative_prompt",
    )
    seed = isnothing(request.seed) ? nothing : request.seed + ordinal - 1
    _set_optional_binding!(workflow, bindings.seed, seed, "seed")
    _set_optional_binding!(workflow, bindings.width, request.width, "width")
    _set_optional_binding!(
        workflow,
        bindings.height,
        request.height,
        "height",
    )
    _set_optional_binding!(workflow, bindings.steps, request.steps, "steps")
    _set_optional_binding!(
        workflow,
        bindings.cfg_scale,
        request.cfg_scale,
        "cfg_scale",
    )
    _set_optional_binding!(
        workflow,
        bindings.sampler_name,
        request.sampler_name,
        "sampler_name",
    )
    _set_optional_binding!(
        workflow,
        bindings.scheduler,
        request.scheduler,
        "scheduler",
    )
    if !isnothing(bindings.batch_size)
        _set_binding!(workflow, bindings.batch_size, batch_size)
    end
    if !isnothing(bindings.filename_prefix)
        suffix = ordinal == 1 ? "" : "-$ordinal"
        _set_binding!(
            workflow,
            bindings.filename_prefix,
            request.filename_prefix * suffix,
        )
    end
    return workflow
end

function _request(
    backend::ComfyBackend,
    method::String,
    path::String;
    body::Union{Vector{UInt8},Nothing} = nothing,
)::HTTP.Response
    url = backend.base_url * path
    response = if method == "GET"
        isnothing(body) || throw(ArgumentError(
            "GET requests must not include a body",
        ))
        HTTP.get(
            url;
            status_exception = false,
            redirect = false,
            proxy = HTTP.ProxyConfig(),
            request_timeout = backend.request_timeout,
        )
    elseif method == "POST"
        payload = isnothing(body) ? UInt8[] : body
        HTTP.post(
            url;
            headers = _JSON_HEADERS,
            body = payload,
            status_exception = false,
            redirect = false,
            proxy = HTTP.ProxyConfig(),
            request_timeout = backend.request_timeout,
        )
    else
        throw(ArgumentError("unsupported ComfyUI HTTP method '$method'"))
    end
    200 <= response.status < 300 || throw(ErrorException(
        "ComfyUI $method $path returned HTTP $(response.status): " *
        first(String(copy(response.body)), 2000),
    ))
    return response
end

function _post_json(
    backend::ComfyBackend,
    path::String,
    payload,
)::Dict{String,Any}
    response = _request(
        backend,
        "POST",
        path;
        body = collect(codeunits(JSON3.write(payload))),
    )
    parsed = try
        JSON3.read(String(copy(response.body)))
    catch error
        throw(ErrorException(
            "ComfyUI $path returned invalid JSON: " *
            sprint(showerror, error),
        ))
    end
    return _plain_workflow(parsed)
end

function _get_json(
    backend::ComfyBackend,
    path::String,
)::Dict{String,Any}
    response = _request(backend, "GET", path)
    parsed = try
        JSON3.read(String(copy(response.body)))
    catch error
        throw(ErrorException(
            "ComfyUI $path returned invalid JSON: " *
            sprint(showerror, error),
        ))
    end
    return _plain_workflow(parsed)
end

function _json_description(value)::String
    value isa AbstractString && return String(value)
    return String(JSON3.write(value))
end

function _queue_prompt(
    backend::ComfyBackend,
    workflow::Dict{String,Any},
)::String
    body = _post_json(
        backend,
        "/prompt",
        Dict{String,Any}(
            "prompt" => workflow,
            "client_id" => backend.client_id,
        ),
    )
    prompt_id = get(body, "prompt_id", nothing)
    if isnothing(prompt_id)
        error_value = get(body, "error", "unknown validation error")
        node_errors = get(body, "node_errors", Dict{String,Any}())
        throw(ErrorException(
            "ComfyUI rejected the workflow: " *
            _json_description(error_value) * "; node_errors=" *
            _json_description(node_errors),
        ))
    end
    return String(prompt_id)
end

function _history_entry(
    history::Dict{String,Any},
    prompt_id::String,
)::Union{Dict{String,Any},Nothing}
    entry = get(history, prompt_id, nothing)
    entry isa AbstractDict && return _plain_workflow(entry)
    haskey(history, "outputs") && return history
    return nothing
end

function _status_messages(status::AbstractDict)::Vector{Any}
    messages = get(status, "messages", Any[])
    messages isa AbstractVector || return Any[messages]
    return Any[message for message in messages]
end

function _execution_error(entry::Dict{String,Any})::Union{String,Nothing}
    status = get(entry, "status", nothing)
    status isa AbstractDict || return nothing
    status_text = lowercase(String(get(status, "status_str", "")))
    messages = _status_messages(status)
    message_error = any(messages) do message
        message isa AbstractVector || return false
        isempty(message) && return false
        event = lowercase(String(first(message)))
        return event in (
            "execution_error",
            "execution_interrupted",
            "execution_cached_error",
        )
    end
    if status_text in ("error", "failed", "failure") || message_error
        return isempty(messages) ? status_text : String(JSON3.write(messages))
    end
    return nothing
end

function _history_complete(entry::Dict{String,Any})::Bool
    status = get(entry, "status", nothing)
    if status isa AbstractDict
        completed = get(status, "completed", nothing)
        completed isa Bool && return completed
        status_text = lowercase(String(get(status, "status_str", "")))
        status_text == "success" && return true
    end
    outputs = get(entry, "outputs", nothing)
    return outputs isa AbstractDict && !isempty(outputs)
end

function _wait_for_history(
    backend::ComfyBackend,
    prompt_id::String,
)::Dict{String,Any}
    deadline = time() + backend.timeout
    while time() < deadline
        history = _get_json(
            backend,
            "/history/" * _url_encode(prompt_id),
        )
        entry = _history_entry(history, prompt_id)
        if !isnothing(entry)
            failure = _execution_error(entry)
            isnothing(failure) || throw(ErrorException(
                "ComfyUI workflow execution failed: $failure",
            ))
            _history_complete(entry) && return entry
        end
        sleep(min(backend.poll_interval, max(0.0, deadline - time())))
    end
    throw(ErrorException(
        "ComfyUI workflow $prompt_id did not complete within " *
        "$(backend.timeout) seconds",
    ))
end

function _url_encode(value::AbstractString)::String
    output = IOBuffer()
    for byte in codeunits(String(value))
        allowed = (UInt8('a') <= byte <= UInt8('z')) ||
                  (UInt8('A') <= byte <= UInt8('Z')) ||
                  (UInt8('0') <= byte <= UInt8('9')) ||
                  byte == UInt8('-') || byte == UInt8('_') ||
                  byte == UInt8('.') || byte == UInt8('~')
        if allowed
            write(output, byte)
        else
            encoded = uppercase(string(byte; base = 16, pad = 2))
            write(output, UInt8('%'))
            write(output, codeunits(encoded))
        end
    end
    return String(take!(output))
end

function _view_path(image::AbstractDict)::String
    filename = get(image, "filename", nothing)
    isnothing(filename) && throw(ErrorException(
        "ComfyUI image output contains no filename",
    ))
    subfolder = String(get(image, "subfolder", ""))
    folder_type = String(get(image, "type", "output"))
    return "/view?filename=" * _url_encode(String(filename)) *
           "&subfolder=" * _url_encode(subfolder) *
           "&type=" * _url_encode(folder_type)
end

function _download_image(
    backend::ComfyBackend,
    image::AbstractDict,
)::Vector{UInt8}
    response = _request(backend, "GET", _view_path(image))
    return copy(response.body)
end

function _extension(image::AbstractDict)::String
    filename = basename(String(get(image, "filename", "image.png")))
    extension = lowercase(splitext(filename)[2])
    isempty(extension) && return ".bin"
    occursin(r"^\.[a-z0-9]{1,8}$", extension) || return ".bin"
    return extension
end

function _collect_images!(
    artifacts::Vector{ArtifactRef},
    backend::ComfyBackend,
    entry::Dict{String,Any},
    store::ArtifactStore.Store,
    request::ImageInterface.ImageGenerationRequest,
    job_index::Int,
)::Nothing
    outputs = get(entry, "outputs", nothing)
    outputs isa AbstractDict || throw(ErrorException(
        "ComfyUI history contains no outputs object",
    ))
    image_index = 0
    for node_id in sort!(String.(collect(keys(outputs))))
        node_output = get(outputs, node_id, nothing)
        node_output isa AbstractDict || continue
        images = get(node_output, "images", Any[])
        images isa AbstractVector || continue
        for image in images
            image isa AbstractDict || continue
            image_index += 1
            bytes = _download_image(backend, image)
            filename = request.filename_prefix * "-" *
                       string(job_index) * "-" *
                       string(image_index) * _extension(image)
            push!(
                artifacts,
                ArtifactStore.write_artifact(
                    store,
                    bytes;
                    filename,
                    prefix = request.filename_prefix,
                ),
            )
        end
    end
    image_index > 0 || throw(ErrorException(
        "ComfyUI workflow completed without image outputs",
    ))
    return nothing
end

"""Return `true` when the configured local ComfyUI server responds."""
function available(backend::ComfyBackend)::Bool
    try
        _request(backend, "GET", "/system_stats")
        return true
    catch
        return false
    end
end

ImageInterface.backend_name(::ComfyBackend)::String = "comfyui"

ImageInterface.is_local_backend(backend::ComfyBackend)::Bool =
    _loopback_host(lowercase(String(HTTP.URI(backend.base_url).host)))

ImageInterface.resource_policy(
    backend::ComfyBackend,
)::ImageInterface.ImageResourcePolicy = backend.resource_policy

function ImageInterface.generate_images(
    backend::ComfyBackend,
    request::ImageInterface.ImageGenerationRequest,
    store::ArtifactStore.Store,
)::ModelResponse
    started = time_ns()
    artifacts = ArtifactRef[]
    batched = !isnothing(backend.bindings.batch_size)
    job_count = batched ? 1 : request.count
    batch_size = batched ? request.count : 1
    for job_index in 1:job_count
        workflow = _workflow_for_request(
            backend,
            request;
            ordinal = job_index,
            batch_size,
        )
        prompt_id = _queue_prompt(backend, workflow)
        history = _wait_for_history(backend, prompt_id)
        _collect_images!(
            artifacts,
            backend,
            history,
            store,
            request,
            job_index,
        )
    end
    return ModelResponse(
        "Generated $(length(artifacts)) image artifact(s) with ComfyUI.";
        artifacts,
        done = true,
        done_reason = "stop",
        metrics = ResponseMetrics(
            client_total_duration_ns = Int(time_ns() - started),
        ),
    )
end

export COMFY_BASE,
       WorkflowBinding,
       ComfyBindings,
       ComfyBackend,
       with_model,
       load_workflow,
       available

end # module ComfyClient
