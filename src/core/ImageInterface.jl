module ImageInterface

import ..ArtifactStore
import ..ProtocolTypes: ModelResponse

"""Abstract interface implemented by image-generation backends."""
abstract type AbstractImageBackend end

"""Sentinel backend used when image generation is not configured."""
struct NoImageBackend <: AbstractImageBackend end

"""
Policy controlling whether the active Ollama model is unloaded before image
inference on a second local GPU runtime.
"""
@enum ImageResourcePolicy begin
    IMAGE_KEEP_LOADED
    IMAGE_UNLOAD_OLLAMA
    IMAGE_AUTO
end

"""
    ImageGenerationRequest

Backend-neutral text-to-image request. Optional values are applied only when a
backend or workflow exposes a corresponding binding.
"""
struct ImageGenerationRequest
    prompt::String
    negative_prompt::Union{String,Nothing}
    width::Union{Int,Nothing}
    height::Union{Int,Nothing}
    count::Int
    seed::Union{Int,Nothing}
    steps::Union{Int,Nothing}
    cfg_scale::Union{Float64,Nothing}
    sampler_name::Union{String,Nothing}
    scheduler::Union{String,Nothing}
    filename_prefix::String

    function ImageGenerationRequest(
        prompt::AbstractString;
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
    )
        text = String(strip(prompt))
        isempty(text) && throw(ArgumentError(
            "image-generation prompt must not be empty",
        ))
        negative = _optional_string(negative_prompt, "negative_prompt")
        image_width = _optional_positive_int(width, "width")
        image_height = _optional_positive_int(height, "height")
        image_count = Int(count)
        image_count > 0 || throw(ArgumentError(
            "count must be positive",
        ))
        image_seed = if isnothing(seed)
            nothing
        else
            value = Int(seed)
            value >= 0 || throw(ArgumentError(
                "seed must be nonnegative",
            ))
            value
        end
        image_steps = _optional_positive_int(steps, "steps")
        scale = _optional_positive_float(cfg_scale, "cfg_scale")
        sampler = _optional_string(sampler_name, "sampler_name")
        schedule = _optional_string(scheduler, "scheduler")
        prefix = String(strip(filename_prefix))
        isempty(prefix) && throw(ArgumentError(
            "filename_prefix must not be empty",
        ))
        occursin('\0', prefix) && throw(ArgumentError(
            "filename_prefix contains a NUL byte",
        ))
        basename(prefix) == prefix || throw(ArgumentError(
            "filename_prefix must not contain directory components",
        ))
        prefix in (".", "..") && throw(ArgumentError(
            "filename_prefix is invalid",
        ))
        return new(
            text,
            negative,
            image_width,
            image_height,
            image_count,
            image_seed,
            image_steps,
            scale,
            sampler,
            schedule,
            prefix,
        )
    end
end

function _optional_string(
    value::Union{AbstractString,Nothing},
    name::String,
)::Union{String,Nothing}
    isnothing(value) && return nothing
    normalized = String(strip(value))
    isempty(normalized) && throw(ArgumentError(
        "$name must not be empty",
    ))
    return normalized
end

function _optional_positive_int(
    value::Union{Integer,Nothing},
    name::String,
)::Union{Int,Nothing}
    isnothing(value) && return nothing
    converted = Int(value)
    converted > 0 || throw(ArgumentError("$name must be positive"))
    return converted
end

function _optional_positive_float(
    value::Union{Real,Nothing},
    name::String,
)::Union{Float64,Nothing}
    isnothing(value) && return nothing
    converted = Float64(value)
    isfinite(converted) && converted > 0.0 || throw(ArgumentError(
        "$name must be positive and finite",
    ))
    return converted
end

"""Return a stable human-readable backend name."""
backend_name(backend::AbstractImageBackend)::String =
    String(nameof(typeof(backend)))

backend_name(::NoImageBackend)::String = "none"

"""Return whether the backend is constrained to the local machine."""
is_local_backend(::AbstractImageBackend)::Bool = false
is_local_backend(::NoImageBackend)::Bool = true

"""Return the backend's Ollama GPU-memory policy."""
resource_policy(::AbstractImageBackend)::ImageResourcePolicy =
    IMAGE_KEEP_LOADED

"""Return `true` when Ollama should be unloaded before generation."""
function should_unload_ollama(
    backend::AbstractImageBackend,
    ollama_is_local::Bool,
)::Bool
    policy = resource_policy(backend)
    policy == IMAGE_UNLOAD_OLLAMA && return true
    policy == IMAGE_KEEP_LOADED && return false
    return ollama_is_local && is_local_backend(backend)
end

"""
    generate_images(backend, request, artifact_store) -> ModelResponse

Execute a backend-neutral image-generation request and import all returned
images into `artifact_store`.
"""
function generate_images(
    backend::AbstractImageBackend,
    ::ImageGenerationRequest,
    ::ArtifactStore.Store,
)::ModelResponse
    throw(ArgumentError(
        "image backend $(backend_name(backend)) does not implement " *
        "generate_images",
    ))
end

function generate_images(
    ::NoImageBackend,
    ::ImageGenerationRequest,
    ::ArtifactStore.Store,
)::ModelResponse
    throw(ArgumentError(
        "image generation is not configured; provide an image_backend",
    ))
end

"""Convenience overload constructing an `ImageGenerationRequest`."""
function generate_images(
    backend::AbstractImageBackend,
    prompt::AbstractString,
    artifact_store::ArtifactStore.Store;
    kwargs...,
)::ModelResponse
    request = ImageGenerationRequest(prompt; kwargs...)
    return generate_images(backend, request, artifact_store)
end

export AbstractImageBackend,
       NoImageBackend,
       ImageResourcePolicy,
       IMAGE_KEEP_LOADED,
       IMAGE_UNLOAD_OLLAMA,
       IMAGE_AUTO,
       ImageGenerationRequest,
       backend_name,
       is_local_backend,
       resource_policy,
       should_unload_ollama,
       generate_images

end # module ImageInterface
