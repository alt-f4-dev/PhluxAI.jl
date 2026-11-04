module ComfyWorkflow

import ..ComfyClient: ComfyBackend, ComfyBindings, WorkflowBinding

"""Standard local text-to-image checkpoint graph. No model weights are downloaded."""
function checkpoint_workflow()::Dict{String,Any}
    return Dict{String,Any}(
        "4" => Dict("class_type" => "CheckpointLoaderSimple", "inputs" => Dict{String,Any}(
            "ckpt_name" => "__SELECT_CHECKPOINT__",
        )),
        "5" => Dict("class_type" => "EmptyLatentImage", "inputs" => Dict{String,Any}(
            "width" => 512, "height" => 512, "batch_size" => 1,
        )),
        "6" => Dict("class_type" => "CLIPTextEncode", "inputs" => Dict{String,Any}(
            "text" => "", "clip" => Any["4", 1],
        )),
        "7" => Dict("class_type" => "CLIPTextEncode", "inputs" => Dict{String,Any}(
            "text" => "", "clip" => Any["4", 1],
        )),
        "3" => Dict("class_type" => "KSampler", "inputs" => Dict{String,Any}(
            "seed" => 0,
            "steps" => 20,
            "cfg" => 7.0,
            "sampler_name" => "euler",
            "scheduler" => "normal",
            "denoise" => 1.0,
            "model" => Any["4", 0],
            "positive" => Any["6", 0],
            "negative" => Any["7", 0],
            "latent_image" => Any["5", 0],
        )),
        "8" => Dict("class_type" => "VAEDecode", "inputs" => Dict{String,Any}(
            "samples" => Any["3", 0], "vae" => Any["4", 2],
        )),
        "9" => Dict("class_type" => "SaveImage", "inputs" => Dict{String,Any}(
            "filename_prefix" => "phluxai", "images" => Any["8", 0],
        )),
    )
end

"""Bindings for the standard checkpoint graph, including optional image parameters."""
function checkpoint_bindings()::ComfyBindings
    return ComfyBindings(
        positive_prompt = WorkflowBinding("6", "text"),
        model = WorkflowBinding("4", "ckpt_name"),
        negative_prompt = WorkflowBinding("7", "text"),
        seed = WorkflowBinding("3", "seed"),
        width = WorkflowBinding("5", "width"),
        height = WorkflowBinding("5", "height"),
        batch_size = WorkflowBinding("5", "batch_size"),
        steps = WorkflowBinding("3", "steps"),
        cfg_scale = WorkflowBinding("3", "cfg"),
        sampler_name = WorkflowBinding("3", "sampler_name"),
        scheduler = WorkflowBinding("3", "scheduler"),
        filename_prefix = WorkflowBinding("9", "filename_prefix"),
    )
end

"""
    checkpoint_backend(model) -> ComfyBackend

Create the built-in standard-checkpoint text-to-image backend. Supports checkpoints
compatible with CheckpointLoaderSimple + CLIPTextEncode + KSampler + VAEDecode.
Other model architectures may require set_default_image_workflow! instead.
"""
function checkpoint_backend(model::AbstractString)::ComfyBackend
    selected = String(strip(model))
    suffix = lowercase(splitext(selected)[2])
    suffix in (".safetensors", ".ckpt") || throw(ArgumentError(
        "No default image workflow is configured for '$selected'. " *
        "The built-in workflow supports standard .safetensors or .ckpt checkpoints; " *
        "use set_default_image_workflow!(workflow, bindings) for another architecture.",
    ))
    basename(selected) == selected || throw(ArgumentError(
        "@imagemodel expects a checkpoint filename, not a filesystem path",
    ))
    return ComfyBackend(checkpoint_workflow(), checkpoint_bindings(); model = selected)
end

export checkpoint_backend, checkpoint_workflow, checkpoint_bindings

end # module DefaultComfyWorkflow
