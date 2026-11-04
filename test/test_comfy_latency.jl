using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

const CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"

node(kind, inputs) = Dict{String,Any}("class_type" => kind, "inputs" => inputs)

# Explicitly identify the existing ComfyUI output location.
comfy_output_root = joinpath(homedir(), ".local", "share", "PhluxAI", "comfy", "ComfyUI", "output")

isdir(comfy_output_root) || error("ComfyUI output directory not found: $comfy_output_root")

artifact_root = joinpath(homedir(), ".local", "share", "PhluxAI", "artifacts", "live-tests")
mkpath(artifact_root)

# This directory is exclusive to the current test run.
# Julia automatically cleans it up on block exit.
mktempdir(comfy_output_root; prefix = "phluxai-live-") do comfy_run_dir

    workflow = Dict{String,Any}(
        "4" => node("CheckpointLoaderSimple", Dict("ckpt_name" => CHECKPOINT)),
        "5" => node("EmptyLatentImage", Dict("width" => 512, "height" => 512, "batch_size" => 1)),
        "6" => node("CLIPTextEncode", Dict("text" => "", "clip" => Any["4", 1])),
        "7" => node("CLIPTextEncode", Dict("text" => "", "clip" => Any["4", 1])),
        "3" => node("KSampler", Dict("seed" => 42, 
                                     "steps" => 12, 
                                     "cfg" => 7.0,
                                     "sampler_name" => "euler", 
                                     "scheduler" => "normal",
                                     "denoise" => 1.0, 
                                     "model" => Any["4", 0], 
                                     "positive" => Any["6", 0], 
                                     "negative" => Any["7", 0],
                                     "latent_image" => Any["5", 0])),
        "8" => node("VAEDecode", Dict("samples" => Any["3", 0], 
                                      "vae" => Any["4", 2])),
        "9" => node("SaveImage", Dict("filename_prefix" => joinpath(basename(comfy_run_dir), "phluxai-phase3"),
                                      "images" => Any["8", 0]))
    )

    bindings = ComfyBindings(positive_prompt = WorkflowBinding("6", "text"),
                             negative_prompt = WorkflowBinding("7", "text"),
                             model = WorkflowBinding("4", "ckpt_name"),
                             width = WorkflowBinding("5", "width"),
                             height = WorkflowBinding("5", "height"),
                             batch_size = WorkflowBinding("5", "batch_size"),
                             seed = WorkflowBinding("3", "seed"),
                             steps = WorkflowBinding("3", "steps"),
                             cfg_scale = WorkflowBinding("3", "cfg"),
                             sampler_name = WorkflowBinding("3", "sampler_name"),
                             scheduler = WorkflowBinding("3", "scheduler"))

    # The filename_prefix binding is intentionally absent.
    # This preserves the unique ComfyUI output subdirectory.

    backend = ComfyBackend(workflow, bindings; model = CHECKPOINT, 
                           base_url = "http://127.0.0.1:8188",
                           poll_interval = 0.1, timeout = 600.0, 
                           request_timeout = 60.0, loopback_only = true)

    request = ImageGenerationRequest(
        "A small red wooden cabin beside a quiet " *
        "mountain lake, daylight, landscape photography";
        negative_prompt = "blurry, distorted, low quality",
        width = 512,
        height = 512,
        count = 1,
        seed = 42,
        steps = 12,
        cfg_scale = 7.0,
        sampler_name = "euler",
        scheduler = "normal",
        filename_prefix = "phluxai-phase3",
    )

    # Automatically clean up the PhluxAI artifact directory.
    mktempdir(artifact_root; prefix = "comfy-live-") do artifact_dir
        store = ArtifactStore.Store(artifact_dir)



        @testset "ComfyUI local generation latency" begin
            @test is_local_backend(backend)
            @test PhluxAI.ComfyClient.available(backend)

            cases = (("A - first new seed", 5101),
                     ("B - warm new seed", 5102),
                     ("C - repeated seed", 5102),)

            timings = Float64[]

            for (label, seed) in cases
                run_request = ImageGenerationRequest(
                    request.prompt;
                    negative_prompt = request.negative_prompt,
                    width = request.width,
                    height = request.height,
                    count = 1,
                    seed = seed,
                    steps = request.steps,
                    cfg_scale = request.cfg_scale,
                    sampler_name = request.sampler_name,
                    scheduler = request.scheduler,
                    filename_prefix = request.filename_prefix,
                )

                started = time_ns()

                response = PhluxAI.ImageInterface.generate_images(backend, run_request, store)

                elapsed = (time_ns() - started) / 1e9
                push!(timings, elapsed)

                @test response.done
                @test length(response.artifacts) == 1

                if length(response.artifacts) == 1
                    artifact = only(response.artifacts)
                    path = ArtifactStore.resolve_artifact(store, artifact)

                    @test artifact.mime_type == "image/png"
                    @test isfile(path)

                    if isfile(path)
                        @test filesize(path) > 1024
                    end
                end

                println(label,": ",round(elapsed; digits = 3)," seconds")
            end

            if length(timings) == 3 && timings[3] > 0
                println("B/C timing ratio: ", round(timings[2] / timings[3]; digits = 2), "x")
            end
        end
    end
end

println("Temporary test image directories cleaned up.")

