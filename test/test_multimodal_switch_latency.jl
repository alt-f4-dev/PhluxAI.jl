using Test
using HTTP
using JSON3

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

const OLLAMA_URL = "http://127.0.0.1:11434"
const COMFY_URL = "http://127.0.0.1:8188"
const TEXT_MODEL = "qwen2.5-coder:latest"
const IMAGE_CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"

seconds(ns) = round(ns / 1e9; digits = 3)

function resident_models()
    response = HTTP.get(
        OLLAMA_URL * "/api/ps";
        request_timeout = 10.0,
        proxy = HTTP.ProxyConfig(),
    )
    data = JSON3.read(response.body)
    return [
        String(get(item, :name, get(item, :model, "")))
        for item in get(data, :models, Any[])
    ]
end

function gpu_memory()
    return strip(read(
        `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits`,
        String,
    ))
end

function report_text(label, response, wall_ns)
    m = response.metrics
    println("\n", label)
    println("  Wall / Ollama client / server: ", seconds(wall_ns),
            " / ", seconds(m.client_total_duration_ns),
            " / ", seconds(m.total_duration_ns), " s")
    println("  Model load: ", seconds(m.load_duration_ns), " s")
    println("  Prompt tokens / evaluation: ", m.prompt_eval_count,
            " / ", seconds(m.prompt_eval_duration_ns), " s")
    println("  Output tokens / generation: ", m.eval_count,
            " / ", seconds(m.eval_duration_ns), " s")
    println("  Generation throughput: ",
            round(generation_rate(m); digits = 2), " tokens/s")
    if !isnothing(m.time_to_first_token_ns)
        println("  Time to first token: ",
                seconds(m.time_to_first_token_ns), " s")
    end
end

function assert_text_response(response)
    @test response.done
    @test !isempty(strip(response.text))
    @test response.metrics.eval_count > 0
    @test !isnothing(response.metrics.time_to_first_token_ns)
end

node(kind, inputs) = Dict{String,Any}(
    "class_type" => kind,
    "inputs" => inputs,
)

root = joinpath(homedir(), ".local", "share", "PhluxAI")
comfy_output_root = joinpath(root, "comfy", "ComfyUI", "output")
isdir(root) || error("PhluxAI directory not found: $root")
isdir(comfy_output_root) || error(
    "ComfyUI output directory not found: $comfy_output_root",
)

# Do not silently interfere with an already loaded Ollama model.
models_at_start = resident_models()
isempty(models_at_start) || error(
    "Unload active Ollama models before the controlled benchmark: " *
    join(models_at_start, ", "),
)

# Tracks whether this benchmark may have loaded the text model.
model_started = Ref(false)

try
    # ComfyUI SaveImage writes only into this run-specific directory.
    mktempdir(comfy_output_root; prefix = "phluxai-switch-") do comfy_dir
        # ModelSession state, downloaded images, and sandbox files share
        # a second run-specific directory, also removed automatically.
        mktempdir(root; prefix = "switch-benchmark-") do bench_dir
            output_prefix = joinpath(basename(comfy_dir), "phluxai-switch")

            workflow = Dict{String,Any}(
                "4" => node("CheckpointLoaderSimple", Dict(
                    "ckpt_name" => IMAGE_CHECKPOINT,
                )),
                "5" => node("EmptyLatentImage", Dict(
                    "width" => 512, "height" => 512, "batch_size" => 1,
                )),
                "6" => node("CLIPTextEncode", Dict(
                    "text" => "", "clip" => Any["4", 1],
                )),
                "7" => node("CLIPTextEncode", Dict(
                    "text" => "", "clip" => Any["4", 1],
                )),
                "3" => node("KSampler", Dict(
                    "seed" => 42, "steps" => 12, "cfg" => 7.0,
                    "sampler_name" => "euler", "scheduler" => "normal",
                    "denoise" => 1.0, "model" => Any["4", 0],
                    "positive" => Any["6", 0],
                    "negative" => Any["7", 0],
                    "latent_image" => Any["5", 0],
                )),
                "8" => node("VAEDecode", Dict(
                    "samples" => Any["3", 0], "vae" => Any["4", 2],
                )),
                "9" => node("SaveImage", Dict(
                    "filename_prefix" => output_prefix,
                    "images" => Any["8", 0],
                )),
            )

            bindings = ComfyBindings(
                positive_prompt = WorkflowBinding("6", "text"),
                negative_prompt = WorkflowBinding("7", "text"),
                model = WorkflowBinding("4", "ckpt_name"),
                width = WorkflowBinding("5", "width"),
                height = WorkflowBinding("5", "height"),
                batch_size = WorkflowBinding("5", "batch_size"),
                seed = WorkflowBinding("3", "seed"),
                steps = WorkflowBinding("3", "steps"),
                cfg_scale = WorkflowBinding("3", "cfg"),
                sampler_name = WorkflowBinding("3", "sampler_name"),
                scheduler = WorkflowBinding("3", "scheduler"),
                # Do not bind filename_prefix: keep server output confined
                # to the dedicated per-run ComfyUI output directory.
            )

            backend = ComfyBackend(
                workflow, bindings;
                model = IMAGE_CHECKPOINT,
                base_url = COMFY_URL,
                poll_interval = 0.1,
                timeout = 600.0,
                request_timeout = 60.0,
                loopback_only = true,
                resource_policy = IMAGE_AUTO,
            )

            PhluxAI.ComfyClient.available(backend) || error(
                "ComfyUI is not ready at $COMFY_URL. " *
                "Start it with `comfy launch --background`.",
            )

            session = ModelSession(
                TEXT_MODEL;
                base_url = OLLAMA_URL,
                cfg = BudgetConfig(
                    num_ctx = 4096,
                    num_thread = 8,
                    max_tokens = 64,
                ),
                system_prompt = "You are a concise software engineering assistant.",
                session_dir = joinpath(bench_dir, "session"),
                artifact_dir = joinpath(bench_dir, "artifacts"),
                sandbox_dir = joinpath(bench_dir, "sandbox"),
                tool_mode = TOOLS_OFF,
                image_backend = backend,
                start_model = false,
                stop_model_on_close = false,
                discover_capabilities = false,
                keep_alive = "5m",
            )

            try
                @testset "PhluxAI text-image-text switching" begin
                    @test is_local_backend(backend)
                    @test should_unload_ollama(backend, true)

                    # Stage 1: first Ollama turn, model initially unloaded.
                    model_started[] = true
                    started = time_ns()
                    first = respond!(
                        session,
                        "For this conversation, remember the marker " *
                        "BLUE-ORBIT-37. Explain multiple dispatch in one sentence.";
                        temperature = 0.0,
                        max_tokens = 64,
                        stream = true,
                        print_tokens = false,
                        tool_mode = TOOLS_OFF,
                        request_timeout = 180.0,
                        read_idle_timeout = 180.0,
                    )
                    first_wall = time_ns() - started
                    assert_text_response(first)
                    report_text("Stage 1 - initial Ollama response", first, first_wall)
                    @test TEXT_MODEL in resident_models()
                    println("  GPU used/free after Ollama: ", gpu_memory(), " MiB")

                    # Stage 2: public ModelSession image API. Its IMAGE_AUTO
                    # policy should unload local Ollama before ComfyUI runs.
                    # Use a fresh seed so previously cached samples do not
                    # masquerade as a new image-generation measurement.
                    seed = Int(mod(time_ns(), 2_147_483_647))
                    image_request = ImageGenerationRequest(
                        "A small red wooden cabin beside a quiet " *
                        "mountain lake, daylight, landscape photography";
                        negative_prompt = "blurry, distorted, low quality",
                        width = 512, height = 512,
                        count = 1, seed = seed, steps = 12,
                        cfg_scale = 7.0,
                        sampler_name = "euler", scheduler = "normal",
                        filename_prefix = "phluxai-switch",
                    )
                    started = time_ns()
                    image_response = generate_images!(session, image_request)
                    image_wall = time_ns() - started
                    @test image_response.done
                    @test length(image_response.artifacts) == 1
                    if length(image_response.artifacts) == 1
                        artifact = only(image_response.artifacts)
                        path = ArtifactStore.resolve_artifact(
                            artifact_store(session), artifact,
                        )
                        @test artifact.mime_type == "image/png"
                        @test isfile(path)
                        if isfile(path)
                            @test filesize(path) > 1024
                        end
                    end
                    @test count(
                        name -> endswith(name, ".png"),
                        readdir(comfy_dir),
                    ) == 1

                    image_client = image_response.metrics.client_total_duration_ns
                    println("\nStage 2 - ComfyUI image after Ollama")
                    println("  Seed: ", seed)
                    println("  Image API wall: ", seconds(image_wall), " s")
                    println("  ComfyUI client: ", seconds(image_client), " s")
                    println("  Difference (unload + session overhead): ",
                            seconds(image_wall - image_client), " s")
                    @test !(TEXT_MODEL in resident_models())
                    println("  GPU used/free after ComfyUI: ", gpu_memory(), " MiB")

                    # Stage 3: return to the exact same ModelSession.
                    # If the text-only model rejects image artifacts in
                    # conversation history, that is an integration issue
                    # to diagnose, not something this benchmark masks.
                    started = time_ns()
                    resumed = try
                        respond!(
                            session,
                            "Resume our text conversation. State the marker " *
                            "I gave you at the beginning, then give one " *
                            "sentence about type stability.";
                            temperature = 0.0,
                            max_tokens = 64,
                            stream = true,
                            print_tokens = false,
                            tool_mode = TOOLS_OFF,
                            request_timeout = 180.0,
                            read_idle_timeout = 180.0,
                        )
                    catch error
                        @error "Ollama resumption failed: inspect GPU memory and image-artifact compatibility with the text-only model" exception = (error, catch_backtrace())
                        rethrow()
                    end
                    resumed_wall = time_ns() - started
                    assert_text_response(resumed)
                    report_text("Stage 3 - Ollama resumes after ComfyUI", resumed, resumed_wall)
                    @test TEXT_MODEL in resident_models()
                    println("  GPU used/free after resumption: ", gpu_memory(), " MiB")

                    println("\nSwitching metrics")
                    println("  Initial text: ", seconds(first_wall), " s")
                    println("  Image including policy: ", seconds(image_wall), " s")
                    println("  Resumed text: ", seconds(resumed_wall), " s")
                    println("  Sum of measured stages: ",
                            seconds(first_wall + image_wall + resumed_wall), " s")
                end
            finally
                close(session)
            end
        end
    end
finally
    if model_started[]
        try
            PhluxAI.OllamaClient.stop_model!(
                TEXT_MODEL, OLLAMA_URL;
                request_timeout = 30.0,
            )
        catch error
            @warn "Ollama model cleanup failed" exception = (
                error, catch_backtrace(),
            )
        end
    end
end

println("\nTemporary session and both image-output directories cleaned up.")
