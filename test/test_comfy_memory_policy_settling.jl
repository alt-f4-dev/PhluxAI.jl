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
const DATA_ROOT = joinpath(homedir(), ".local", "share", "PhluxAI")
const COMFY_OUTPUT = joinpath(DATA_ROOT, "comfy", "ComfyUI", "output")

secs(ns) = round(ns / 1e9; digits = 3)
node(kind, inputs) = Dict{String,Any}("class_type" => kind, "inputs" => inputs)

function resident_models()
    response = HTTP.get(OLLAMA_URL * "/api/ps"; request_timeout = 10.0, proxy = HTTP.ProxyConfig())
    data = JSON3.read(response.body)
    return [String(get(item, :name, get(item, :model, ""))) for item in get(data, :models, Any[])]
end

function gpu_used_mib()
    result = strip(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String))
    lines = split(result, '\n')
    length(lines) == 1 || error("Expected one GPU for this benchmark; got $(length(lines))")
    return parse(Int, strip(only(lines)))
end

function ensure_comfy_idle(; timeout = 15.0)
    deadline = time() + timeout
    while true
        response = HTTP.get(COMFY_URL * "/queue"; request_timeout = 10.0, proxy = HTTP.ProxyConfig())
        queue = JSON3.read(response.body)
        running = get(queue, :queue_running, Any[])
        pending = get(queue, :queue_pending, Any[])
        isempty(running) && isempty(pending) && return nothing
        time() >= deadline && error("ComfyUI queue is busy; benchmark requires exclusive idle access")
        sleep(0.2)
    end
end

function print_qwen_residency()
    response = HTTP.get(OLLAMA_URL * "/api/ps"; request_timeout = 10.0, proxy = HTTP.ProxyConfig())
    data = JSON3.read(response.body)
    for item in get(data, :models, Any[])
        String(get(item, :name, get(item, :model, ""))) == TEXT_MODEL || continue
        size_bytes = Int(get(item, :size, 0))
        vram_bytes = Int(get(item, :size_vram, 0))
        if size_bytes > 0
            println("  Qwen VRAM residency: ", round(100 * vram_bytes / size_bytes; digits = 1), "%")
        end
        return nothing
    end
    return nothing
end

# This GPU-specific sampling belongs in the benchmark, not ComfyClient.
# A fall of at least 1024 MiB is expected when our 512x512 image model is cached.
# Two stable successive samples help avoid accepting a transient partial release.
function wait_for_gpu_release(initial_mib; min_drop_mib = 1024, timeout_s = 10.0)
    deadline = time() + timeout_s
    previous = initial_mib
    stable = 0
    while time() < deadline
        sleep(0.1)
        current = gpu_used_mib()
        sufficient_drop = initial_mib - current >= min_drop_mib
        stable = sufficient_drop && abs(current - previous) <= 32 ? stable + 1 : 0
        stable >= 2 && return current
        previous = current
    end
    error("ComfyUI release was not observed: initial=$(initial_mib) MiB, last=$(previous) MiB")
end

function release_comfy!(label)
    ensure_comfy_idle()
    before = gpu_used_mib()
    started = time_ns()
    PhluxAI.ComfyClient.release_memory!(make_backend("phluxai-release-probe"))
    acknowledged_ns = time_ns() - started

    # Low baseline GPU usage indicates no large ComfyUI model cache remains.
    # Otherwise require a substantial observable decrease before proceeding.
    after = before >= 1536 ? wait_for_gpu_release(before) : gpu_used_mib()
    elapsed = time_ns() - started
    println("  $label: acknowledgement $(secs(acknowledged_ns)) s; observed settling $(secs(elapsed - acknowledged_ns)) s")
    println("  GPU used $(before) → $(after) MiB (observed release $(before - after) MiB)")
    return (elapsed_ns = elapsed, acknowledgement_ns = acknowledged_ns,
            settling_ns = elapsed - acknowledged_ns, used_mib = after)
end

function workflow_for(output_prefix)
    return Dict{String,Any}(
        "4" => node("CheckpointLoaderSimple", Dict("ckpt_name" => IMAGE_CHECKPOINT)),
        "5" => node("EmptyLatentImage", Dict("width" => 512, "height" => 512, "batch_size" => 1)),
        "6" => node("CLIPTextEncode", Dict("text" => "", "clip" => Any["4", 1])),
        "7" => node("CLIPTextEncode", Dict("text" => "", "clip" => Any["4", 1])),
        "3" => node("KSampler", Dict(
            "seed" => 42, "steps" => 12, "cfg" => 7.0, "sampler_name" => "euler",
            "scheduler" => "normal", "denoise" => 1.0, "model" => Any["4", 0],
            "positive" => Any["6", 0], "negative" => Any["7", 0], "latent_image" => Any["5", 0],
        )),
        "8" => node("VAEDecode", Dict("samples" => Any["3", 0], "vae" => Any["4", 2])),
        "9" => node("SaveImage", Dict("filename_prefix" => output_prefix, "images" => Any["8", 0])),
    )
end

function make_backend(output_prefix)
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
        # Deliberately leave filename_prefix unbound: SaveImage must
        # write only inside the current test's temporary directory.
    )
    return ComfyBackend(
        workflow_for(output_prefix), bindings;
        model = IMAGE_CHECKPOINT,
        base_url = COMFY_URL,
        poll_interval = 0.1,
        timeout = 600.0,
        request_timeout = 60.0,
        loopback_only = true,
        resource_policy = IMAGE_AUTO,
    )
end

function image_request(seed)
    return ImageGenerationRequest(
        "A small red wooden cabin beside a quiet mountain lake, daylight, landscape photography";
        negative_prompt = "blurry, distorted, low quality",
        width = 512, height = 512, count = 1,
        seed = seed, steps = 12, cfg_scale = 7.0,
        sampler_name = "euler", scheduler = "normal", filename_prefix = "phluxai-policy",
    )
end

function run_text(session, prompt)
    started = time_ns()
    response = respond!(
        session, prompt;
        temperature = 0.0, max_tokens = 64,
        stream = true, print_tokens = false, tool_mode = TOOLS_OFF,
        request_timeout = 180.0, read_idle_timeout = 180.0,
    )
    wall = time_ns() - started
    @test response.done
    @test !isempty(strip(response.text))
    @test response.metrics.eval_count > 0
    @test !isnothing(response.metrics.time_to_first_token_ns)
    return (wall_ns = wall, metrics = response.metrics)
end

function run_image(session, seed, comfy_dir, expected_pngs)
    started = time_ns()
    response = generate_images!(session, image_request(seed))
    wall = time_ns() - started
    @test response.done
    @test length(response.artifacts) == 1
    if length(response.artifacts) == 1
        artifact = only(response.artifacts)
        path = ArtifactStore.resolve_artifact(artifact_store(session), artifact)
        @test artifact.mime_type == "image/png"
        @test isfile(path)
        if isfile(path)
            @test filesize(path) > 1024
        end
    end
    @test count(name -> endswith(name, ".png"), readdir(comfy_dir)) == expected_pngs
    @test !(TEXT_MODEL in resident_models())
    return (wall_ns = wall, client_ns = response.metrics.client_total_duration_ns)
end

function stop_qwen_if_resident!()
    try
        if TEXT_MODEL in resident_models()
            PhluxAI.OllamaClient.stop_model!(TEXT_MODEL, OLLAMA_URL; request_timeout = 30.0)
        end
    catch error
        @warn "Unable to unload Qwen after benchmark" exception = (error, catch_backtrace())
    end
    return nothing
end

function run_case(label, do_release, seed)
    ensure_comfy_idle()
    isempty(resident_models()) || error("Ollama model already resident before $label; cannot compare cases safely")
    println("\n========== $label ==========")
    println("  Initial GPU used: ", gpu_used_mib(), " MiB")

    return mktempdir(COMFY_OUTPUT; prefix = "phluxai-policy-") do comfy_dir
        mktempdir(DATA_ROOT; prefix = "policy-benchmark-") do bench_dir
            prefix = joinpath(basename(comfy_dir), "phluxai-policy")
            backend = make_backend(prefix)
            @test is_local_backend(backend)
            @test should_unload_ollama(backend, true)
            session = ModelSession(
                TEXT_MODEL;
                base_url = OLLAMA_URL,
                cfg = BudgetConfig(num_ctx = 4096, num_thread = 8, max_tokens = 64),
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
                first = run_text(session, "Explain Julia's multiple dispatch in one sentence.")
                @test TEXT_MODEL in resident_models()
                println("  Initial text: $(secs(first.wall_ns)) s; model load $(secs(first.metrics.load_duration_ns)) s")
                print_qwen_residency()

                image1 = run_image(session, seed, comfy_dir, 1)
                used_after_image = gpu_used_mib()
                println("  Image 1: $(secs(image1.wall_ns)) s; GPU used $used_after_image MiB")

                policy = if do_release
                    release_comfy!("After first image")
                else
                    # No release request or artificial settling delay in the retain arm.
                    (elapsed_ns = 0, acknowledgement_ns = 0, settling_ns = 0,
                     used_mib = gpu_used_mib())
                end
                println("  GPU used before text resumption: $(policy.used_mib) MiB")

                resumed = run_text(
                    session,
                    "Resume the same conversation. In one sentence, explain Julia's type stability.",
                )
                @test TEXT_MODEL in resident_models()
                used_after_text = gpu_used_mib()
                println("  Resumed text: $(secs(resumed.wall_ns)) s; model load $(secs(resumed.metrics.load_duration_ns)) s")
                println("  Resumed throughput: $(round(generation_rate(resumed.metrics); digits = 2)) tokens/s")
                println("  GPU used after text: $used_after_text MiB")
                println("  Resumed first token: $(secs(resumed.metrics.time_to_first_token_ns)) s")
                print_qwen_residency()

                image2 = run_image(session, seed + 1, comfy_dir, 2)
                println("  Image 2: $(secs(image2.wall_ns)) s (includes Ollama unload)")
                println("  GPU used after image 2: ", gpu_used_mib(), " MiB")

                return (
                    after_image_mib = used_after_image,
                    before_text_mib = policy.used_mib,
                    after_text_mib = used_after_text,
                    release_ns = policy.elapsed_ns,
                    initial_image_ns = image1.wall_ns,
                    resumed_text_ns = resumed.wall_ns,
                    resumed_load_ns = resumed.metrics.load_duration_ns,
                    resumed_tps = generation_rate(resumed.metrics),
                    next_image_ns = image2.wall_ns,
                )
            finally
                try
                    close(session)
                catch error
                    @warn "Session close failed" exception = (error, catch_backtrace())
                end
                stop_qwen_if_resident!()
            end
        end
    end
end

isdir(DATA_ROOT) || error("PhluxAI directory not found: $DATA_ROOT")
isdir(COMFY_OUTPUT) || error("ComfyUI output directory not found: $COMFY_OUTPUT")
ensure_comfy_idle()
isempty(resident_models()) || error("Stop loaded Ollama models before this benchmark")

# Two distinct sampler seeds per case avoid reusing cached image outputs.
seed_base = Int(mod(time_ns(), 2_000_000_000))
seed_base = max(seed_base, 100_000)
results = Dict{Symbol,Any}()
try
    @testset "PhluxAI ComfyUI memory policy A/B" begin
        for (label, release, seed) in (
            ("A - retain ComfyUI cache", false, seed_base),
            ("B - release ComfyUI cache", true, seed_base + 100),
        )
            # Normalize both cases to an unloaded ComfyUI model/cache.
            release_comfy!("Before $label (normalization)")
            results[release ? :release : :retain] = run_case(label, release, seed)
        end
    end
finally
    stop_qwen_if_resident!()
    # Leave ComfyUI running but request that it relinquish cached VRAM.
    try
        release_comfy!("Final cleanup")
    catch error
        @warn "ComfyUI final memory-release request failed" exception = (error, catch_backtrace())
    end
end

if haskey(results, :retain) && haskey(results, :release)
    a, b = results[:retain], results[:release]
    println("\n========== A/B comparison (different idle durations; interpret wall-time deltas cautiously) ==========")
    println("GPU used after first image (retain / release): $(a.after_image_mib) / $(b.after_image_mib) MiB")
    println("GPU used before Ollama (retain / release): $(a.before_text_mib) / $(b.before_text_mib) MiB")
    println("Observed VRAM reduction after release: $(b.after_image_mib - b.before_text_mib) MiB")
    println("GPU used after Ollama (retain / release): $(a.after_text_mib) / $(b.after_text_mib) MiB")
    println("Resumed Ollama wall (retain / release): $(secs(a.resumed_text_ns)) / $(secs(b.resumed_text_ns)) s")
    println("Resumed Ollama load (retain / release): $(secs(a.resumed_load_ns)) / $(secs(b.resumed_load_ns)) s")
    println("Resumed Ollama throughput (retain / release): $(round(a.resumed_tps; digits = 2)) / $(round(b.resumed_tps; digits = 2)) tokens/s")
    println("Next image wall (retain / release): $(secs(a.next_image_ns)) / $(secs(b.next_image_ns)) s")
    println("Observed release acknowledgement + settling: $(secs(b.release_ns)) s")
end

println("\nScoped test session data and all four generated images cleaned up.")
