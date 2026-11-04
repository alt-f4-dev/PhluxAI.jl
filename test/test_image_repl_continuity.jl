using Test
using PhluxAI

const MS = PhluxAI.ModelSessions
const CC = PhluxAI.ComfyClient

@testset "Phase 4: image REPL continuity (CPU only)" begin
    @testset "@imagemodel keeps backend state without returning full workflow" begin
        configured = @imagemodel "v1-5-pruned-emaonly-fp16.safetensors"
        @test isnothing(configured)
        backend = PhluxAI._take_image_backend!()
        @test backend isa CC.ComfyBackend
        @test backend.model == "v1-5-pruned-emaonly-fp16.safetensors"
    end

    @testset "Offload and restore state transitions" begin
        root = joinpath(pwd(), "test")
        mktempdir(root; prefix = "image-repl-continuity-") do directory
            ms = PhluxAI.ModelSession(
                "qwen2.5-coder:latest";
                session_dir = joinpath(directory, "session"),
                artifact_dir = joinpath(directory, "artifacts"),
                sandbox_dir = joinpath(directory, "sandbox"),
                start_model = false,
                discover_capabilities = false,
                stop_model_on_close = false,
            )
            try
                @test !ms.ollama_unloaded_for_image
                stops = Ref(0)
                starts = Ref(0)

                stop_callback = (model; request_timeout = 10.0) -> begin
                    @test model.name == "qwen2.5-coder:latest"
                    @test request_timeout == 10.0
                    stops[] += 1
                    return nothing
                end
                start_callback = (name, url) -> begin
                    @test name == "qwen2.5-coder:latest"
                    @test url == ms.session.model.base_url
                    starts[] += 1
                    return nothing
                end

                MS._offload_ollama_for_image!(ms; stop_model = stop_callback)
                @test ms.ollama_unloaded_for_image
                @test stops[] == 1
                MS._offload_ollama_for_image!(ms; stop_model = stop_callback)
                @test stops[] == 1

                @test_throws ErrorException MS._restore_ollama_after_image!(
                    ms;
                    start_model = (_name, _url) -> error("synthetic startup failure"),
                )
                @test ms.ollama_unloaded_for_image
                @test starts[] == 0

                MS._restore_ollama_after_image!(ms; start_model = start_callback)
                @test !ms.ollama_unloaded_for_image
                @test starts[] == 1
                MS._restore_ollama_after_image!(ms; start_model = start_callback)
                @test starts[] == 1

                @test_throws ErrorException MS._offload_ollama_for_image!(
                    ms;
                    stop_model = (_model; request_timeout = 10.0) -> error("synthetic shutdown failure"),
                )
                @test !ms.ollama_unloaded_for_image

                # With the marker true, close must persist the session WITHOUT
                # invoking an Ollama stop request, even when stop-on-close is set.
                MS._offload_ollama_for_image!(ms; stop_model = stop_callback)
                @test ms.ollama_unloaded_for_image
                ms.stop_model_on_close = true
                close(ms)
                @test !isopen(ms)
            finally
                isopen(ms) && close(ms)
            end
        end
    end
end

println("CPU-only continuity checks complete. No Ollama/ComfyUI inference or CLI launch/stop was requested.")
