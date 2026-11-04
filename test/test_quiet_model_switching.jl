using Test
using Logging
using PhluxAI

const OL = PhluxAI.OllamaClient
const MS = PhluxAI.ModelSessions

@testset "Phase 4: quiet transitions and truthful close text (CPU only)" begin
    @testset "Info gate leaves warnings/errors intact" begin
        output = IOBuffer()
        with_logger(SimpleLogger(output, Logging.Info)) do
            OL._model_lifecycle_info(false, "HIDDEN_INTERNAL_TRANSITION")
            OL._model_lifecycle_info(true, "VISIBLE_SESSION_BOUNDARY")
            @warn "VISIBLE_WARNING"
        end
        text = String(take!(output))
        @test !occursin("HIDDEN_INTERNAL_TRANSITION", text)
        @test occursin("VISIBLE_SESSION_BOUNDARY", text)
        @test occursin("VISIBLE_WARNING", text)
    end

    @testset "Conversation persistence messages" begin
        @test MS._repl_close_message(nothing) == "Session closed. Goodbye."
        @test MS._repl_close_message("sessions/observatory") == "Session saved. Goodbye."
        @test MS._repl_close_message("/home/example/research/session") == "Session saved. Goodbye."
    end

    @testset "Injected callbacks remain compatible with transient offload state" begin
        mktempdir(pwd(); prefix = "phluxai-quiet-switch-") do directory
            session = PhluxAI.ModelSession(
                "qwen2.5-coder:latest";
                session_dir = joinpath(directory, "conversation"),
                artifact_dir = joinpath(directory, "artifacts"),
                sandbox_dir = joinpath(directory, "sandbox"),
                start_model = false,
                discover_capabilities = false,
                stop_model_on_close = false,
            )
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
                @test url == session.session.model.base_url
                starts[] += 1
                return nothing
            end
            try
                @test isdefined(MS, :_quiet_start_model_after_image!)
                @test MS._restore_ollama_after_image!(session) === nothing
                @test !session.ollama_unloaded_for_image
                MS._offload_ollama_for_image!(session; stop_model = stop_callback)
                @test session.ollama_unloaded_for_image
                MS._offload_ollama_for_image!(session; stop_model = stop_callback)
                @test stops[] == 1
                MS._restore_ollama_after_image!(session; start_model = start_callback)
                @test !session.ollama_unloaded_for_image
                MS._restore_ollama_after_image!(session; start_model = start_callback)
                @test starts[] == 1
            finally
                isopen(session) && close(session)
            end
        end
    end
end

println("CPU-only checks finished. No Ollama or ComfyUI subprocess was intentionally started, stopped, or queried.")
