using Test
using PhluxAI

@testset "ModelSession honors initial offload state" begin
    mktempdir() do directory
        ms = ModelSession(
            "qwen2.5-coder:latest";
            session_dir = joinpath(directory, "session"),
            artifact_dir = joinpath(directory, "artifacts"),
            sandbox_dir = joinpath(directory, "sandbox"),
            start_model = false,
            discover_capabilities = false,
            stop_model_on_close = false,
            ollama_unloaded_for_image = true,
        )
        try
            @test ms.ollama_unloaded_for_image
        finally
            close(ms)
        end
    end
end
