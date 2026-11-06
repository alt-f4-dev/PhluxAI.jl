using Test

const ROOT = dirname(@__DIR__)
const CPU_SUITES = (
    "test_comfy_lifecycle.jl",
    "test_comfy_interactive_api.jl",
    "test_image_repl_continuity.jl",
    "test_quiet_model_switching.jl",
    "test_comfy_registry.jl",
    "test_model_session_initial_state.jl",
)

# Independent Julia processes avoid module redefinitions and shared mutable state.
ENV["PHLUXAI_BANNER"] = "0"
ENV["PHLUXAI_COMFY_AUTOSTART"] = "0"
ENV["RESEAU_PRECOMPILE_ONLY"] = "eventloops,internal_poll,socket_ops,tcp,host_resolvers"

@testset "PhluxAI CPU-only integration regressions" begin
    for suite in CPU_SUITES
        @testset "$suite" begin
            command = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, suite))`
            @test success(pipeline(command; stdout = stdout, stderr = stderr))
        end
    end
end
