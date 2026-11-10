using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "suite_manifest.jl"))

const TEST_PROFILE = lowercase(strip(get(ENV, "PHLUXAI_TEST_PROFILE", "cpu")))
const TEST_PROFILES = ("cpu", "sandbox", "live", "benchmark", "all")
TEST_PROFILE in TEST_PROFILES || throw(ArgumentError(
    "PHLUXAI_TEST_PROFILE must be one of $(join(TEST_PROFILES, ", ")); got '$(TEST_PROFILE)'",
))

# Child processes avoid module redefinitions and shared mutable package state.
ENV["PHLUXAI_BANNER"] = "0"
ENV["PHLUXAI_COMFY_AUTOSTART"] = "0"
get!(ENV, "RESEAU_PRECOMPILE_ONLY", "eventloops,internal_poll,socket_ops,tcp,host_resolvers")

@testset "PhluxAI $(TEST_PROFILE) regressions" begin
    @testset "Suite manifest integrity" begin
        registered = [suite.file for suite in SUITE_MANIFEST]
        discovered = filter(name -> startswith(name, "test_") && endswith(name, ".jl"), readdir(@__DIR__))
        @test length(registered) == length(unique(registered))
        @test Set(registered) == Set(discovered)
        @test all(suite -> suite.tier in SUITE_TIERS, SUITE_MANIFEST)
    end

    selected = filter(suite -> TEST_PROFILE == "all" || string(suite.tier) == TEST_PROFILE, SUITE_MANIFEST)
    @test !isempty(selected)
    println("Selected $(length(selected)) of $(length(SUITE_MANIFEST)) registered test suites; profile=$(TEST_PROFILE).")

    for suite in selected
        @testset "$(suite.file)" begin
            command = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, suite.file))`
            @test success(pipeline(command; stdout = stdout, stderr = stderr))
        end
    end
end
