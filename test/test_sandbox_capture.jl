using Test

include("../src/tools/Sandbox.jl")
using .Sandboxing

@testset "Subprocess output and deadline" begin
    if Sys.isunix()
        mktempdir() do workspace
            config = SandboxConfig(workspace; backend = SANDBOX_LOCAL, network = true, allow_unsafe_local = true,
                                   max_output_bytes = 8, timeout_seconds = 4.0)
            exact = run_sandboxed(config, ["/bin/sh", "-c", "printf abcd; printf efgh >&2"])
            @test exact.success
            @test exact.stdout == "abcd"
            @test exact.stderr == "efgh"
            @test !exact.output_limited

            overflow = run_sandboxed(config, ["/bin/sh", "-c", "printf abcde; printf fghi >&2; exec sleep 8"])
            @test !overflow.success
            @test overflow.output_limited
            @test !overflow.timed_out
            @test ncodeunits(overflow.stdout) + ncodeunits(overflow.stderr) == 8

            split_utf8 = run_sandboxed(config, ["/bin/sh", "-c", raw"printf '1234567\303\251'"])
            @test split_utf8.output_limited
            @test split_utf8.stdout == "1234567?"

            failure = run_sandboxed(config, ["/bin/sh", "-c", "printf error >&2; exit 7"])
            @test failure.exit_code == 7
            @test failure.stderr == "error"
            @test !failure.success
            @test !failure.output_limited

            deadline = SandboxConfig(workspace; backend = SANDBOX_LOCAL, network = true, allow_unsafe_local = true,
                                     timeout_seconds = 0.25)
            timeout = run_sandboxed(deadline, ["/bin/sh", "-c", "trap '' TERM; printf started; exec sleep 8"])
            @test timeout.timed_out
            @test !timeout.success
            @test !timeout.output_limited
            @test timeout.elapsed_seconds < 4.0
            isolated_local = SandboxConfig(workspace; backend = SANDBOX_LOCAL, network = false, allow_unsafe_local = true)
            @test_throws ArgumentError run_sandboxed(isolated_local, ["/bin/true"])
        end
    else
        @test_skip Sys.isunix()
    end
end
