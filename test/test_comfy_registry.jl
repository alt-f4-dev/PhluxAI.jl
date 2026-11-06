using Test
using PhluxAI

const CL = PhluxAI.ComfyLifecycle

@testset "Managed ComfyUI registry and interactive lifetime (CPU only)" begin
    @testset "Atomic last-session transition" begin
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 0
        CL.enter_interactive_repl!()
        shutdown_started = Channel{Nothing}(1)
        finish_shutdown = Channel{Nothing}(1)
        shutdown = @async CL._leave_interactive_repl!(true, () -> begin
            put!(shutdown_started, nothing)
            take!(finish_shutdown)
            return nothing
        end)
        take!(shutdown_started)
        entering = @async CL.enter_interactive_repl!()
        yield()
        @test !istaskdone(entering)
        put!(finish_shutdown, nothing)
        wait(shutdown)
        wait(entering)
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 1
        CL.leave_interactive_repl!(; close_on_exit = false)
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 0
    end

    @testset "Cross-process metadata and registry lock" begin
        if Sys.islinux()
            previous = get(ENV, "XDG_RUNTIME_DIR", nothing)
            try
                mktempdir() do directory
                    ENV["XDG_RUNTIME_DIR"] = directory
                    stamp = CL._process_stamp(getpid())
                    @test !isnothing(stamp)
                    owner = (pid = getpid(), stamp = stamp)
                    registered = CL._with_registry_lock() do
                        CL._write_managed_owner!(owner)
                        return CL._read_managed_owner() == owner
                    end
                    @test registered
                    @test isfile(CL._owner_path())
                    stops = Ref(0)
                    mocked_stop = _owner -> begin
                        stops[] += 1
                        return true
                    end
                    @test CL._finish_managed_owner!(owner; borrowers = () -> true, stop = mocked_stop) == :borrowed
                    @test stops[] == 0
                    @test isfile(CL._owner_path())
                    @test CL._finish_managed_owner!(owner; borrowers = () -> false, stop = mocked_stop) == :stopped
                    @test stops[] == 1
                    @test CL._read_managed_owner() === nothing
                    @test isfile(joinpath(CL._lease_directory(), "registry.lock"))
                end
            finally
                if isnothing(previous)
                    pop!(ENV, "XDG_RUNTIME_DIR", nothing)
                else
                    ENV["XDG_RUNTIME_DIR"] = previous
                end
            end
        else
            @test CL._with_registry_lock(() -> true)
        end
    end

    @testset "Startup readiness helper refuses closed endpoint" begin
        @test !CL._wait_for_api("http://127.0.0.1:1"; timeout = 0.01, poll_interval = 0.005)
    end
end

println("CPU-only registry tests complete. No Ollama/ComfyUI launch, stop, or image generation was requested.")
