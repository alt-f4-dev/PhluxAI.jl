using Test
using Sockets

include("../src/tools/Sandbox.jl")
using .Sandboxing

@testset "Bubblewrap filesystem layout" begin
    mktempdir() do workspace
        cfg = SandboxConfig(workspace; backend = SANDBOX_BWRAP, network = false)

        command = Sandboxing._bwrap_command(cfg, ["/bin/true"], cfg.workspace)

        argv = command.exec
        operations = [argv[i:(i + 2)] for i in 1:(length(argv) - 2)]

        @test !(["--ro-bind", "/", "/"] in operations)
        @test ["--ro-bind", "/usr", "/usr"] in operations
        @test ["--bind", cfg.workspace, "/workspace"] in operations
        @test "--unshare-net" in argv
        @test any(i -> argv[i:(i + 1)] == ["--tmpfs", "/home"], 1:(length(argv) - 1))
    end
end

@testset "Bubblewrap filesystem boundaries" begin
    mktempdir() do workspace
        cfg = SandboxConfig(
            workspace;
            backend = SANDBOX_BWRAP,
            network = false,
        )

        # Test whether a host file is visible.
        probe = run_sandboxed(
            cfg,
            ["/usr/bin/test", "-e", "/etc/passwd"],
        )

        @test !probe.timed_out
        @test probe.exit_code == 1

        # Test writing within the authorized workspace.
        filename = "phluxai_probe.txt"

        result = run_sandboxed(
            cfg,
            ["/usr/bin/touch", "/workspace/$filename"],
        )

        @test result.success
        @test !result.timed_out
        @test isfile(joinpath(workspace, filename))
    end
end

@testset "Bubblewrap network isolation" begin
    mktempdir() do workspace
        cfg = SandboxConfig(
            workspace;
            backend = SANDBOX_BWRAP,
            network = false,
        )

        # 1. Verify that Bubblewrap creates a separate
        # network namespace.
        readlink_exe = Sys.which("readlink")
        @test !isnothing(readlink_exe)

        if !isnothing(readlink_exe)
            host_ns = readlink("/proc/self/ns/net")

            result = run_sandboxed(
                cfg,
                [readlink_exe, "/proc/self/ns/net"],
            )

            @test result.success
            @test strip(result.stdout) != host_ns
        end

        # 2. Verify that the sandbox cannot connect to
        # a host-side loopback TCP listener.
        python = Sys.which("python3")
        @test !isnothing(python)

        if !isnothing(python)
            server = listen(ip"127.0.0.1", 0)

            try
                _, port = getsockname(server)

                # Confirm the listener is accessible
                # from the host.
                connection = connect(ip"127.0.0.1", port)
                close(connection)

                probe = """
                import socket
                import sys

                try:
                    connection = socket.create_connection(
                        ("127.0.0.1", int(sys.argv[1])),
                        timeout=1,
                    )
                except OSError:
                    print("BLOCKED")
                else:
                    connection.close()
                    print("CONNECTED")
                """

                result = run_sandboxed(
                    cfg,
                    [python, "-c", probe, string(port)],
                )

                @test result.success
                @test strip(result.stdout) == "BLOCKED"
            finally
                close(server)
            end
        end
    end
end

@testset "Unsafe local backend policy" begin
    mktempdir() do workspace
        cfg = SandboxConfig(
            workspace;
            backend = SANDBOX_LOCAL,
            network = false,
            allow_unsafe_local = true,
        )

        @test_throws ArgumentError Sandboxing._local_command(
            cfg,
            ["/bin/true"],
            cfg.workspace,
        )
    end
end

@testset "Bubblewrap process timeout" begin
    mktempdir() do workspace
        cfg = SandboxConfig(
            workspace;
            backend = SANDBOX_BWRAP,
            network = false,
            timeout_seconds = 0.25,
        )

        executable = Sys.which("sleep")
        @test !isnothing(executable)

        if !isnothing(executable)
            result = run_sandboxed(
                cfg,
                [executable, "3"],
            )

            @test result.timed_out
            @test !result.success
            @test result.elapsed_seconds < 2.5
        end
    end
end

@testset "Bubblewrap descendant containment" begin
    mktempdir() do workspace
        cfg = SandboxConfig(
            workspace;
            backend = SANDBOX_BWRAP,
            network = false,
            timeout_seconds = 0.75,
        )

        python = Sys.which("python3")
        @test !isnothing(python)

        if !isnothing(python)
            parent_code = raw"""
            import subprocess
            import sys
            import time

            child_code = (
                "from pathlib import Path; import time; "
                "Path('/workspace/child_started').write_text('started'); "
                "time.sleep(1.5); "
                "Path('/workspace/child_survived').write_text('survived')"
            )

            subprocess.Popen(
                [sys.executable, "-c", child_code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )

            time.sleep(5)
            """

            result = run_sandboxed(
                cfg,
                [python, "-c", parent_code],
            )

            started = joinpath(workspace, "child_started")
            survived = joinpath(workspace, "child_survived")

            @test result.timed_out
            @test !result.success
            @test isfile(started)

            # Allow enough time for an orphaned child
            # to complete its delayed write.
            sleep(2.0)

            @test !isfile(survived)
        end
    end
end
