using Test
using SHA: sha256
using PhluxAI

struct SandboxPolicyFixture <: AbstractToolHandler
    declaration::ToolHandlerDeclaration
    argv::Vector{String}
end

PhluxAI.tool_declaration(handler::SandboxPolicyFixture) = handler.declaration

function PhluxAI.tool_command(handler::SandboxPolicyFixture, arguments, sources)
    return append!(copy(handler.argv), values(sources))
end

function register_fixture!(registry, name, argv; policy = ExecutionPolicy(approval = :automatic, timeout_seconds = 20))
    schema = "{\"type\":\"object\",\"properties\":{\"source_ids\":{\"type\":\"array\"}}}"
    spec = ToolSpec(name, "Run a controlled CI fixture", schema)
    handler = SandboxPolicyFixture(ToolHandlerDeclaration("$name.v1", spec, TOOL_READ; policy), argv)
    return register!(registry, handler)
end

@testset "Policy dispatch under Bubblewrap and delegated cgroup v2" begin
    workers = ENV["PHLUXAI_WORKER_CGROUP"]
    before = Set(readdir(workers))
    @test Sys.islinux()
    @test Sys.which("bwrap") !== nothing
    @test isfile("/usr/bin/python3")
    mktempdir() do root
        sandbox = SandboxConfig(joinpath(root, "workspace"); network = true)
        private_path = joinpath(sandbox.workspace, "private.txt")
        write(private_path, "private")
        visible_path = joinpath(root, "visible")
        extra_path = joinpath(root, "extra")
        write(visible_path, "authorized")
        write(extra_path, "ungranted!")
        source(path) = SourceRef(bytes2hex(open(sha256, path)), :text, filesize(path))
        visible_grant = SourceGrant(source(visible_path), visible_path)
        extra_grant = SourceGrant(source(extra_path), extra_path)
        registry = ToolRegistry(; policy_only = true)
        inspect = raw"""
            import os, pathlib, sys

            path = pathlib.Path(sys.argv[3])
            assert path.read_text() == "authorized"
            assert not pathlib.Path("/inputs/extra").exists()
            assert not pathlib.Path(sys.argv[1]).exists()
            assert not pathlib.Path("/workspace/private.txt").exists()
            assert not pathlib.Path("/sys/fs/cgroup").exists()
            assert os.readlink("/proc/self/ns/net") != sys.argv[2]
            try:
                path.write_text("changed")
            except OSError as error:
                assert error.errno in (13, 30)
            else:
                raise AssertionError("source mount is writable")
            print("source checks passed")
            """
        net_namespace = readlink("/proc/self/ns/net")
        register_fixture!(registry, "inspect", ["/usr/bin/python3", "-c", inspect, private_path, net_namespace])
        memory = raw"""
            import subprocess, sys

            def allocate():
                allocation = bytearray(48 * 1024 * 1024)
                allocation[::4096] = b"x" * (len(allocation) // 4096)
                return allocation

            if sys.argv[1] == "child":
                allocation = allocate()
                print("ready", flush=True)
                sys.stdin.read()
            elif sys.argv[1] == "tree":
                child = subprocess.Popen(
                    [sys.executable, "-c", sys.argv[2], "child"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
                )
                assert child.stdout.readline() == b"ready\n"
                allocation = allocate()
                child.communicate()
            else:
                allocation = allocate()
                print("single worker fits")
            """
        memory_policy = ExecutionPolicy(approval = :automatic, max_memory_bytes = 80 * 1024 * 1024, timeout_seconds = 20)
        register_fixture!(registry, "single", ["/usr/bin/python3", "-c", memory, "single"]; policy = memory_policy)
        register_fixture!(registry, "tree", ["/usr/bin/python3", "-c", memory, "tree", memory]; policy = memory_policy)
        register_fixture!(registry, "output", ["/usr/bin/python3", "-c", "print('x' * 8192)"];
                          policy = ExecutionPolicy(approval = :automatic, max_output_bytes = 1024))
        register_fixture!(registry, "timeout", ["/usr/bin/python3", "-c", "import time; time.sleep(20)"];
                          policy = ExecutionPolicy(approval = :automatic, timeout_seconds = 1))
        function grants(call, declaration)
            @test declaration.handler_id == "inspect.v1"
            selected = call.id == "bad_digest" ? SourceGrant(visible_grant.source, extra_path) : visible_grant
            return Dict("visible" => selected, "extra" => extra_grant)
        end
        artifacts = PhluxAI.ArtifactStore.Store(joinpath(root, "artifacts"))
        context = ToolContext(sandbox, artifacts; registry, worker_cgroup = workers, authorize_sources = grants)
        dispatch(id, name, args = "{}") = PhluxAI.JupyTools.execute_tool_call(context, ToolCall(id, name, args); mode = TOOLS_AUTO)

        allowed = dispatch("allowed", "inspect", "{\"source_ids\":[\"visible\"]}")
        @test allowed.success || (@info "worker output" allowed.output; false)
        @test occursin("source checks passed", allowed.output)
        @test read(visible_path, String) == "authorized"
        denied = dispatch("denied", "inspect", "{\"source_ids\":[\"unknown\"]}")
        @test !denied.success && occursin("not authorized", denied.output)
        mismatch = dispatch("bad_digest", "inspect", "{\"source_ids\":[\"visible\"]}")
        @test !mismatch.success && occursin("source digest mismatch", mismatch.output)

        single = dispatch("single", "single")
        @test single.success || (@info "worker output" single.output; false)
        @test occursin("single worker fits", single.output)
        tree = dispatch("tree", "tree")
        @test !tree.success
        @test occursin("memory_limit_exceeded: true", tree.output) || (@info "worker output" tree.output; false)
        @test occursin("worker_oom_killed: true", tree.output)
        overflow = dispatch("output", "output")
        @test !overflow.success && occursin("output_limit_exceeded: true", overflow.output)
        timeout = dispatch("timeout", "timeout")
        @test !timeout.success && occursin("timed_out: true", timeout.output)
    end
    @test Set(readdir(workers)) == before
end
