using Test
using SHA: sha256
using PhluxAI

const Tools = PhluxAI.JupyTools

struct PolicyDispatchFixture <: AbstractToolHandler
    declaration::ToolHandlerDeclaration
    commands::Vector{String}
end

PhluxAI.tool_declaration(handler::PolicyDispatchFixture) = handler.declaration

function PhluxAI.tool_command(handler::PolicyDispatchFixture, arguments, sources)
    push!(handler.commands, handler.declaration.spec.name)
    return ["/usr/bin/true"]
end

@testset "Declared handlers deny calls before worker launch" begin
    mktempdir() do root
        sandbox = SandboxConfig(joinpath(root, "workspace"))
        artifacts = PhluxAI.ArtifactStore.Store(joinpath(root, "artifacts"))
        registry = ToolRegistry(; policy_only = true)
        spec = ToolSpec("inspect", "Inspect a granted source", "{\"type\":\"object\"}")
        handler = PolicyDispatchFixture(ToolHandlerDeclaration("inspect.v1", spec, TOOL_READ), String[])
        register!(registry, handler)
        @test Tools.tool_specs(registry) == [spec]
        @test_throws ArgumentError register!(registry, handler)
        @test_throws ArgumentError register!(registry, Tools.RegisteredTool(spec, TOOL_READ))
        @test_throws ArgumentError register!(ToolRegistry(), handler)

        approvals = String[]
        function approve(call, risk)
            push!(approvals, call.id)
            return risk == TOOL_READ
        end
        inspected = String[]
        function grants(call, declaration)
            push!(inspected, declaration.handler_id)
            return Dict{String,SourceGrant}()
        end
        context = ToolContext(sandbox, artifacts; registry, authorize_sources = grants)
        unknown = ToolCall("unknown", "inspect", "{\"source_ids\":[\"ungranted\"]}")
        @test occursin("not approved", Tools.execute_tool_call(context, unknown; mode = TOOLS_AUTO).output)
        @test isempty(inspected)
        denied = Tools.execute_tool_call(context, unknown; mode = TOOLS_AUTO, approval_callback = approve)
        @test !denied.success
        @test occursin("source access was not authorized", denied.output)
        @test inspected == ["inspect.v1"]

        invalid = ToolCall("invalid", "inspect", "{\"source_ids\":[\"../bad\"]}")
        @test occursin("invalid source identifier", Tools.execute_tool_call(context, invalid; approval_callback = approve).output)
        duplicate = ToolCall("duplicate", "inspect", "{\"source_ids\":[\"same\",\"same\"]}")
        @test occursin("duplicate source identifier", Tools.execute_tool_call(context, duplicate; approval_callback = approve).output)
        missing_cgroup = ToolCall("missing_cgroup", "inspect", "{}")
        @test occursin("delegated worker cgroup", Tools.execute_tool_call(context, missing_cgroup; approval_callback = approve).output)
        local_only = SandboxConfig(joinpath(root, "local"); backend = SANDBOX_LOCAL, network = true, allow_unsafe_local = true)
        local_context = ToolContext(local_only, artifacts; registry)
        local_call = ToolCall("local", "inspect", "{}")
        @test occursin("require Bubblewrap", Tools.execute_tool_call(local_context, local_call; approval_callback = approve).output)
        @test approvals == ["unknown", "invalid", "duplicate", "missing_cgroup", "local"]
        @test isempty(handler.commands)
    end
end

@testset "Verify granted source bytes before mounting" begin
    mktempdir() do root
        original = joinpath(root, "source")
        snapshot = joinpath(root, "staged")
        write(original, "trusted")
        ref = SourceRef(bytes2hex(open(sha256, original)), :text, filesize(original))
        grant = SourceGrant(ref, original)
        @test PhluxAI.PolicyWorkers._copy_grant(snapshot, grant) === nothing
        @test read(snapshot, String) == "trusted"
        write(original, "changed")
        @test_throws ArgumentError PhluxAI.PolicyWorkers._copy_grant(snapshot, grant)
        write(original, "too long")
        @test_throws ArgumentError PhluxAI.PolicyWorkers._copy_grant(snapshot, grant)
    end
end

@testset "Built-in tool dispatch and approval" begin
    mktempdir() do root
        sandbox = SandboxConfig(joinpath(root, "workspace"))
        artifacts = PhluxAI.ArtifactStore.Store(joinpath(root, "artifacts"))
        context = ToolContext(sandbox, artifacts; jupy_executable = "unused", jupip_executable = "unused")
        path = joinpath(sandbox.workspace, "example.txt")
        write_call = ToolCall("write", "write_text_file", "{\"path\":\"example.txt\",\"content\":\"test\"}")
        approvals = Tuple{String,ToolRisk}[]
        approve(call, risk) = (push!(approvals, (call.id, risk)); true)

        for mode in (TOOLS_OFF, TOOLS_ASK, TOOLS_AUTO)
            @test !Tools.execute_tool_call(context, write_call; mode).success
            @test !isfile(path)
        end
        @test !Tools.execute_tool_call(context, write_call; mode = TOOLS_OFF, approval_callback = approve).success
        @test isempty(approvals)
        broken_approval(call, risk) = error("approval unavailable")
        denied = Tools.execute_tool_call(context, write_call; mode = TOOLS_ASK, approval_callback = broken_approval)
        @test !denied.success
        @test denied.output == "tool execution was not approved"
        @test !isfile(path)

        @test Tools.execute_tool_call(context, write_call; mode = TOOLS_AUTO, approval_callback = approve).success
        @test approvals == [("write", TOOL_WRITE)]
        @test read(path, String) == "test"

        read_call = ToolCall("read", "read_text_file", "{\"path\":\"example.txt\"}")
        @test !Tools.execute_tool_call(context, read_call; mode = TOOLS_ASK).success
        read_result = Tools.execute_tool_call(context, read_call; mode = TOOLS_AUTO, approval_callback = approve)
        @test read_result.success
        @test read_result.output == "test"
        @test approvals == [("write", TOOL_WRITE)]

        for (name, risk) in (("run_julia_script", TOOL_EXECUTE), ("install_python_packages", TOOL_NETWORK))
            call = ToolCall(name, name, "{}")
            @test !Tools.execute_tool_call(context, call; mode = TOOLS_AUTO).success
            seen = Ref(false)
            deny(candidate, actual_risk) = (seen[] = candidate.id == name && actual_risk == risk; false)
            execution = Tools.execute_tool_call(context, call; mode = TOOLS_AUTO, approval_callback = deny)
            @test !execution.success
            @test execution.output == "tool execution was not approved"
            @test seen[]
        end

        malformed = ToolCall("invalid", "write_text_file", "[]")
        @test !Tools.execute_tool_call(context, malformed; mode = TOOLS_AUTO, approval_callback = approve).success
        unknown = ToolCall("unknown", "missing_tool", "{}")
        @test !Tools.execute_tool_call(context, unknown; mode = TOOLS_AUTO, approval_callback = approve).success
        @test approvals == [("write", TOOL_WRITE)]

        legacy = ToolRegistry()
        legacy_context = ToolContext(sandbox, artifacts; registry = legacy, jupy_executable = "unused", jupip_executable = "unused")
        read_spec = context.registry.tools[read_call.name].spec
        register!(legacy, Tools.RegisteredTool(read_spec, TOOL_READ))
        @test Tools.execute_tool_call(legacy_context, read_call; mode = TOOLS_AUTO).output == "test"
        write_spec = context.registry.tools[write_call.name].spec
        register!(legacy, Tools.RegisteredTool(write_spec, TOOL_READ))
        downgrade = Tools.execute_tool_call(legacy_context, write_call; mode = TOOLS_AUTO)
        @test !downgrade.success
        @test occursin("risk does not match", downgrade.output)
    end
end
