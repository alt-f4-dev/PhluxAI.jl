using Test
using JSON3
using PhluxAI

struct ReadOnlyToolContractStub <: AbstractToolHandler end

@testset "Tool-handler and execution-policy contracts (CPU only)" begin
    @testset "Safe defaults and validation" begin
        policy = ExecutionPolicy()
        @test policy.isolation == :bubblewrap
        @test policy.approval == :required
        @test !policy.network
        @test policy.source_access == :authorized_only
        @test policy.max_memory_bytes > 0
        @test_throws ArgumentError ExecutionPolicy(isolation = :unknown)
        @test_throws ArgumentError ExecutionPolicy(isolation = :trusted_local)
        @test ExecutionPolicy(isolation = :trusted_local, allow_unsafe_local = true).isolation == :trusted_local
        @test_throws ArgumentError ExecutionPolicy(isolation = :trusted_local, allow_unsafe_local = true, approval = :automatic)
        @test_throws ArgumentError ExecutionPolicy(approval = :automatic, network = true)
        @test_throws ArgumentError ExecutionPolicy(source_access = :unrestricted)
        @test_throws ArgumentError ExecutionPolicy(timeout_seconds = Inf)
        @test_throws ArgumentError ExecutionPolicy(timeout_seconds = 0)
        @test_throws ArgumentError ExecutionPolicy(max_memory_bytes = 0)
        @test_throws ArgumentError ExecutionPolicy(max_output_bytes = -1)
    end

    @testset "Handler declaration uses existing ToolSpec and ToolRisk" begin
        schema = "{\"type\":\"object\",\"properties\":{\"page\":{\"type\":\"integer\"}}}"
        spec = ToolSpec("pdf.inspect", "Inspect a source", schema)
        declaration = ToolHandlerDeclaration("pdf.inspect.v1", spec, TOOL_READ)
        @test declaration.spec === spec
        @test declaration.risk == TOOL_READ
        @test declaration.policy.isolation == :bubblewrap
        @test_throws ArgumentError ToolHandlerDeclaration("../evil", spec, TOOL_READ)
        @test_throws ArgumentError ToolHandlerDeclaration("pdf", ToolSpec("x", "", "{}"), TOOL_READ)
        @test_throws ArgumentError ToolHandlerDeclaration("pdf", ToolSpec("x", "", "[]"), TOOL_READ)
        @test_throws ArgumentError ToolHandlerDeclaration("pdf", ToolSpec("x", "", "{"), TOOL_READ)
        @test_throws ArgumentError ToolHandlerDeclaration("writer", spec, TOOL_WRITE; policy = ExecutionPolicy(approval = :automatic))
        @test_throws ArgumentError ToolHandlerDeclaration("network", spec, TOOL_NETWORK)
        network_policy = ExecutionPolicy(network = true, approval = :required)
        @test ToolHandlerDeclaration("network", spec, TOOL_NETWORK; policy = network_policy).risk == TOOL_NETWORK
    end

    @testset "Round-trip metadata; reject tampered fields" begin
        policy = ExecutionPolicy(network = true, timeout_seconds = 2.5)
        schema = "{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}"
        spec = ToolSpec("pkg.inspect", "Inspect metadata", schema)
        declaration = ToolHandlerDeclaration("pkg.inspect.v1", spec, TOOL_NETWORK; policy)
        for (type, value) in ((ExecutionPolicy, policy), (ToolHandlerDeclaration, declaration))
            encoded = contract_json(value)
            @test JSON3.read(encoded).schema_version == 1
            decoded = parse_contract(type, encoded)
            @test contract_json(decoded) == encoded
        end
        encoded = contract_json(declaration)
        json = Dict{String,Any}(String(k) => v for (k,v) in pairs(JSON3.read(encoded)))
        json["risk"] = "TOOL_WRITE"
        @test_throws ArgumentError parse_contract(ToolHandlerDeclaration, JSON3.write(json))
        json["risk"] = "UNKNOWN"
        @test_throws ArgumentError parse_contract(ToolHandlerDeclaration, JSON3.write(json))
        json["risk"] = "TOOL_NETWORK"
        json["extra"] = 1
        @test_throws ArgumentError parse_contract(ToolHandlerDeclaration, JSON3.write(json))
    end

    @testset "No execution handler activated by Stage 2" begin
        @test_throws MethodError PhluxAI.ToolContracts.tool_declaration(ReadOnlyToolContractStub())
        @test !isdefined(PhluxAI.ToolContracts, :execute_handler!)
        @test length(PhluxAI.JupyTools.default_registry().tools) == 7
    end
end
