using Test
using Dates
using JSON3
using PhluxAI

const Analysis = PhluxAI.AnalysisTools

@testset "Source analysis API and provenance" begin
    mktempdir() do root
        store = SourceStore(joinpath(root, "sources"))
        path = joinpath(root, "input.pdf")
        write(path, "%PDF-1.7\nfixture")
        source = import_source(store, path)
        sandbox = SandboxConfig(joinpath(root, "workspace"))
        artifacts = PhluxAI.ArtifactStore.Store(joinpath(root, "artifacts"))
        authorizations = String[]
        function authorize(call, declaration)
            push!(authorizations, call.id)
            return Dict(source.sha256 => source_grant(store, source))
        end
        context = analysis_context(sandbox, artifacts, store; authorize)
        session = ModelSession("analysis-fixture"; session_dir = joinpath(root, "session"), artifact_dir = artifacts.root,
                                start_model = false, discover_capabilities = false, stop_model_on_close = false)
        try
            set_tool_context!(session, context)
            set_tool_mode!(session, TOOLS_ASK)
            @test session.tool_context === context
            @test session.tool_context.registry.policy_only
        finally
            close(session)
        end
        @test length(PhluxAI.JupyTools.tool_specs(context.registry)) == 8
        @test context.registry.policy_only
        @test_throws ArgumentError analysis_registry(store; limits = AnalysisLimits(max_pages = 0))
        @test_throws ArgumentError analysis_registry(store; python = "python3")
        @test_throws ArgumentError analysis_registry(store; policy = ExecutionPolicy(network = true))
        @test_throws ArgumentError analyze_source(context, source, :invalid)
        @test_throws ArgumentError analyze_source(context, source, :hdf5_read)
        @test_throws ArgumentError analyze_source(context, source, :pdf_text; source_ids = [])
        denied = analyze_source(context, source, :pdf_metadata; mode = TOOLS_AUTO)
        @test !denied.execution.success
        @test denied.error_code == :execution_rejected
        @test denied.operation.status == :failed
        @test isempty(authorizations)
        approved = analyze_source(context, source, :pdf_text; approval_callback = (call, risk) -> true, pages = [1])
        @test !approved.execution.success
        @test occursin("delegated worker cgroup", approved.message)
        @test length(authorizations) == 1

        handler = context.registry.tools["pdf_text"].handler
        parameters = JSON3.write((source_ids = [source.sha256], pages = [1]))
        call = ToolCall("model-tool-call", "pdf_text", parameters)
        command = PhluxAI.tool_command(handler, JSON3.read(parameters), Dict(source.sha256 => "/inputs/$(source.sha256)"))
        @test command[1:3] == ["/usr/bin/python3", "-I", "-c"]
        @test JSON3.read(command[5]).limits.max_pages == 16
        @test command[end] == "/inputs/$(source.sha256)"
        @test PhluxAI.JupyTools._validate_sources(handler, JSON3.read(parameters), authorize(call, handler)) === nothing
        wrong = SourceGrant(SourceRef(repeat("0", 64), :pdf, source.size_bytes), path)
        @test_throws ArgumentError PhluxAI.JupyTools._validate_sources(handler, JSON3.read(parameters), Dict(source.sha256 => wrong))

        payload = JSON3.write((ok = true, data = [(page = 1, text = "fixture")],
                               evidence = [(scheme = "pdf_page", locator = (page = 1,))],
                               method = "pdfplumber", versions = (pdfplumber = "fixture",)))
        process = ProcessResult(command, true, false, 0, payload, "", now(UTC), 0.1)
        result = PhluxAI.JupyTools._worker_result(handler, call, process, false, false)
        envelope = JSON3.read(result.output)
        @test result.success
        @test envelope.analysis_schema_version == 1
        record = parse_contract(OperationRecord, JSON3.write(envelope.operation))
        evidence = parse_contract(EvidenceRef, JSON3.write(only(envelope.evidence)))
        @test record.status == :succeeded
        @test record.inputs == [source]
        @test record.outputs[1].sha256 == evidence.extraction_sha256
        @test record.outputs[1].size_bytes == ncodeunits(JSON3.write(envelope.data))
        @test evidence.source == source
        @test JSON3.read(evidence.locator).page == 1

        for (code, timeout, output_limited, memory) in (("timeout", true, false, false), ("output_limit", false, true, false),
                                                       ("memory_limit", false, false, true), ("worker_failed", false, false, false))
            failed = ProcessResult(command, false, timeout, nothing, "", "", now(UTC), 0.1, output_limited)
            execution = PhluxAI.JupyTools._worker_result(handler, call, failed, memory, memory)
            @test !execution.success
            @test JSON3.read(execution.output).error == code
        end
        parser_failure = ProcessResult(command, true, false, 0, "{\"ok\":false,\"error\":\"encrypted_source\",\"message\":\"encrypted\"}",
                                        "", now(UTC), 0.1)
        @test !PhluxAI.JupyTools._worker_result(handler, call, parser_failure, false, false).success
    end
end
