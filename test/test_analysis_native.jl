using Test
using JSON3
using PhluxAI

include("analysis/roundtrip.jl")

@testset "Native parser worker protocol and recovery" begin
    mktempdir() do root
        python = get(ENV, "PHLUXAI_ANALYSIS_PYTHON", "/usr/bin/python3")
        store, pdf, hdf5, authorize, sandbox, artifacts = analysis_fixture_setup(root; python)
        registry = analysis_registry(store; python)
        native = SandboxConfig(joinpath(root, "native"); backend = SANDBOX_LOCAL, allow_unsafe_local = true, network = true)
        function execute_native(source, operation; kwargs...)
            handler = registry.tools[String(operation)].handler
            call = ToolCall("native-$operation", String(operation), JSON3.write((; source_ids = [source.sha256], kwargs...)))
            parameters = JSON3.read(call.arguments_json)
            grant = source_grant(store, source)
            arguments = tool_command(handler, parameters, Dict(source.sha256 => grant.path))
            result = PhluxAI.Sandboxing._run_command(native, arguments, Cmd(arguments))
            return PhluxAI.JupyTools._worker_result(handler, call, result, false, false)
        end
        for (operation, parameters) in ((:pdf_metadata, (;)), (:pdf_text, (pages = [1],)),
                                         (:pdf_tables, (pages = [2],)), (:pdf_ocr, (pages = [3], dpi = 216)))
            execution = execute_native(pdf, operation; parameters...)
            payload = check_analysis_payload(execution, pdf, operation)
            @test payload.method !== nothing
        end
        for (operation, parameters) in ((:hdf5_hierarchy, (;)), (:hdf5_metadata, (dataset = "/data",)),
                                         (:hdf5_read, (dataset = "/data", selection = [(start = 0, stop = 1), (start = 0, stop = 2)])),
                                         (:hdf5_statistics, (dataset = "/nonfinite",)))
            execution = execute_native(hdf5, operation; parameters...)
            check_analysis_payload(execution, hdf5, operation)
        end
        failed = execute_native(hdf5, :hdf5_read; dataset = "/external_link")
        @test !failed.success
        @test JSON3.read(failed.output).error == "unauthorized_reference"
        @test isempty(JSON3.read(failed.output).evidence)
        @test execute_native(pdf, :pdf_text; pages = [1]).success
        @test verify_source(store, pdf) === nothing
        @test verify_source(store, hdf5) === nothing
    end
end
