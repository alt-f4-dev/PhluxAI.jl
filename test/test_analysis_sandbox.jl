using Test
using JSON3
using PhluxAI

include("analysis/roundtrip.jl")

@testset "PDF/HDF5 operations through isolated public dispatch" begin
    workers = ENV["PHLUXAI_WORKER_CGROUP"]
    before = Set(readdir(workers))
    mktempdir() do root
        store, pdf, hdf5, authorize, sandbox, artifacts = analysis_fixture_setup(root)
        context = analysis_context(sandbox, artifacts, store; authorize, worker_cgroup = workers)
        approve(call, risk) = risk == TOOL_READ
        for (source, operation, parameters) in ((pdf, :pdf_metadata, (;)), (pdf, :pdf_text, (pages = [1],)),
                                                 (pdf, :pdf_tables, (pages = [2],)), (pdf, :pdf_ocr, (pages = [3], dpi = 216)),
                                                 (hdf5, :hdf5_hierarchy, (;)), (hdf5, :hdf5_metadata, (dataset = "/data",)),
                                                 (hdf5, :hdf5_read, (dataset = "/huge", selection = [(start = 100, stop = 110)],)),
                                                 (hdf5, :hdf5_statistics, (dataset = "/nonfinite",)))
            result = analyze_source(context, source, operation; approval_callback = approve, parameters...)
            check_analysis_payload(result.execution, source, operation)
            @test result.operation.status == :succeeded
            @test result.error_code === nothing
            @test haskey(result.versions, "python")
        end
        denied = analyze_source(context, pdf, :pdf_text; mode = TOOLS_AUTO)
        @test !denied.execution.success && denied.error_code == :execution_rejected
        ungranted = analysis_context(sandbox, artifacts, store; worker_cgroup = workers,
                                     authorize = (call, declaration) -> Dict{String,SourceGrant}())
        @test !analyze_source(ungranted, pdf, :pdf_text; approval_callback = approve).execution.success
        failed = analyze_source(context, hdf5, :hdf5_read; dataset = "/external_link", approval_callback = approve)
        @test !failed.execution.success && failed.error_code == :unauthorized_reference
        bounded = analysis_context(sandbox, artifacts, store; worker_cgroup = workers, authorize,
                                   limits = AnalysisLimits(max_elements = 2))
        oversized = analyze_source(bounded, hdf5, :hdf5_read; dataset = "/data", approval_callback = approve)
        @test !oversized.execution.success && oversized.error_code == :limit_exceeded
        for (code, policy) in ((:timeout, ExecutionPolicy(timeout_seconds = 0.001)),
                                (:output_limit, ExecutionPolicy(max_output_bytes = 128)),
                                (:memory_limit, ExecutionPolicy(max_memory_bytes = 16 * 1024 * 1024)))
            limited = analysis_context(sandbox, artifacts, store; worker_cgroup = workers, authorize, policy)
            result = analyze_source(limited, hdf5, :hdf5_read; dataset = "/data", approval_callback = approve)
            @test !result.execution.success
            @test result.error_code == code
            @test result.operation.status == :failed
            @test isempty(result.evidence)
        end
        @test analyze_source(context, pdf, :pdf_text; pages = [1], approval_callback = approve).execution.success
        @test verify_source(store, pdf) === nothing
        @test verify_source(store, hdf5) === nothing
    end
    @test Set(readdir(workers)) == before
end
