using Dates

function analysis_fixture_setup(root::String; python::String = get(ENV, "PHLUXAI_ANALYSIS_PYTHON", "/usr/bin/python3"))
    fixture_root = joinpath(root, "fixtures")
    run(Cmd([python, joinpath(@__DIR__, "fixtures.py"), fixture_root]))
    store = SourceStore(joinpath(root, "sources"))
    pdf = import_source(store, joinpath(fixture_root, "sample.pdf"))
    hdf5 = import_source(store, joinpath(fixture_root, "sample.h5"))
    function authorize(call, declaration)
        return Dict(pdf.sha256 => source_grant(store, pdf), hdf5.sha256 => source_grant(store, hdf5))
    end
    sandbox = SandboxConfig(joinpath(root, "sandbox"))
    artifacts = PhluxAI.ArtifactStore.Store(joinpath(root, "artifacts"))
    return store, pdf, hdf5, authorize, sandbox, artifacts
end

function check_analysis_payload(execution, source, operation)
    @test execution.success
    payload = JSON3.read(execution.output)
    @test payload.ok
    record = parse_contract(OperationRecord, JSON3.write(payload.operation))
    @test record.status == :succeeded
    @test record.inputs == [source]
    @test record.action == String(operation)
    for item in payload.evidence
        evidence = parse_contract(EvidenceRef, JSON3.write(item))
        @test evidence.source == source
        @test evidence.extraction_sha256 == record.outputs[1].sha256
    end
    return payload
end
