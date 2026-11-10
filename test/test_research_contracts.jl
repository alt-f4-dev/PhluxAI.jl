using Test
using UUIDs
using PhluxAI

const DIGEST_A = repeat("a", 64)
const DIGEST_B = repeat("b", 64)

@testset "Research contract invariants (CPU only)" begin
    @testset "Content identity and validation" begin
        source = SourceRef(uppercase(DIGEST_A), :hdf5, 100)
        @test source.sha256 == DIGEST_A
        @test source.format == :hdf5
        @test source.size_bytes == 100
        @test_throws ArgumentError SourceRef("not-a-hash", :pdf, 1)
        @test_throws ArgumentError SourceRef(repeat("g", 64), :pdf, 1)
        @test_throws ArgumentError SourceRef(DIGEST_A, Symbol(""), 1)
        @test_throws ArgumentError SourceRef(DIGEST_A, :pdf, -1)
    end

    @testset "Source-grounded evidence" begin
        source = SourceRef(DIGEST_A, :pdf, 1024)
        evidence = EvidenceRef(source, :pdf_page, "page=12"; extraction_sha256 = uppercase(DIGEST_B))
        @test evidence.source === source
        @test evidence.scheme == :pdf_page
        @test evidence.locator == "page=12"
        @test evidence.extraction_sha256 == DIGEST_B
        @test_throws ArgumentError EvidenceRef(source, Symbol(""), "page=1")
        @test_throws ArgumentError EvidenceRef(source, :pdf_page, " ")
        @test_throws ArgumentError EvidenceRef(source, :pdf_page, "page=1"; extraction_sha256 = "invalid")
    end

    @testset "Operation metadata and defensive copying" begin
        source = SourceRef(DIGEST_A, :hdf5, 4096)
        sources = SourceRef[source]
        record = OperationRecord(uuid4(), " hdf5.inspect "; inputs = sources, parameters_json = "{\"path\":\"/entry\"}", status = :succeeded)
        @test record.action == "hdf5.inspect"
        @test record.inputs == sources
        @test record.inputs !== sources
        @test isempty(record.outputs)
        @test record.status == :succeeded
        @test record.parameters_json == "{\"path\":\"/entry\"}"
        empty!(sources)
        @test length(record.inputs) == 1
        @test_throws ArgumentError OperationRecord(uuid4(), "")
        @test_throws ArgumentError OperationRecord(uuid4(), "inspect"; parameters_json = "[]")
        @test_throws ArgumentError OperationRecord(uuid4(), "inspect"; parameters_json = "{")
        @test_throws ArgumentError OperationRecord(uuid4(), "inspect"; status = :unknown)
    end
end
