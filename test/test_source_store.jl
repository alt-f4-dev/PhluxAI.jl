using Test
using SHA: sha256
using PhluxAI

@testset "Bounded source snapshots" begin
    mktempdir() do root
        store = SourceStore(joinpath(root, "sources"); max_source_bytes = 1024)
        original = joinpath(root, "input")
        bytes = "%PDF-1.7\nfixture"
        write(original, bytes)
        source = import_source(store, original)
        @test source.format == :pdf
        @test source.sha256 == bytes2hex(sha256(bytes))
        @test source.size_bytes == ncodeunits(bytes)
        @test import_source(store, original) == source
        @test length(readdir(store.root)) == 1
        @test verify_source(store, source) === nothing
        concurrent = [@async import_source(store, original) for _ in 1:4]
        @test all(task -> fetch(task) == source, concurrent)
        @test length(readdir(store.root)) == 1
        write(original, "modified original")
        @test read(source_grant(store, source).path, String) == bytes
        @test_throws ArgumentError import_source(store, original)
        @test_throws ArgumentError import_source(store, original; format = :xml)
        write(original, bytes)
        @test_throws ArgumentError import_source(store, original; format = :hdf5)
        write(original, zeros(UInt8, 1025))
        @test_throws ArgumentError import_source(store, original)
        @test length(readdir(store.root)) == 1
        write(original, vcat(zeros(UInt8, 512), UInt8[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]))
        @test import_source(store, original).format == :hdf5
        write(original, "%PDF-" * repeat("x", 1019))
        @test import_source(store, original).size_bytes == store.max_source_bytes
        snapshot = source_grant(store, source).path
        chmod(snapshot, 0o600)
        write(snapshot, "%PDF-1.7\ntampered")
        @test_throws ArgumentError verify_source(store, source)
        @test_throws ArgumentError SourceStore(root; max_source_bytes = 0)
    end
end
