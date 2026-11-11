using Test
using JSON3
using UUIDs
using PhluxAI

const MODEL_DIGEST = repeat("c", 64)

@testset "Model and serialization contracts (CPU only)" begin
    @testset "Model identity" begin
        ref = ModelRef(:ollama, " Qwen/Qwen2.5-Coder:latest "; revision = "main", sha256 = uppercase(MODEL_DIGEST))
        @test ref.backend == :ollama
        @test ref.model_id == "Qwen/Qwen2.5-Coder:latest"
        @test ref.revision == "main"
        @test ref.sha256 == MODEL_DIGEST
        equivalent = ModelRef(:ollama, ref.model_id; revision = "main", sha256 = MODEL_DIGEST)
        @test equivalent == ref
        @test hash(equivalent) == hash(ref)
        @test ref != ModelRef(:comfyui, ref.model_id; revision = "main", sha256 = MODEL_DIGEST)
        @test_throws ArgumentError ModelRef(Symbol(""), "model")
        @test_throws ArgumentError ModelRef(:ollama, " ")
        @test_throws ArgumentError ModelRef(:ollama, "bad\nname")
        @test_throws ArgumentError ModelRef(:ollama, repeat("a", 1025))
        @test_throws ArgumentError ModelRef(:ollama, "model"; revision = "")
        @test_throws ArgumentError ModelRef(:ollama, "model"; sha256 = "bad")
    end

    @testset "Capability set and provider adapter" begin
        names = Symbol[:vision, :completion, :vision, :image_generation]
        caps = ModelCapabilitySet(names)
        @test caps.features == (:completion, :image_generation, :vision)
        push!(names, :training)
        @test !supports_capability(caps, :training)
        @test supports_capability(caps, :vision)
        @test supports_capability(caps, "VISION")
        @test !supports_capability(caps, :finetuning)
        @test isnothing(require_model_capability(caps, :completion))
        @test_throws ArgumentError require_model_capability(caps, :training)
        @test_throws ArgumentError ModelCapabilitySet(["invalid capability"])
        @test ModelCapabilitySet(["vision", "completion"]) == ModelCapabilitySet([:completion, :vision])
        @test capability_set(caps) === caps
        existing = ModelCapabilities(["completion", "vision", "tools", "image_generation"])
        @test supports_capability(capability_set(existing), :tools)
        @test !supports_capability(capability_set(existing), :training)
    end

    @testset "Strict, versioned JSON round-trip" begin
        source = SourceRef(MODEL_DIGEST, :hdf5, 2048)
        evidence = EvidenceRef(source, :hdf5_selection, "/entry/data[1:10]")
        record = OperationRecord(uuid4(), "hdf5.inspect"; inputs = SourceRef[source], status = :planned)
        model = ModelRef(:ollama, "qwen2.5-coder:latest"; revision = "v1")
        caps = ModelCapabilitySet([:completion, :tools])
        descriptor = ModelDescriptor(model, caps)
        for (kind, value) in ((SourceRef, source), (EvidenceRef, evidence), (OperationRecord, record),
                              (ModelRef, model), (ModelCapabilitySet, caps), (ModelDescriptor, descriptor))
            encoded = contract_json(value)
            @test contract_json(value) == encoded
            @test JSON3.read(encoded).schema_version == 1
            @test contract_json(parse_contract(kind, encoded)) == encoded
        end
        @test parse_contract(SourceRef, contract_json(source)) == source
        @test parse_contract(ModelRef, contract_json(model)) == model
        @test parse_contract(ModelCapabilitySet, contract_json(caps)) == caps
        @test parse_contract(ModelDescriptor, contract_json(descriptor)).ref == model
        @test_throws ArgumentError parse_contract(SourceRef, "[]")
        @test_throws ArgumentError parse_contract(SourceRef, "{")
        data = Dict{String,Any}(String(k) => v for (k,v) in pairs(JSON3.read(contract_json(source))))
        data["schema_version"] = 2
        @test_throws ArgumentError parse_contract(SourceRef, JSON3.write(data))
        data["schema_version"] = 1
        data["extra"] = true
        @test_throws ArgumentError parse_contract(SourceRef, JSON3.write(data))
        delete!(data, "extra")
        data["size_bytes"] = true
        @test_throws ArgumentError parse_contract(SourceRef, JSON3.write(data))
    end
end
