using Test

# Phase 3.11D1: no model startup, network calls, or GPU inference.
ENV["PHLUXAI_BANNER"] = "0"
include(joinpath(@__DIR__, "..", "src", "PhluxAI.jl"))
using .PhluxAI

const OC = PhluxAI.OllamaClient
const PT = PhluxAI.ProtocolTypes

const VALID_SUMMARY = """
<assumptions>Pressure is measured in pascals.</assumptions>
<results>Earlier engineering discussion was summarized.</results>
<notation></notation>
<deferred></deferred>
<rejected></rejected>
<narrative>Continue the investigation.</narrative>
"""

function fake_response(body::String; done::Bool = true, reason::Union{Nothing,String} = "stop")
    return OC.GenerateResponse(
        "fixture-only", body, nothing, PT.ArtifactRef[], done, reason,
        nothing, nothing, PT.ResponseMetrics(),
    )
end

function test_config()
    return OC.BudgetConfig(
        num_ctx = 4096,
        num_thread = 8,
        max_tokens = 512,
        system_budget = 256,
        summary_budget = 128,
        response_budget = 512,
        safety_margin = 128,
    )
end

function history_fixture(marker::String)
    messages = PT.ChatMessage[]
    for turn in 1:4
        statement = turn == 1 ? "Release identifier: $marker. It is required later." :
                                "Engineering check $turn is underway."
        push!(messages, PT.ChatMessage(PT.ROLE_USER, statement))
        push!(messages, PT.ChatMessage(PT.ROLE_ASSISTANT, "Acknowledged turn $turn."))
    end
    return messages
end

function commit!(session, candidate, tail, revision)
    return lock(session.lock) do
        OC._commit_compression_candidate_unlocked!(session, candidate, tail, revision)
    end
end

parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected existing PhluxAI data directory: $parent")

@testset "Phase 3.11D1: transactional compression (CPU only)" begin
    @testset "Candidate validation" begin
        @test OC._validated_summary(fake_response(VALID_SUMMARY)) == strip(VALID_SUMMARY)
        @test isnothing(OC._validated_summary(fake_response(VALID_SUMMARY; reason = "length")))
        @test isnothing(OC._validated_summary(fake_response(VALID_SUMMARY; done = false)))
        @test isnothing(OC._validated_summary(fake_response("")))
        @test isnothing(OC._validated_summary(fake_response("<results>Only one section.</results>")))
        @test isnothing(OC._validated_summary(fake_response(VALID_SUMMARY; reason = nothing)))
    end

    @testset "Protected facts and revision-guarded commit" begin
        mktempdir(parent; prefix = "compression-transactional-") do root
            session_path = joinpath(root, "session")
            model = OC.OllamaModel("qwen2.5-coder:latest", "http://127.0.0.1:11434")
            session = OC.ChatSession(model, "Preserve user-approved facts."; cfg = test_config(), session_dir = session_path)
            marker = "P311D1-RELEASE-71"
            pinned_fact = "Release identifier: $marker"
            history = history_fixture(marker)
            tail = copy(history[3:end])

            try
                session.messages = history
                @test_throws ArgumentError OC.pin_context!(session, "  ")
                OC.pin_context!(session, pinned_fact)
                @test session.protected_facts == [pinned_fact]
                @test occursin(marker, only(filter(m -> m.role == PT.ROLE_SYSTEM, OC.assemble_messages(session))).content)
                @test_throws ArgumentError OC.pin_context!(session, repeat("x", 1600))
                OC.pin_context!(session, pinned_fact)
                @test session.protected_facts == [pinned_fact]
                snapshot = session.revision

                @test !commit!(session, fake_response(VALID_SUMMARY; reason = "length"), tail, snapshot)
                @test !commit!(session, fake_response("invalid structure"), tail, snapshot)
                @test !commit!(session, fake_response(VALID_SUMMARY), tail, snapshot - UInt64(1))
                @test isnothing(session.summary)
                @test session.messages === history
                @test session.revision == snapshot

                @test commit!(session, fake_response(VALID_SUMMARY), tail, snapshot)
                @test session.summary == strip(VALID_SUMMARY)
                @test session.messages === tail
                @test session.revision == snapshot + UInt64(1)
                @test !any(m -> occursin(marker, m.content), session.messages)
                @test any(m -> occursin(marker, m.content), OC.assemble_messages(session))
                @test !commit!(session, fake_response(VALID_SUMMARY), tail, snapshot)

                hot = read(joinpath(session_path, "hot.json"), String)
                @test occursin(marker, hot)
                @test occursin("protected_facts", hot)
                @test occursin("<results>", hot)
            finally
                OC.close_session(session)
            end

            loaded = OC.load_session(model, session_path)
            try
                @test loaded.protected_facts == [pinned_fact]
                @test loaded.summary == strip(VALID_SUMMARY)
                @test any(m -> occursin(marker, m.content), OC.assemble_messages(loaded))
                @test length(loaded.messages) == length(tail)
            finally
                OC.close_session(loaded)
            end
        end
    end
end

println("CPU-only validation finished. No Ollama model was started, stopped, or queried.")
println("Tests do not establish semantic completeness of summaries or nonblocking compression scheduling.")
