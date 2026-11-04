using Test

# Phase 3.11C: CPU-only structural tests for conversation compression.
# No Ollama endpoint requests, generation, ComfyUI calls, or GPU measurements.
ENV["PHLUXAI_BANNER"] = "0"
include(joinpath(@__DIR__, "..", "src", "PhluxAI.jl"))
using .PhluxAI

const OC = PhluxAI.OllamaClient
const PT = PhluxAI.ProtocolTypes

function test_config(; compression_frac = 0.6)
    return OC.BudgetConfig(
        num_ctx = 4096,
        num_thread = 8,
        max_tokens = 512,
        system_budget = 256,
        summary_budget = 128,
        response_budget = 512,
        safety_margin = 128,
        compression_frac = compression_frac,
    )
end

function fabricated_history(marker::String)
    # Exactly four user/assistant pairs: first pair contains the fact we must retain.
    messages = PT.ChatMessage[]
    for turn in 1:4
        user_text = turn == 1 ? "Release identifier: $marker. Confirm its significance." :
                                  "Engineering discussion turn $turn with ordinary context."
        push!(messages, PT.ChatMessage(PT.ROLE_USER, user_text))
        push!(messages, PT.ChatMessage(PT.ROLE_ASSISTANT, "Discussion response for turn $turn."))
    end
    return messages
end

contains_marker(messages, marker::String) = any(m -> occursin(marker, m.content), messages)

function exercise_pending_task_gate!(session::OC.ChatSession)
    release = Base.Event()
    started = Channel{Bool}(1)
    entered = Channel{Bool}(1)

    pending = @async begin
        put!(started, true)
        wait(release)
    end
    lock(session.lock) do
        session.compression_task = pending
    end
    take!(started)

    contender = @async begin
        OC._lock_when_idle!(session)
        try
            put!(entered, true)
        finally
            unlock(session.lock)
        end
    end

    try
        yield()
        @test !isready(entered)  # A pending compression task gates admission.
    finally
        notify(release)
        wait(pending)
        wait(contender)
    end

    @test take!(entered)
    @test isnothing(session.compression_task)
    return nothing
end

parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected existing PhluxAI data directory: $parent")

@testset "Phase 3.11C: CPU-only compression invariants" begin
    default = test_config()
    deferred = test_config(compression_frac = 1.0)

    @testset "Budget geometry" begin
        @test OC.history_budget(default) == 3072
        @test OC.compression_trigger(default) == 1843
        @test OC.compression_trigger(deferred) == 3072
        @test default.num_ctx == deferred.num_ctx
        @test default.max_tokens == deferred.max_tokens
        @test default.summary_budget == deferred.summary_budget
    end

    @testset "Cutoff and context retention (synthetic summary, no inference)" begin
        marker = "P311-STRUCTURAL-FACT-71"
        history = fabricated_history(marker)
        cutoff = OC._compression_cutoff(history, default.m_verbatim)
        @test default.m_verbatim == 6
        @test cutoff == 2
        @test history[cutoff + 1].role == PT.ROLE_USER
        @test contains_marker(history[1:cutoff], marker)
        @test !contains_marker(history[cutoff + 1:end], marker)

        mktempdir(parent; prefix = "context-invariants-") do root
            model = OC.OllamaModel("qwen2.5-coder:latest", "http://127.0.0.1:11434")
            chat = OC.ChatSession(model, "Preserve relevant engineering facts.";
                                  cfg = default, session_dir = joinpath(root, "session"))
            try
                # Mimic append-only audit logging without running inference.
                for message in history
                    OC._log_cold_unlocked(chat, message)
                end

                # Model a summary which omits a crucial fact. This tests routing,
                # NOT the probability of a real model producing such a summary.
                chat.messages = copy(history[cutoff + 1:end])
                chat.summary = "<results>Prior discussions covered engineering safety.</results>"
                assembled = OC.assemble_messages(chat)
                @test length(assembled) == 1 + 2 + default.m_verbatim
                @test !contains_marker(assembled, marker)
                @test contains_marker(history, marker)
                OC._save_hot_unlocked(chat)
                @test occursin(marker, read(joinpath(root, "session", "cold.jsonl"), String))
                @test !occursin(marker, read(joinpath(root, "session", "hot.json"), String))

                # Exact retention in summary restores model-visible access.
                chat.summary = "<results>Release identifier: $marker.</results>"
                @test contains_marker(OC.assemble_messages(chat), marker)

                # The pending-task gate can be exercised entirely on the CPU.
                exercise_pending_task_gate!(chat)
            finally
                OC.close_session(chat)
            end
        end
    end
end

println("CPU-only structural checks completed. No model was started or stopped.")
println("Scope: cutoff, synthetic summary visibility, and pending-task gating.")
println("Not tested: LLM summary quality, tokenizer accuracy, GPU latency, or production behavior under concurrency.")
