using Test

# Phase 3.11D2: CPU-only tests of admission and compression-task ordering.
# No model startup, HTTP requests, GPU inference, or ComfyUI.
ENV["PHLUXAI_BANNER"] = "0"
include(joinpath(@__DIR__, "..", "src", "PhluxAI.jl"))
using .PhluxAI

const OC = PhluxAI.OllamaClient
const PT = PhluxAI.ProtocolTypes

const VALID_SUMMARY = """
<assumptions>Units use SI.</assumptions>
<results>Earlier turns have been summarized.</results>
<notation></notation>
<deferred></deferred>
<rejected></rejected>
<narrative>Proceed with the next turn.</narrative>
"""

function fake_response(body::String; reason::String = "stop")
    return OC.GenerateResponse(
        "fixture-only", body, nothing, PT.ArtifactRef[], true,
        reason, nothing, nothing, PT.ResponseMetrics(),
    )
end

function config()
    return OC.BudgetConfig(
        num_ctx = 4096, num_thread = 8, max_tokens = 512,
        system_budget = 256, summary_budget = 128,
        response_budget = 512, safety_margin = 128,
    )
end

function new_chat(root::String)
    model = OC.OllamaModel("qwen2.5-coder:latest", "http://127.0.0.1:11434")
    return OC.ChatSession(
        model, "Maintain established engineering constraints.";
        cfg = config(), session_dir = joinpath(root, "session"),
    )
end

user(text) = PT.ChatMessage(PT.ROLE_USER, text)
assistant(text) = PT.ChatMessage(PT.ROLE_ASSISTANT, text)

function short_history()
    return PT.ChatMessage[
        user("Release identifier: P311D2-ALPHA"), assistant("Recorded."),
        user("Retain this discussion."), assistant("Acknowledged."),
        user("Check stability."), assistant("Proceed."),
        user("Continue validation."), assistant("Ready."),
    ]
end

function oversized_history()
    history = short_history()
    history[1] = user(repeat("q", 16000))
    return history
end

function install_task!(session, task)
    lock(session.lock) do
        session.compression_task = task
    end
end

function admit_and_unlock!(session, message; max_tokens = 512)
    OC._lock_when_idle!(session; incoming = message, max_tokens = max_tokens)
    try
        return true
    finally
        unlock(session.lock)
    end
end

parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected existing PhluxAI data directory: $parent")

@testset "Phase 3.11D2: capacity-aware scheduling (CPU only)" begin
    @testset "Capacity and requested response budget" begin
        mktempdir(parent; prefix = "compression-d2-capacity-") do root
            session = new_chat(root)
            try
                ordinary = user(repeat("x", 9500))
                @test OC._context_capacity_ok_unlocked(session, ordinary, 512)
                @test !OC._context_capacity_ok_unlocked(session, ordinary, 1700)
                @test admit_and_unlock!(session, ordinary; max_tokens = 512)
                @test_throws ArgumentError admit_and_unlock!(
                    session, ordinary; max_tokens = 1700,
                )
                @test isnothing(session.compression_task)
                @test isempty(session.messages)
            finally
                OC.close_session(session)
            end
        end
    end

    @testset "Pending task is bypassed only when there is headroom" begin
        mktempdir(parent; prefix = "compression-d2-bypass-") do root
            session = new_chat(root)
            gate = Base.Event()
            started = Channel{Bool}(1)
            committed = Channel{Bool}(1)
            session.messages = short_history()
            snapshot = session.revision
            tail = copy(session.messages[3:end])
            pending = @async begin
                put!(started, true)
                wait(gate)
                result = lock(session.lock) do
                    OC._commit_compression_candidate_unlocked!(
                        session, fake_response(VALID_SUMMARY), tail, snapshot,
                    )
                end
                put!(committed, result)
            end
            install_task!(session, pending)
            take!(started)
            try
                @test admit_and_unlock!(session, user("A new short request."))
                @test !istaskdone(pending)
                # Model a successful request's revision mutation while the
                # old summary was still in flight.
                lock(session.lock) do
                    push!(session.messages, user("A new short request."))
                    session.revision += UInt64(1)
                end
            finally
                notify(gate)
                wait(pending)
            end
            @test !take!(committed)   # Old snapshot cannot replace new history.
            @test isnothing(session.summary)
            @test length(session.messages) == 9
            OC.close_session(session)
        end
    end

    @testset "Capacity pressure waits for validated compression" begin
        mktempdir(parent; prefix = "compression-d2-pressure-") do root
            session = new_chat(root)
            session.messages = oversized_history()
            next_user = user("Continue.")
            @test !OC._context_capacity_ok_unlocked(session, next_user, 512)

            gate = Base.Event()
            started = Channel{Bool}(1)
            committed = Channel{Bool}(1)
            snapshot = session.revision
            tail = copy(session.messages[3:end])
            pending = @async begin
                put!(started, true)
                wait(gate)
                result = lock(session.lock) do
                    OC._commit_compression_candidate_unlocked!(
                        session, fake_response(VALID_SUMMARY), tail, snapshot,
                    )
                end
                put!(committed, result)
            end
            install_task!(session, pending)
            take!(started)

            admitted = Channel{Bool}(1)
            contender = @async begin
                result = admit_and_unlock!(session, next_user)
                put!(admitted, result)
            end
            try
                yield()
                @test !isready(admitted)
                notify(gate)
                @test Base.timedwait(() -> isready(admitted), 2.0) == :ok
                @test take!(admitted)
                @test take!(committed)
                @test OC._context_capacity_ok_unlocked(session, next_user, 512)
                @test length(session.messages) == 6
                @test !isnothing(session.summary)
            finally
                notify(gate)
                wait(pending)
                istaskdone(contender) && wait(contender)
            end
            OC.close_session(session)
        end
    end

    @testset "Failed pending compression rejects oversized context" begin
        mktempdir(parent; prefix = "compression-d2-reject-") do root
            session = new_chat(root)
            session.messages = oversized_history()
            gate = Base.Event()
            started = Channel{Bool}(1)
            pending = @async begin
                put!(started, true)
                wait(gate)  # Simulate rejected/failed compression: no commit.
            end
            install_task!(session, pending)
            take!(started)
            rejected = Channel{Bool}(1)
            contender = @async begin
                try
                    OC._lock_when_idle!(
                        session; incoming = user("Continue."), max_tokens = 512,
                    )
                    try
                        put!(rejected, false)
                    finally
                        unlock(session.lock)
                    end
                catch error
                    put!(rejected, error isa ArgumentError)
                end
            end
            try
                yield()
                @test !isready(rejected)
                notify(gate)
                @test Base.timedwait(() -> isready(rejected), 2.0) == :ok
                @test take!(rejected)
                @test length(session.messages) == 8
                @test isnothing(session.summary)
            finally
                notify(gate)
                wait(pending)
                istaskdone(contender) && wait(contender)
            end
            OC.close_session(session)
        end
    end

    @testset "Legacy close/save gate remains blocking" begin
        mktempdir(parent; prefix = "compression-d2-legacy-") do root
            session = new_chat(root)
            gate = Base.Event()
            started = Channel{Bool}(1)
            entered = Channel{Bool}(1)
            pending = @async begin
                put!(started, true)
                wait(gate)
            end
            install_task!(session, pending)
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
                @test !isready(entered)
                notify(gate)
                @test Base.timedwait(() -> isready(entered), 2.0) == :ok
                @test take!(entered)
            finally
                notify(gate)
                wait(pending)
                istaskdone(contender) && wait(contender)
            end
            OC.close_session(session)
            @test isnothing(OC.close_session(session))  # Idempotent close.
        end
    end
end

println("CPU-only D2 admission tests complete. No Ollama model was started, stopped, or queried.")
println("Warning: context capacity uses approx_tokens; actual Ollama tokenization can differ.")
println("The forced real-LLM compression path and GPU contention are not tested here.")
