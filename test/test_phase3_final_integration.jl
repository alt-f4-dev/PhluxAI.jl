using Test

# Phase 3 closure gate: existing CPU regressions plus a short live Ollama
# integration of D1 validation/pinning and D2 emergency compression admission.
# This script does not change production files, GPU limits, or system settings.
ENV["PHLUXAI_BANNER"] = "0"
include(joinpath(@__DIR__, "..", "src", "PhluxAI.jl"))
using .PhluxAI

const OC = PhluxAI.OllamaClient
const PT = PhluxAI.ProtocolTypes
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const MODEL_TIMEOUT = 120.0
const OUTPUT_LIMIT = 64
const MARKER = "PH3-CLOSURE-R71"

seconds(ns::Integer) = round(Float64(ns) / 1.0e9; digits = 3)

function run_cpu_regressions()
    repo = normpath(joinpath(@__DIR__, ".."))
    scripts = (
        "test_ollama_compression_invariants.jl",
        "test_ollama_compression_transactional.jl",
        "test_ollama_compression_scheduling.jl",
    )
    @testset "Previously validated CPU-only regressions" begin
        for filename in scripts
            path = joinpath(@__DIR__, filename)
            @test isfile(path)
            isfile(path) || continue
            println("\n[CPU] ", filename)
            command = `$(Base.julia_cmd()) --project=$repo $path`
            @test success(run(ignorestatus(command)))
        end
    end
end

function require_exclusive_model_ownership()
    # Do not unload any model another user session has already loaded.
    listing = strip.(split(read(`ollama ps`, String), '\n'))
    occupied = filter(!isempty, listing[2:end])
    isempty(occupied) || error(
        "An Ollama model is already resident. This test will not reuse or " *
        "unload it. Run the GPU segment when no other model is resident:\n" *
        join(occupied, "\n"),
    )
end

function config()
    return BudgetConfig(
        num_ctx = 4096,
        num_thread = 8,
        max_tokens = OUTPUT_LIMIT,
        system_budget = 256,
        summary_budget = 512,
        response_budget = OUTPUT_LIMIT,
        safety_margin = 128,
        compression_frac = 1.0,
        m_verbatim = 2,
    )
end

function new_session(root::String, name::String)
    return ModelSession(
        MODEL;
        base_url = URL,
        cfg = config(),
        system_prompt = "You are a careful technical assistant. State protected identifiers exactly when requested.",
        session_dir = joinpath(root, "session-" * name),
        artifact_dir = joinpath(root, "artifacts-" * name),
        sandbox_dir = joinpath(root, "sandbox-" * name),
        tool_mode = TOOLS_OFF,
        discover_capabilities = false,
        start_model = false,
        stop_model_on_close = false,
        keep_alive = "5m",
    )
end

function ask_short!(session::ModelSession, prompt::String)
    started = time_ns()
    response = respond!(
        session, prompt;
        temperature = 0.0,
        max_tokens = OUTPUT_LIMIT,
        stream = true,
        print_tokens = false,
        tool_mode = TOOLS_OFF,
        request_timeout = MODEL_TIMEOUT,
        read_idle_timeout = MODEL_TIMEOUT,
    )
    return (response = response, wall_ns = time_ns() - started)
end

function synthetic_history()
    # Approximately 4,000 tokens according to PhluxAI's byte/4 estimator.
    # Repetitive engineering notes limit the real summarizer's input complexity.
    # The oldest identifier is separately pinned; other details are disposable.
    sentence = "Audit record: verify local inference, reversible artifact cleanup, " *
               "sandbox boundaries, and numerical checks. "
    history = PT.ChatMessage[]
    for turn in 1:4
        for role in (PT.ROLE_USER, PT.ROLE_ASSISTANT)
            preface = role == PT.ROLE_USER ? "User engineering note $turn. " : "Assistant engineering note $turn. "
            if turn == 1 && role == PT.ROLE_USER
                preface *= "Release identifier: $MARKER. "
            end
            content = preface
            while ncodeunits(content) < 2050
                content *= sentence
            end
            push!(history, PT.ChatMessage(role, content))
        end
    end
    return history
end

function print_live_result(label::String, result)
    response = result.response
    m = response.metrics
    outside_client_ns = result.wall_ns - m.client_total_duration_ns
    println("[GPU] $label")
    println("    wall $(seconds(result.wall_ns)) s | Ollama $(seconds(m.total_duration_ns)) s | outside HTTP client $(seconds(outside_client_ns)) s")
    println("    prompt $(m.prompt_eval_count) tokens / $(seconds(m.prompt_eval_duration_ns)) s" *
            " | output $(m.eval_count) tokens / $(seconds(m.eval_duration_ns)) s" *
            " | completion $(response.done_reason)")
    return nothing
end

function run_live_gate(root::String)
    println("\n[GPU] Warmup: one short response; model stays resident.")
    warm = new_session(root, "warmup")
    try
        result = ask_short!(warm, "Reply with OK.")
        print_live_result("Warmup", result)
        @test result.response.done
        @test !isempty(strip(result.response.text))
    finally
        close(warm)
    end

    println("\n[GPU] D2 estimated-capacity pressure: one real compression followed by one short response.")
    session = new_session(root, "compression")
    try
        chat = session.session
        OC.pin_context!(chat, "Release identifier: $MARKER")
        history = synthetic_history()
        incoming = PT.ChatMessage(PT.ROLE_USER, "State the protected release identifier exactly, with no explanation.")
        before = lock(chat.lock) do
            append!(chat.messages, history)
            chat.revision += UInt64(1)
            return (
                history_length = length(chat.messages),
                estimated_input = OC.approx_tokens(OC._assemble_messages_unlocked(chat)) + OC.approx_tokens(incoming),
                fits = OC._context_capacity_ok_unlocked(chat, incoming, OUTPUT_LIMIT),
                cutoff = OC._compression_cutoff(chat.messages, chat.cfg.m_verbatim),
                trigger = OC.compression_trigger(chat.cfg),
            )
        end
        println("    preflight estimate: $(before.estimated_input) | trigger: $(before.trigger) | compressible messages: $(before.cutoff)")
        @test !before.fits
        @test before.cutoff > 0
        @test before.history_length == 8
        (!before.fits && before.cutoff > 0) || error("Synthetic history did not create the required admission pressure")

        result = ask_short!(session, incoming.content)
        print_live_result("Compressed-context reply", result)
        after = lock(chat.lock) do
            visible = OC._assemble_messages_unlocked(chat)
            return (
                summary = chat.summary,
                remaining_history = length(chat.messages),
                protected = copy(chat.protected_facts),
                marker_visible = any(message -> occursin(MARKER, message.content), visible),
                pending = !isnothing(chat.compression_task) && !istaskdone(chat.compression_task),
            )
        end
        println("    summary committed: $(!isnothing(after.summary)) | remaining messages: $(after.remaining_history)")
        println("    protected identifier visible: $(after.marker_visible) | recalled: $(occursin(MARKER, result.response.text))")

        @test !isnothing(after.summary)  # Real compression must pass the D1 completion and structure validator.
        @test after.remaining_history < before.history_length + 2
        @test "Release identifier: $MARKER" in after.protected
        @test after.marker_visible
        @test !after.pending
        @test result.response.done
        @test result.response.done_reason == "stop"  # Do not accept output-budget truncation as a complete answer.
        @test occursin(MARKER, result.response.text)
        @test 0 < result.response.metrics.eval_count <= OUTPUT_LIMIT
        @test result.response.metrics.prompt_eval_count > 0
        @test result.response.metrics.client_total_duration_ns > 0
        @test result.wall_ns > 0

        # D1 persistence: the protected fact must survive a real compression commit.
        hot_path = joinpath(root, "session-compression", "hot.json")
        @test isfile(hot_path)
        @test occursin(MARKER, read(hot_path, String))
        @test occursin("protected_facts", read(hot_path, String))
        println("[GPU] Real forced compression, pinned recall, and persisted context checked.")
    finally
        # close() intentionally joins a pending compression task before the
        # temporary session directory is deleted.
        close(session)
    end
end

println("Phase 3 closure gate: three CPU suites, then limited live Ollama inference.")
println("GPU intensity: moderate; the compression pass uses GPU inference and may sustain high utilization briefly.")

run_cpu_regressions()

# The model ownership preflight runs only after CPU validation. The test must
# not terminate or unload a model that was resident before this script began.
require_exclusive_model_ownership()
parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected existing PhluxAI data directory: $parent")
owned_model = Ref(false)
try
    @testset "Phase 3 final live integration" begin
        mktempdir(parent; prefix = "phase3-final-gate-") do root
            owned_model[] = true
            run_live_gate(root)
        end
    end
finally
    if owned_model[]
        println("[Cleanup] Stopping only the benchmark-owned Ollama model...")
        try
            OC.stop_model!(OC.OllamaModel(MODEL, URL))
        catch error
            @warn "Could not stop benchmark-owned Ollama model; inspect `ollama ps`" exception = (error, catch_backtrace())
        end
    end
end
println("Phase 3 integration gate finished; temporary sessions and artifacts cleaned up.")
println("Scope: one real compression at estimated capacity pressure and protected-fact recall.")
println("Not established: semantic losslessness for unpinned facts, exact tokenizer fit, or all concurrency interleavings.")
