using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

# Phase 3.11B: compare ordinary and deferred history-compression thresholds.
# Diagnostic only: no production changes, no forced compression-task cancellation,
# and no unloading between requests. The two policies have the same num_ctx.
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const SYSTEM_PROMPT = "You are a careful local software engineering assistant. Distinguish observations from assumptions."
const NUM_CTX = 4096
const NUM_THREADS = 8
const MAX_TOKENS = 512
const KEEP_ALIVE = "5m"
const TIMEOUT = 180.0
const PASSES = 2
const THINK_TIME_SECONDS = 0.0  # Optional simulated time between conversational turns, outside request timing
const POLICIES = (:default, :deferred)

seconds(ns) = Float64(ns) / 1.0e9
rounded(x) = round(x; digits = 3)

function median_value(values)
    xs = sort(collect(values))
    isempty(xs) && throw(ArgumentError("Cannot take median of an empty collection"))
    n = length(xs)
    return isodd(n) ? xs[(n + 1) ÷ 2] : (xs[n ÷ 2] + xs[n ÷ 2 + 1]) / 2
end

function benchmark_config(policy::Symbol)
    policy in POLICIES || throw(ArgumentError("Unknown policy: $policy"))
    return BudgetConfig(
        num_ctx = NUM_CTX,
        num_thread = NUM_THREADS,
        max_tokens = MAX_TOKENS,
        system_budget = 256,
        summary_budget = 128,
        response_budget = MAX_TOKENS,
        safety_margin = 128,
        compression_frac = policy == :default ? 0.6 : 1.0,
    )
end

function new_session(root::String, label::String, policy::Symbol)
    return ModelSession(
        MODEL;
        base_url = URL,
        system_prompt = SYSTEM_PROMPT,
        cfg = benchmark_config(policy),
        session_dir = joinpath(root, "session-" * label),
        artifact_dir = joinpath(root, "artifacts-" * label),
        sandbox_dir = joinpath(root, "sandbox-" * label),
        tool_mode = TOOLS_OFF,
        start_model = false,
        stop_model_on_close = false,
        discover_capabilities = false,
        keep_alive = KEEP_ALIVE,
    )
end

function dialogue_prompts(pass::Int)
    # Reuse the Phase 3.11 engineering workload so that this experiment tests
    # the existing failure case, not a newly simplified recall prompt.
    marker = "P311-RC-$(pass * 97 + 13)"
    prompts = [
        "We are reviewing release $marker of a local Julia AI agent. Its sandbox permits writes only within a specified workspace, and the model communicates with a loopback Ollama server. Explain which security properties these observations establish and which they do not.",
        "The same release generates images through ComfyUI. Each image exists both in the server output directory and in a client artifact store. Design a failure-safe cleanup procedure that preserves unrelated files and works when generation fails partway through.",
        "Our profiling now shows that first-token latency increases with uncached prompt length, while generation dominates completion time for long answers. Explain how you would distinguish prompt caching, model loading, and GPU memory contention without weakening the agent's answer quality.",
        "Propose a controlled regression plan for prompt-prefix caching and model residency that does not change the correctness requirements of the agent. Include what would falsify the performance hypothesis.",
        "Prepare a release-review conclusion that connects the sandbox, dual image artifacts, and observed inference latency. State the release identifier given at the start of our conversation and distinguish what we measured from what remains unverified.",
    ]
    return marker, prompts
end

function snapshot(session::ModelSession, marker::String)
    # These internals are read-only diagnostic observations. Holding the session
    # lock makes the summary, message list, and task reference one snapshot.
    chat = session.session
    return lock(chat.lock) do
        messages = PhluxAI.OllamaClient._assemble_messages_unlocked(chat)
        signature = [(message.role, message.content) for message in messages]
        task = chat.compression_task
        summary = chat.summary
        return (
            signature = signature,
            task = task,
            pending = !isnothing(task) && !istaskdone(task),
            summary_active = !isnothing(summary),
            marker_in_summary = !isnothing(summary) && occursin(marker, summary),
            marker_in_context = any(message -> occursin(marker, message.content), messages),
            approx_tokens = PhluxAI.OllamaClient.approx_tokens(messages),
        )
    end
end

function request!(session::ModelSession, prompt::String, marker::String)
    before = snapshot(session, marker)
    task_finished_ns = Ref{Int}(0)
    summary_at_completion = Ref{Union{Nothing,String}}(nothing)
    observer = if before.pending
        # Passive observer: it never delays or changes when compression runs.
        # Its timestamp is approximate because task scheduling is asynchronous.
        @async begin
            wait(before.task)
            task_finished_ns[] = time_ns()
            summary_at_completion[] = lock(session.session.lock) do
                session.session.summary
            end
        end
    else
        nothing
    end

    started_ns = time_ns()
    response = respond!(
        session, prompt;
        temperature = 0.0,
        stream = true,
        print_tokens = false,
        tool_mode = TOOLS_OFF,
        request_timeout = TIMEOUT,
        read_idle_timeout = TIMEOUT,
    )
    ended_ns = time_ns()
    !isnothing(observer) && wait(observer)

    after = snapshot(session, marker)
    m = response.metrics
    wall_ns = ended_ns - started_ns
    # The model request's client duration starts only after the session is
    # prepared; wall-client includes waits and finalization, not just locking.
    prepost_ns = wall_ns - m.client_total_duration_ns
    pending_wait_estimate_ns = before.pending ? clamp(task_finished_ns[] - started_ns, 0, wall_ns) : 0
    unchanged_prefix = length(after.signature) >= length(before.signature) &&
                       after.signature[1:length(before.signature)] == before.signature
    return (
        response = response,
        before = before,
        after = after,
        wall_ns = wall_ns,
        server_ns = m.total_duration_ns,
        client_ns = m.client_total_duration_ns,
        prepost_ns = prepost_ns,
        wall_minus_server_ns = wall_ns - m.total_duration_ns,
        pending_wait_estimate_ns = pending_wait_estimate_ns,
        summary_when_prior_task_finished = summary_at_completion[],
        load_ns = m.load_duration_ns,
        prompt_ns = m.prompt_eval_duration_ns,
        prompt_tokens = m.prompt_eval_count,
        output_ns = m.eval_duration_ns,
        output_tokens = m.eval_count,
        ttft_ns = m.time_to_first_token_ns,
        reason = response.done_reason,
        unchanged_prefix = unchanged_prefix,
        marker_in_answer = occursin(marker, response.text),
        marker_in_prior_context_after = any(item -> occursin(marker, item[2]),
                                            after.signature[1:max(0, length(after.signature) - 2)]),
    )
end

function validate_result(r)
    @test r.response.done
    @test !isempty(strip(r.response.text))
    @test r.reason in ("stop", "length")
    @test r.prompt_tokens > 0
    @test r.prompt_ns > 0
    @test r.output_tokens > 0
    @test r.output_ns > 0
    @test !isnothing(r.ttft_ns)
    @test r.server_ns > 0
    @test r.wall_ns > 0
    @test r.before.approx_tokens > 0
    @test r.wall_ns >= r.client_ns - 5_000_000
end

function print_result(policy::Symbol, turn::Int, r)
    ttft = isnothing(r.ttft_ns) ? NaN : seconds(r.ttft_ns)
    rate = r.output_ns > 0 ? r.output_tokens / seconds(r.output_ns) : NaN
    println("  ", rpad("$(policy) turn $turn", 18), " | prompt tokens ", r.prompt_tokens,
            " | evaluation ", rounded(seconds(r.prompt_ns)), " s | TTFT ", rounded(ttft), " s")
    println("    wall ", rounded(seconds(r.wall_ns)), " s | Ollama ", rounded(seconds(r.server_ns)),
            " s | wall-Ollama ", rounded(seconds(r.wall_minus_server_ns)), " s",
            " | wall-client ", rounded(seconds(r.prepost_ns)), " s")
    println("    model load ", rounded(seconds(r.load_ns)), " s | output ", r.output_tokens,
            " tokens | generation ", rounded(seconds(r.output_ns)), " s | rate ", rounded(rate),
            " tok/s | completion ", r.reason)
    println("    compression pending before/after: ", r.before.pending, "/", r.after.pending,
            " | approximate time until prior task finished: ", rounded(seconds(r.pending_wait_estimate_ns)), " s")
    println("    summary before/after: ", r.before.summary_active, "/", r.after.summary_active,
            " | original prefix preserved afterward: ", r.unchanged_prefix,
            " | preflight history estimate: ", r.before.approx_tokens)
end

function preflight_model_check()
    lines = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
    length(lines) == 1 || error(
        "Other Ollama models are already resident. Stop existing sessions manually before this exclusive benchmark.\n" *
        join(lines, "\n"),
    )
end

preflight_model_check()
parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("PhluxAI data directory does not exist: $parent")

results = NamedTuple[]
benchmark_owned_model = Ref(false)
try
    mktempdir(parent; prefix = "ollama-compression-policy-") do root
        println("Warming Qwen once; model residency is preserved across both policies...")
        benchmark_owned_model[] = true
        warm = new_session(root, "warmup", :default)
        try
            warmup = request!(warm, "Reply with READY.", "WARMUP-MARKER")
            warmup.response.done || error("Warmup failed")
            println("Warmup wall: ", rounded(seconds(warmup.wall_ns)), " s",
                    " | initial load ", rounded(seconds(warmup.load_ns)), " s")
        finally
            close(warm)
        end
        println("Model: $MODEL | num_ctx: $NUM_CTX | max output: $MAX_TOKENS | threads: $NUM_THREADS")
        cfg_default = benchmark_config(:default)
        cfg_deferred = benchmark_config(:deferred)
        println("History budget: ", PhluxAI.OllamaClient.history_budget(cfg_default),
                " | default trigger: ", PhluxAI.OllamaClient.compression_trigger(cfg_default),
                " | deferred trigger: ", PhluxAI.OllamaClient.compression_trigger(cfg_deferred))
        current_models = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
        any(line -> occursin(MODEL, line), current_models[2:end]) ||
            error("Qwen is not resident after warmup; this is not a valid warm-session benchmark")
        println("Resident-model status: ", join(current_models[2:end], " | "))
        println("No model unload/preload occurs between measured turns or policies.")
        println("Simulated inter-turn think time: $THINK_TIME_SECONDS s (not counted toward wall time)")

        @testset "Phase 3.11B: compression wait and context retention" begin
            for pass in 1:PASSES
                marker, prompts = dialogue_prompts(pass)
                order = isodd(pass) ? POLICIES : reverse(POLICIES)
                println("\n========== Pass $pass / $PASSES: ", join(string.(order), " -> "), " ==========")
                for policy in order
                    session = new_session(root, "pass-$pass-$policy", policy)
                    println("\n----- Policy $policy | release $marker -----")
                    try
                        @test session.session.cfg.num_ctx == NUM_CTX
                        @test session.session.cfg.max_tokens == MAX_TOKENS
                        for (turn, prompt) in enumerate(prompts)
                            if turn > 1 && THINK_TIME_SECONDS > 0.0
                                sleep(THINK_TIME_SECONDS)
                            end
                            measurement = request!(session, prompt, marker)
                            validate_result(measurement)
                            print_result(policy, turn, measurement)
                            push!(results, merge((pass = pass, policy = policy, turn = turn), measurement))

                            if turn == 5
                                prior_summary = measurement.summary_when_prior_task_finished
                                println("    release identifier in response: ", measurement.marker_in_answer,
                                        " | in prior history after response: ", measurement.marker_in_prior_context_after,
                                        " | in current summary: ", measurement.after.marker_in_summary)
                                println("    summary from pending compression task contains identifier: ",
                                        isnothing(prior_summary) ? "not observed" : string(occursin(marker, prior_summary)))
                                if measurement.before.approx_tokens + MAX_TOKENS >= NUM_CTX
                                    @warn "Estimated incoming context plus output budget approaches num_ctx" pass policy
                                end
                                if policy == :deferred
                                    @test !measurement.before.pending
                                    @test !measurement.before.summary_active
                                end
                            end
                        end
                    finally
                        # close() waits for any unfinished compression, outside the
                        # timed requests. Test data stays inside mktempdir.
                        close(session)
                    end
                end
            end

            println("\n========== Median measurements by turn and policy ==========")
            println("Turn | Policy    | Prompt eval | TTFT | Wall | Wall-client | Wall-server | Length responses")
            for turn in 1:5, policy in POLICIES
                rows = filter(r -> r.turn == turn && r.policy == policy, results)
                length_hits = count(r -> r.reason == "length", rows)
                println(lpad(string(turn), 4), " | ", rpad(string(policy), 9), " | ",
                        rounded(seconds(median_value(r.prompt_ns for r in rows))), " s | ",
                        rounded(seconds(median_value(r.ttft_ns for r in rows))), " s | ",
                        rounded(seconds(median_value(r.wall_ns for r in rows))), " s | ",
                        rounded(seconds(median_value(r.prepost_ns for r in rows))), " s | ",
                        rounded(seconds(median_value(r.wall_minus_server_ns for r in rows))), " s | ",
                        length_hits, "/", length(rows))
            end

            println("\n========== Final-turn context and recall ==========")
            println("Pass | Policy    | Pending task | Approx task wait | Summary | Identifier in prior history | Identifier in answer")
            for r in filter(r -> r.turn == 5, results)
                println(lpad(string(r.pass), 4), " | ", rpad(string(r.policy), 9), " | ",
                        r.before.pending, " | ", rounded(seconds(r.pending_wait_estimate_ns)), " s | ",
                        r.after.summary_active, " | ", r.marker_in_prior_context_after, " | ", r.marker_in_answer)
            end
        end
    end
finally
    if benchmark_owned_model[]
        try
            PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
        catch error
            @warn "Unable to stop the benchmark-owned Ollama model" exception = (error, catch_backtrace())
        end
    end
end

println("\nTemporary benchmark sessions and artifacts cleaned up.")
println("A summary observed after a request is not, by itself, proof of the exact on-wire request context.")
println("Wall-client is outside the measured HTTP stream, but includes session work beyond compression waiting.")
println("The watcher measures task completion approximately; it does not instrument the internal lock directly.")
println("Recall checks are diagnostic, not a complete accuracy evaluation. 'length' denotes output-budget truncation.")
