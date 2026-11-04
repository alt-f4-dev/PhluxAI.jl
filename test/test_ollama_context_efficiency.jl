using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

# Phase 3.11: prompt-prefix reuse and growing conversation history.
# No production code is changed. The model remains resident after warmup;
# there is deliberately no unload/preload cycle between measured requests.
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const SYSTEM_PROMPT = "You are a careful local software engineering assistant. Distinguish observations from assumptions."
const NUM_CTX = 4096
const NUM_THREADS = 8
const MAX_TOKENS = 512
const KEEP_ALIVE = "5m"
const TIMEOUT = 180.0
const PREFIX_TARGET = 2048   # Approximate, based on character count, not tokenizer output
const PREFIX_PASSES = 3
const DIALOGUE_PASSES = 2

seconds(ns) = Float64(ns) / 1.0e9
rounded(value) = round(value; digits = 3)

function median_value(values)
    sorted = sort(collect(values))
    isempty(sorted) && throw(ArgumentError("Cannot calculate the median of an empty collection"))
    n = length(sorted)
    return isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

function fresh_session(root::String, label::String)
    return ModelSession(
        MODEL;
        base_url = URL,
        system_prompt = SYSTEM_PROMPT,
        cfg = BudgetConfig(
            num_ctx = NUM_CTX,
            num_thread = NUM_THREADS,
            max_tokens = MAX_TOKENS,
            system_budget = 256,
            summary_budget = 128,
            response_budget = MAX_TOKENS,
            safety_margin = 128,
            # Deliberately retain the default compression policy in the
            # realistic conversation arm, so its effect is observable.
        ),
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

function request!(session::ModelSession, prompt::String)
    before = PhluxAI.OllamaClient.assemble_messages(session.session)
    approx_input = PhluxAI.OllamaClient.approx_tokens(before) +
                   PhluxAI.OllamaClient.approx_tokens(prompt) + 4
    start = time_ns()
    response = respond!(
        session, prompt;
        temperature = 0.0,
        stream = true,
        print_tokens = false,
        tool_mode = TOOLS_OFF,
        request_timeout = TIMEOUT,
        read_idle_timeout = TIMEOUT,
    )
    wall_ns = time_ns() - start
    m = response.metrics
    return (
        response = response,
        approx_input = approx_input,
        wall_ns = wall_ns,
        server_ns = m.total_duration_ns,
        client_ns = m.client_total_duration_ns,
        load_ns = m.load_duration_ns,
        prompt_tokens = m.prompt_eval_count,
        prompt_ns = m.prompt_eval_duration_ns,
        output_tokens = m.eval_count,
        output_ns = m.eval_duration_ns,
        ttft_ns = m.time_to_first_token_ns,
        reason = response.done_reason,
    )
end

function once(root::String, label::String, prompt::String)
    session = fresh_session(root, label)
    try
        return request!(session, prompt)
    finally
        close(session)
    end
end

function print_measurement(label::String, result)
    ttft = isnothing(result.ttft_ns) ? NaN : seconds(result.ttft_ns)
    rate = result.output_ns > 0 ? result.output_tokens / seconds(result.output_ns) : NaN
    println("  ", rpad(label, 19), " | estimated input ", result.approx_input,
            " | Ollama prompt count ", result.prompt_tokens,
            " | prompt eval ", rounded(seconds(result.prompt_ns)), " s",
            " | TTFT ", rounded(ttft), " s")
    println("    wall ", rounded(seconds(result.wall_ns)), " s",
            " | server ", rounded(seconds(result.server_ns)), " s",
            " | load ", rounded(seconds(result.load_ns)), " s",
            " | wall-server ", rounded(seconds(result.wall_ns - result.server_ns)), " s")
    println("    output ", result.output_tokens, " tokens",
            " | generation ", rounded(seconds(result.output_ns)), " s",
            " | rate ", rounded(rate), " tokens/s",
            " | completion ", result.reason)
end

function validate_result(result)
    @test result.response.done
    @test !isempty(strip(result.response.text))
    @test result.prompt_tokens > 0
    @test result.prompt_ns > 0
    @test result.output_tokens > 0
    @test result.output_ns > 0
    @test !isnothing(result.ttft_ns)
    @test result.server_ns > 0
    @test result.wall_ns > 0
    @test result.reason in ("stop", "length")
end

function shared_records(pass::Int)::String
    io = IOBuffer()
    println(io, "Engineering review P311 case $pass. Treat these records as input evidence.")
    println(io, "Assess software reliability, security boundaries, failure handling, and verification strategy.")
    index = 0
    while position(io) < PREFIX_TARGET * 4
        index += 1
        id = pass * 10000 + index
        println(io, "Record $id: the local agent validates array bounds and input schema, ",
                "maintains reversible artifact storage, checks numerical tolerances, ",
                "and distinguishes loopback binding from actual network isolation. ",
                "A failure report must separate observed behavior from inferred causes. ",
                "Check $(mod(id * 37, 1009)) is an independent audit identifier.")
    end
    return String(take!(io))
end

function prefix_prompts(pass::Int)
    base = shared_records(pass)
    primer = base * "\nWhich verification steps would best detect silent failures? Explain the reasoning."
    probe = base * "\nWhich architectural risks require explicit regression tests? Explain the reasoning."
    # A distinct first line disrupts the long shared prefix while preserving
    # the engineering records and question. The system prompt is still shared.
    control = "Distinct control request $(pass * 7919 + 31).\n" * probe
    @assert startswith(primer, base) && startswith(probe, base)
    return (control = control, primer = primer, probe = probe)
end

function dialogue_prompts(pass::Int)
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

function message_signature(session::ModelSession)
    messages = PhluxAI.OllamaClient.assemble_messages(session.session)
    return [(message.role, message.content) for message in messages]
end

function resident_model_present()::Bool
    lines = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
    return any(line -> occursin(MODEL, line), lines[2:end])
end

lines = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
length(lines) == 1 || error(
    "Ollama already has a resident model. Run this exclusive benchmark only after " *
    "manually stopping existing sessions. Current state:\n" * join(lines, "\n"),
)

parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("PhluxAI data directory does not exist: $parent")

prefix_results = NamedTuple[]
dialogue_results = NamedTuple[]
benchmark_owned_model = Ref(false)
try
    mktempdir(parent; prefix = "ollama-context-efficiency-") do root
        println("Warming Ollama once. Warmup is excluded from reported measurements...")
        benchmark_owned_model[] = true
        warmup = once(root, "warmup", "Confirm the inference service is responding normally.")
        warmup.response.done || error("Warmup did not finish")
        resident_model_present() || error("Qwen is not resident after warmup; cannot perform a warm interactive benchmark")
        println("Warmup wall: ", rounded(seconds(warmup.wall_ns)), " s",
                " | warmup model load: ", rounded(seconds(warmup.load_ns)), " s")
        println("Model: $MODEL | num_ctx: $NUM_CTX | max output tokens: $MAX_TOKENS")
        println("Model is kept resident throughout all measured requests.\n")

        @testset "Phase 3.11: PhluxAI prompt and context efficiency" begin
            @testset "A. Stable-prefix versus disrupted-prefix inputs" begin
                for pass in 1:PREFIX_PASSES
                    println("\n----- Prefix experiment $pass / $PREFIX_PASSES -----")
                    prompts = prefix_prompts(pass)
                    order = isodd(pass) ? (:control, :primer, :probe) : (:primer, :probe, :control)
                    observations = Dict{Symbol,Any}()
                    for kind in order
                        prompt = getproperty(prompts, kind)
                        result = once(root, "prefix-$pass-$kind", prompt)
                        validate_result(result)
                        observations[kind] = result
                        push!(prefix_results, merge((pass = pass, kind = kind), result))
                        print_measurement(string(kind), result)
                        if seconds(result.load_ns) > 0.5
                            @warn "Unexpected model loading in warm-prefix experiment" pass kind load_seconds = seconds(result.load_ns)
                        end
                    end
                    @test length(observations) == 3
                    control = observations[:control]
                    probe = observations[:probe]
                    println("  Probe/control prompt-evaluation ratio: ",
                            rounded(probe.prompt_ns / control.prompt_ns),
                            " (less than 1 suggests a prefix-reuse benefit, not proof of a cache hit)")
                end

                println("\n========== Prefix experiment: median of $PREFIX_PASSES passes ==========")
                println("Input kind | Prompt count | Prompt eval | TTFT | Load | Output | Completion length count")
                for kind in (:control, :primer, :probe)
                    rows = filter(r -> r.kind == kind, prefix_results)
                    length_hits = count(r -> r.reason == "length", rows)
                    println(rpad(string(kind), 10), " | ",
                            rounded(median_value(r.prompt_tokens for r in rows)), " | ",
                            rounded(seconds(median_value(r.prompt_ns for r in rows))), " s | ",
                            rounded(seconds(median_value(r.ttft_ns for r in rows))), " s | ",
                            rounded(seconds(median_value(r.load_ns for r in rows))), " s | ",
                            rounded(median_value(r.output_tokens for r in rows)), " | ", length_hits, "/", length(rows))
                end
            end

            @testset "B. Realistic persistent conversations with policy enabled" begin
                for pass in 1:DIALOGUE_PASSES
                    println("\n----- Dialogue experiment $pass / $DIALOGUE_PASSES -----")
                    marker, prompts = dialogue_prompts(pass)
                    session = fresh_session(root, "dialogue-$pass")
                    prior_signature = nothing
                    try
                        for (turn, prompt) in enumerate(prompts)
                            before_signature = message_signature(session)
                            prefix_unchanged = isnothing(prior_signature) ||
                                               (length(before_signature) >= length(prior_signature) &&
                                                before_signature[1:length(prior_signature)] == prior_signature)
                            summary_before = !isnothing(session.session.summary)
                            result = request!(session, prompt)
                            validate_result(result)
                            after_signature = message_signature(session)
                            summary_after = !isnothing(session.session.summary)
                            prior_signature = after_signature
                            marker_found = turn == length(prompts) ? occursin(marker, result.response.text) : missing
                            push!(dialogue_results, (
                                pass = pass,
                                turn = turn,
                                prefix_unchanged = prefix_unchanged,
                                summary_before = summary_before,
                                summary_after = summary_after,
                                marker_found = marker_found,
                                result = result,
                            ))
                            print_measurement("dialogue turn $turn", result)
                            println("    history prefix unchanged since prior turn: ", prefix_unchanged,
                                    " | summary before/after: ", summary_before, "/", summary_after,
                                    " | assembled messages: ", length(after_signature))
                            if !ismissing(marker_found)
                                println("    release identifier present in final answer: ", marker_found,
                                        " (limited recall check, not an accuracy score)")
                            end
                            if result.approx_input + MAX_TOKENS >= NUM_CTX
                                @warn "Estimated context plus output budget approaches or exceeds num_ctx; interpretation may be affected by truncation" pass turn approx_input = result.approx_input
                            end
                            if !prefix_unchanged
                                @warn "Conversation prefix changed. Check summarization/compression before attributing latency to cache reuse" pass turn
                            end
                            @test length(after_signature) >= 2
                        end
                    finally
                        close(session)
                    end
                end
                println("\n========== Persistent-dialogue observations ==========")
                println("Turn | Estimated input | Ollama prompt count | Prompt eval | TTFT | Length completions | Summary active")
                for turn in 1:5
                    rows = filter(r -> r.turn == turn, dialogue_results)
                    results = [r.result for r in rows]
                    println(lpad(string(turn), 4), " | ",
                            lpad(string(rounded(median_value(r.approx_input for r in results))), 15), " | ",
                            lpad(string(rounded(median_value(r.prompt_tokens for r in results))), 19), " | ",
                            rounded(seconds(median_value(r.prompt_ns for r in results))), " s | ",
                            rounded(seconds(median_value(r.ttft_ns for r in results))), " s | ",
                            count(r -> r.reason == "length", results), "/", length(results), " | ",
                            count(r -> r.summary_before || r.summary_after, rows), "/", length(rows))
                end
            end
        end
    end
finally
    if benchmark_owned_model[]
        try
            PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
        catch error
            @warn "Unable to unload benchmark-owned Ollama model" exception = (error, catch_backtrace())
        end
    end
end

println("\nTemporary benchmark sessions and artifacts cleaned up.")
println("Interpretation: Ollama's prompt_eval_count is not necessarily the full context length.")
println("Prefix caching is inferred from measured evaluation time, not assumed from prompt similarity.")
println("Model responses are not comprehensively scored for correctness; 'length' means output-budget truncation.")
