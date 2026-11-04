using Test

# Phase 3.12: one bounded inference-configuration and completeness benchmark.
# Run only on the user's local development machine, with no other Ollama model resident.
# GPU workload: moderate; 16K context allocation and the output-budget comparison
# can briefly demand substantial GPU resources. No GPU power/utilization limits are changed.
ENV["PHLUXAI_BANNER"] = "0"
include(joinpath(@__DIR__, "..", "src", "PhluxAI.jl"))
using .PhluxAI

const OC = PhluxAI.OllamaClient
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const SYSTEM_PROMPT = "You are an accurate local engineering assistant. Preserve identifiers exactly and state unsupported claims as uncertain."
const CONFIGS = ((4096, 8), (4096, 4), (8192, 8), (8192, 4), (16384, 8), (16384, 4))
const REPETITIONS = 2
const DEFAULT_OUTPUT = 512
const TIMEOUT = 180.0
const KEEP_ALIVE = "10m"

seconds(ns::Integer) = Float64(ns) / 1.0e9
rounded(x::Real) = round(Float64(x); digits = 3)

function midpoint(values)
    numbers = sort(collect(values))
    isempty(numbers) && throw(ArgumentError("Cannot take the median of an empty collection"))
    n = length(numbers)
    return isodd(n) ? numbers[(n + 1) ÷ 2] : (numbers[n ÷ 2] + numbers[n ÷ 2 + 1]) / 2
end

function ensure_exclusive_ollama()
    lines = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
    length(lines) == 1 || error(
        "An Ollama model is already resident. This exclusive benchmark will not alter it. " *
        "Run the test when those sessions are finished.\n" * join(lines, "\n"),
    )
end

function memory_status()
    isnothing(Sys.which("nvidia-smi")) && return "nvidia-smi unavailable"
    try
        return strip(read(
            `nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits`,
            String,
        ))
    catch error
        return "nvidia-smi unavailable: $(sprint(showerror, error))"
    end
end

function session_config(context::Int, threads::Int, max_output::Int)
    return BudgetConfig(
        num_ctx = context,
        num_thread = threads,
        max_tokens = max_output,
        system_budget = 256,
        summary_budget = 512,
        response_budget = max_output,
        safety_margin = 128,
    )
end

function new_session(root::String, label::String, context::Int, threads::Int, max_output::Int)
    return ModelSession(
        MODEL;
        base_url = URL,
        system_prompt = SYSTEM_PROMPT,
        cfg = session_config(context, threads, max_output),
        session_dir = joinpath(root, "session-" * label),
        artifact_dir = joinpath(root, "artifacts-" * label),
        sandbox_dir = joinpath(root, "sandbox-" * label),
        tool_mode = TOOLS_OFF,
        discover_capabilities = false,
        start_model = false,
        stop_model_on_close = false,
        keep_alive = KEEP_ALIVE,
    )
end

function request(root::String, label::String, prompt::String, context::Int, threads::Int, max_output::Int)
    session = new_session(root, label, context, threads, max_output)
    try
        started = time_ns()
        response = respond!(
            session, prompt;
            temperature = 0.0,
            max_tokens = max_output,
            stream = true,
            print_tokens = false,
            tool_mode = TOOLS_OFF,
            request_timeout = TIMEOUT,
            read_idle_timeout = TIMEOUT,
        )
        wall_ns = time_ns() - started
        m = response.metrics
        return (
            wall = seconds(wall_ns),
            server = seconds(m.total_duration_ns),
            load = seconds(m.load_duration_ns),
            prompt = seconds(m.prompt_eval_duration_ns),
            prompt_tokens = m.prompt_eval_count,
            generation = seconds(m.eval_duration_ns),
            output_tokens = m.eval_count,
            rate = m.eval_duration_ns > 0 ? m.eval_count / seconds(m.eval_duration_ns) : 0.0,
            ttft = isnothing(m.time_to_first_token_ns) ? NaN : seconds(m.time_to_first_token_ns),
            reason = response.done_reason,
            done = response.done,
            text = response.text,
        )
    finally
        close(session)
    end
end

function prompt_with_marker(marker::String, target_bytes::Int)
    io = IOBuffer()
    println(io, "Independent test case: $marker")
    println(io, "Verified engineering facts: release identifier = $marker; artifact handling = remove partial outputs; " *
                "concurrency rule = reject stale history-summary revisions.")
    index = 0
    while position(io) < target_bytes
        index += 1
        println(io, "Audit entry $index: an agent validates numeric input bounds, checks local sandbox permissions, " *
                    "compares expected and actual file hashes, and records whether state changes can be rolled back. " *
                    "The observed sample number is $(mod(index * 137, 991)).")
    end
    println(io, "From the verified facts, state the release identifier exactly, identify the artifact-handling " *
                "and concurrency requirements, and give one concrete verification check. Be concise and accurate.")
    return String(take!(io))
end

function substantial_prompt()
    return """
    Review the following proposed design for a local Julia multimodal agent framework. The framework invokes a local text model,
    generates images using a separate local backend, stores conversation history on disk, and allows supervised sandboxed tools.
    The design must satisfy these eight engineering constraints:
    (1) avoid losing user-designated critical facts during history compression;
    (2) never commit summaries generated from stale history revisions;
    (3) keep inference local and avoid unintended external requests;
    (4) manage GPU residency when text and image backends alternate;
    (5) avoid deleting outputs owned by unrelated processes;
    (6) reject oversized prompts rather than silently truncating essential facts;
    (7) measure latency separately from correctness and response completeness;
    (8) preserve session persistence and recovery after failures.

    Produce a technical assessment addressing each constraint, its principal failure mode, and a practical validation strategy.
    Explain important interactions between the constraints and distinguish established evidence from assumptions.
    Begin with the exact identifier PH312-BUDGET-CHECK. Do not invent observations or benchmark measurements.
    """
end

function print_measurement(label::String, value)
    println("  ", rpad(label, 18), " | input ", value.prompt_tokens,
            " | output ", value.output_tokens, " | reason ", value.reason)
    println("    wall ", rounded(value.wall), " s | load ", rounded(value.load),
            " s | prompt ", rounded(value.prompt), " s | TTFT ", rounded(value.ttft),
            " s | generation ", rounded(value.generation), " s | ", rounded(value.rate), " tok/s")
    if value.load > 0.5
        @warn "In-request loading is substantial; interpret this run as not fully warm" label load_seconds = value.load
    end
end

function assert_basic(value, max_output::Int)
    @test value.done
    @test value.prompt_tokens > 0
    @test 0 < value.output_tokens <= max_output
    @test isfinite(value.ttft) && value.ttft > 0
    @test !isempty(strip(value.text))
end

function warm_config(root::String, label::String, context::Int, threads::Int)
    value = request(root, "warm-" * label, "Reply exactly READY.", context, threads, 32)
    @test value.done
    @test !isempty(strip(value.text))
    println("    Ollama residency:\n", strip(read(`ollama ps`, String)))
    println("    VRAM used / total / temperature (MiB, MiB, C): ", memory_status())
    return value
end

function run_config_grid(root::String)
    measurements = NamedTuple[]
    println("\n========== A. Context window and CPU thread count ==========")
    for (context, threads) in CONFIGS
        println("\nConfig: num_ctx=$context | num_thread=$threads | num_predict=$DEFAULT_OUTPUT")
        warm_config(root, "ctx-$context-threads-$threads", context, threads)
        for pass in 1:REPETITIONS
            marker = "PH312-$context-$threads-$pass"
            prompt = prompt_with_marker(marker, 5600)
            value = request(root, "ctx-$context-threads-$threads-pass-$pass", prompt,
                            context, threads, DEFAULT_OUTPUT)
            print_measurement("pass $pass", value)
            assert_basic(value, DEFAULT_OUTPUT)
            recalled = occursin(marker, value.text)
            println("    release identifier recalled: ", recalled)
            @test recalled
            push!(measurements, merge(value, (context = context, threads = threads, pass = pass, recalled = recalled)))
        end
    end
    println("\nContext | Threads | Median wall | Median prompt eval | Median TTFT | Median tok/s | Recall")
    for (context, threads) in CONFIGS
        selected = filter(x -> x.context == context && x.threads == threads, measurements)
        println(lpad(string(context), 7), " | ", lpad(string(threads), 7), " | ",
                lpad(string(rounded(midpoint(x.wall for x in selected))), 11), " s | ",
                lpad(string(rounded(midpoint(x.prompt for x in selected))), 18), " s | ",
                lpad(string(rounded(midpoint(x.ttft for x in selected))), 11), " s | ",
                lpad(string(rounded(midpoint(x.rate for x in selected))), 12), " | ",
                count(x -> x.recalled, selected), "/", length(selected))
    end
    return measurements
end

function run_long_context(root::String)
    println("\n========== B. Larger, genuinely uncached input ==========")
    println("The 4K configuration is omitted because this input is deliberately larger than its safe budget.")
    for context in (8192, 16384)
        warm_config(root, "long-$context", context, 8)
        marker = "PH312-LONG-$context"
        prompt = prompt_with_marker(marker, 23000)
        value = request(root, "long-context-$context", prompt, context, 8, DEFAULT_OUTPUT)
        print_measurement("context $context", value)
        assert_basic(value, DEFAULT_OUTPUT)
        recalled = occursin(marker, value.text)
        println("    release identifier recalled: ", recalled)
        @test recalled
        println("    Actual prompt tokens: ", value.prompt_tokens,
                "; remaining nominal context: ", context - value.prompt_tokens - DEFAULT_OUTPUT)
        @test value.prompt_tokens + DEFAULT_OUTPUT < context
    end
end

function run_output_budget(root::String)
    println("\n========== C. Output limit versus answer completeness ==========")
    println("The prompt is identical. Different limits affect maximum work, not the amount of work the model must perform.")
    prompt = substantial_prompt()
    context = 16384
    threads = 8
    for limit in (512, 1024)
        warm_config(root, "budget-$limit", context, threads)
        value = request(root, "output-budget-$limit", prompt, context, threads, limit)
        print_measurement("limit $limit", value)
        assert_basic(value, limit)
        recalled = occursin("PH312-BUDGET-CHECK", value.text)
        println("    identifier included: ", recalled, " | natural completion: ", value.reason == "stop")
        @test recalled
        println("    Assessment: ", value.reason == "length" ?
                "output budget terminated the response; do not infer completeness" :
                "response terminated naturally; semantic correctness is not established")
    end
end

println("Phase 3.12 — single-pass inference-configuration decision benchmark")
println("GPU intensity: moderate; 16K context and the 1024-token output budget may cause sustained GPU work.")
println("No explicit model unload/preload occurs between measurements; Ollama may reconfigure itself. Warmups are excluded.")
println("Model: $MODEL | production reference: num_ctx=16384, num_thread=8, max_tokens=512")

ensure_exclusive_ollama()
parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected existing PhluxAI data directory: $parent")

model_owned = false
try
    global model_owned = true
    OC.start_model!(MODEL, URL; timeout = 120)
    mktempdir(parent; prefix = "phase3-12-inference-configuration-") do root
        @testset "Phase 3.12: inference configuration and response completeness" begin
            run_config_grid(root)
            run_long_context(root)
            run_output_budget(root)
        end
    end
finally
    if model_owned
        println("\n[Cleanup] Unloading only the benchmark-owned Qwen model...")
        try
            OC.stop_model!(OC.OllamaModel(MODEL, URL))
        catch error
            @warn "Benchmark-owned Qwen unload failed; check `ollama ps`" exception = (error, catch_backtrace())
        end
    end
end

println("\nTemporary sessions and artifacts cleaned up.")
println("Interpretation rules: compare like-for-like prompt counts and GPU residency; large load_duration indicates a cold/confounded run.")
println("A 'length' completion establishes budget truncation, not semantic correctness. Stable improvements require more than tiny timing differences.")
