using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

# Phase 3.9: independent, uncached-input prompt-scaling benchmark.
# Each measured request uses a fresh ModelSession; the Ollama model stays warm.
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const CONTEXT_WINDOW = 4096
const TARGETS = (256, 768, 1536, 3072)  # APPROXIMATE input tokens, not actual

# Reversing/mixing size order reduces the confounding effect of run order.
const ORDERS = (
    (256, 768, 1536, 3072),
    (3072, 1536, 768, 256),
    (768, 3072, 256, 1536),
)

sec(ns) = Float64(ns) / 1.0e9
rounded(x) = round(x; digits = 3)

function median_value(values)
    sorted = sort(collect(values))
    n = length(sorted)
    n > 0 || error("Cannot take median of empty observations")
    return isodd(n) ? sorted[(n + 1) ÷ 2] :
           (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

function synthetic_prompt(target_tokens::Int, run_id::Int)::String
    # The run identifier appears before the long content to invalidate
    # reuse of the prior request's user-prompt KV-cache prefix.
    io = IOBuffer()
    println(io, "UNIQUE BENCHMARK REQUEST $(run_id); target $(target_tokens).")
    println(io, "Read the following independent engineering records. ",
            "After the final record, provide a detailed technical ",
            "explanation of their engineering implications.")
    k = 0
    while position(io) < target_tokens * 4
        k += 1
        id = run_id * 10000 + k
        println(io, "Record $(id): the local parser validates array bounds, ",
                "checks numerical stability, and records a reversible ",
                "checksum. The test fixture $(mod(id * 37, 1009)) must not ",
                "modify any external files or network services.")
    end
    println(io, "Explain the implications for correctness, numerical stability, and software reliability.")
    return String(take!(io))
end

function new_session(root::String, label::String)
    return ModelSession(
        MODEL;
        base_url = URL,
        system_prompt = "You are a concise software engineering assistant.",
        cfg = BudgetConfig(
            num_ctx = CONTEXT_WINDOW,
            num_thread = 8,
            max_tokens = 512,
            system_budget = 256,
            summary_budget = 128,
            response_budget = 32,
            safety_margin = 128,
        ),
        session_dir = joinpath(root, "session-" * label),
        artifact_dir = joinpath(root, "artifacts-" * label),
        sandbox_dir = joinpath(root, "sandbox-" * label),
        tool_mode = TOOLS_OFF,
        start_model = false,
        stop_model_on_close = false,
        discover_capabilities = false,
        keep_alive = "5m",
    )
end

function run_prompt(root::String, label::String, prompt::String)
    session = new_session(root, label)
    try
        started = time_ns()
        response = respond!(
            session, prompt;
            temperature = 0.0,
            stream = true,
            print_tokens = false,
            tool_mode = TOOLS_OFF,
            request_timeout = 180.0,
            read_idle_timeout = 180.0,
        )
        wall = sec(time_ns() - started)
        return response, wall
    finally
        close(session)
    end
end

# Do not unload another user's existing Ollama session on completion.
ps_lines = filter(line -> !isempty(strip(line)),
                  split(read(`ollama ps`, String), '\n'))
length(ps_lines) == 1 || error(
    "Ollama already has a resident model; unload it manually before " *
    "running this exclusive benchmark. Current state:\n" *
    join(ps_lines, "\n"),
)

root_parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(root_parent) || error("Expected PhluxAI data directory: $root_parent")

try
    mktempdir(root_parent; prefix = "ollama-prompt-scaling-") do root
        println("Warming Ollama model (excluded from timing table)...")
        warm_response, warm_wall = run_prompt(
            root, "warmup", "Warmup request: answer with exactly READY.",
        )
        warm_response.done || error("Ollama warmup did not complete")
        println("Warmup wall: ", rounded(warm_wall), " s")
        println("Context window: ", CONTEXT_WINDOW, " tokens")
        println("Model: ", MODEL, "\n")

        results = NamedTuple[]
        @testset "PhluxAI uncached prompt scaling" begin
            run_id = 0
            for (pass, order) in enumerate(ORDERS)
                println("----- Pass $pass / $(length(ORDERS)) -----")
                for target in order
                    run_id += 1
                    prompt = synthetic_prompt(target, run_id)
                    response, wall = run_prompt(root, "run-$run_id", prompt)
                    m = response.metrics
                    ttft = isnothing(m.time_to_first_token_ns) ? NaN :
                           sec(m.time_to_first_token_ns)
                    evaluation = sec(m.prompt_eval_duration_ns)
                    generation = sec(m.eval_duration_ns)
                    total = sec(m.total_duration_ns)
                    push!(results, (
                        target = target,
                        prompt_tokens = m.prompt_eval_count,
                        prompt_seconds = evaluation,
                        first_token_seconds = ttft,
                        generation_seconds = generation,
                        output_tokens = m.eval_count,
                        wall_seconds = wall,
                    ))

                    @test response.done
                    @test !isempty(strip(response.text))
                    @test m.prompt_eval_count > 0
                    @test m.prompt_eval_duration_ns > 0
                    @test m.eval_count > 0
                    @test !isnothing(m.time_to_first_token_ns)

                    println("Target ~", target,
                            " | actual prompt tokens ", m.prompt_eval_count,
                            " | prompt eval ", rounded(evaluation), " s",
                            " | TTFT ", rounded(ttft), " s",
                            " | wall ", rounded(wall), " s")
                    println("  Model load ", rounded(sec(m.load_duration_ns)),
                            " s; generation ", rounded(generation),
                            " s / ", m.eval_count, " tokens; server total ",
                            rounded(total), " s")
                    println("  Completion reason: ", response.done_reason)
                    if m.prompt_eval_count >= CONTEXT_WINDOW - 64
                        @warn "Prompt approaches context limit; compare counts for truncation" target m.prompt_eval_count
                    end
                end
            end

            println("\n========== Median per input length (3 runs each) ==========")
            println("Approx target | Actual prompt tokens | Prompt eval | TTFT | Wall")
            for target in TARGETS
                selected = filter(r -> r.target == target, results)
                actual = median_value(r.prompt_tokens for r in selected)
                eval_s = median_value(r.prompt_seconds for r in selected)
                ttft_s = median_value(r.first_token_seconds for r in selected)
                wall_s = median_value(r.wall_seconds for r in selected)
                println(lpad(string(target), 13), " | ",
                        lpad(string(rounded(actual)), 20), " | ",
                        rounded(eval_s), " s | ",
                        rounded(ttft_s), " s | ",
                        rounded(wall_s), " s")
            end
        end
    end
finally
    try
        PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
    catch error
        @warn "Could not unload the benchmark Ollama model" exception = (
            error, catch_backtrace(),
        )
    end
end

println("\nTemporary benchmark sessions and artifacts cleaned up.")
