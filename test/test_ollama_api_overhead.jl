using Test
using HTTP
using JSON3

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

# Phase 3.10: compare direct Ollama streaming with the full PhluxAI agent path.
# Every paired comparison uses identical input and wire parameters. The model
# is unloaded and preloaded between measurements to limit prompt-cache bias.
const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"
const SYSTEM_PROMPT = "You are a careful software engineering assistant. Be accurate and identify uncertainty."
const NUM_CTX = 4096
const MAX_TOKENS = 512
const NUM_THREADS = 8
const KEEP_ALIVE = "5m"
const TARGETS = (256, 2048)
const REPETITIONS = 3
const TIMEOUT = 180.0

sec(ns) = Float64(ns) / 1.0e9
rounded(x) = round(x; digits = 3)

function median_value(values)
    xs = sort(collect(values))
    isempty(xs) && throw(ArgumentError("cannot take median of empty values"))
    n = length(xs)
    return isodd(n) ? xs[(n + 1) ÷ 2] : (xs[n ÷ 2] + xs[n ÷ 2 + 1]) / 2
end

function benchmark_prompt(target::Int, run_id::Int)::String
    io = IOBuffer()
    println(io, "Unique evaluation case $run_id. Review these engineering observations and their implications.")
    println(io, "Explain likely failure modes, verification checks, and tradeoffs. Do not invent evidence.")
    record = 0
    while position(io) < target * 4
        record += 1
        id = 10000 * run_id + record
        println(io, "Observation $id: a local service validates input bounds, protects artifact writes, ",
                "measures numerical error, and reports whether a rollback succeeded. ",
                "The test case $(mod(id * 37, 1009)) exercises fault handling and independent process state.")
    end
    println(io, "Discuss which checks would distinguish a real failure from a misleading measurement.")
    return String(take!(io))
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

function run_session(root::String, label::String, prompt::String)
    setup_started = time_ns()
    session = fresh_session(root, label)
    setup_ns = time_ns() - setup_started
    try
        started = time_ns()
        response = respond!(
            session,
            prompt;
            temperature = 0.0,
            stream = true,
            print_tokens = false,
            tool_mode = TOOLS_OFF,
            request_timeout = TIMEOUT,
            read_idle_timeout = TIMEOUT,
        )
        wall_ns = time_ns() - started
        m = response.metrics
        return (
            path = :phluxai,
            wall_ns = wall_ns,
            server_ns = m.total_duration_ns,
            client_ns = m.client_total_duration_ns,
            ttft_ns = m.time_to_first_token_ns,
            load_ns = m.load_duration_ns,
            prompt_tokens = m.prompt_eval_count,
            prompt_ns = m.prompt_eval_duration_ns,
            output_tokens = m.eval_count,
            output_ns = m.eval_duration_ns,
            done = response.done,
            reason = response.done_reason,
            text = response.text,
            setup_ns = setup_ns,
        )
    finally
        close(session)
    end
end

function direct_payload(prompt::String)::Dict{String,Any}
    return Dict{String,Any}(
        "model" => MODEL,
        "messages" => [
            Dict("role" => "system", "content" => SYSTEM_PROMPT),
            Dict("role" => "user", "content" => prompt),
        ],
        "stream" => true,
        "options" => Dict(
            "temperature" => 0.0,
            "num_predict" => MAX_TOKENS,
            "num_ctx" => NUM_CTX,
            "num_thread" => NUM_THREADS,
        ),
        "keep_alive" => KEEP_ALIVE,
    )
end

function run_direct(prompt::String)
    started = time_ns()
    payload = direct_payload(prompt)
    buffer = IOBuffer()
    first_token_ns = nothing
    final_chunk = nothing

    HTTP.open(
        "POST", URL * "/api/chat", ["Content-Type" => "application/json"];
        request_timeout = TIMEOUT,
        read_idle_timeout = TIMEOUT,
    ) do stream
        write(stream, JSON3.write(payload))
        HTTP.closewrite(stream)
        while !eof(stream)
            line = readline(stream)
            isempty(line) && continue
            chunk = JSON3.read(line)
            message = get(chunk, :message, nothing)
            if message !== nothing
                piece = String(get(message, :content, ""))
                thought = String(get(message, :thinking, ""))
                if (!isempty(piece) || !isempty(thought)) && isnothing(first_token_ns)
                    first_token_ns = Int(time_ns() - started)
                end
                write(buffer, piece)
            end
            if get(chunk, :done, false) === true
                final_chunk = chunk
                break
            end
        end
    end

    wall_ns = time_ns() - started
    final_chunk === nothing && error("Direct Ollama stream ended without a final response")
    return (
        path = :direct,
        wall_ns = wall_ns,
        server_ns = Int(get(final_chunk, :total_duration, 0)),
        client_ns = wall_ns,
        ttft_ns = first_token_ns,
        load_ns = Int(get(final_chunk, :load_duration, 0)),
        prompt_tokens = Int(get(final_chunk, :prompt_eval_count, 0)),
        prompt_ns = Int(get(final_chunk, :prompt_eval_duration, 0)),
        output_tokens = Int(get(final_chunk, :eval_count, 0)),
        output_ns = Int(get(final_chunk, :eval_duration, 0)),
        done = Bool(get(final_chunk, :done, false)),
        reason = get(final_chunk, :done_reason, nothing),
        text = String(take!(buffer)),
        setup_ns = 0,
    )
end

function no_resident_ollama_models()::Bool
    lines = filter(!isempty, strip.(split(read(`ollama ps`, String), '\n')))
    return length(lines) == 1
end

function stop_benchmark_model!()
    PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
end

# This benchmark owns only its own model residency. Do not disturb active sessions.
no_resident_ollama_models() || error(
    "Ollama already has a resident model. Stop it manually before this exclusive benchmark.",
)

parent = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(parent) || error("Expected PhluxAI data directory: $parent")

results = NamedTuple[]
try
    mktempdir(parent; prefix = "ollama-api-overhead-") do root
        @testset "PhluxAI versus direct Ollama streaming" begin
            for pass in 1:REPETITIONS
                println("\n----- Pass $pass / $REPETITIONS -----")
                targets = isodd(pass) ? TARGETS : reverse(TARGETS)
                for target in targets
                    prompt = benchmark_prompt(target, pass * 100 + target)
                    paths = isodd(pass) ? (:direct, :phluxai) : (:phluxai, :direct)
                    pair = NamedTuple[]
                    for path in paths
                        label = "pass-$pass-target-$target-path-$path"
                        # Preload is excluded from request timings. Unloading between
                        # arms prevents the second path from reusing the first path's KV cache.
                        PhluxAI.OllamaClient.start_model!(MODEL, URL; timeout = 120)
                        measurement = try
                            path == :direct ? run_direct(prompt) : run_session(root, label, prompt)
                        finally
                            stop_benchmark_model!()
                        end
                        push!(pair, measurement)
                        push!(results, merge(measurement, (target = target, pass = pass)))

                        @test measurement.done
                        @test !isempty(strip(measurement.text))
                        @test measurement.server_ns > 0
                        @test measurement.prompt_tokens > 0
                        @test measurement.output_tokens > 0
                        @test !isnothing(measurement.ttft_ns)
                        @test measurement.wall_ns > 0

                        overhead = sec(measurement.wall_ns - measurement.server_ns)
                        println("  ", rpad(string(path), 8), " ~", target,
                                " | actual input ", measurement.prompt_tokens,
                                " | output ", measurement.output_tokens,
                                " | reason ", measurement.reason)
                        println("    wall ", rounded(sec(measurement.wall_ns)),
                                " s | server ", rounded(sec(measurement.server_ns)),
                                " s | wall-server ", rounded(overhead), " s")
                        println("    prompt eval ", rounded(sec(measurement.prompt_ns)),
                                " s | generation ", rounded(sec(measurement.output_ns)),
                                " s | TTFT ", rounded(sec(measurement.ttft_ns)), " s")
                        println("    model load ",
                                rounded(sec(measurement.load_ns)),
                                " s | other server time ",
                                rounded(sec(
                                    measurement.server_ns -
                                    measurement.load_ns -
                                    measurement.prompt_ns -
                                    measurement.output_ns
                                )),
                                " s")
                        if path == :phluxai
                            println("    client ", rounded(sec(measurement.client_ns)),
                                    " s | session-only overhead ",
                                    rounded(sec(measurement.wall_ns - measurement.client_ns)),
                                    " s | session setup ", rounded(sec(measurement.setup_ns)), " s (separate)")
                        end
                    end
                    @test length(pair) == 2
                    @test pair[1].prompt_tokens == pair[2].prompt_tokens
                    if pair[1].prompt_tokens != pair[2].prompt_tokens
                        @warn "Input token counts differ: wire parameters may not match" target pass
                    end
                end
            end

            println("\n========== Median by input size and path ==========")
            println("Input size | Path    | Prompt tokens | Prompt eval | TTFT | Wall | Wall - Ollama server")
            for target in TARGETS, path in (:direct, :phluxai)
                rows = filter(r -> r.target == target && r.path == path, results)
                println(lpad(string(target), 10), " | ", rpad(string(path), 7), " | ",
                        lpad(string(rounded(median_value(r.prompt_tokens for r in rows))), 13), " | ",
                        rounded(sec(median_value(r.prompt_ns for r in rows))), " s | ",
                        rounded(sec(median_value(r.ttft_ns for r in rows))), " s | ",
                        rounded(sec(median_value(r.wall_ns for r in rows))), " s | ",
                        rounded(sec(median_value(r.wall_ns - r.server_ns for r in rows))), " s")
            end
            println("\nResponse content is not scored for correctness. 'length' means the 512-token API budget was reached.")
        end
    end
finally
    # Do not mask an earlier benchmark failure with a cleanup failure.
    try
        stop_benchmark_model!()
    catch error
        @warn "Could not unload benchmark Qwen model" exception = (error, catch_backtrace())
    end
end

println("\nTemporary benchmark sessions and artifacts cleaned up.")
