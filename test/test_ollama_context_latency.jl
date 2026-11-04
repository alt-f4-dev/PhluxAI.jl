using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"

const NOTES = (
    "A filesystem sandbox should confine tool-created files to an explicitly assigned workspace. A test must verify that a permitted write succeeds and a write outside the boundary fails. A passing test does not prove that the entire Julia process has been sandboxed.",
    "An agent that communicates with a local inference server should distinguish binding to the loopback interface from blocking outbound network traffic. These are different security properties. The verification procedure must test the actual network boundary instead of inferring it from a URL.",
    "An image-generation integration test produces an image in the server output directory and a downloaded artifact in the client store. Both copies must be confined to per-run directories. Ordinary failures should trigger scoped cleanup without removing unrelated images.",
    "The ComfyUI execution benchmark must distinguish new sampling work from repeated workflows that reuse cached intermediate results. Changing the random seed creates a more meaningful uncached sample. Client polling time remains part of the end-to-end measurement.",
    "An Ollama text-generation benchmark should report model loading, prompt evaluation, token generation, and the time until the first streamed token. A short total response time does not necessarily mean the model has a high token-generation rate.",
    "The text model and image model may compete for the same graphics memory. A model-lifecycle policy should account for the cost of unloading and reloading models, not only steady-state inference speed. Changes to this policy require measurements from both workloads.",
    "A growing conversation contains prior user messages and assistant replies. Keeping one session open is necessary to measure the effect of this history. Opening a fresh session for every request would measure independent requests instead of context growth.",
    "Prompt evaluation timing may be affected by a model server's reuse of cached prompt prefixes. Report the estimated session context and the server's prompt-token metrics separately. Avoid treating cached prompt evaluation as a full recomputation of the entire conversation.",
    "Benchmark sessions may write small conversation logs and temporary artifacts even when no image is produced. Isolate these files under a dedicated test directory and delete it when the test ends. Do not write benchmark output files unless they are explicitly needed.",
    "A performance optimization should be proposed only after the experiment identifies a reproducible bottleneck. Preserve the existing public API and working tests. Record cold and warm measurements separately, and never report a single cached run as representative inference latency.",
)

const BENCHMARK_CONTEXT = """
This is a synthetic local software-engineering review. Each note describes a bounded test or an architectural constraint, not a request to change production code. Treat the information as context for this conversation. Preserve the distinction between a measured result, an inference, and an unverified assumption. The response should remain short so that changes in conversation length, rather than changes in requested output length, are the main variable under examination. No external tools or network services are needed for the answer.
"""

seconds(ns) = round(ns / 1e9; digits = 3)

benchmark_root = joinpath(homedir(), ".local", "share", "PhluxAI")
isdir(benchmark_root) || error("PhluxAI data directory does not exist: $benchmark_root")

try
    mktempdir(benchmark_root; prefix = "ollama-context-benchmark-") do root
        @testset "PhluxAI Ollama growing-context latency" begin
            session = ModelSession(
                MODEL;
                base_url = URL,
                system_prompt = "You are a concise software engineering assistant. Reply to each new note with one short sentence.",
                cfg = BudgetConfig(
                    num_ctx = 4096,
                    num_thread = 8,
                    max_tokens = 64,
                    system_budget = 256,
                    summary_budget = 128,
                    response_budget = 64,
                    safety_margin = 128,
                    compression_frac = 1.0,
                ),
                session_dir = joinpath(root, "session"),
                artifact_dir = joinpath(root, "artifacts"),
                sandbox_dir = joinpath(root, "sandbox"),
                tool_mode = TOOLS_OFF,
                start_model = false,
                stop_model_on_close = false,
                discover_capabilities = false,
                keep_alive = "5m",
            )

            wall_times = Float64[]
            previous_context = 0

            try
                for (i, note) in enumerate(NOTES)
                    prompt = "$BENCHMARK_CONTEXT\nObservation $i: $note\nRespond with one short sentence about this observation."
                    estimated_context = PhluxAI.OllamaClient.approx_tokens(
                        PhluxAI.OllamaClient.assemble_messages(session.session),
                    ) + PhluxAI.OllamaClient.approx_tokens(prompt) + 4

                    if i > 1
                        @test estimated_context > previous_context
                    end
                    previous_context = estimated_context

                    started = time_ns()
                    response = respond!(
                        session, prompt;
                        temperature = 0.0,
                        max_tokens = 64,
                        stream = true,
                        print_tokens = false,
                        tool_mode = TOOLS_OFF,
                        request_timeout = 180.0,
                        read_idle_timeout = 180.0,
                    )
                    wall = seconds(time_ns() - started)
                    push!(wall_times, wall)
                    metrics = response.metrics

                    @test response.done
                    @test !isempty(strip(response.text))
                    @test metrics.eval_count > 0
                    @test !isnothing(metrics.time_to_first_token_ns)

                    ttft = isnothing(metrics.time_to_first_token_ns) ? NaN : seconds(metrics.time_to_first_token_ns)
                    println("\nTurn $i / $(length(NOTES)) — estimated context: $estimated_context tokens")
                    println("  Wall / client / Ollama total: ", wall, " / ",
                            seconds(metrics.client_total_duration_ns), " / ", seconds(metrics.total_duration_ns), " s")
                    println("  Time to first token: ", ttft, " s")
                    println("  Prompt tokens / evaluation: ", metrics.prompt_eval_count,
                            " / ", seconds(metrics.prompt_eval_duration_ns), " s")
                    println("  Output tokens / generation: ", metrics.eval_count, " / ", seconds(metrics.eval_duration_ns), " s")
                    println("  Generation throughput: ", round(generation_rate(metrics); digits = 2), " tokens/s")
                end

                warm = wall_times[2:end]
                println("\nMean wall time over turns 2–$(length(NOTES)) (growing context): ",
                        round(sum(warm) / length(warm); digits = 3), " s")
            finally
                close(session)
            end
        end
    end
finally
    try
        PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
    catch error
        @warn "Ollama model cleanup failed" exception = (error, catch_backtrace())
    end
end

println("\nTemporary benchmark files cleaned up.")
