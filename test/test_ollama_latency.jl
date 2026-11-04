using Test

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

const MODEL = "qwen2.5-coder:latest"
const URL = "http://127.0.0.1:11434"

seconds(ns) = round(ns / 1e9; digits = 3)

benchmark_root = joinpath(homedir(), ".local", "share", "PhluxAI")

isdir(benchmark_root) || error("PhluxAI data directory does not exist")

try
    mktempdir(benchmark_root; prefix = "ollama-benchmark-") do root
        @testset "PhluxAI Ollama agent latency" begin
            prompts = ("Explain Julia's multiple dispatch in one sentence.",
                       "Explain Julia's parametric types in one sentence.",
                       "Explain Julia's type stability in one sentence.",)
            for i in 1:3
                session = ModelSession(MODEL; 
                                       base_url = URL,
                                       cfg = BudgetConfig(num_ctx = 4096, num_thread = 8, max_tokens = 64),
                                       session_dir = joinpath(root, "session-$i"),
                                       artifact_dir = joinpath(root, "artifacts-$i"),
                                       sandbox_dir = joinpath(root, "sandbox-$i"),
                                       tool_mode = TOOLS_OFF,
                                       start_model = false,
                                       stop_model_on_close = false,
                                       discover_capabilities = false,
                                       keep_alive = "5m")

                try
                    started = time_ns()

                    response = respond!(session, prompts[i];
                                        temperature = 0.0,
                                        max_tokens = 64,
                                        stream = true,
                                        print_tokens = false,
                                        tool_mode = TOOLS_OFF,
                                        request_timeout = 120.0,
                                        read_idle_timeout = 120.0)

                    wall = time_ns() - started
                    m = response.metrics

                    @test response.done
                    @test !isempty(strip(response.text))
                    @test m.eval_count > 0

                    if i == 1
                        println("\nOllama model residency:")
                        run(`ollama ps`)

                        println("\nGPU memory after model loading:")
                        run(`nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv`)
                    end
                    
                    println("\nRun $i")
                    println("  Wall time: ", seconds(wall), " s")
                    println("  Model load: ", seconds(m.load_duration_ns), " s")
                    println("  Prompt evaluation: ", seconds(m.prompt_eval_duration_ns), " s")
                    println("  Token generation: ", seconds(m.eval_duration_ns), " s")
                    println("  Output tokens: ", m.eval_count)
                    println("  Generation rate: ", round(generation_rate(m); digits = 2), " tokens/s")

                    if !isnothing(m.time_to_first_token_ns)
                        println("  Time to first token: ", seconds(m.time_to_first_token_ns), " s")
                    end

                finally
                    close(session)
                end
            end
        end
    end
finally
    # Restore the initially unloaded model state.
    try
        PhluxAI.OllamaClient.stop_model!(MODEL, URL; request_timeout = 30.0)
    catch error
        @warn "Ollama model cleanup failed" exception = (error, catch_backtrace())
    end
end

println("\nTemporary benchmark files cleaned up.")
