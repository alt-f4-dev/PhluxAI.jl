# Test inventory for PhluxAI.jl.
# Every test_*.jl file must occur exactly once. The runner checks this at startup.
# :cpu includes pure tests and local mock-HTTP integration without live model inference.
# :sandbox requires a working bubblewrap / namespace-capable host.
# :live requires explicitly installed and running local model services and weights.
# :benchmark runs device- and model-dependent performance experiments.

const SUITE_TIERS = (:cpu, :sandbox, :live, :benchmark)
const SUITE_MANIFEST = (
    (file = "test_comfy_http.jl", tier = :cpu),
    (file = "test_comfy_interactive_api.jl", tier = :cpu),
    (file = "test_comfy_latency.jl", tier = :benchmark),
    (file = "test_comfy_lifecycle.jl", tier = :cpu),
    (file = "test_comfy_live.jl", tier = :live),
    (file = "test_comfy_memory_policy.jl", tier = :benchmark),
    (file = "test_comfy_memory_policy_settling.jl", tier = :benchmark),
    (file = "test_comfy_registry.jl", tier = :cpu),
    (file = "test_comfy_release_mock.jl", tier = :cpu),
    (file = "test_image_model_api.jl", tier = :cpu),
    (file = "test_image_repl_continuity.jl", tier = :cpu),
    (file = "test_inference_configuration.jl", tier = :benchmark),
    (file = "test_model_contracts.jl", tier = :cpu),
    (file = "test_model_session_initial_state.jl", tier = :cpu),
    (file = "test_multimodal_switch_latency.jl", tier = :benchmark),
    (file = "test_ollama_api_overhead.jl", tier = :benchmark),
    (file = "test_ollama_compression_invariants.jl", tier = :cpu),
    (file = "test_ollama_compression_policy.jl", tier = :benchmark),
    (file = "test_ollama_compression_scheduling.jl", tier = :cpu),
    (file = "test_ollama_compression_transactional.jl", tier = :cpu),
    (file = "test_ollama_context_efficiency.jl", tier = :benchmark),
    (file = "test_ollama_context_latency.jl", tier = :benchmark),
    (file = "test_ollama_latency.jl", tier = :benchmark),
    (file = "test_ollama_prompt_scaling.jl", tier = :benchmark),
    (file = "test_phase3_final_integration.jl", tier = :live),
    (file = "test_research_contracts.jl", tier = :cpu),
    (file = "test_quiet_model_switching.jl", tier = :cpu),
    (file = "test_sandbox_layout.jl", tier = :sandbox),
    (file = "test_syntax.jl", tier = :cpu),
)
