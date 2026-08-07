# test/test_syntax.jl
#
# Network-free tests for the text-based PhluxAI API.
#
# Deliberately excluded from this file:
#   - artifact management
#   - vision understanding
#   - image generation
#   - tool-call execution
#   - sandbox execution
#
# The tests cover:
#   - module loading and text-oriented exports
#   - token-budget validation and one-shot @budget behavior
#   - model, agent, and resume macro expansion
#   - thinking normalization and response metadata
#   - text-only ModelInput, ChatMessage, and ModelResponse behavior
#   - low-level text request options and token counting
#   - network-free ModelSession and PhysicsAssistant configuration
#   - interactive text commands that do not call Ollama

using Test

ENV["PHLUXAI_BANNER"] = "0"

include("../src/PhluxAI.jl")
using .PhluxAI

const OC = PhluxAI.OllamaClient
const MS = PhluxAI.ModelSessions
const PA = PhluxAI.PhysicalAssistant
const PT = PhluxAI.ProtocolTypes

function caught_exception(
    callable::F,
)::Union{Exception,Nothing} where {F<:Function}
    try
        callable()
    catch error
        return error
    end
    return nothing
end

function captured_stdout(
    callable::F,
)::Tuple{Any,String} where {F<:Function}
    path, stream = mktemp()

    try
        result = redirect_stdout(stream) do
            callable()
        end

        flush(stream)
        seekstart(stream)
        return result, read(stream, String)
    finally
        close(stream)
        rm(path; force = true)
    end
end

@testset "PhluxAI text API" begin
    @testset "Module identity and exports" begin
        @test PHLUXAI_VERSION == v"0.2.0"

        for macro_name in (
            Symbol("@model"),
            Symbol("@agent"),
            Symbol("@resume"),
            Symbol("@budget"),
        )
            @test isdefined(PhluxAI, macro_name)
        end

        for name in (
            :OllamaModel,
            :BudgetConfig,
            :ModelSession,
            :PhysicsAssistant,
            :PhysicalAgent,
            :AssistantMode,
            :GENERAL,
            :DERIVE,
            :CODE,
            :DRAFT,
            :ModelInput,
            :TextPart,
            :ChatMessage,
            :ModelResponse,
            :ResponseMetrics,
            :ThinkingLevel,
            :THINK_LOW,
            :THINK_MEDIUM,
            :THINK_HIGH,
            :THINK_MAX,
            :respond!,
            :ask!,
            :save,
            :repl,
            :load_model_session,
            :load_assistant,
            :load_physical_agent,
            :compression_running,
            :await_compression!,
            :last_response,
            :last_response_metrics,
            :set_temperature!,
            :default_temperature,
            :set_thinking!,
            :default_thinking,
            :set_reasoning_visible!,
            :set_mode!,
            :default_mode,
            :set_keep_alive!,
            :set_timeouts!,
            :default_keep_alive,
            :model_name,
            :base_url,
            :session_directory,
            :system_prompt,
            :effective_response_budget,
            :history_budget,
            :compression_trigger,
            :generation_rate,
            :prompt_rate,
            :has_reasoning,
        )
            @test isdefined(PhluxAI, name)
        end
    end

    @testset "Banner output" begin
        buffer = IOBuffer()
        @test welcome(buffer; check_ollama = false) === nothing
        output = String(take!(buffer))
        @test occursin("PhluxAI v0.2.0", output)
        @test occursin("@model", output)
        @test occursin("@agent :physics", output)
    end

    @testset "BudgetConfig defaults" begin
        cfg = BudgetConfig()

        @test cfg.num_ctx == 4096
        @test cfg.system_budget == 600
        @test cfg.summary_budget == 600
        @test cfg.response_budget == 400
        @test cfg.safety_margin == 200
        @test cfg.compression_frac == 0.6
        @test cfg.m_verbatim == 6
        @test cfg.exact_threshold == 200
        @test cfg.num_thread == 0
        @test cfg.max_tokens == 512
    end

    @testset "Budget arithmetic" begin
        cfg = BudgetConfig(
            num_ctx = 16384,
            num_thread = 8,
        )

        expected_response = max(
            cfg.response_budget,
            cfg.max_tokens,
        )
        expected_history = cfg.num_ctx -
                           cfg.system_budget -
                           cfg.summary_budget -
                           expected_response -
                           cfg.safety_margin

        @test effective_response_budget(cfg) == 512
        @test history_budget(cfg) == 14472
        @test history_budget(cfg) == expected_history
        @test compression_trigger(cfg) ==
              round(Int, cfg.compression_frac * expected_history)

        response_dominant = BudgetConfig(
            num_ctx = 8192,
            system_budget = 300,
            summary_budget = 300,
            response_budget = 900,
            safety_margin = 100,
            compression_frac = 0.7,
            max_tokens = 512,
        )
        @test effective_response_budget(response_dominant) == 900
        @test history_budget(response_dominant) ==
              8192 - 300 - 300 - 900 - 100

        generation_dominant = BudgetConfig(
            num_ctx = 8192,
            system_budget = 300,
            summary_budget = 300,
            response_budget = 200,
            safety_margin = 100,
            compression_frac = 0.7,
            max_tokens = 512,
        )
        @test effective_response_budget(generation_dominant) == 512
        @test history_budget(generation_dominant) ==
              8192 - 300 - 300 - 512 - 100
        @test compression_trigger(generation_dominant) ==
              round(
                  Int,
                  0.7 * history_budget(generation_dominant),
              )
    end

    @testset "BudgetConfig validation" begin
        @test_throws ArgumentError BudgetConfig(num_ctx = 0)
        @test_throws ArgumentError BudgetConfig(summary_budget = 0)
        @test_throws ArgumentError BudgetConfig(
            compression_frac = 0.0,
        )
        @test_throws ArgumentError BudgetConfig(
            compression_frac = 1.1,
        )
        @test_throws ArgumentError BudgetConfig(m_verbatim = 3)
        @test_throws ArgumentError BudgetConfig(num_thread = -1)
        @test_throws ArgumentError BudgetConfig(max_tokens = 0)
        @test_throws ArgumentError BudgetConfig(
            num_ctx = 1900,
            system_budget = 600,
            summary_budget = 600,
            response_budget = 400,
            safety_margin = 200,
            max_tokens = 512,
        )
    end

    @testset "One-shot @budget behavior" begin
        # Consume any state left by an earlier workload and restore defaults.
        PhluxAI._take_budget!()

        @test PhluxAI._BUDGET[].num_ctx == 16384
        @test PhluxAI._BUDGET[].num_thread == 8
        @test !PhluxAI._BUDGET_EXPLICIT[]

        @budget num_ctx=32768 num_thread=4 max_tokens=1024

        @test PhluxAI._BUDGET[].num_ctx == 32768
        @test PhluxAI._BUDGET[].num_thread == 4
        @test PhluxAI._BUDGET[].max_tokens == 1024
        @test PhluxAI._BUDGET[].m_verbatim == 6
        @test PhluxAI._BUDGET_EXPLICIT[]

        consumed, explicit = PhluxAI._take_budget!()

        @test explicit
        @test consumed.num_ctx == 32768
        @test consumed.num_thread == 4
        @test consumed.max_tokens == 1024

        reset, reset_explicit = PhluxAI._take_budget!()

        @test !reset_explicit
        @test reset.num_ctx == 16384
        @test reset.num_thread == 8
        @test reset.max_tokens == 512
        @test !PhluxAI._BUDGET_EXPLICIT[]
    end

    @testset "@budget rejects invalid syntax" begin
        @test_throws Exception macroexpand(
            @__MODULE__,
            :(@budget),
        )
        @test_throws Exception macroexpand(
            @__MODULE__,
            :(@budget 12345),
        )
        @test_throws Exception macroexpand(
            @__MODULE__,
            :(@budget unknown_field = 1),
        )
    end

    @testset "Interactive macro expansion is network-free" begin
        model_expr = macroexpand(
            @__MODULE__,
            :(@model "gpt-oss:20b"),
        )
        model_text = sprint(show, model_expr)

        @test model_expr isa Expr
        @test occursin("_take_budget!", model_text)
        @test occursin("_build_model_session", model_text)
        @test occursin("repl", model_text)

        agent_expr = macroexpand(
            @__MODULE__,
            :(@agent :physics "gpt-oss:20b" "sessions/test"),
        )
        agent_text = sprint(show, agent_expr)

        @test agent_expr isa Expr
        @test occursin("_build_agent", agent_text)
        @test occursin("Val", agent_text)
        @test occursin("repl", agent_text)

        resume_expr = macroexpand(
            @__MODULE__,
            :(@resume "gpt-oss:20b" "sessions/test"),
        )
        resume_text = sprint(show, resume_expr)

        @test resume_expr isa Expr
        @test occursin("_resume_session", resume_text)
        @test occursin("_take_budget!", resume_text)
        @test occursin("repl", resume_text)
    end

    @testset "Agent profile dispatch" begin
        cfg = BudgetConfig(
            num_ctx = 16384,
            num_thread = 8,
        )

        @test hasmethod(
            PhluxAI._build_agent,
            Tuple{
                Val{:physics},
                String,
                BudgetConfig,
                Nothing,
            },
        )

        for profile in (:coding, :math, :general)
            error = caught_exception() do
                PhluxAI._build_agent(
                    Val(profile),
                    "gpt-oss:20b",
                    cfg,
                    nothing,
                )
            end

            @test error isa ArgumentError
            message = if error isa Exception
                lowercase(sprint(showerror, error))
            else
                ""
            end
            @test occursin(":$profile", message)
            @test occursin("available profiles: :physics", message)
        end
    end

    @testset "Assistant modes" begin
        @test GENERAL isa AssistantMode
        @test DERIVE isa AssistantMode
        @test CODE isa AssistantMode
        @test DRAFT isa AssistantMode
        @test length(Set((GENERAL, DERIVE, CODE, DRAFT))) == 4

        @test PA._mode_temperature(GENERAL) == 0.7
        @test PA._mode_temperature(DERIVE) == 0.15
        @test PA._mode_temperature(CODE) == 0.1
        @test PA._mode_temperature(DRAFT) == 0.5

        @test PA._mode_prefix(GENERAL) == ""
        @test startswith(PA._mode_prefix(DERIVE), "[DERIVE]")
        @test startswith(PA._mode_prefix(CODE), "[CODE]")
        @test startswith(PA._mode_prefix(DRAFT), "[DRAFT]")

        mode, prompt = PA._parse_mode_sigil(
            "\\derive Derive the limiting form.",
            GENERAL,
        )
        @test mode == DERIVE
        @test prompt == "Derive the limiting form."

        mode, prompt = PA._parse_mode_sigil(
            "Continue normally.",
            CODE,
        )
        @test mode == CODE
        @test prompt == "Continue normally."

        @test_throws ArgumentError PA._parse_mode_sigil(
            "\\code",
            GENERAL,
        )
    end

    @testset "Thinking normalization" begin
        @test PT.normalize_thinking(nothing) === nothing
        @test PT.normalize_thinking(false) === false
        @test PT.normalize_thinking(true) === true
        @test PT.normalize_thinking(:low) == THINK_LOW
        @test PT.normalize_thinking(" medium ") == THINK_MEDIUM
        @test PT.normalize_thinking("HIGH") == THINK_HIGH
        @test PT.normalize_thinking(:max) == THINK_MAX
        @test PT.normalize_thinking("none") === false

        @test PT.thinking_wire_value(nothing) === nothing
        @test PT.thinking_wire_value(true) === true
        @test PT.thinking_wire_value(THINK_LOW) == "low"
        @test PT.thinking_wire_value(THINK_MEDIUM) == "medium"
        @test PT.thinking_wire_value(THINK_HIGH) == "high"
        @test PT.thinking_wire_value(THINK_MAX) == "max"

        @test PT.thinking_label(nothing) == "default"
        @test PT.thinking_label(false) == "off"
        @test PT.thinking_label(true) == "on"
        @test PT.thinking_label(THINK_HIGH) == "high"

        @test_throws ArgumentError PT.normalize_thinking("extreme")
    end

    @testset "Text ModelInput and message roles" begin
        input = ModelInput("Explain the response.")

        @test length(input.parts) == 1
        @test input.parts[1] isa TextPart
        @test PT.text_content(input) == "Explain the response."
        @test_throws ArgumentError ModelInput("")
        @test_throws ArgumentError TextPart("   ")

        @test PT.role_name(PT.ROLE_SYSTEM) == "system"
        @test PT.role_name(PT.ROLE_USER) == "user"
        @test PT.role_name(PT.ROLE_ASSISTANT) == "assistant"
        @test PT.parse_role("SYSTEM") == PT.ROLE_SYSTEM
        @test PT.parse_role("user") == PT.ROLE_USER
        @test PT.parse_role("assistant") == PT.ROLE_ASSISTANT
        @test_throws ArgumentError PT.parse_role("invalid")

        message = ChatMessage(
            PT.ROLE_USER,
            "Explain the response.",
        )
        @test message.role == PT.ROLE_USER
        @test message.content == "Explain the response."
        @test isnothing(message.reasoning)
    end

    @testset "Response metrics and reasoning fields" begin
        metrics = ResponseMetrics(
            total_duration_ns = 2_000_000_000,
            prompt_eval_count = 100,
            prompt_eval_duration_ns = 500_000_000,
            eval_count = 40,
            eval_duration_ns = 2_000_000_000,
            client_total_duration_ns = 2_100_000_000,
            time_to_first_token_ns = 100_000_000,
        )

        @test prompt_rate(metrics) == 200.0
        @test generation_rate(metrics) == 20.0
        @test prompt_rate(ResponseMetrics()) == 0.0
        @test generation_rate(ResponseMetrics()) == 0.0

        response = ModelResponse(
            "Final answer.";
            reasoning = "Internal reasoning trace.",
            done = true,
            done_reason = "stop",
            metrics,
        )

        @test response.text == "Final answer."
        @test response.reasoning == "Internal reasoning trace."
        @test response.done
        @test response.done_reason == "stop"
        @test response.metrics === metrics
        @test has_reasoning(response)

        plain = ModelResponse("Final answer.")
        @test isnothing(plain.reasoning)
        @test !has_reasoning(plain)
    end

    @testset "Text message storage compatibility" begin
        original = ChatMessage(
            PT.ROLE_ASSISTANT,
            "Final answer.";
            reasoning = "Reasoning trace.",
        )

        stored = OC._message_storage(
            original;
            include_reasoning = true,
        )
        @test stored["role"] == "assistant"
        @test stored["content"] == "Final answer."
        @test stored["reasoning"] == "Reasoning trace."

        restored = OC._message_from_storage(stored)
        @test restored.role == PT.ROLE_ASSISTANT
        @test restored.content == "Final answer."
        @test restored.reasoning == "Reasoning trace."

        redacted = OC._message_storage(
            original;
            include_reasoning = false,
        )
        @test isnothing(redacted["reasoning"])
    end

    @testset "OllamaModel construction" begin
        model = OllamaModel("gpt-oss:20b")

        @test model.name == "gpt-oss:20b"
        @test model.base_url == "http://localhost:11434"

        normalized = OllamaModel(
            "  gpt-oss:20b  ",
            "http://127.0.0.1:11434/",
        )
        @test normalized.name == "gpt-oss:20b"
        @test normalized.base_url ==
              "http://127.0.0.1:11434"

        @test_throws ArgumentError OllamaModel("")
        @test_throws ArgumentError OllamaModel(
            "gpt-oss:20b",
            "   ",
        )
    end

    @testset "Text token counting" begin
        @test OC.approx_tokens("") == 1
        @test OC.approx_tokens("abcd") == 1
        @test OC.approx_tokens("abcde") == 2
        @test OC.approx_tokens("a" ^ 100) == 25

        user = ChatMessage(PT.ROLE_USER, "abcd")
        assistant = ChatMessage(
            PT.ROLE_ASSISTANT,
            "abcde",
        )

        @test OC.approx_tokens(user) == 5
        @test OC.approx_tokens(assistant) == 6
        @test OC.approx_tokens([user, assistant]) == 11

        legacy = Dict(
            "role" => "user",
            "content" => "abcd",
        )
        @test OC.approx_tokens(legacy) == 5
        @test OC.approx_tokens([legacy]) == 5
    end

    @testset "Text request options" begin
        options = OC._build_options(
            0.25,
            512;
            num_ctx = 8192,
            num_thread = 8,
        )

        @test options["temperature"] == 0.25
        @test options["num_predict"] == 512
        @test options["num_ctx"] == 8192
        @test options["num_thread"] == 8

        minimal = OC._build_options(
            0.7,
            128;
            num_ctx = 0,
            num_thread = 0,
        )
        @test !haskey(minimal, "num_ctx")
        @test !haskey(minimal, "num_thread")

        payload = Dict{String,Any}()
        @test OC._set_thinking!(payload, :high) === nothing
        @test payload["think"] == "high"

        @test OC._set_keep_alive!(payload, "10m") === nothing
        @test payload["keep_alive"] == "10m"

        @test OC._timeout_keywords(0.0, 0.0) == (;)
        @test OC._timeout_keywords(5.0, 0.0) ==
              (request_timeout = 5.0,)
        @test OC._timeout_keywords(0.0, 30.0) ==
              (read_idle_timeout = 30.0,)
        @test OC._timeout_keywords(5.0, 30.0) == (
            request_timeout = 5.0,
            read_idle_timeout = 30.0,
        )

        @test_throws ArgumentError OC._build_options(
            -0.1,
            128,
        )
        @test_throws ArgumentError OC._build_options(
            0.7,
            0,
        )
        @test_throws ArgumentError OC._timeout_keywords(
            -1.0,
            0.0,
        )
    end

    @testset "Network-free ModelSession text configuration" begin
        mktempdir(prefix = "phluxai-text-test-") do root
            cfg = BudgetConfig(
                num_ctx = 8192,
                num_thread = 2,
                max_tokens = 256,
            )
            session = ModelSession(
                "phluxai-text-test";
                cfg,
                system_prompt = "Text-only system prompt.",
                artifact_dir = joinpath(root, "artifacts"),
                sandbox_dir = joinpath(root, "sandbox"),
                temperature = 0.4,
                thinking = :medium,
                show_reasoning = false,
                keep_alive = "10m",
                request_timeout = 5.0,
                read_idle_timeout = 30.0,
                discover_capabilities = false,
                start_model = false,
                stop_model_on_close = false,
            )

            try
                @test isopen(session)
                @test model_name(session) == "phluxai-text-test"
                @test base_url(session) ==
                      "http://localhost:11434"
                @test session_directory(session) === nothing
                @test system_prompt(session) ==
                      "Text-only system prompt."
                @test default_temperature(session) == 0.4
                @test default_thinking(session) == THINK_MEDIUM
                @test default_keep_alive(session) == "10m"
                @test last_response(session) === nothing
                @test last_response_metrics(session) === nothing
                @test !compression_running(session)
                @test await_compression!(session) === nothing

                @test applicable(
                    PhluxAI.respond!,
                    session,
                    "Test prompt.",
                )
                @test applicable(
                    PhluxAI.ask!,
                    session,
                    "Test prompt.",
                )

                set_temperature!(session, 0.2)
                @test default_temperature(session) == 0.2

                set_thinking!(session, :high)
                @test default_thinking(session) == THINK_HIGH

                set_reasoning_visible!(session, true)
                @test session.show_reasoning

                set_keep_alive!(session, -1)
                @test default_keep_alive(session) == -1

                set_timeouts!(
                    session;
                    request_timeout = 12.0,
                    read_idle_timeout = 45.0,
                )
                @test session.request_timeout == 12.0
                @test session.read_idle_timeout == 45.0

                handled, output = captured_stdout() do
                    MS._handle_common_command!(
                        session,
                        "\\temp 0.3",
                    )
                end
                @test handled
                @test default_temperature(session) == 0.3
                @test occursin(
                    "Temperature set to 0.3",
                    output,
                )

                handled, output = captured_stdout() do
                    MS._handle_common_command!(
                        session,
                        "\\think low",
                    )
                end
                @test handled
                @test default_thinking(session) == THINK_LOW
                @test occursin("Thinking set to low", output)

                handled, output = captured_stdout() do
                    MS._handle_common_command!(
                        session,
                        "\\reasoning hide",
                    )
                end
                @test handled
                @test !session.show_reasoning
                @test occursin(
                    "Reasoning display: hide",
                    output,
                )

                @test !MS._handle_common_command!(
                    session,
                    "\\unknown",
                )
                @test MS._is_exit_command("\\exit")
                @test MS._is_exit_command("  \\QUIT  ")
                @test MS._is_exit_command("\\bye")
                @test !MS._is_exit_command("continue")

                assembled = OC.assemble_messages(
                    session.session,
                )
                @test length(assembled) == 1
                @test assembled[1].role == PT.ROLE_SYSTEM
                @test assembled[1].content ==
                      "Text-only system prompt."

                push!(
                    session.session.messages,
                    ChatMessage(
                        PT.ROLE_USER,
                        "Question.",
                    ),
                )
                push!(
                    session.session.messages,
                    ChatMessage(
                        PT.ROLE_ASSISTANT,
                        "Answer.",
                    ),
                )

                assembled = OC.assemble_messages(
                    session.session,
                )
                @test length(assembled) == 3
                @test assembled[2].content == "Question."
                @test assembled[3].content == "Answer."

                metrics = ResponseMetrics(eval_count = 7)
                response = ModelResponse(
                    "Answer.";
                    reasoning = "Trace.",
                    metrics,
                )
                session.session.last_response = response

                @test last_response(session) === response
                @test last_response_metrics(session) === metrics
                @test save(session) === nothing
            finally
                close(session)
            end

            @test !isopen(session)
            @test_throws ArgumentError set_temperature!(
                session,
                0.5,
            )
        end
    end

    @testset "Network-free PhysicsAssistant text configuration" begin
        mktempdir(prefix = "phluxai-physics-test-") do root
            assistant = PhysicsAssistant(
                "phluxai-physics-test";
                mode = DERIVE,
                artifact_dir = joinpath(root, "artifacts"),
                sandbox_dir = joinpath(root, "sandbox"),
                discover_capabilities = false,
                start_model = false,
                stop_model_on_close = false,
            )

            try
                @test isopen(assistant)
                @test assistant isa PhysicsAssistant
                @test assistant isa PhysicalAgent
                @test default_mode(assistant) == DERIVE
                @test default_temperature(assistant) == 0.15
                @test model_name(assistant) ==
                      "phluxai-physics-test"
                @test system_prompt(assistant) ==
                      PA._SYSTEM_PROMPT

                @test applicable(
                    PhluxAI.respond!,
                    assistant,
                    "Test prompt.",
                )
                @test applicable(
                    PhluxAI.ask!,
                    assistant,
                    "Test prompt.",
                )

                set_mode!(assistant, CODE)
                @test default_mode(assistant) == CODE
                @test default_temperature(assistant) == 0.1

                set_temperature!(assistant, 0.05)
                @test default_temperature(assistant) == 0.05

                set_mode!(assistant, DRAFT)
                @test default_mode(assistant) == DRAFT
                @test default_temperature(assistant) == 0.05

                set_temperature!(assistant, nothing)
                @test default_temperature(assistant) == 0.5

                set_thinking!(assistant, :high)
                @test default_thinking(assistant) == THINK_HIGH

                input = PA._mode_input(
                    "Derive the result.",
                    DERIVE,
                )
                @test startswith(
                    PT.text_content(input),
                    "[DERIVE]",
                )
                @test occursin(
                    "Derive the result.",
                    PT.text_content(input),
                )

                handled, output = captured_stdout() do
                    PA._handle_temperature_command!(
                        assistant,
                        "\\temp 0.08",
                    )
                end
                @test handled
                @test default_temperature(assistant) == 0.08
                @test occursin(
                    "Temperature override set to 0.08",
                    output,
                )

                handled, output = captured_stdout() do
                    PA._handle_temperature_command!(
                        assistant,
                        "\\temp auto",
                    )
                end
                @test handled
                @test default_temperature(assistant) == 0.5
                @test occursin(
                    "Temperature reset to mode default",
                    output,
                )
            finally
                close(assistant)
            end

            @test !isopen(assistant)
        end
    end
end
