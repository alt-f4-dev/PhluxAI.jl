using Test, JSON3, HTTP, Sockets, SHA
using PhluxAI
const OC = PhluxAI.OllamaClient
const PT = PhluxAI.ProtocolTypes

function events(session)
    output = JournalEvent[]
    foreach_event(event -> push!(output, event), session_journal(session))
    return output
end

@testset "Journal session replay and derived summaries" begin
    mktempdir() do root
        model = OllamaModel("fixture")
        session = OC.ChatSession(model, "System α\nβ"; session_dir = root)
        expected = [ChatMessage(PT.ROLE_USER, "user $i α\nβ") for i in 1:20]
        for message in expected
            OC._log_cold(session, message)
            push!(session.messages, message)
        end
        trace = ChatMessage(PT.ROLE_ASSISTANT, "answer"; reasoning = "received reasoning")
        OC._log_cold(session, trace)
        push!(session.messages, trace)
        OC.finalize_interaction!(session)
        @test session.messages[end].reasoning === nothing
        @test JSON3.read(filter(e -> e.kind == "message", events(session))[end].payload_json).reasoning == "received reasoning"
        before = read(session.journal.path)
        anchor = journal_head(session.journal)
        OC.save_session(session)
        @test read(session.journal.path) == before
        summary = "<assumptions></assumptions><results>condensed</results><notation></notation><deferred></deferred><rejected></rejected><narrative>next</narrative>"
        candidate = OC.GenerateResponse("fixture", summary, nothing, ArtifactRef[], true, "stop", nothing, nothing, ResponseMetrics())
        @test OC._commit_compression_candidate_unlocked!(session, candidate, session.messages[end-1:end], session.revision)
        @test read(session.journal.path)[1:length(before)] == before
        @test length(filter(e -> e.kind == "message", events(session))) == 21
        summary_event = only(filter(e -> e.kind == "summary_created", events(session)))
        @test JSON3.read(summary_event.payload_json).source_sha256 == anchor.sha256
        OC.close_session(session)
        # An absent or stale hot file cannot erase committed history.
        rm(joinpath(root, "hot.json"))
        loaded = OC.load_session(model, root; checkpoint = anchor)
        @test loaded.summary == summary
        @test length(loaded.messages) == 2
        bytes = read(loaded.journal.path)
        OC.save_session(loaded)
        OC.close_session(loaded)
        @test read(joinpath(root, "journal.jsonl")) == bytes
        @test verify_journal(joinpath(root, "journal.jsonl"); checkpoint = anchor).sequence > anchor.sequence
    end
end

@testset "Interactive settings, interrupted calls, and artifact identity" begin
    mktempdir() do root
        session = ModelSession("fixture"; session_dir = root, start_model = false, discover_capabilities = false)
        set_temperature!(session, 0.25)
        set_thinking!(session, :high)
        source_store = SourceStore(joinpath(root, "sources"))
        context = analysis_context(session.sandbox, artifact_store(session), source_store)
        set_tool_context!(session, context)
        set_tool_mode!(session, TOOLS_ASK)
        file = joinpath(root, "source.txt")
        write(file, "artifact bytes")
        artifact = attach!(session, file)
        OC._log_cold(session.session, ChatMessage(PT.ROLE_USER, "artifact"; artifacts = [artifact]))
        @test isempty(verify_session(session).artifact_issues)
        write(PhluxAI.ArtifactStore.resolve_artifact(session.session.artifact_store, artifact), "altered")
        @test only(verify_session(session).artifact_issues).status == :changed
        OC.record_session_event!(session.session, :operation_started, (id = "uncertain-call", operation = "tool", name = "write"))
        @test only(pending_operations(session)).status == :uncertain
        close(session)
        loaded = load_model_session("fixture", root; start_model = false, discover_capabilities = false)
        @test default_temperature(loaded) == 0.25
        @test default_thinking(loaded) == PT.THINK_HIGH
        @test loaded.tool_mode == TOOLS_ASK
        @test only(pending_attachments(loaded)).sha256 == artifact.sha256
        @test length(pending_operations(loaded)) == 1
        set_tool_context!(loaded, analysis_context(loaded.sandbox, artifact_store(loaded), source_store))
        @test_throws ArgumentError PhluxAI.ModelSessions._execute_journaled_tool!(loaded, loaded.tool_context,
                                         ToolCall("uncertain-call", "write", "{}"), TOOLS_ASK, nothing)
        close(loaded)
    end
end

@testset "Model HTTP history and failed streaming" begin
    count = Ref(0)
    server = HTTP.serve!("127.0.0.1", 0; listenany = true) do request
        count[] += 1
        if count[] == 1
            return HTTP.Response(200; body = JSON3.write((message = (content = "answer α", thinking = "trace β"), done = true, done_reason = "stop")))
        end
        return HTTP.Response(200; body = JSON3.write((message = (content = "partial", thinking = "partial trace"), done = false)) * "\n")
    end
    try
        port = HTTP.port(server)
        mktempdir() do root
            session = OC.ChatSession(OllamaModel("fixture", "http://127.0.0.1:$port"), "system"; session_dir = root)
            response = OC.respond!(session, "request"; stream = false)
            @test response.text == "answer α"
            @test_throws Exception OC.respond!(session, "failed request"; stream = true, print_tokens = false)
            history = events(session)
            @test any(e -> e.kind == "message" && JSON3.read(e.payload_json).content == "failed request", history)
            @test any(e -> e.kind == "model_chunk" && occursin("partial trace", JSON3.read(e.payload_json).raw_json), history)
            @test isempty(pending_operations(session))
            @test length(session.messages) == 2
            OC.close_session(session)
        end
    finally
        close(server)
    end
end

@testset "Non-destructive legacy migration" begin
    mktempdir() do root
        legacy = Dict("schema_version" => 1, "session_type" => "model", "model_name" => "fixture", "prompt" => "legacy system",
                      "summary" => nothing, "messages" => [Dict("role" => "user", "content" => "legacy α")],
                      "cfg" => OC._budget_dict(BudgetConfig()), "persist_reasoning" => false)
        hot = JSON3.write(legacy)
        cold = "{\"timestamp\":\"original\",\"message\":{\"role\":\"user\",\"content\":\"legacy α\"}}\n"
        write(joinpath(root, "hot.json"), hot)
        write(joinpath(root, "cold.jsonl"), cold)
        session = OC.load_session(OllamaModel("fixture"), root)
        @test only(session.messages).content == "legacy α"
        @test read(joinpath(root, "legacy-originals", "hot.json"), String) == hot
        @test read(joinpath(root, "legacy-originals", "cold.jsonl"), String) == cold
        record = only(filter(e -> e.kind == "legacy_record", events(session)))
        @test JSON3.read(record.payload_json).raw_json == chomp(cold)
        before = read(session.journal.path)
        OC.close_session(session)
        @test read(joinpath(root, "journal.jsonl")) == before
        resumed = OC.load_session(OllamaModel("fixture"), root)
        @test length(filter(e -> e.kind == "legacy_import", events(resumed))) == 1
        @test length(filter(e -> e.kind == "legacy_record", events(resumed))) == 1
        OC.close_session(resumed)
    end
    mktempdir() do root
        write(joinpath(root, "hot.json"), "{\"schema_version\":999}")
        @test_throws ArgumentError OC.load_session(OllamaModel("fixture"), root)
    end
end

@testset "Interactive tool loop approvals and result replay" begin
    count = Ref(0)
    server = HTTP.serve!("127.0.0.1", 0; listenany = true) do request
        count[] += 1
        if isodd(count[])
            call = Dict("id" => "call-$(count[])", "function" => Dict("name" => "write_text_file",
                        "arguments" => Dict("path" => "output.txt", "content" => "approved bytes")))
            return HTTP.Response(200; body = JSON3.write((message = (role = "assistant", content = "", tool_calls = [call]), done = true)))
        end
        messages = JSON3.read(String(request.body)).messages
        @test any(message -> message.role == "tool", messages)
        return HTTP.Response(200; body = JSON3.write((message = (content = "finished",), done = true, done_reason = "stop")))
    end
    try
        mktempdir() do root
            session = ModelSession("fixture"; session_dir = root, base_url = "http://127.0.0.1:$(HTTP.port(server))",
                                   start_model = false, discover_capabilities = false)
            context = ToolContext(session.sandbox, artifact_store(session); jupy_executable = "unused", jupip_executable = "unused")
            set_tool_context!(session, context)
            set_tool_mode!(session, TOOLS_ASK)
            allowed = respond!(session, "write"; stream = false, approval_callback = (call, risk) -> true)
            @test only(allowed.tool_executions).success
            @test read(joinpath(session.sandbox.workspace, "output.txt"), String) == "approved bytes"
            denied = respond!(session, "deny"; stream = false, approval_callback = (call, risk) -> false)
            @test !only(denied.tool_executions).success
            decisions = filter(e -> e.kind == "approval_decision", events(session))
            @test [JSON3.read(e.payload_json).approved for e in decisions] == [true, false]
            @test isempty(pending_operations(session))
            close(session)
            loaded = load_model_session("fixture", root; base_url = "http://127.0.0.1:$(HTTP.port(server))",
                                        start_model = false, discover_capabilities = false)
            @test last_response(loaded).text == "finished"
            @test !only(last_response(loaded).tool_executions).success
            @test isempty(pending_operations(loaded))
            close(loaded)
        end
    finally
        close(server)
    end
end

@testset "Long public session, asynchronous compression, and repeated resume" begin
    answers = Ref(0)
    summary = "<assumptions></assumptions><results>derived view</results><notation></notation><deferred></deferred><rejected></rejected><narrative>continue</narrative>"
    server = HTTP.serve!("127.0.0.1", 0; listenany = true) do request
        if String(request.target) == "/api/generate"
            return HTTP.Response(200; body = JSON3.write((model = "fixture", response = summary, done = true, done_reason = "stop")))
        end
        answers[] += 1
        return HTTP.Response(200; body = JSON3.write((message = (content = "answer-$(answers[])", thinking = "trace-$(answers[])"), done = true, done_reason = "stop")))
    end
    try
        mktempdir() do root
            url = "http://127.0.0.1:$(HTTP.port(server))"
            config = BudgetConfig(num_ctx = 1024, system_budget = 64, summary_budget = 128, response_budget = 128,
                                  safety_margin = 64, max_tokens = 128, compression_frac = 0.3)
            session = ModelSession("fixture"; session_dir = root, base_url = url, cfg = config,
                                   start_model = false, discover_capabilities = false)
            expected = ["turn-$index " * repeat("αβ", 20) for index in 1:120]
            prefix = read(session_journal(session).path)
            for text in expected
                respond!(session, text; stream = false)
                await_compression!(session)
            end
            history = events(session)
            messages = [JSON3.read(event.payload_json) for event in history if event.kind == "message"]
            @test [String(message.content) for message in messages if message.role == "user"] == expected
            @test length(messages) == 240
            @test count(event -> event.kind == "summary_created", history) > 0
            @test length(session.session.messages) < length(messages)
            @test read(session_journal(session).path)[1:length(prefix)] == prefix
            save(session)
            anchor = journal_head(session_journal(session))
            close(session)
            bytes = read(joinpath(root, "journal.jsonl"))
            for cycle in 1:3
                session = load_model_session("fixture", root; base_url = url, start_model = false,
                                            discover_capabilities = false, checkpoint = anchor)
                @test isempty(pending_operations(session))
                @test last_response(session).text == "answer-120"
                save(session)
                close(session)
                @test read(joinpath(root, "journal.jsonl")) == bytes
            end
        end
    finally
        close(server)
    end
end
