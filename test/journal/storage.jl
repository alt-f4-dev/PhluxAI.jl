using Test, JSON3, UUIDs, SHA
include("../../src/core/SessionJournal.jl")
using .SessionJournals

@testset "Durable journal storage" begin
    mktempdir() do root
        path = joinpath(root, "history.jsonl")
        writer = open_journal(path)
        @test_throws ArgumentError open_journal(path)
        raw = "{\"text\":\"α\\nβ\", \"integer\":9007199254740993}"
        first = append_event!(writer, :message; payload_json = raw)
        anchor = journal_head(writer)
        original = read(path)
        @test first.payload_json == raw
        @test first.kind isa String && first.kind == "message"
        @test first.id == uuid5(first.session_id, "1")
        @test first.sha256 == bytes2hex(sha256(JSON3.read(String(copy(original))).body_json))
        append_event!(writer, :message, (text = "second",))
        close(writer)
        @test read(path)[1:length(original)] == original
        events = JournalEvent[]
        @test foreach_event(event -> push!(events, event), path; checkpoint = anchor).sequence == 2
        @test getfield.(events, :sequence) == [1, 2]
        @test events[1].payload_json == raw
        intact = read(path)
        for changed in (reverse(readlines(path)), [readlines(path)[2]], [readlines(path)[1], readlines(path)[1]])
            write(path, join(changed, '\n') * "\n")
            @test_throws JournalError verify_journal(path)
        end
        write(path, original)
        @test_throws JournalError verify_journal(path; checkpoint = JournalHead(first.session_id, 2, events[2].sha256, length(intact)))
        write(path, intact)
        open(path, "a") do io
            write(io, "{\"torn\":")
        end
        @test_throws JournalError open_journal(path)
        repaired = open_journal(path; recover = true)
        @test read(path) == intact
        @test length(filter(name -> occursin(".incomplete-", name), readdir(root))) == 1
        @test journal_head(repaired).sequence == 2
        tasks = [Threads.@spawn append_event!(repaired, :parallel, (index = i,)) for i in 1:64]
        foreach(fetch, tasks)
        @test verify_journal(repaired).sequence == 66
        close(repaired)
        @test_throws ArgumentError append_event!(repaired, :closed)
        limited = open_journal(joinpath(root, "limited"); max_record_bytes = 500)
        @test_throws ArgumentError append_event!(limited, :large, repeat("x", 1000))
        @test journal_head(limited).sequence == 0
        close(limited)
        custom = open_journal(joinpath(root, "custom"))
        @test append_event!(custom, "user-kind-α").kind == "user-kind-α"
        @test_throws ArgumentError append_event!(custom, "")
        @test_throws ArgumentError append_event!(custom, repeat("x", 257))
        close(custom)
    end
end

@testset "Corruption and incremental long history" begin
    mktempdir() do root
        path = joinpath(root, "long.jsonl")
        writer = open_journal(path)
        payload = repeat("α", 512)
        started = time_ns()
        for index in 1:2000
            append_event!(writer, :sample, (; index, payload))
        end
        first_seconds = (time_ns() - started) / 1e9
        first_head = journal_head(writer)
        prefix_digest = bytes2hex(open(sha256, path))
        started = time_ns()
        for index in 2001:4000
            append_event!(writer, :sample, (; index, payload))
        end
        second_seconds = (time_ns() - started) / 1e9
        last_head = journal_head(writer)
        close(writer)
        @test last_head.sequence == 4000
        @test last_head.offset - first_head.offset < 1.05 * first_head.offset
        open(path, "r") do io
            @test bytes2hex(sha256(read(io, first_head.offset))) == prefix_digest
        end
        count = Ref(0)
        foreach_event(path) do event
            count[] += 1
            @test JSON3.read(event.payload_json).index == count[]
        end
        @test count[] == 4000
        @info "Fixed-size durable append batches" first_seconds second_seconds bytes = last_head.offset
        damaged = joinpath(root, "damaged")
        first_line = open(readline, path)
        envelope = JSON3.read(first_line)
        body = replace(String(envelope.body_json), "sample" => "tamper")
        write(damaged, JSON3.write((body_json = body, sha256 = envelope.sha256)) * "\n")
        @test_throws JournalError verify_journal(damaged)
        # A self-consistent replacement requires an independent checkpoint to detect.
        write(damaged, JSON3.write((body_json = body, sha256 = bytes2hex(sha256(body)))) * "\n")
        @test verify_journal(damaged).sequence == 1
        @test_throws JournalError verify_journal(damaged; checkpoint = first_head)
        future = replace(String(envelope.body_json), "\"schema_version\":1" => "\"schema_version\":999")
        write(damaged, JSON3.write((body_json = future, sha256 = bytes2hex(sha256(future)))) * "\n")
        @test_throws JournalError open_journal(damaged; recover = true)
    end
end

@testset "Failed writes require explicit reopening" begin
    if Sys.islinux() && ispath("/dev/full")
        mktempdir() do root
            path = joinpath(root, "failed.jsonl")
            writer = open_journal(path)
            close(writer.io)
            writer.io = open("/dev/full", "w")
            @test_throws Exception append_event!(writer, :cannot_commit)
            @test writer.poisoned
            @test_throws ArgumentError append_event!(writer, :cannot_retry)
            try
                close(writer)
            catch
                # A repeated flush of the injected failing device may fail during close.
            end
            restarted = open_journal(path)
            @test journal_head(restarted).sequence == 0
            append_event!(restarted, :healthy)
            @test verify_journal(restarted).sequence == 1
            close(restarted)
        end
    end
end
