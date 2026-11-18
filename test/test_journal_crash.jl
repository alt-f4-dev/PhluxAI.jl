using Test
include("../src/core/SessionJournal.jl")
using .SessionJournals

@testset "Process crash boundaries and writer leases" begin
    mktempdir() do root
        for mode in ("before", "during", "before_sync", "after_sync", "acknowledged")
            path = joinpath(root, mode * ".jsonl")
            writer = open_journal(path)
            append_event!(writer, :seed, (value = "retained",))
            anchor = journal_head(writer)
            prefix = read(path)
            close(writer)
            marker = path * ".ready"
            command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, "journal", "crash_writer.jl")) $path $marker $mode`
            process = run(pipeline(command; stdout = devnull, stderr = stderr); wait = false)
            try
                @test timedwait(() -> isfile(marker) || process_exited(process), 30.0; pollint = 0.01) == :ok
                @test isfile(marker)
            finally
                process_exited(process) || kill(process, Base.SIGKILL)
                wait(process)
            end
            @test read(path)[1:length(prefix)] == prefix
            if mode == "during"
                @test_throws JournalError verify_journal(path; checkpoint = anchor)
                recovered = open_journal(path; recover = true, checkpoint = anchor)
                @test journal_head(recovered) == anchor
            else
                recovered = open_journal(path; checkpoint = anchor)
                @test journal_head(recovered).sequence == (mode == "before" ? 1 : 2)
            end
            @test parse_checkpoint(checkpoint_json(anchor)) == anchor
            # A killed writer releases the OS lease without deleting or trusting stale lock-file contents.
            append_event!(recovered, :restarted)
            close(recovered)
        end
        path = joinpath(root, "exclusive.jsonl")
        writer = open_journal(path)
        script = "using JSON3; include(ARGS[1]); using .SessionJournals; try open_journal(ARGS[2]); exit(2); catch e; occursin(\"exclusive writer\", sprint(showerror,e)) || rethrow(); end"
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(@__DIR__)) -e $script $(joinpath(@__DIR__, "..", "src", "core", "SessionJournal.jl")) $path`)
        close(writer)
    end
end
