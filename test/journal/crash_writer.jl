using JSON3
include("../../src/core/SessionJournal.jl")
using .SessionJournals
path, marker, mode = ARGS
journal = open_journal(path)
if mode in ("before_sync", "after_sync")
    @eval SessionJournals function _sync(io::IOStream)::Nothing
        flush(io)
        if Main.mode == "after_sync"
            ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || error("sync failed")
        end
        write(Main.marker, "boundary")
        while true
            sleep(0.1)
        end
    end
end
if mode == "during"
    write(journal.io, "{\"body_json\":")
    flush(journal.io)
elseif mode != "before"
    append_event!(journal, :child, (value = "durable payload α",))
end
write(marker, "ready")
while true
    sleep(0.1)
end
