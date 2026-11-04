using Test
using HTTP
using JSON3

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

@testset "ComfyUI release_memory! mock" begin
    state = Ref(:idle)
    queue_calls = Ref(0)
    free_calls = Ref(0)
    free_payload = Ref("")

    handler = function (req)
        target = String(req.target)
        if req.method == "GET" && target == "/queue"
            queue_calls[] += 1
            state[] == :queue_error && return HTTP.Response(503; body = "Unavailable")
            state[] == :malformed && return HTTP.Response(200; body = JSON3.write(Dict("queue_running" => [])))
            busy = state[] == :busy
            return HTTP.Response(200; body = JSON3.write(Dict(
                "queue_running" => busy ? ["running-job"] : Any[],
                "queue_pending" => Any[],
            )))
        elseif req.method == "POST" && target == "/free"
            free_calls[] += 1
            free_payload[] = String(req.body)
            state[] == :free_error && return HTTP.Response(503; body = "Unavailable")
            return HTTP.Response(200)
        end
        return HTTP.Response(404; body = "Not found")
    end

    server = HTTP.serve!(handler, "127.0.0.1", 0; listenany = true)
    try
        workflow = Dict{String,Any}(
            "6" => Dict("class_type" => "CLIPTextEncode", "inputs" => Dict("text" => "initial")),
        )
        bindings = ComfyBindings(positive_prompt = WorkflowBinding("6", "text"))
        url = "http://127.0.0.1:$(HTTP.port(server))"
        backend = ComfyBackend(workflow, bindings; base_url = url, request_timeout = 5.0)

        @test PhluxAI.ComfyClient.release_memory!(backend) === nothing
        @test queue_calls[] == 1
        @test free_calls[] == 1
        @test JSON3.read(free_payload[])["unload_models"] == true
        @test JSON3.read(free_payload[])["free_memory"] == true

        state[] = :busy
        @test_throws ErrorException PhluxAI.ComfyClient.release_memory!(backend)
        @test free_calls[] == 1

        state[] = :malformed
        @test_throws ErrorException PhluxAI.ComfyClient.release_memory!(backend)
        @test free_calls[] == 1

        state[] = :queue_error
        @test_throws Exception PhluxAI.ComfyClient.release_memory!(backend)
        @test free_calls[] == 1

        state[] = :free_error
        @test_throws Exception PhluxAI.ComfyClient.release_memory!(backend; require_idle = false)
        @test free_calls[] == 2

        state[] = :busy
        @test PhluxAI.ComfyClient.release_memory!(backend; require_idle = false) === nothing
        @test free_calls[] == 3

        remote_backend = ComfyBackend(
            workflow, bindings;
            base_url = "http://192.0.2.1:8188", loopback_only = false,
        )
        @test_throws ArgumentError PhluxAI.ComfyClient.release_memory!(remote_backend; require_idle = false)
    finally
        HTTP.forceclose(server)
    end
end
