using Test
using PhluxAI

const CL = PhluxAI.ComfyLifecycle

@testset "Managed ComfyUI lifecycle: CPU-only structural tests" begin
    @testset "comfy-cli JSON envelopes" begin
        launched = """
            Launching ComfyUI from: /some/local/workspace
            {"schema":"envelope/1","type":"envelope","ok":true,"command":"launch","data":{"background":true,"listen":"127.0.0.1","port":8188,"url":"http://127.0.0.1:8188","pid":24680}}
            """
        packet = CL._cli_envelope(launched)
        @test packet.data.pid == 24680
        @test packet.data.url == "http://127.0.0.1:8188"
        @test packet.ok === true

        failed = """{"schema":"envelope/1","type":"envelope","ok":false,"command":"launch","error":{"code":"port_in_use"}}"""
        @test_throws ErrorException CL._cli_envelope(failed)
        @test_throws ErrorException CL._cli_envelope("non-JSON CLI output")
    end

    @testset "CLI background identity" begin
        environment = CL._cli_envelope("""{"type":"envelope","ok":true,"command":"env","data":{"config":{"background":{"host":"127.0.0.1","port":8188,"pid":24680}}}}""")
        background = CL._background_data(environment)
        @test background.pid == 24680
        @test background.port == 8188
        @test CL._background_data(CL._cli_envelope("""{"type":"envelope","ok":true,"data":{}}""")) === nothing
    end

    @testset "Linux process identity and lease behavior" begin
        if Sys.islinux()
            stamp = CL._process_stamp(getpid())
            @test !isnothing(stamp)
            @test CL._process_stamp(-1) === nothing
            @test CL._process_stamp(99999999) === nothing
        else
            @test CL._process_stamp(getpid()) === nothing
        end
    end

    @testset "autostart switch and safe unavailable endpoint" begin
        prior = get(ENV, "PHLUXAI_COMFY_AUTOSTART", nothing)
        try
            ENV["PHLUXAI_COMFY_AUTOSTART"] = "0"
            @test !CL._enabled()
            ENV["PHLUXAI_COMFY_AUTOSTART"] = "off"
            @test !CL._enabled()
            ENV["PHLUXAI_COMFY_AUTOSTART"] = "1"
            @test CL._enabled()
        finally
            if isnothing(prior)
                pop!(ENV, "PHLUXAI_COMFY_AUTOSTART", nothing)
            else
                ENV["PHLUXAI_COMFY_AUTOSTART"] = prior
            end
        end
        @test CL._available("http://127.0.0.1:1") === false
        @test CL.ensure_backend!(PhluxAI.NoImageBackend()) === :not_comfy
        status = CL.lifecycle_status()
        @test status.endpoint == PhluxAI.COMFY_BASE
        @test status.running isa Bool
        @test status.owned === false
    end
end

println("CPU-only checks complete. No comfy-cli launch, stop, or GPU inference requested.")
