using Test
using PhluxAI

const CL = PhluxAI.ComfyLifecycle
const DW = PhluxAI.ComfyWorkflow
const CC = PhluxAI.ComfyClient
const II = PhluxAI.ImageInterface

@testset "Phase 4: interactive image API (CPU only)" begin
    @testset "Ready-to-use standard checkpoint graph" begin
        workflow = DW.checkpoint_workflow()
        bindings = DW.checkpoint_bindings()
        @test length(workflow) == 7
        @test workflow["4"]["class_type"] == "CheckpointLoaderSimple"
        @test workflow["9"]["class_type"] == "SaveImage"
        @test bindings.model.node_id == "4"
        @test bindings.positive_prompt.input_name == "text"
        backend = DW.checkpoint_backend("v1-5-pruned-emaonly-fp16.safetensors")
        @test backend isa CC.ComfyBackend
        @test backend.model == "v1-5-pruned-emaonly-fp16.safetensors"
        @test backend.base_url == PhluxAI.COMFY_BASE
        @test_throws ArgumentError DW.checkpoint_backend("not-a-checkpoint.bin")
        @test_throws ArgumentError DW.checkpoint_backend("/tmp/weights.safetensors")
    end

    @testset "Bound prompt generation without live inference" begin
        backend = DW.checkpoint_backend("example.safetensors")
        request = II.ImageGenerationRequest(
            "A small observatory under a clear night sky";
            negative_prompt = "blurry",
            width = 512,
            height = 512,
            seed = 12345,
            steps = 12,
            cfg_scale = 6.5,
        )
        graph = CC._workflow_for_request(backend, request; ordinal = 1, batch_size = 1)
        @test graph["4"]["inputs"]["ckpt_name"] == "example.safetensors"
        @test graph["6"]["inputs"]["text"] == request.prompt
        @test graph["7"]["inputs"]["text"] == "blurry"
        @test graph["5"]["inputs"]["width"] == 512
        @test graph["3"]["inputs"]["seed"] == 12345
        @test graph["3"]["inputs"]["steps"] == 12
        @test graph["3"]["inputs"]["cfg"] == 6.5
        @test backend.workflow["6"]["inputs"]["text"] == ""
    end

    @testset "Interactive REPL nesting (no process launch or stop)" begin
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 0
        CL.enter_interactive_repl!()
        CL.enter_interactive_repl!()
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 2
        CL.leave_interactive_repl!(close_on_exit = false)
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 1
        CL.leave_interactive_repl!(close_on_exit = false)
        @test CL._ACTIVE_INTERACTIVE_REPLS[] == 0
        @test_throws ArgumentError CL.leave_interactive_repl!(close_on_exit = false)
    end
end

println("CPU-only tests complete. No model inference or intentional comfy-cli launch/stop was performed.")
