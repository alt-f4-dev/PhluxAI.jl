using Test

include("../src/PhluxAI.jl")
using .PhluxAI

const TEST_WORKFLOW = Dict{String,Any}(
    "2" => Dict{String,Any}(
        "class_type" => "CheckpointLoaderSimple",
        "inputs" => Dict{String,Any}(
            "ckpt_name" => "default.safetensors",
        ),
    ),
    "6" => Dict{String,Any}(
        "class_type" => "CLIPTextEncode",
        "inputs" => Dict{String,Any}("text" => "prompt"),
    ),
)

const TEST_BINDINGS = ComfyBindings(
    positive_prompt = WorkflowBinding("6", "text"),
    model = WorkflowBinding("2", "ckpt_name"),
)

@testset "PhluxAI image-model API" begin
    @test isdefined(PhluxAI, Symbol("@imagemodel"))
    @test isdefined(PhluxAI, :set_default_image_workflow!)
    @test isdefined(PhluxAI, :default_image_workflow)
    @test isdefined(PhluxAI, :clear_default_image_workflow!)

    backend = ComfyBackend(
        TEST_WORKFLOW,
        TEST_BINDINGS;
        model = "sd_xl_base_1.0.safetensors",
    )
    @test backend.model == "sd_xl_base_1.0.safetensors"

    request = ImageGenerationRequest("test prompt")
    workflow = PhluxAI.ComfyClient._workflow_for_request(
        backend,
        request;
        ordinal = 1,
        batch_size = 1,
    )
    @test workflow["2"]["inputs"]["ckpt_name"] ==
          "sd_xl_base_1.0.safetensors"
    @test workflow["6"]["inputs"]["text"] == "test prompt"

    replacement = PhluxAI.ComfyClient.with_model(
        backend,
        "replacement.safetensors",
    )
    @test replacement.model == "replacement.safetensors"
    @test backend.model == "sd_xl_base_1.0.safetensors"

    no_model_binding = ComfyBindings(
        positive_prompt = WorkflowBinding("6", "text"),
    )
    @test_throws ArgumentError ComfyBackend(
        TEST_WORKFLOW,
        no_model_binding;
        model = "sd_xl_base_1.0.safetensors",
    )
    @test_throws ArgumentError set_default_image_workflow!(
        TEST_WORKFLOW,
        no_model_binding,
    )

    clear_default_image_workflow!()
    @test isnothing(default_image_workflow())
    @test_throws ArgumentError PhluxAI._resolve_image_backend(
        "sd_xl_base_1.0.safetensors",
    )

    template = set_default_image_workflow!(
        TEST_WORKFLOW,
        TEST_BINDINGS,
    )
    @test isnothing(template.model)
    @test default_image_workflow() isa ComfyBackend

    selected = @imagemodel "sd_xl_base_1.0.safetensors"
    @test selected isa ComfyBackend
    @test selected.model == "sd_xl_base_1.0.safetensors"
    @test PhluxAI._take_image_backend!().model ==
          "sd_xl_base_1.0.safetensors"
    @test isnothing(PhluxAI._take_image_backend!())

    model_named = macroexpand(
        PhluxAI,
        :(@model "qwen3-vl:4b" image = "sd_xl_base_1.0.safetensors"),
    )
    @test occursin("_resolve_image_backend", sprint(show, model_named))

    explicit_backend = macroexpand(
        PhluxAI,
        :(@model "qwen3-vl:4b" image = ComfyBackend(
            TEST_WORKFLOW,
            TEST_BINDINGS;
            model = "sd_xl_base_1.0.safetensors",
        )),
    )
    @test occursin("ComfyBackend", sprint(show, explicit_backend))

    pending_model = macroexpand(
        PhluxAI,
        :(@model "qwen3-vl:4b"),
    )
    @test occursin("_take_image_backend!", sprint(show, pending_model))

    agent_named = macroexpand(
        PhluxAI,
        :(@agent :physics "qwen3-vl:4b" image =
            "sd_xl_base_1.0.safetensors"),
    )
    @test occursin("_resolve_image_backend", sprint(show, agent_named))

    @test_throws Exception macroexpand(
        PhluxAI,
        :(@model "qwen3-vl:4b" unknown = 1),
    )

    clear_default_image_workflow!()
end
