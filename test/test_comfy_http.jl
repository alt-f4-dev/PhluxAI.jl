
using Test
using HTTP
using JSON3
using Base64

ENV["PHLUXAI_BANNER"] = "0"
include("../src/PhluxAI.jl")
using .PhluxAI

@testset "ComfyUI mock HTTP integration" begin
    # A small, pre-encoded image fixture.
    image_bytes = base64decode(
        "R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
    )

    submitted_body = Ref("")
    history_calls = Ref(0)
    view_target = Ref("")

    handler = function (req)
        target = String(req.target)

        if req.method == "POST" && target == "/prompt"
            submitted_body[] = String(req.body)
            return HTTP.Response(
                200;
                body = JSON3.write(Dict("prompt_id" => "mock-1")),
            )
        elseif req.method == "GET" &&
               target == "/history/mock-1"

            history_calls[] += 1
            completed = history_calls[] >= 2

            entry = Dict(
                "status" => Dict(
                    "completed" => completed,
                    "status_str" => completed ?
                        "success" : "running",
                ),
                "outputs" => Dict(
                    "9" => Dict(
                        "images" => [
                            Dict(
                                "filename" => "mock.gif",
                                "subfolder" => "",
                                "type" => "output",
                            ),
                        ],
                    ),
                ),
            )

            return HTTP.Response(
                200;
                body = JSON3.write(Dict("mock-1" => entry)),
            )
        elseif req.method == "GET" &&
               startswith(target, "/view?")

            view_target[] = target
            return HTTP.Response(
                200;
                headers = ["Content-Type" => "image/gif"],
                body = image_bytes,
            )
        end

        return HTTP.Response(404; body = "Not found")
    end

    # Bind only to the host loopback interface.
    # Port 0 requests an automatically assigned free port.
    server = HTTP.serve!(
        handler, "127.0.0.1", 0;
        listenany = true,
    )

    try
        mktempdir() do directory
            workflow = Dict{String,Any}(
                "6" => Dict(
                    "class_type" => "CLIPTextEncode",
                    "inputs" => Dict("text" => "initial"),
                ),
            )

            bindings = ComfyBindings(
                positive_prompt = WorkflowBinding("6", "text"),
            )

            backend = ComfyBackend(
                workflow,
                bindings;
                base_url = "http://127.0.0.1:$(HTTP.port(server))",
                poll_interval = 0.01,
                timeout = 3.0,
                request_timeout = 10.0,
            )

            store = PhluxAI.ArtifactStore.Store(
                joinpath(directory, "artifacts"),
            )

            request = ImageGenerationRequest("test prompt")

            result = PhluxAI.ImageInterface.generate_images(
                backend,
                request,
                store,
            )

            @test result.done
            @test length(result.artifacts) == 1
            @test history_calls[] == 2
            @test occursin("filename=mock.gif", view_target[])

            @test !isempty(submitted_body[])
            if !isempty(submitted_body[])
                submitted = JSON3.read(submitted_body[])
                @test submitted["prompt"]["6"]["inputs"]["text"] ==
                      "test prompt"
            end

            if length(result.artifacts) == 1
                artifact = only(result.artifacts)
                @test artifact.mime_type == "image/gif"

                path = PhluxAI.ArtifactStore.resolve_artifact(
                    store,
                    artifact,
                )
                @test read(path) == image_bytes
            end
        end
    finally
        HTTP.forceclose(server)
    end
end
@testset "ComfyUI mock HTTP error handling" begin
    scenario = Ref(:http_error)

    handler = function (req)
        target = String(req.target)

        if req.method == "POST" && target == "/prompt"
            if scenario[] == :http_error
                return HTTP.Response(
                    503;
                    body = "Service unavailable",
                )
            elseif scenario[] == :invalid_json
                return HTTP.Response(
                    200;
                    body = "{invalid json",
                )
            elseif scenario[] == :rejected
                return HTTP.Response(
                    200;
                    body = JSON3.write(Dict(
                        "error" => "Invalid workflow",
                        "node_errors" => Dict(
                            "6" => "Invalid input",
                        ),
                    )),
                )
            end

            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "prompt_id" => "mock-failed",
                )),
            )
        elseif req.method == "GET" &&
               target == "/history/mock-failed"

            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "mock-failed" => Dict(
                        "status" => Dict(
                            "completed" => false,
                            "status_str" => "error",
                        ),
                        "outputs" => Dict(),
                    ),
                )),
            )
        end

        return HTTP.Response(404; body = "Not found")
    end

    server = HTTP.serve!(
        handler, "127.0.0.1", 0;
        listenany = true,
    )

    try
        mktempdir() do directory
            workflow = Dict{String,Any}(
                "6" => Dict(
                    "class_type" => "CLIPTextEncode",
                    "inputs" => Dict("text" => "initial"),
                ),
            )

            bindings = ComfyBindings(
                positive_prompt = WorkflowBinding("6", "text"),
            )

            backend = ComfyBackend(
                workflow,
                bindings;
                base_url = "http://127.0.0.1:$(HTTP.port(server))",
                poll_interval = 0.01,
                timeout = 2.0,
                # Allow cold HTTP/JIT compilation on CI runners without
                # changing the ComfyUI workflow-completion deadline.
                request_timeout = 20.0,
            )

            store = PhluxAI.ArtifactStore.Store(
                joinpath(directory, "artifacts"),
            )

            request = ImageGenerationRequest("test prompt")

            cases = (
                (:http_error, "HTTP 503"),
                (:invalid_json, "invalid JSON"),
                (:rejected, "rejected the workflow"),
                (:execution_error, "workflow execution failed"),
            )

            for (mode, expected_message) in cases
                scenario[] = mode

                error = try
                    PhluxAI.ImageInterface.generate_images(
                        backend,
                        request,
                        store,
                    )
                    nothing
                catch caught
                    caught
                end

                @test error isa ErrorException
                message = isnothing(error) ?
                    "" : sprint(showerror, error)
                @test occursin(expected_message, message)
            end
        end
    finally
        HTTP.forceclose(server)
    end
end
@testset "ComfyUI invalid image payload" begin
    handler = function (req)
        target = String(req.target)

        if req.method == "POST" && target == "/prompt"
            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "prompt_id" => "invalid-image",
                )),
            )
        elseif req.method == "GET" &&
               target == "/history/invalid-image"
            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "invalid-image" => Dict(
                        "status" => Dict(
                            "completed" => true,
                        ),
                        "outputs" => Dict(
                            "9" => Dict(
                                "images" => [
                                    Dict(
                                        "filename" => "invalid.png",
                                        "subfolder" => "",
                                        "type" => "output",
                                    ),
                                ],
                            ),
                        ),
                    ),
                )),
            )
        elseif req.method == "GET" &&
               startswith(target, "/view?")
            return HTTP.Response(
                200;
                headers = ["Content-Type" => "text/plain"],
                body = "This is not an image",
            )
        end

        return HTTP.Response(404)
    end

    server = HTTP.serve!(
        handler, "127.0.0.1", 0;
        listenany = true,
    )

    try
        mktempdir() do directory
            workflow = Dict{String,Any}(
                "6" => Dict(
                    "class_type" => "CLIPTextEncode",
                    "inputs" => Dict("text" => "initial"),
                ),
            )

            bindings = ComfyBindings(
                positive_prompt = WorkflowBinding("6", "text"),
            )

            backend = ComfyBackend(
                workflow,
                bindings;
                base_url = "http://127.0.0.1:$(HTTP.port(server))",
                poll_interval = 0.01,
                timeout = 2.0,
            )

            store = PhluxAI.ArtifactStore.Store(
                joinpath(directory, "artifacts"),
            )

            request = ImageGenerationRequest("test prompt")

            caught = try
                PhluxAI.ImageInterface.generate_images(
                    backend, request, store,
                )
                nothing
            catch error
                error
            end

            # Invalid bytes must cause a failure.
            @test caught isa Exception

            # The invalid image must not be persisted.
            @test isempty(readdir(store.root))
        end
    finally
        HTTP.forceclose(server)
    end
end
@testset "ComfyUI partial artifact failure" begin
    valid_image = base64decode(
        "R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
    )

    handler = function (req)
        target = String(req.target)

        if req.method == "POST" && target == "/prompt"
            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "prompt_id" => "partial-failure",
                )),
            )

        elseif req.method == "GET" &&
               target == "/history/partial-failure"

            images = [
                Dict(
                    "filename" => "valid.gif",
                    "subfolder" => "",
                    "type" => "output",
                ),
                Dict(
                    "filename" => "invalid.png",
                    "subfolder" => "",
                    "type" => "output",
                ),
            ]

            return HTTP.Response(
                200;
                body = JSON3.write(Dict(
                    "partial-failure" => Dict(
                        "status" => Dict("completed" => true),
                        "outputs" => Dict(
                            "9" => Dict("images" => images),
                        ),
                    ),
                )),
            )

        elseif req.method == "GET" &&
               startswith(target, "/view?filename=valid.gif")

            return HTTP.Response(200; body = valid_image)

        elseif req.method == "GET" &&
               startswith(target, "/view?filename=invalid.png")

            return HTTP.Response(
                200;
                body = "Not an image",
            )
        end

        return HTTP.Response(404)
    end

    server = HTTP.serve!(
        handler, "127.0.0.1", 0;
        listenany = true,
    )

    try
        mktempdir() do directory
            workflow = Dict{String,Any}(
                "6" => Dict(
                    "class_type" => "CLIPTextEncode",
                    "inputs" => Dict("text" => "initial"),
                ),
            )

            bindings = ComfyBindings(
                positive_prompt = WorkflowBinding("6", "text"),
            )

            backend = ComfyBackend(
                workflow,
                bindings;
                base_url = "http://127.0.0.1:$(HTTP.port(server))",
                poll_interval = 0.01,
                timeout = 2.0,
            )

            # Initialize the artifact store first.
            store = PhluxAI.ArtifactStore.Store(
                joinpath(directory, "artifacts"),
            )

            # Simulate an artifact from a previous successful request.
            existing_bytes = copy(valid_image)

            existing = PhluxAI.ArtifactStore.write_artifact(
                store,
                existing_bytes;
                filename = "preexisting.gif",
            )

            existing_path = PhluxAI.ArtifactStore.resolve_artifact(
                store,
                existing,
            )
            
            # Continue with the intentionally failing generation request.
            request = ImageGenerationRequest("test prompt")

            caught = try
                PhluxAI.ImageInterface.generate_images(
                    backend, request, store,
                )
                nothing
            catch error
                error
            end

            @test caught isa ErrorException

            message = caught isa Exception ?
                sprint(showerror, caught) : ""

            @test occursin("invalid image data", message)

            # The previous artifact must survive the failed request.
            @test isfile(existing_path)
            @test read(existing_path) == existing_bytes

            # The failed request must not leave additional artifacts.
            @test readdir(store.root) == [basename(existing_path)]
        end
    finally
        HTTP.forceclose(server)
    end
end
