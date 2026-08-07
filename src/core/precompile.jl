using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    sample_show = JSON3.read("""
    {
      "capabilities": ["completion", "vision", "thinking", "tools"],
      "modified_at": "2026-08-05T00:00:00Z",
      "parameters": "temperature 0.7",
      "template": "{{ .Prompt }}",
      "details": {
        "family": "example",
        "format": "gguf",
        "parameter_size": "8B",
        "quantization_level": "Q4_K_M"
      },
      "model_info": {"example.context_length": 32768}
    }
    """)

    sample_workflow = JSON3.read("""
    {
      "2": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "precompile.safetensors"}
      },
      "3": {
        "class_type": "KSampler",
        "inputs": {
          "cfg": 7.0,
          "negative": ["7", 0],
          "positive": ["6", 0],
          "sampler_name": "euler",
          "scheduler": "normal",
          "seed": 1,
          "steps": 20
        }
      },
      "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
          "batch_size": 1,
          "height": 512,
          "width": 512
        }
      },
      "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "positive"}
      },
      "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "negative"}
      },
      "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI"}
      }
    }
    """)

    @compile_workload begin
        cfg = BudgetConfig(
            num_ctx = 4096,
            system_budget = 256,
            summary_budget = 256,
            response_budget = 128,
            safety_margin = 128,
            compression_frac = 0.6,
            m_verbatim = 4,
            exact_threshold = 64,
            num_thread = 0,
            max_tokens = 128,
        )
        effective_response_budget(cfg)
        history_budget(cfg)
        compression_trigger(cfg)

        ProtocolTypes.normalize_thinking(:medium)
        ProtocolTypes.thinking_wire_value(ProtocolTypes.THINK_MEDIUM)
        ProtocolTypes.ModelInput("Precompile multimodal input")
        ProtocolTypes.ModelResponse(
            "Precompiled response";
            reasoning = "Representative reasoning",
        )
        image_request = ImageInterface.ImageGenerationRequest(
            "Precompile image request";
            negative_prompt = "low quality",
            width = 512,
            height = 512,
            seed = 1,
            steps = 20,
            cfg_scale = 7.0,
        )
        ImageInterface.backend_name(ImageInterface.NoImageBackend())
        ImageInterface.resource_policy(ImageInterface.NoImageBackend())
        ProtocolTypes.ToolCall(
            "call-1",
            "list_files",
            "{\"path\":\".\"}",
        )

        details = CapabilityDiscovery.parse_model_details(
            "phluxai-precompile",
            sample_show,
        )
        CapabilityDiscovery.supports(details.capabilities, :vision)
        CapabilityDiscovery.capability_names(details.capabilities)

        mktempdir(prefix = "phluxai-precompile-") do directory
            store = ArtifactStore.Store(joinpath(directory, "artifacts"))
            artifact = ArtifactStore.write_artifact(
                store,
                collect(codeunits("example"));
                filename = "example.txt",
                mime_type = "text/plain",
            )
            ArtifactStore.verify_artifact(store, artifact)
            ArtifactStore.encode_artifact_base64(store, artifact)

            encoded_pixel =
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC" *
                "AAAAC0lEQVR42mP8/x8AAusB9Y9Zk8sAAAAASUVORK5CYII="
            image_body = Dict{Symbol,Any}(
                :data => Any[
                    Dict{Symbol,Any}(:b64_json => encoded_pixel),
                ],
            )
            image_response = OllamaClient._image_generation_response(
                image_body,
                store;
                prefix = "precompile-image",
                client_total_duration_ns = 1,
            )
            ProtocolTypes.response_images(image_response)

            sandbox = Sandboxing.SandboxConfig(
                joinpath(directory, "sandbox");
                allow_unsafe_local = true,
            )
            Sandboxing.resolve_workspace_path(sandbox, "script.jl")
            JupyTools.default_registry()

            model = OllamaModel(
                "phluxai-precompile",
                "http://127.0.0.1:11434",
            )
            ollama_image_backend = OllamaImageBackend(model)
            ImageInterface.backend_name(ollama_image_backend)

            bindings = ComfyClient.ComfyBindings(
                positive_prompt = ComfyClient.WorkflowBinding("6", "text"),
                model = ComfyClient.WorkflowBinding("2", "ckpt_name"),
                negative_prompt = ComfyClient.WorkflowBinding("7", "text"),
                seed = ComfyClient.WorkflowBinding("3", "seed"),
                width = ComfyClient.WorkflowBinding("5", "width"),
                height = ComfyClient.WorkflowBinding("5", "height"),
                batch_size = ComfyClient.WorkflowBinding(
                    "5",
                    "batch_size",
                ),
                steps = ComfyClient.WorkflowBinding("3", "steps"),
                cfg_scale = ComfyClient.WorkflowBinding("3", "cfg"),
                sampler_name = ComfyClient.WorkflowBinding(
                    "3",
                    "sampler_name",
                ),
                scheduler = ComfyClient.WorkflowBinding(
                    "3",
                    "scheduler",
                ),
                filename_prefix = ComfyClient.WorkflowBinding(
                    "9",
                    "filename_prefix",
                ),
            )
            comfy_backend = ComfyClient.ComfyBackend(
                sample_workflow,
                bindings;
                model = "precompile.safetensors",
                timeout = 1.0,
                request_timeout = 1.0,
            )
            ComfyClient.with_model(
                comfy_backend,
                "alternate-precompile.safetensors",
            )
            ImageInterface.backend_name(comfy_backend)
            ImageInterface.is_local_backend(comfy_backend)
            ImageInterface.should_unload_ollama(comfy_backend, true)
            session = OllamaClient.ChatSession(
                model,
                "Precompile system prompt";
                cfg,
                session_dir = joinpath(directory, "session"),
                session_type = "model",
                artifact_store = store,
            )
            OllamaClient.assemble_messages(session)
            OllamaClient.save_session(session)
            OllamaClient.close_session(session)

            restored = OllamaClient.load_session(
                model,
                joinpath(directory, "session");
                artifact_store = store,
            )
            OllamaClient.assemble_messages(restored)
            OllamaClient.close_session(restored)

            model_session = ModelSessions.ModelSession(
                model.name;
                cfg,
                base_url = model.base_url,
                session_dir = joinpath(directory, "model-session"),
                artifact_dir = joinpath(directory, "model-artifacts"),
                sandbox_dir = joinpath(directory, "model-sandbox"),
                start_model = false,
                stop_model_on_close = false,
                discover_capabilities = false,
                image_backend = comfy_backend,
            )
            ModelSessions.set_temperature!(model_session, 0.5)
            ModelSessions.set_thinking!(model_session, :medium)
            ModelSessions.default_temperature(model_session)
            ModelSessions.default_thinking(model_session)
            ModelSessions.default_image_backend(model_session)
            ModelSessions.set_image_backend!(
                model_session,
                ollama_image_backend,
            )
            ModelSessions.save(model_session)
            close(model_session)

            assistant = PhysicalAssistant.PhysicsAssistant(
                model.name;
                cfg,
                base_url = model.base_url,
                session_dir = joinpath(directory, "physics-session"),
                artifact_dir = joinpath(directory, "physics-artifacts"),
                sandbox_dir = joinpath(directory, "physics-sandbox"),
                start_model = false,
                stop_model_on_close = false,
                discover_capabilities = false,
                image_backend = comfy_backend,
            )
            PhysicalAssistant.set_mode!(
                assistant,
                PhysicalAssistant.DERIVE,
            )
            PhysicalAssistant.default_mode(assistant)
            PhysicalAssistant.default_image_backend(assistant)
            PhysicalAssistant.save(assistant)
            close(assistant)
        end

        macroexpand(
            @__MODULE__,
            :(@budget num_ctx = 8192 max_tokens = 256),
        )
        macroexpand(@__MODULE__, :(@imagemodel "precompile.safetensors"))
        macroexpand(@__MODULE__, :(@model "phluxai-precompile"))
        macroexpand(
            @__MODULE__,
            :(@model "phluxai-precompile" image = "precompile.safetensors"),
        )
        macroexpand(
            @__MODULE__,
            :(@agent :physics "phluxai-precompile"),
        )
        macroexpand(
            @__MODULE__,
            :(@resume "phluxai-precompile" "sessions/example"),
        )
        buffer = IOBuffer()
        welcome(buffer; check_ollama = false)
        String(take!(buffer))
    end
end
