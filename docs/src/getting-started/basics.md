# PhluxAI Basics

PhluxAI is a local multimodal framework for stateful Ollama text and vision
sessions, ComfyUI image generation, exposed reasoning fields, multimodal
artifacts, and model-requested tool execution through `jupy` and `jupip`.

The text/vision model and image-generation model are independent:

- `@model` and `@agent` select the Ollama conversational model;
- `@imagemodel` or `image=...` selects a ComfyUI image model when a default
  ComfyUI workflow has been configured;
- `\image PROMPT` generates through the image backend attached to the active
  session.

This guide covers two styles of use:

1. **Interactive use** through `@model`, `@agent`, `@imagemodel`, `@resume`,
   and the terminal REPL.
2. **Programmatic use** through `ModelSession`, `PhysicsAssistant`, `respond!`,
   `ask!`, `generate_images!`, and `ComfyBackend`.

The examples assume PhluxAI has been built successfully. See `BUILD.md` for the
external runtime installation and build flags.

---

## 1. Install and verify models

### Ollama text or vision model

Install a conversational model through the Ollama CLI:

```bash
ollama pull gpt-oss:20b
```

For a vision-capable model:

```bash
ollama pull qwen3-vl:4b
```

List locally installed Ollama models:

```bash
ollama list
```

Run a direct smoke test:

```bash
ollama run gpt-oss:20b
```

Press `Ctrl+D` or enter `/bye` to leave the Ollama CLI.

PhluxAI can preload an Ollama model automatically when a session is
constructed with `start_model=true`, but pulling it explicitly first gives
clearer installation and storage errors.

### ComfyUI image model

The PhluxAI build installs and configures `comfy-cli` and ComfyUI, but it does
not download image-model weights. Select and install those separately.

A checkpoint download typically uses:

```bash
comfy model download \
  --url <MODEL_URL> \
  --relative-path models/checkpoints
```

With the default Linux workspace, checkpoints normally live under:

```text
~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/
```

Other model classes can require different ComfyUI model directories. Follow the
model and workflow documentation rather than assuming every file is a
checkpoint.

Image-model files are runtime/user data and do not belong in the PhluxAI
repository.

---

## 2. Start PhluxAI and the local runtimes

From the package repository during development, launch Julia through Jupy:

```bash
jupy
```

`jupy` resolves the nearest Julia project, activates it, and binds PythonCall
to that project's `.venv`. Run it from the intended project directory or one
of its descendants.

Then import the package:

```julia
using PhluxAI
```

Print the splash manually:

```julia
welcome()
```

Suppress the automatic interactive splash:

```bash
PHLUXAI_BANNER=0 jupy
```

Ollama must be running for text/vision sessions. When installed as a local
service this is normally automatic; otherwise start it separately:

```bash
ollama serve
```

ComfyUI is required only when using a `ComfyBackend`. Start it with:

```bash
comfy launch --background
```

Verify the default local endpoint:

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

Stop it with:

```bash
comfy stop
```

---

## 3. Configure ComfyUI image generation

ComfyUI model selection and workflow selection are separate concerns. A model
filename identifies the model loaded by a workflow; the API-format workflow
defines the graph that loads, conditions, samples, decodes, and saves the
image.

PhluxAI therefore requires explicit workflow bindings. `positive_prompt` is
required for prompt injection. `model` is also required when selecting a model
by filename through `@imagemodel` or `image="..."`.

For example:

```julia
bindings = ComfyBindings(
    positive_prompt = WorkflowBinding("6", "text"),
    model = WorkflowBinding("4", "ckpt_name"),
    negative_prompt = WorkflowBinding("7", "text"),
    seed = WorkflowBinding("3", "seed"),
    width = WorkflowBinding("5", "width"),
    height = WorkflowBinding("5", "height"),
    batch_size = WorkflowBinding("5", "batch_size"),
    steps = WorkflowBinding("3", "steps"),
    cfg_scale = WorkflowBinding("3", "cfg"),
    sampler_name = WorkflowBinding("3", "sampler_name"),
    scheduler = WorkflowBinding("3", "scheduler"),
    filename_prefix = WorkflowBinding("9", "filename_prefix"),
)
```

The node IDs above are examples only. They must match the actual API-format
workflow being used.

Configure a model-neutral default workflow:

```julia
set_default_image_workflow!(
    "/path/to/sdxl_api.json",
    bindings,
)
```

The workflow can live anywhere on the filesystem. It does not have to exist in
the PhluxAI repository. Small maintained reference workflows may be committed
under a directory such as `examples/workflows/`, while personal workflows are
better kept in user data.

Inspect or clear the process-local default:

```julia
default_image_workflow()
clear_default_image_workflow!()
```

### Select an image model for the next session

`@imagemodel` stores a one-shot image backend consumed by the next `@model` or
`@agent` call:

```julia
@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

The same selection can be written inline:

```julia
@model "qwen3-vl:4b" image="sd_xl_base_1.0.safetensors"
```

or for an agent:

```julia
@agent :physics "qwen3-vl:4b" \
    image="sd_xl_base_1.0.safetensors"
```

An explicit backend is also accepted:

```julia
backend = ComfyBackend(
    "/path/to/sdxl_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)

@model "qwen3-vl:4b" image=backend
```

An explicit `image=...` argument overrides and consumes a pending
`@imagemodel` selection. `@imagemodel` does not apply to `@resume` in the
current API. To attach a Comfy backend to a resumed session, load it
programmatically, call `set_image_backend!`, and then enter `repl`.

The default workflow is process-local and is not persisted with chat-session
state. The selected image model is likewise not persisted as part of the chat
session.

---

# Interactive usage

The macros launch blocking PhluxAI terminal loops. They are intended for direct
conversation. The session object is internal to the macro and the macro returns
after its REPL exits.

Use the programmatic constructors later in this guide when the session object
must remain available for inspection or further configuration.

## 4. Generic interactive model session

Start an in-memory agentless session:

```julia
@model "gpt-oss:20b"
```

Example interaction:

```text
Model ready. Type '\help' for commands.
────────────────────────────────────────────────────────────

you> Explain the physical meaning of a branch cut.

assistant> ...

you> Give a more formal statement.

assistant> ...

you> \exit
```

The model response streams to the terminal.

### Persistent generic session

Supply a session directory:

```julia
@model "gpt-oss:20b" "sessions/general"
```

PhluxAI stores resumable state under that directory. The default artifact and
sandbox locations are:

```text
sessions/general/artifacts/
sessions/general/sandbox/
```

Resume the session later:

```julia
@resume "gpt-oss:20b" "sessions/general"
```

`@resume` reads the saved session type and selects the generic or physics
wrapper automatically.

---

## 5. Physics-assistant session

Start the condensed-matter physics profile:

```julia
@agent :physics "gpt-oss:20b"
```

Start a persistent physics session:

```julia
@agent :physics "gpt-oss:20b" "sessions/scco"
```

Resume it later:

```julia
@resume "gpt-oss:20b" "sessions/scco"
```

The physics REPL supports turn-specific mode sigils:

```text
you> \general Summarize the current argument.

you> \derive Derive the response-function pole condition.

you> \code Implement the susceptibility in Julia and plot it.

you> \draft Write the corresponding manuscript paragraph.
```

The mode temperatures are:

| Mode | Default temperature | Intended use |
|---|---:|---|
| `GENERAL` | `0.70` | General technical discussion |
| `DERIVE` | `0.15` | Formal derivations |
| `CODE` | `0.10` | Complete Julia implementations |
| `DRAFT` | `0.50` | Manuscript or proposal drafting |

A mode sigil applies to that turn. The configured default mode remains
unchanged.

---

## 6. Interactive command reference

Enter `\help` inside either REPL to print the command list.

| Command | Effect |
|---|---|
| `\temp` | Show the current temperature |
| `\temp VALUE` | Set the session temperature |
| `\temp default` | Reset a generic session to `0.7` |
| `\temp auto` | Physics REPL only: return to the active mode temperature |
| `\think` | Show the current thinking setting |
| `\think off` | Disable requested thinking |
| `\think on` | Request thinking without a named level |
| `\think low` | Request low thinking |
| `\think medium` | Request medium thinking |
| `\think high` | Request high thinking |
| `\think max` | Request the maximum named level |
| `\think default` | Return to the model default |
| `\reasoning show` | Display reasoning chunks during streaming |
| `\reasoning hide` | Hide reasoning chunks |
| `\tools` | Show the current tool mode |
| `\tools off` | Do not expose or execute tools |
| `\tools ask` | Ask before executing every tool request |
| `\tools auto` | Auto-approve non-network tools |
| `\attach PATH` | Import a file as a pending attachment |
| `\attachments` | List pending attachments |
| `\clearattachments` | Clear pending attachments |
| `\image PROMPT` | Generate through the image backend attached to the active session |
| `\artifacts` | List files in the session artifact store |
| `\sandbox` | Show sandbox workspace, backend, and network state |
| `\save` | Save the current session |
| `\exit` | Save, close, and leave |
| `\quit` | Save, close, and leave |
| `\bye` | Generic REPL only: save, close, and leave |

### Change temperature interactively

```text
you> \temp
Temperature: 0.7

you> \temp 0.2
Temperature set to 0.2
```

In the physics REPL:

```text
you> \temp 0.05
Temperature override set to 0.05

you> \temp auto
Temperature reset to mode default: 0.7
```

### Request and display reasoning

```text
you> \think high
Thinking set to high

you> \reasoning show
Reasoning display: show

you> Derive the result and then state the final conclusion.
```

Thinking support depends on the selected model. PhluxAI keeps reasoning
separate from final response text.

### Attach an image or other file

```text
you> \attach data/detector_scan.png
Attached: detector_scan.png

you> Identify the dominant features in this scan.
```

The interactive loop automatically includes pending attachments in the next
turn and clears the pending list after a successful response.

### Generate an image

A ComfyUI-enabled session can generate through the backend selected when the
session was created:

```text
you> \image Publication-style schematic of a two-leg spin ladder
```

`\image` does not choose a checkpoint. It sends the prompt to the image backend
already attached to the session. For a named ComfyUI model, configure a default
workflow and use `@imagemodel` or `image=...` before entering the REPL.

Generated files are imported into the session artifact store and listed after
the request completes.

### Enable code-execution tools

```text
you> \tools ask
Tool mode set to TOOLS_ASK

you> Write a Julia script that plots sin(x), run it, and save the PNG.
```

In `TOOLS_ASK` mode, PhluxAI prints the requested tool, its risk class, and its
JSON arguments before asking for approval.

---

## 7. Configure the next interactive session with `@budget`

`@budget` changes the `BudgetConfig` consumed by the next `@model`, `@agent`, or
`@resume` invocation. The setting is one-shot and resets after use.

```julia
@budget num_ctx=32768 max_tokens=1024 num_thread=8
@model "gpt-oss:20b"
```

A larger persistent physics session:

```julia
@budget num_ctx=32768 \
        summary_budget=1200 \
        response_budget=1024 \
        safety_margin=512 \
        compression_frac=0.7 \
        m_verbatim=8 \
        num_thread=8 \
        max_tokens=1024

@agent :physics "gpt-oss:20b" "sessions/scco-large"
```

Available fields:

| Field | Meaning |
|---|---|
| `num_ctx` | Context size sent to Ollama |
| `system_budget` | Context reserved for the system prompt |
| `summary_budget` | Context and generation budget for compressed memory |
| `response_budget` | Explicit response reservation |
| `safety_margin` | Additional context withheld from history |
| `compression_frac` | Fraction of history capacity that triggers compression |
| `m_verbatim` | Newest raw messages retained after compression |
| `exact_threshold` | Threshold used by explicit hybrid token counting |
| `num_thread` | CPU thread setting sent to Ollama; `0` leaves it to Ollama |
| `max_tokens` | Default output-token cap |

Important constraints:

```text
num_ctx >
    system_budget +
    summary_budget +
    max(response_budget, max_tokens) +
    safety_margin

0 < compression_frac <= 1
m_verbatim must be nonnegative and even
num_thread must be nonnegative
max_tokens must be positive
```

For `@resume`, the persisted budget is retained unless `@budget` was explicitly
set immediately before the resume call.

---

# Programmatic usage

Use programmatic sessions when code needs to inspect responses, configure
callbacks, retain artifacts, or continue using a session after a terminal loop.

## 8. Create a generic `ModelSession`

```julia
using PhluxAI

cfg = BudgetConfig(
    num_ctx = 32768,
    system_budget = 600,
    summary_budget = 1200,
    response_budget = 1024,
    safety_margin = 512,
    compression_frac = 0.7,
    m_verbatim = 8,
    exact_threshold = 200,
    num_thread = 8,
    max_tokens = 1024,
)

session = ModelSession(
    "gpt-oss:20b";
    cfg,
    system_prompt = "You are a precise technical research assistant.",
    session_dir = "sessions/programmatic",
    temperature = 0.2,
    thinking = :high,
    show_reasoning = false,
    persist_reasoning = false,
    keep_alive = "30m",
    request_timeout = 300.0,
    read_idle_timeout = 60.0,
    tool_mode = TOOLS_OFF,
    sandbox_backend = SANDBOX_AUTO,
    sandbox_network = false,
    sandbox_timeout = 120.0,
    discover_capabilities = true,
    timeout = 180,
    poll_interval = 2.0,
    start_model = true,
    stop_model_on_close = true,
)
```

Common constructor controls:

| Keyword | Purpose |
|---|---|
| `cfg` | Context, compression, thread, and output-token configuration |
| `system_prompt` | Generic system prompt |
| `session_dir` | Hot/cold persistence directory |
| `artifact_dir` | Override the artifact-store directory |
| `sandbox_dir` | Override the tool workspace directory |
| `base_url` | Ollama API endpoint |
| `temperature` | Default sampling temperature |
| `thinking` | `nothing`, boolean, symbol, string, or named thinking level |
| `show_reasoning` | Print reasoning during streamed responses |
| `persist_reasoning` | Persist reasoning traces in session messages |
| `keep_alive` | Ollama model-residency policy |
| `request_timeout` | Overall HTTP timeout; `0` disables it |
| `read_idle_timeout` | Streaming inactivity timeout; `0` disables it |
| `tool_mode` | `TOOLS_OFF`, `TOOLS_ASK`, or `TOOLS_AUTO` |
| `sandbox_backend` | `SANDBOX_AUTO`, `SANDBOX_BWRAP`, or `SANDBOX_LOCAL` |
| `sandbox_network` | Allow network access inside the sandbox |
| `sandbox_timeout` | Tool-process timeout in seconds |
| `allow_unsafe_local` | Permit the non-isolating local backend |
| `sandbox_readonly_paths` | Host paths made available read-only |
| `sandbox_environment` | Explicit environment values for tools |
| `discover_capabilities` | Query Ollama model metadata during construction |
| `image_backend` | Image-generation backend; pass a `ComfyBackend` for ComfyUI |
| `timeout` | Model startup timeout |
| `poll_interval` | Model startup polling interval |
| `start_model` | Preload the model |
| `stop_model_on_close` | Unload the model when closing |

For an externally managed or remote server:

```julia
session = ModelSession(
    "gpt-oss:20b";
    base_url = "http://192.168.1.50:11434",
    start_model = false,
    stop_model_on_close = false,
)
```

---

## 9. `respond!` versus `ask!`

Use `respond!` for the full structured response:

```julia
response = respond!(
    session,
    "Explain the result and expose reasoning separately.";
    thinking = :high,
    temperature = 0.15,
    max_tokens = 768,
)
```

`respond!` returns a `ModelResponse`:

```julia
response.text
response.reasoning
response.artifacts
response.tool_calls
response.tool_executions
response.done
response.done_reason
response.metrics
```

Convenience predicates:

```julia
has_reasoning(response)
has_tool_calls(response)
response_images(response)
```

Use `ask!` when only final text is required:

```julia
text = ask!(
    session,
    "State only the final result.";
    temperature = 0.1,
    max_tokens = 384,
)
```

The return value of `ask!` is a `String`.

### Non-streaming structured response

```julia
response = respond!(
    session,
    "Return a concise answer.";
    stream = false,
    print_tokens = false,
)
```

### Stream into a selected IO object

```julia
buffer = IOBuffer()

response = respond!(
    session,
    "Explain the derivation.";
    io = buffer,
    stream = true,
    print_tokens = true,
)

streamed_text = String(take!(buffer))
```

---

## 10. Change session defaults

```julia
set_temperature!(session, 0.25)
set_thinking!(session, :medium)
set_reasoning_visible!(session, true)
set_keep_alive!(session, "30m")

set_timeouts!(
    session;
    request_timeout = 600.0,
    read_idle_timeout = 120.0,
)
```

Inspect the defaults:

```julia
default_temperature(session)
default_thinking(session)
default_keep_alive(session)
default_tool_mode(session)
```

A per-turn keyword passed to `respond!` or `ask!` overrides the stored default
for that call only.

---

## 11. Create a programmatic `PhysicsAssistant`

```julia
assistant = PhysicsAssistant(
    "gpt-oss:20b";
    cfg,
    session_dir = "sessions/scco-programmatic",
    mode = DERIVE,
    thinking = :high,
    show_reasoning = false,
    persist_reasoning = false,
    tool_mode = TOOLS_ASK,
    sandbox_backend = SANDBOX_AUTO,
    sandbox_network = false,
)
```

Use its default mode:

```julia
response = respond!(
    assistant,
    "Derive the response-function pole condition.",
)
```

Select a mode for one turn:

```julia
response = respond!(
    assistant,
    "Implement the result in Julia and produce a figure.";
    mode = CODE,
)
```

Text-only compatibility:

```julia
text = ask!(
    assistant,
    "State the final instability condition.";
    mode = DERIVE,
)
```

Change the persistent mode:

```julia
set_mode!(assistant, CODE)
default_mode(assistant)
```

Override the mode temperature:

```julia
set_temperature!(assistant, 0.05)
default_temperature(assistant)
```

Return to mode-dependent temperature selection:

```julia
set_temperature!(assistant, nothing)
```

`PhysicalAgent` is an alias for `PhysicsAssistant`:

```julia
assistant = PhysicalAgent(
    "gpt-oss:20b";
    mode = CODE,
)
```

---

## 12. Multimodal input

### Import and send an image

```julia
image = attach!(
    session,
    "data/detector_scan.png",
)

input = ModelInput(
    "Identify the dominant scattering features in this image.",
    [image],
)

response = respond!(session, input)
```

`attach!` imports the file into the session artifact store and also places it
in the pending-attachment list.

Programmatic `respond!` uses exactly the `ModelInput` supplied to it. Pending
attachments are included automatically only by the interactive REPL.

Clear pending attachments when they are no longer needed:

```julia
clear_attachments!(session)
```

Inspect pending attachments:

```julia
pending_attachments(session)
```

### Construct ordered multimodal parts

```julia
input = ModelInput([
    TextPart("Compare these two scans."),
    ArtifactPart(scan_a),
    ArtifactPart(scan_b),
])

response = respond!(session, input)
```

### Inspect the artifact store

```julia
store = artifact_store(session)

store.root
ArtifactStore.list_artifacts(store)
ArtifactStore.verify_artifact(store, image)
```

Artifact references contain:

```julia
image.path
image.mime_type
image.sha256
image.size_bytes
```

---

## 13. Generate images

### ComfyUI backend

Construct a backend from an API-format workflow and bindings:

```julia
bindings = ComfyBindings(
    positive_prompt = WorkflowBinding("6", "text"),
    model = WorkflowBinding("4", "ckpt_name"),
    negative_prompt = WorkflowBinding("7", "text"),
    seed = WorkflowBinding("3", "seed"),
    width = WorkflowBinding("5", "width"),
    height = WorkflowBinding("5", "height"),
    steps = WorkflowBinding("3", "steps"),
    cfg_scale = WorkflowBinding("3", "cfg"),
)

backend = ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)
```

`ComfyBackend` defaults to `http://127.0.0.1:8188` and
`loopback_only=true`. It communicates with ComfyUI through local HTTP polling
using `/prompt`, `/history/{prompt_id}`, and `/view`.

Check server availability:

```julia
comfy_available(backend)
```

Load and validate a workflow without constructing a backend:

```julia
workflow = load_comfy_workflow("/path/to/workflow_api.json")
```

### Attach the backend to a session

```julia
session = ModelSession(
    "qwen3-vl:4b";
    image_backend = backend,
)
```

A backend can also be changed later:

```julia
set_image_backend!(session, backend)
default_image_backend(session)
```

Generate through the session:

```julia
response = generate_images!(
    session,
    "Publication-style schematic of a two-leg spin ladder";
    negative_prompt = "blurry, illegible labels",
    width = 1024,
    height = 1024,
    count = 1,
    seed = 1234,
    steps = 30,
    cfg_scale = 7.0,
    sampler_name = "euler_ancestral",
    scheduler = "normal",
    filename_prefix = "spin-ladder",
)
```

Optional values are applied only when the selected workflow exposes the
corresponding binding. For example, passing `width=1024` requires a configured
`width` binding.

Inspect generated files:

```julia
for image in response_images(response)
    println(image.path)
    println(image.mime_type)
    println(image.sha256)
end
```

### Backend-neutral request object

```julia
request = ImageGenerationRequest(
    "A reciprocal-space intensity map with fourfold symmetry";
    width = 1024,
    height = 1024,
    seed = 1234,
    steps = 30,
    cfg_scale = 6.5,
)

response = generate_images!(session, request)
```

### Generate directly through a backend

The low-level form requires an explicit artifact store:

```julia
store = ArtifactStore.Store("generated-images")

response = generate_images(
    backend,
    "A publication-style scientific schematic";
    artifact_store = store,
    width = 1024,
    height = 1024,
)
```

### Explicit backend construction in an interactive macro

```julia
@model "qwen3-vl:4b" image=ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)
```

### Ollama image-generation compatibility

A `ModelSession` without an explicit `image_backend` currently receives an
`OllamaImageBackend` for compatibility. The older Ollama image transports can
still be selected with `backend=:auto`, `:http`, or `:cli` and their related
keywords.

For the local ComfyUI path, prefer a concrete `ComfyBackend` or the
`@imagemodel`/`image=...` user-facing API. Do not mix `backend=` and
`image_backend=` in the same image-generation call.

---

## 14. Tool execution with `jupy` and `jupip`

Tool execution requires:

```text
jupy
jupip
```

Both executables must be on `PATH`, unless explicit paths are supplied through
a custom `ToolContext`.

The default registered tools are:

| Tool | Risk | Purpose |
|---|---|---|
| `write_text_file` | `TOOL_WRITE` | Write UTF-8 text in the workspace |
| `read_text_file` | `TOOL_READ` | Read a bounded UTF-8 text file |
| `list_files` | `TOOL_READ` | List workspace files |
| `run_julia_script` | `TOOL_EXECUTE` | Run a Julia script through `jupy` |
| `install_python_packages` | `TOOL_NETWORK` | Install packages through `jupip` |
| `inspect_python_package` | `TOOL_READ` | Run `jupip show` |
| `collect_artifacts` | `TOOL_READ` | Copy workspace files to the artifact store |

### Tool modes

```julia
set_tool_mode!(session, TOOLS_OFF)
set_tool_mode!(session, TOOLS_ASK)
set_tool_mode!(session, TOOLS_AUTO)
```

Behavior:

- `TOOLS_OFF` does not expose tools to the model.
- `TOOLS_ASK` requires an approval callback for every tool.
- `TOOLS_AUTO` automatically approves non-network tools.
- Network-capable package installation always requires explicit approval.

### Programmatic approval callback

```julia
function approve_tool(
    call::ToolCall,
    risk::ToolRisk,
)::Bool
    println("Requested tool: ", call.name)
    println("Risk: ", risk)
    println("Arguments: ", call.arguments_json)

    # Example policy: deny network mutation, approve other registered tools.
    return risk != TOOL_NETWORK
end

response = respond!(
    session,
    """
    Write a Julia script that samples sin(x), creates a PNG plot, runs the
    script through jupy, and return the generated artifact.
    """;
    tool_mode = TOOLS_ASK,
    approval_callback = approve_tool,
    max_tool_steps = 8,
)
```

Inspect the executions:

```julia
for execution in response.tool_executions
    println("Tool: ", execution.name)
    println("Success: ", execution.success)
    println("Exit code: ", execution.exit_code)
    println(execution.output)

    for artifact in execution.artifacts
        println("Artifact: ", artifact.path)
    end
end
```

### Use custom `jupy` and `jupip` paths

```julia
context = ToolContext(
    sandbox_config(session),
    artifact_store(session);
    jupy_executable = "/home/user/.local/bin/jupy",
    jupip_executable = "/home/user/.local/bin/jupip",
)

set_tool_context!(session, context)
set_tool_mode!(session, TOOLS_ASK)
```

---

## 15. Sandbox configuration

The preferred Linux backend is bubblewrap:

```julia
session = ModelSession(
    "gpt-oss:20b";
    tool_mode = TOOLS_ASK,
    sandbox_backend = SANDBOX_BWRAP,
    sandbox_network = false,
)
```

Automatic selection:

```julia
sandbox_backend = SANDBOX_AUTO
```

`SANDBOX_AUTO` selects bubblewrap when `bwrap` is available. It does not fall
back to local execution unless `allow_unsafe_local=true`.

Explicit local execution:

```julia
session = ModelSession(
    "gpt-oss:20b";
    tool_mode = TOOLS_ASK,
    sandbox_backend = SANDBOX_LOCAL,
    allow_unsafe_local = true,
)
```

`SANDBOX_LOCAL` constrains paths and uses a session-local home and Julia depot,
but it is not a security boundary.

Useful settings:

```julia
session = ModelSession(
    "gpt-oss:20b";
    tool_mode = TOOLS_ASK,
    sandbox_backend = SANDBOX_AUTO,
    sandbox_network = false,
    sandbox_timeout = 120.0,
    sandbox_readonly_paths = [pwd()],
    sandbox_environment = Dict(
        "JULIA_NUM_THREADS" => "8",
    ),
)
```

Inspect the active configuration:

```julia
config = sandbox_config(session)

config.workspace
config.backend
config.network
config.timeout_seconds
config.max_output_bytes
config.readonly_paths
config.environment
```

All model-supplied file paths must remain relative to the sandbox workspace.
Absolute paths and path traversal outside the workspace are rejected.

---

## 16. Diagnostics

Obtain metrics from the structured response:

```julia
response = respond!(
    session,
    "Give a concise explanation.";
)

metrics = response.metrics
```

Or inspect the most recent session result:

```julia
last_response(session)
last_response_metrics(session)
```

Available metric fields:

```julia
metrics.total_duration_ns
metrics.load_duration_ns
metrics.prompt_eval_count
metrics.prompt_eval_duration_ns
metrics.eval_count
metrics.eval_duration_ns
metrics.client_total_duration_ns
metrics.time_to_first_token_ns
```

Convert durations and calculate throughput:

```julia
ns_to_s(value::Integer)::Float64 = value / 1.0e9

println(
    "Client time: ",
    round(ns_to_s(metrics.client_total_duration_ns); digits = 3),
    " s",
)

if !isnothing(metrics.time_to_first_token_ns)
    println(
        "Time to first token: ",
        round(ns_to_s(metrics.time_to_first_token_ns); digits = 3),
        " s",
    )
end

println(
    "Prompt throughput: ",
    round(prompt_rate(metrics); digits = 2),
    " tokens/s",
)

println(
    "Generation throughput: ",
    round(generation_rate(metrics); digits = 2),
    " tokens/s",
)
```

A reusable formatter:

```julia
function print_diagnostics(value)::Nothing
    metrics = last_response_metrics(value)

    if isnothing(metrics)
        println("No response metrics are available.")
        return nothing
    end

    ns_to_s(value::Integer)::Float64 = value / 1.0e9

    println("Server total:  ",
            round(ns_to_s(metrics.total_duration_ns); digits = 3), " s")
    println("Model load:    ",
            round(ns_to_s(metrics.load_duration_ns); digits = 3), " s")
    println("Client total:  ",
            round(ns_to_s(metrics.client_total_duration_ns); digits = 3), " s")
    println("Prompt tokens: ", metrics.prompt_eval_count)
    println("Output tokens: ", metrics.eval_count)
    println("Prompt rate:   ",
            round(prompt_rate(metrics); digits = 2), " tokens/s")
    println("Output rate:   ",
            round(generation_rate(metrics); digits = 2), " tokens/s")

    if isnothing(metrics.time_to_first_token_ns)
        println("First token:   unavailable")
    else
        println(
            "First token:   ",
            round(
                ns_to_s(metrics.time_to_first_token_ns);
                digits = 3,
            ),
            " s",
        )
    end

    return nothing
end
```

This works with both `ModelSession` and `PhysicsAssistant`:

```julia
print_diagnostics(session)
print_diagnostics(assistant)
```

---

## 17. Model capability inspection

Query a model directly:

```julia
caps = model_capabilities(
    "gpt-oss:20b";
    base_url = "http://localhost:11434",
)

capability_names(caps)
supports(caps, :completion)
supports(caps, :thinking)
supports(caps, :tools)
supports(caps, :vision)
supports(caps, :image_generation)
```

Require a capability:

```julia
require_capability(caps, :vision)
```

Inspect additional model metadata:

```julia
details = show_model_details(
    "gpt-oss:20b";
    verbose = true,
)

details.model
details.family
details.format
details.parameter_size
details.quantization_level
details.capabilities
```

Inspect capability discovery cached by a session:

```julia
caps = model_capabilities(session)
```

The value can be `nothing` when discovery was disabled or the metadata request
failed.

---

## 18. Persistence and resume

Save a session explicitly:

```julia
save(session)
```

Wait for background compression before continuing:

```julia
compression_running(session)
await_compression!(session)
save(session)
```

Close the session:

```julia
close(session)
```

Resume a generic session programmatically:

```julia
session = load_model_session(
    "gpt-oss:20b",
    "sessions/programmatic";
    temperature = 0.2,
    thinking = :high,
    tool_mode = TOOLS_OFF,
)
```

Resume a physics session:

```julia
assistant = load_assistant(
    "gpt-oss:20b",
    "sessions/scco-programmatic";
    mode = DERIVE,
    thinking = :high,
)
```

Equivalent alias:

```julia
assistant = load_physical_agent(
    "gpt-oss:20b",
    "sessions/scco-programmatic",
)
```

Inspect session identity and locations:

```julia
model_name(session)
base_url(session)
session_directory(session)
system_prompt(session)
artifact_store(session).root
sandbox_config(session).workspace
isopen(session)
```

---

## 19. Enter a terminal loop from a retained object

A programmatic session can enter the same interactive loop without losing the
object:

```julia
session = ModelSession(
    "gpt-oss:20b";
    session_dir = "sessions/retained",
)

repl(session; close_on_exit = false)

# Execution continues here after \exit or \quit.
save(session)
close(session)
```

Physics assistant:

```julia
assistant = PhysicsAssistant(
    "gpt-oss:20b";
    session_dir = "sessions/physics-retained",
)

repl(assistant; close_on_exit = false)

save(assistant)
close(assistant)
```

---

## 20. Low-level model lifecycle

The low-level client is available as `PhluxAI.OllamaClient`:

```julia
const OC = PhluxAI.OllamaClient
```

Preload a model:

```julia
OC.start_model!(
    "gpt-oss:20b";
    timeout = 180,
    poll_interval = 2.0,
)
```

Unload it:

```julia
OC.stop_model!("gpt-oss:20b")
```

Custom endpoint:

```julia
OC.start_model!(
    "gpt-oss:20b",
    "http://192.168.1.50:11434";
    timeout = 180,
    poll_interval = 2.0,
)

OC.stop_model!(
    "gpt-oss:20b",
    "http://192.168.1.50:11434",
)
```

---

## 21. Complete generic example

```julia
using PhluxAI

cfg = BudgetConfig(
    num_ctx = 32768,
    summary_budget = 1200,
    response_budget = 1024,
    safety_margin = 512,
    m_verbatim = 8,
    num_thread = 8,
    max_tokens = 1024,
)

session = ModelSession(
    "gpt-oss:20b";
    cfg,
    session_dir = "sessions/full-example",
    temperature = 0.2,
    thinking = :high,
    show_reasoning = false,
    persist_reasoning = false,
    keep_alive = "30m",
    request_timeout = 300.0,
    read_idle_timeout = 60.0,
    tool_mode = TOOLS_ASK,
    sandbox_backend = SANDBOX_AUTO,
    sandbox_network = false,
    sandbox_timeout = 120.0,
)

function approve_tool(call::ToolCall, risk::ToolRisk)::Bool
    println("Tool request: ", call.name, " [", risk, "]")
    return risk != TOOL_NETWORK
end

try
    response = respond!(
        session,
        """
        Write a Julia script that evaluates sin(x), creates a PNG plot,
        executes it through jupy, and returns the figure artifact.
        """;
        approval_callback = approve_tool,
        max_tool_steps = 8,
    )

    println("\nFinal text:")
    println(response.text)

    if has_reasoning(response)
        println("\nReasoning was returned separately.")
    end

    for execution in response.tool_executions
        println("\nTool: ", execution.name)
        println("Success: ", execution.success)
        println(execution.output)

        for artifact in execution.artifacts
            println("Generated: ", artifact.path)
        end
    end

    print_diagnostics(session)
    save(session)
finally
    close(session)
end
```

---

## 22. Complete multimodal physics example

```julia
using PhluxAI

assistant = PhysicsAssistant(
    "gpt-oss:20b";
    session_dir = "sessions/multimodal-physics",
    mode = DERIVE,
    thinking = :high,
    show_reasoning = false,
    tool_mode = TOOLS_OFF,
)

try
    scan = attach!(
        assistant,
        "data/detector_scan.png",
    )

    input = ModelInput(
        """
        Analyze the attached detector scan. Identify the dominant feature,
        state the assumptions used, and distinguish observation from
        interpretation.
        """,
        [scan],
    )

    response = respond!(
        assistant,
        input;
        mode = DERIVE,
        temperature = 0.1,
        max_tokens = 1024,
    )

    println(response.text)
    print_diagnostics(assistant)
    save(assistant)
finally
    close(assistant)
end
```

---

## 23. Operational notes

- `ask!` returns only final text. Use `respond!` for reasoning, artifacts,
  tool calls, tool results, and diagnostics.
- `@model` and `@agent` select the Ollama text/vision model. They do not imply a
  ComfyUI image checkpoint unless `image=...` is supplied or a pending
  `@imagemodel` selection exists.
- `@imagemodel` is one-shot and is consumed by the next `@model` or `@agent`.
- `@imagemodel` requires a default ComfyUI workflow with `bindings.model`.
- `\image PROMPT` uses the image backend already attached to the active
  session; it does not select a model on its own.
- A ComfyUI image-model filename and a ComfyUI workflow are separate. The model
  name is written into the workflow loader input identified by
  `bindings.model`.
- ComfyUI must be running before a `ComfyBackend` can generate images.
- Image-model weights, the ComfyUI checkout, `.venv`, and generated images are
  runtime/user data and should not be committed to the PhluxAI source
  repository.
- A workflow may be stored anywhere. Commit only small reference workflows that
  are intentionally part of project examples or tests.
- Tool execution is disabled by default.
- `TOOLS_AUTO` does not automatically approve network-capable package
  installation.
- Keep sandbox networking disabled unless a specific task requires it.
- Prefer `SANDBOX_BWRAP` or `SANDBOX_AUTO` on Linux.
- `SANDBOX_LOCAL` is not a security boundary.
- Image input requires a compatible Ollama vision model. Image output depends
  on the configured image backend and workflow.
- Named thinking levels require a model that understands them.
- Reasoning display and reasoning persistence are separate controls.
- Programmatic pending attachments are not added implicitly; construct a
  `ModelInput` with the desired artifacts.
- Call `close` in a `finally` block for programmatic sessions.
