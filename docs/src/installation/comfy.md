# ComfyUI Models and Workflows for PhluxAI.jl

This guide covers the image-model and workflow installation required to use
PhluxAI.jl's local ComfyUI image-generation backend.

`Pkg.build("PhluxAI")` installs or validates `jupy-cli`, `comfy-cli`, and a
local ComfyUI runtime. It deliberately does **not** download image-model weights
or choose a workflow. Those remain explicit user choices.

PhluxAI separates three things that should not be confused:

1. the Ollama text/vision model selected by `@model` or `@agent`;
2. the ComfyUI image model selected by `@imagemodel` or `image=...`; and
3. the ComfyUI API-format workflow that defines how the image model is loaded,
   prompted, sampled, decoded, and saved.

A checkpoint filename alone is not a complete ComfyUI generation pipeline. The
workflow and model must be compatible.

## 1. Verify the ComfyUI runtime

After building PhluxAI, check the installed CLI and its Python package
through the PhluxAI/Jupy environment:

```bash
comfy --version
jupip show comfy-cli
```

The default PhluxAI installation configures a persistent local ComfyUI checkout
under:

```text
Linux:
~/.local/share/PhluxAI/comfy/ComfyUI
```

The path can differ when `PHLUXAI_COMFY_WORKSPACE` was set during the build.
See `BUILD.md` for all build flags and platform-specific paths.

Start the configured local server:

```bash
comfy launch --background
```

Check the local endpoint:

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

On Linux, verify that it is bound only to loopback:

```bash
ss -ltnp | grep ':8188'
```

Stop the server with:

```bash
comfy stop
```

The PhluxAI build configures local launch arguments including
`--listen 127.0.0.1`, `--port 8188`, `--disable-api-nodes`, and
`--disable-auto-launch`.

## 2. Understand ComfyUI model files

ComfyUI models are weight files stored under the ComfyUI `models/` tree.
Different workflows may require different classes of weights, for example:

```text
ComfyUI/models/checkpoints/
ComfyUI/models/vae/
ComfyUI/models/loras/
ComfyUI/models/controlnet/
ComfyUI/models/clip/
```

A simple checkpoint-based workflow may need only one `.safetensors` checkpoint.
More advanced workflows may also require a VAE, text encoder, LoRA, ControlNet,
upscaler, or architecture-specific components.

Always use the model author's or workflow author's directory instructions. Do
not assume that every downloaded file belongs in `models/checkpoints`.

Model weights are runtime/user data. They do **not** belong in the PhluxAI.jl
Git repository.

## 3. Choose a model and workflow together

Before downloading anything, determine:

- the model architecture or family;
- the exact model filename(s);
- the expected ComfyUI model directories;
- whether the workflow uses only ComfyUI core nodes or needs custom nodes;
- whether a separate VAE/text encoder is required;
- recommended resolution, sampler, scheduler, steps, and CFG values;
- model and workflow licenses; and
- expected VRAM/RAM requirements.

A workflow built for SDXL is not automatically suitable for Flux, SD 1.5,
SD3, or another architecture. The loader nodes and required model components
may differ.

For a first PhluxAI smoke test, a core-node checkpoint workflow is preferable
because it minimizes external dependencies.

## 4. Download a checkpoint with comfy-cli

The general command is:

```bash
comfy model download \
    --url <MODEL_URL> \
    --relative-path models/checkpoints
```

The Comfy CLI documentation defines `models/checkpoints` as the default
relative model-download path. Use a different `--relative-path` when the model
belongs to another ComfyUI model class.

List downloaded checkpoint models:

```bash
comfy model list --relative-path models/checkpoints
```

Remove a downloaded checkpoint when needed:

```bash
comfy model remove \
    --relative-path models/checkpoints \
    --model-names <MODEL_FILENAME>
```

## 5. Concrete first-model example: SDXL base 1.0

A straightforward first checkpoint is Stability AI's SDXL base model:

```text
sd_xl_base_1.0.safetensors
```

Official model page:

<https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>

The current official file page reports a size of approximately 6.94 GB and the
following SHA-256 digest:

```text
31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b
```

Download it through the PhluxAI-managed Comfy CLI:

```bash
comfy model download \
    --url https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors \
    --relative-path models/checkpoints
```

Verify that ComfyUI sees the file:

```bash
comfy model list --relative-path models/checkpoints
```

For a default Linux PhluxAI installation, the file should ultimately be under:

```text
~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/
```

Optionally verify the file digest directly:

```bash
sha256sum \
    ~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors
```

If a custom workspace was configured, adjust the path accordingly.

The SDXL refiner is not required for the first PhluxAI generation test. Start
with one base checkpoint and one simple workflow.

## 6. Obtain or create a workflow

PhluxAI requires an **API-format** ComfyUI workflow JSON for direct workflow
submission.

ComfyUI workflows are node graphs. A normal text-to-image checkpoint workflow
usually contains nodes that perform roles similar to:

```text
checkpoint/model loader
        │
        ├── positive prompt encoder
        ├── negative prompt encoder
        │
        └── sampler
             │
        latent image
             │
        VAE decode
             │
        save image
```

Node classes and graph structure depend on the model family.

### Use a built-in/template workflow

Start ComfyUI:

```bash
comfy launch --background
```

Open the local UI:

```text
http://127.0.0.1:8188
```

ComfyUI includes workflow templates accessible from its workflow/template UI.
Templates using only core nodes are a good starting point because they avoid
third-party custom-node dependencies.

Load or construct a workflow compatible with the model you downloaded, select
the exact model filename in its loader node, and run the workflow once in the
ComfyUI UI before integrating it with PhluxAI.

### Export API format

PhluxAI needs the API representation, not merely the regular editable UI
workflow JSON.

Use the ComfyUI frontend's **Save (API Format)** / API export option. In
versions where API export is hidden, enable the frontend's developer-mode/API
options first.

An API-format workflow is typically a JSON object whose node IDs are keys and
whose nodes contain entries such as:

```json
{
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
  }
}
```

PhluxAI uses those node IDs and input names to bind the prompt, model filename,
seed, dimensions, sampler settings, and other request values.

## 7. Where to store workflows

Personal workflows should normally live outside the PhluxAI source repository,
for example:

```text
~/.local/share/PhluxAI/workflows/
```

Example:

```bash
mkdir -p ~/.local/share/PhluxAI/workflows
cp /path/to/exported_sdxl_api.json \
    ~/.local/share/PhluxAI/workflows/sdxl_base_api.json
```

A workflow may be committed to the PhluxAI repository only when it is
intentionally maintained as a small package example or test fixture, such as:

```text
examples/workflows/sdxl_base_api.json
```

Do not commit model weights, generated images, caches, or a complete ComfyUI
installation.

## 8. Identify the workflow bindings

Open the API-format JSON and locate the node IDs and input names that PhluxAI
should control.

For example, a `CheckpointLoaderSimple` node may contain:

```json
"4": {
  "inputs": {
    "ckpt_name": "sd_xl_base_1.0.safetensors"
  },
  "class_type": "CheckpointLoaderSimple"
}
```

Its model binding would be:

```julia
WorkflowBinding("4", "ckpt_name")
```

A positive prompt encoder might be:

```json
"6": {
  "inputs": {
    "text": "a photograph of a mountain observatory"
  },
  "class_type": "CLIPTextEncode"
}
```

with:

```julia
WorkflowBinding("6", "text")
```

Node IDs are workflow-specific. Do not copy node IDs from another workflow
without checking the JSON you actually exported.

## 9. Configure `ComfyBindings`

Start the PhluxAI Julia environment through Jupy from the project/application
directory that contains PhluxAI:

```bash
jupy
```

Then configure the workflow in Julia. `positive_prompt` is required. The other
bindings are optional, except that `model` is required when a model filename
will be supplied through
`@imagemodel` or `image="..."`.

A typical checkpoint workflow configuration looks like:

```julia
using PhluxAI

bindings = ComfyBindings(
    positive_prompt = WorkflowBinding("<positive-node>", "text"),
    model = WorkflowBinding("<loader-node>", "ckpt_name"),
    negative_prompt = WorkflowBinding("<negative-node>", "text"),
    seed = WorkflowBinding("<sampler-node>", "seed"),
    width = WorkflowBinding("<latent-node>", "width"),
    height = WorkflowBinding("<latent-node>", "height"),
    batch_size = WorkflowBinding("<latent-node>", "batch_size"),
    steps = WorkflowBinding("<sampler-node>", "steps"),
    cfg_scale = WorkflowBinding("<sampler-node>", "cfg"),
    sampler_name = WorkflowBinding("<sampler-node>", "sampler_name"),
    scheduler = WorkflowBinding("<sampler-node>", "scheduler"),
    filename_prefix = WorkflowBinding("<save-node>", "filename_prefix"),
)
```

Only bind fields that actually exist in the workflow.

## 10. Configure the default PhluxAI image workflow

Point PhluxAI at the exported API workflow:

```julia
workflow = expanduser(
    "~/.local/share/PhluxAI/workflows/sdxl_base_api.json",
)

set_default_image_workflow!(workflow, bindings)
```

Inspect the configured default:

```julia
default_image_workflow()
```

Clear it when necessary:

```julia
clear_default_image_workflow!()
```

The default workflow is model-neutral. The selected image-model filename is
inserted into the loader input identified by `bindings.model` when an image
backend is created.

The default workflow setting is process-local Julia state. Configure it again
after starting a new Julia process, or place the configuration in the
application/startup code that creates the PhluxAI session.

## 11. Select the image model with `@imagemodel`

After configuring a default workflow:

```julia
@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

`@imagemodel` is a one-shot selection consumed by the next `@model` or
`@agent` call.

The two model names have different roles:

```text
qwen3-vl:4b
    Ollama text/vision model

sd_xl_base_1.0.safetensors
    ComfyUI image-generation model
```

Inside the interactive PhluxAI session:

```text
you> \image A small observatory on a snowy mountain beneath the Milky Way
```

The prompt is submitted to the configured ComfyUI workflow, not to the Ollama
model as an image-generation request.

## 12. Select the image model inline

The same model can be selected directly when creating the text session:

```julia
@model "qwen3-vl:4b" image="sd_xl_base_1.0.safetensors"
```

This also requires a default workflow configured by
`set_default_image_workflow!`.

The corresponding agent form is:

```julia
@agent :physics "qwen3-vl:4b" image="sd_xl_base_1.0.safetensors"
```

## 13. Use an explicit backend instead

For maximum control, bypass the default workflow state and construct the
backend explicitly:

```julia
backend = ComfyBackend(
    expanduser("~/.local/share/PhluxAI/workflows/sdxl_base_api.json"),
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)

@model "qwen3-vl:4b" image=backend
```

The explicit backend form is useful when a program needs multiple workflows or
needs to configure polling/timeouts independently.

A model filename supplied to `ComfyBackend(...; model=...)` requires a
`bindings.model` entry.

## 14. Programmatic image generation

A retained session with an attached image backend can generate images through:

```julia
response = generate_images!(
    session,
    "A publication-style schematic of a two-leg spin ladder";
    width = 1024,
    height = 1024,
    n = 1,
)
```

Inspect generated artifacts:

```julia
for image in response_images(response)
    println(image.path)
    println(image.mime_type)
    println(image.sha256)
end
```

PhluxAI retrieves ComfyUI outputs and imports them into the session artifact
store. Generated artifacts are separate from the ComfyUI model files and
should not be committed accidentally.

## 15. Validate the complete pipeline

Before diagnosing PhluxAI, validate each layer independently.

### Check the model exists

```bash
comfy model list --relative-path models/checkpoints
```

### Check ComfyUI starts

```bash
comfy launch --background
curl -fsS http://127.0.0.1:8188/system_stats
```

### Check the workflow in ComfyUI itself

Load the editable workflow in the local ComfyUI UI, select the downloaded
model, and generate one image successfully.

### Check the API-format workflow

Confirm that the exported JSON contains the node IDs and inputs referenced by
`ComfyBindings`.

### Check PhluxAI configuration

```julia
default_image_workflow()
```

Then run:

```julia
@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

and issue a `\image` command.

## 16. Multiple model files and advanced workflows

Some model families do not use a single checkpoint loader. A workflow may
require separate model, CLIP/text-encoder, VAE, or other files. In that case:

1. download every required file to its documented ComfyUI model directory;
2. validate the workflow directly in ComfyUI;
3. export the working graph in API format;
4. bind the input that should receive the user-selected model name; and
5. leave the other required component names hard-coded in the workflow unless
   PhluxAI needs to vary them too.

The current `@imagemodel` API selects one explicit model string through
`bindings.model`. It does not automatically infer or install companion model
files.

## 17. Custom nodes

A downloaded workflow may reference custom nodes that are not included in the
base ComfyUI installation. The default PhluxAI build skips ComfyUI-Manager.

For the first image-generation test, prefer a workflow using only core nodes.
Third-party custom nodes expand the code trusted by the local ComfyUI process
and should be reviewed before installation.

If a workflow requires custom nodes, install and test those separately before
using the workflow through PhluxAI.

## 18. Model and workflow updates

Model filenames are part of the PhluxAI configuration. If the filename changes,
update the `@imagemodel`, inline `image=`, or `ComfyBackend(...; model=...)`
value accordingly.

If workflow node IDs or input names change, update `ComfyBindings` to match the
new API-format JSON.

A workflow update can therefore require no PhluxAI source change at all; it may
only require a new JSON file and bindings configuration.

## 19. Troubleshooting

### `comfy: command not found`

Check that the user-local Jupy command directory is on `PATH`. On a default
Linux installation:

```bash
command -v comfy
printf '%s\n' "$PATH" | tr ':' '\n' | grep "$HOME/.local/bin"
```

See `BUILD.md` for launcher and build details.

### ComfyUI cannot find the model

Confirm the filename and directory:

```bash
comfy model list --relative-path models/checkpoints
```

The filename supplied by `@imagemodel` must match the filename expected by the
workflow loader.

### `@imagemodel` reports that no default workflow is configured

Configure one first:

```julia
set_default_image_workflow!(workflow_path, bindings)
```

### PhluxAI reports that model selection requires `bindings.model`

Add the loader input to `ComfyBindings`, for example:

```julia
model = WorkflowBinding("4", "ckpt_name")
```

Use the actual node ID and input name from your workflow.

### The workflow runs in the UI but fails through PhluxAI

Confirm that the file supplied to PhluxAI is the API-format export rather than
the normal editable UI workflow JSON.

Then compare every bound node ID and input name against the API JSON.

### A downloaded workflow has missing nodes

The workflow depends on custom nodes. Install and validate those dependencies
separately, or choose a core-node workflow for the initial smoke test.

### The server is listening on all network interfaces

Stop it and launch through the PhluxAI-configured `comfy` launcher. The build
is intended to bind ComfyUI to `127.0.0.1:8188`.

## 20. Repository hygiene

Keep the package source and runtime data separate.

Recommended separation:

```text
PhluxAI.jl repository
├── src/
├── deps/
├── test/
├── README.md
├── BASICS.md
├── BUILD.md
├── OLLAMA.md
├── COMFY.md
└── examples/workflows/       # optional curated small workflows only

User/runtime data
~/.local/share/PhluxAI/comfy/ComfyUI/
    models/
    input/
    output/
    user/

~/.local/share/PhluxAI/workflows/
    personal API workflows
```

Do not commit:

- checkpoint/model weights;
- ComfyUI itself;
- `.venv`;
- generated images;
- caches;
- personal runtime configuration; or
- large user workflow collections.

Small, intentionally maintained example workflow JSON files may be committed
because workflows are configuration, not model weights.

## 21. Minimal first-image checklist

1. Build PhluxAI and verify `comfy --version`.
2. Download one compatible checkpoint.
3. Start ComfyUI locally.
4. Create or load a core-node workflow compatible with that checkpoint.
5. Run one image successfully in the ComfyUI UI.
6. Export the workflow in API format.
7. Store the API JSON outside the PhluxAI repository unless it is an intentional
   example.
8. Start the Julia environment with `jupy`.
9. Create `ComfyBindings` from the exported node IDs.
10. Call `set_default_image_workflow!`.
11. Select the checkpoint with `@imagemodel` or inline `image=...`.
12. Enter `\image <prompt>` in the PhluxAI interactive session.
13. Inspect the generated PhluxAI artifact.

## Upstream references

- ComfyUI model concepts: <https://docs.comfy.org/development/core-concepts/models>
- ComfyUI workflow concepts: <https://docs.comfy.org/development/core-concepts/workflow>
- Comfy CLI getting started: <https://docs.comfy.org/comfy-cli/getting-started>
- Comfy CLI reference: <https://docs.comfy.org/comfy-cli/reference>
- ComfyUI repository: <https://github.com/Comfy-Org/ComfyUI>
- Comfy CLI repository: <https://github.com/Comfy-Org/comfy-cli>
- SDXL base 1.0 model: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
