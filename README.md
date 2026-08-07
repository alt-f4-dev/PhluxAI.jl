# PhluxAI.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alt-f4-dev.github.io/PhluxAI.jl/dev/)

PhluxAI.jl is a local multimodal agent framework for Julia that combines:

- Ollama for text and vision models;
- ComfyUI for local image generation;
- `jupy` and `jupip` for Julia/Python tool execution in a shared
  project-local Python environment;
- stateful model and agent sessions with persistence, reasoning fields,
  artifacts, multimodal inputs, and sandboxed tool execution.

The package provides both interactive macros such as `@model`, `@agent`, and
`@imagemodel`, and programmatic APIs such as `ModelSession`,
`PhysicsAssistant`, `respond!`, `generate_images!`, and `ComfyBackend`.

Current development version: `0.0.1`.

## Architecture

PhluxAI keeps the text/vision model and image-generation model separate.

| Role | Runtime | Typical selection |
|---|---|---|
| Text and reasoning | Ollama | `@model "gpt-oss:20b"` |
| Vision understanding | Ollama | `@model "qwen3-vl:4b"` |
| Image generation | ComfyUI | `@imagemodel "model.safetensors"` |
| Julia/Python tools | Jupy | `jupy`, `jupip` |

An Ollama model name identifies the model used for conversation. A ComfyUI
image-model filename identifies a model input inside an explicit ComfyUI API
workflow. The workflow remains explicit because it defines how the model is
loaded and sampled.

## Installation

### Prerequisites

The build expects:

- Julia;
- Git;
- Python 3.10 or newer with virtual-environment support;
- network access for the initial runtime installation;
- enough storage for ComfyUI, PyTorch/accelerator packages, and any models the
  user later chooses to download.

On Debian, Ubuntu, and Linux Mint, Python virtual-environment support may be a
separate system package:

```bash
sudo apt install python3-venv
```

### Install from GitHub

PhluxAI uses `jupy` as its normal Julia launcher and `jupip` as its Python
package manager. Run PhluxAI commands from the intended Julia project
directory so `jupy` resolves the correct `Project.toml`.

When `jupy` is already installed, install PhluxAI from GitHub with:

```bash
jupy -e 'using Pkg; Pkg.add(url = "https://github.com/alt-f4-dev/PhluxAI.jl")'
```

A first installation on a machine without `jupy` has one bootstrap exception:
PhluxAI's build installs `jupy-cli`, so the initial package installation must
be started with Julia itself:

```bash
julia --project=. -e 'using Pkg; Pkg.add(url = "https://github.com/alt-f4-dev/PhluxAI.jl")'
```

After that build succeeds, use `jupy` for normal Julia execution and `jupip`
for Python package operations associated with the project. To rerun the
external-runtime build:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.build("PhluxAI"; verbose = true)'
```

The build configures the software runtime but deliberately does not download
LLM or image-model weights.

### Development checkout

```bash
git clone https://github.com/alt-f4-dev/PhluxAI.jl.git
cd PhluxAI.jl
```

When `jupy` is already available, use it for the development environment:

```bash
jupy --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.build(; verbose = true)
Pkg.precompile()
'
```

On a clean machine where `jupy` has not yet been installed, bootstrap the first
build once with Julia:

```bash
julia --project=. --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.build(; verbose = true)
'
```

Then use `jupy` for subsequent precompilation, tests, scripts, and REPL use:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.precompile()'
```

The build performs four external-runtime steps:

1. install or detect Ollama;
2. install or validate user-local `jupy-cli`;
3. create or reuse the project-local `.venv` and install `comfy-cli`;
4. install and configure a persistent local ComfyUI workspace.

See [BUILD.md](BUILD.md) for build flags, paths, accelerator selection,
recovery behavior, and CI guidance.

## Verify the installed runtime

```bash
ollama --version
jupy --jupy-tool-version
jupip --jupy-tool-version
jupip show comfy-cli
comfy --version
```

For image generation, start the locally configured ComfyUI server:

```bash
comfy launch --background
```

The default PhluxAI configuration binds ComfyUI to:

```text
http://127.0.0.1:8188
```

Check it with:

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

Stop it with:

```bash
comfy stop
```

## Install models

### Ollama text or vision model

PhluxAI does not choose or download an Ollama model automatically. For
example:

```bash
ollama pull gpt-oss:20b
```

or for a vision-capable model:

```bash
ollama pull qwen3-vl:4b
```

### ComfyUI image model

Image-model weights are also user-selected. A checkpoint can be downloaded
through the configured `comfy` command:

```bash
comfy model download \
  --url <MODEL_URL> \
  --relative-path models/checkpoints
```

With the default Linux workspace, checkpoint files normally live under:

```text
~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/
```

Model weights do **not** belong in the PhluxAI source repository.

## Quick start: text session

From the project or application directory containing PhluxAI, start Julia
through Jupy:

```bash
jupy
```

Then:

```julia
using PhluxAI

@model "gpt-oss:20b"
```

Start the physics assistant profile:

```julia
@agent :physics "gpt-oss:20b"
```

A persistent session directory can be supplied as the second positional
argument:

```julia
@agent :physics "gpt-oss:20b" "sessions/research"
```

Resume it later:

```julia
@resume "gpt-oss:20b" "sessions/research"
```

## Quick start: ComfyUI image generation

PhluxAI requires an API-format ComfyUI workflow and bindings that identify the
workflow inputs PhluxAI is allowed to change.

For example, for a workflow whose model loader is node `4` and positive prompt
encoder is node `6`:

```julia
using PhluxAI

bindings = ComfyBindings(
    positive_prompt = WorkflowBinding("6", "text"),
    model = WorkflowBinding("4", "ckpt_name"),
)

set_default_image_workflow!(
    "/path/to/workflow_api.json",
    bindings,
)
```

The node IDs and input names are workflow-specific. They must match the
API-format JSON being used.

Select the image model for the next interactive model or agent session:

```julia
@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

Inside the session:

```text
you> \image A moonlit observatory on a snowy mountain, realistic photography
```

The same image model can be selected inline:

```julia
@model "qwen3-vl:4b" image="sd_xl_base_1.0.safetensors"
```

or through a fully explicit backend:

```julia
backend = ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)

@model "qwen3-vl:4b" image=backend
```

`@imagemodel` is one-shot: it is consumed by the next `@model` or `@agent`.
An explicit `image=...` argument takes precedence. The default workflow stays
configured for the current Julia process until it is replaced or cleared.

## Programmatic image generation

```julia
using PhluxAI

backend = ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)

session = ModelSession(
    "qwen3-vl:4b";
    image_backend = backend,
)

response = generate_images!(
    session,
    "A publication-style schematic of a two-leg spin ladder";
    width = 1024,
    height = 1024,
    steps = 30,
    seed = 1234,
)

for image in response_images(response)
    println(image.path)
end
```

## Repository source versus runtime data

The Git repository should contain source code, tests, documentation, and any
small example workflows that are intentionally maintained as examples. It
should not contain installed runtimes, model weights, or generated artifacts.

| Content | Typical location | Commit to repository? |
|---|---|---|
| Julia package source | `src/` | Yes |
| Build installers | `deps/` | Yes |
| Tests | `test/` | Yes |
| Documentation | `README.md`, `BASICS.md`, `BUILD.md` | Yes |
| Maintained example workflow | `examples/workflows/` | Optional |
| Project Python environment | `.venv/` | No |
| Frozen Python snapshot | `requirements.txt` | Usually no; generated and platform-specific |
| ComfyUI checkout | user data directory | No |
| Image-model weights | ComfyUI `models/` | No |
| Ollama model weights | Ollama-managed storage | No |
| Generated images | session/ComfyUI runtime data | No |
| User workflows | user-selected location | No |

The build-generated `requirements.txt` is a freeze of the local Python
environment, including platform- and accelerator-specific packages. Treat it
as a diagnostic/reproducibility snapshot rather than a universal
cross-platform lock file unless the project deliberately curates it.

A workflow is configuration, not a model weight. PhluxAI accepts a workflow
from any filesystem location. A project may commit small reference workflows
under `examples/`, but user workflows do not need to live in the package
repository.

## Locality and network behavior

The initial build uses network access to obtain external software and Python
packages. Model downloads are explicit user actions.

The configured ComfyUI server is local by default. PhluxAI uses the local HTTP
endpoints `/prompt`, `/history/{prompt_id}`, and `/view`, and its Comfy backend
rejects non-loopback URLs when `loopback_only=true`.

ComfyUI API-service nodes are disabled by the default launch configuration,
and `comfy-cli` telemetry is disabled by the installer.

## Documentation

- [BASICS.md](BASICS.md) — interactive and programmatic API usage.
- [BUILD.md](BUILD.md) — external-runtime build, flags, paths, and recovery.
- [OLLAMA.md](OLLAMA.md) — Ollama model installation and validation.
- [COMFY.md](COMFY.md) — ComfyUI model and workflow installation.
