# PhluxAI.jl

PhluxAI.jl is a local multimodal agent framework for Julia. It combines
Ollama-backed text and vision sessions, ComfyUI image generation, Jupy-managed
Julia/Python tooling, persistent artifacts and sessions, reasoning fields, and
sandboxed model-requested tools.

The package deliberately separates conversational models from image-generation
models:

| Role | Runtime | Selection |
|---|---|---|
| Text and reasoning | Ollama | `@model "gpt-oss:20b"` |
| Vision understanding | Ollama | `@model "qwen3-vl:4b"` |
| Image generation | ComfyUI | `@imagemodel "model.safetensors"` |
| Julia/Python tooling | Jupy | `jupy`, `jupip` |

## Start here

1. [Build and external runtimes](installation/build.md)
2. [Ollama models](installation/ollama.md)
3. [ComfyUI models and workflows](installation/comfy.md)
4. [PhluxAI basics](getting-started/basics.md)

After the initial build has installed Jupy, the normal Julia entry point for a
PhluxAI project is:

```bash
jupy
```

Use `jupip` for Python package operations associated with the same project.

## Quick text session

```julia
using PhluxAI

@model "gpt-oss:20b"
```

For the physics profile:

```julia
@agent :physics "gpt-oss:20b"
```

## Quick image-enabled session

After installing a compatible ComfyUI model and exporting an API-format
workflow:

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

@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

Then, inside the interactive session:

```text
you> \image A moonlit observatory on a snowy mountain
```

## Manual

```@contents
Pages = [
    "getting-started/basics.md",
    "installation/build.md",
    "installation/ollama.md",
    "installation/comfy.md",
    "reference/api.md",
    "development/documentation.md",
]
Depth = 2
```

## Source and runtime data

PhluxAI source, tests, documentation, and intentionally maintained small
example workflows belong in the Git repository. Ollama weights, ComfyUI model
weights, ComfyUI itself, `.venv`, generated images, personal workflows, and
session artifacts are runtime/user data and should remain outside source
control.

The repository is hosted at
[github.com/alt-f4-dev/PhluxAI.jl](https://github.com/alt-f4-dev/PhluxAI.jl).
