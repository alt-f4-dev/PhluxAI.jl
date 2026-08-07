# Ollama Models for PhluxAI.jl

This guide covers the model-download and verification steps required to use
Ollama-backed text and vision models with PhluxAI.jl.

`Pkg.build("PhluxAI")` installs or validates the Ollama runtime, but PhluxAI
deliberately does **not** choose or download Ollama model weights. Model
selection remains a user decision because model size, capabilities, license,
hardware requirements, and intended use vary substantially.

For image **generation**, use the ComfyUI backend described in `COMFY.md`.
Ollama vision models are used for understanding images supplied to a text
session; they are not the ComfyUI image-generation model selected by
`@imagemodel`.

## 1. Verify the Ollama runtime

After installing or building PhluxAI, confirm that Ollama is available:

```bash
ollama --version
```

List models already installed on the machine:

```bash
ollama ls
```

If the local Ollama service is not already running, start it with:

```bash
ollama serve
```

PhluxAI communicates with the local Ollama HTTP service. A normal local
installation uses Ollama's default local endpoint on port `11434`.

## 2. Choose a model

Browse the Ollama model library before downloading a model:

<https://ollama.com/search>

Check at least the following before pulling a model:

- the exact model name and tag;
- whether the model supports text, vision, tools, or thinking as required;
- model size and expected RAM/VRAM use;
- context-window information;
- the model license; and
- any minimum Ollama-version requirement listed on the model page.

PhluxAI passes the Ollama model name directly to Ollama. There is no PhluxAI
alias or preset layer between the string used in `@model`/`@agent` and the
locally installed Ollama model.

### Example text/reasoning model

A text model can be installed with:

```bash
ollama pull gpt-oss:20b
```

The corresponding PhluxAI call is:

```julia
using PhluxAI

@model "gpt-oss:20b"
```

### Example vision model

A vision-capable model can be installed with:

```bash
ollama pull qwen3-vl:4b
```

Then use the same exact model name in PhluxAI:

```julia
using PhluxAI

@model "qwen3-vl:4b"
```

The Ollama library currently identifies `qwen3-vl:4b` as a text-and-image
model. Always check the model page before installation because available tags
and requirements can change.

## 3. Download the selected model

The general command is:

```bash
ollama pull <model-name[:tag]>
```

Examples:

```bash
ollama pull gpt-oss:20b
ollama pull qwen3-vl:4b
```

The download can be large. Ollama manages its own model storage; the model
files should **not** be copied into the PhluxAI.jl repository.

Do not commit Ollama model weights, caches, or Ollama runtime data to the
PhluxAI repository.

## 4. Verify the download

List installed models:

```bash
ollama ls
```

Run the model directly before testing PhluxAI:

```bash
ollama run gpt-oss:20b
```

or:

```bash
ollama run qwen3-vl:4b
```

A direct Ollama smoke test is useful because it separates model-download or
runtime problems from PhluxAI API problems.

Exit the Ollama interactive session when finished.

## 5. Use the model from PhluxAI

Start Julia through Jupy from the environment that contains PhluxAI:

```bash
jupy
```

Run `jupy` from the intended project directory so it resolves the correct
`Project.toml` and project-local `.venv`.

Then:

```julia
using PhluxAI
```

### Interactive generic session

```julia
@model "gpt-oss:20b"
```

### Interactive physics agent

```julia
@agent :physics "gpt-oss:20b"
```

### Vision-capable session

```julia
@model "qwen3-vl:4b"
```

Inside the interactive session, attach an image and then ask about it:

```text
you> \attach data/detector_scan.png
you> Identify the dominant features in this scan.
```

The selected Ollama model must support image input for this to work.

## 6. Check model capabilities from PhluxAI

PhluxAI can query Ollama model metadata:

```julia
caps = model_capabilities("qwen3-vl:4b")

capability_names(caps)
supports(caps, :completion)
supports(caps, :thinking)
supports(caps, :tools)
supports(caps, :vision)
```

For a session that performed capability discovery:

```julia
session = ModelSession("qwen3-vl:4b")
caps = model_capabilities(session)
```

Capability discovery is useful for validating a model before relying on
vision, thinking, or tool support.

## 7. Model names must match exactly

PhluxAI does not automatically translate model names. If Ollama lists:

```text
qwen3-vl:4b
```

then use:

```julia
@model "qwen3-vl:4b"
```

Do not substitute a filename, Hugging Face repository name, or ComfyUI
checkpoint name. Ollama and ComfyUI use separate model namespaces.

## 8. Keep local and cloud model tags distinct

PhluxAI is designed primarily around local runtimes. Some Ollama library
entries may offer cloud-tagged variants. A cloud model is not equivalent to a
fully local model even though it may be selected through Ollama.

For a local-only workflow, choose a model tag whose weights execute locally and
avoid tags explicitly identified as cloud models.

## 9. Update or replace a model

Pulling the same model name again lets Ollama retrieve the current version
associated with that tag:

```bash
ollama pull qwen3-vl:4b
```

After an update, perform the direct Ollama smoke test again before diagnosing a
PhluxAI regression.

To remove a model:

```bash
ollama rm <model-name[:tag]>
```

Example:

```bash
ollama rm qwen3-vl:4b
```

## 10. Inspect running models

List models currently loaded by Ollama:

```bash
ollama ps
```

Unload a model explicitly:

```bash
ollama stop <model-name[:tag]>
```

PhluxAI sessions can also manage model residency through their `keep_alive`,
`start_model`, and `stop_model_on_close` controls.

## 11. A minimal first-use sequence

For a text model:

```bash
ollama pull gpt-oss:20b
ollama ls
ollama run gpt-oss:20b
```

Then:

```julia
using PhluxAI
@model "gpt-oss:20b"
```

For a vision model:

```bash
ollama pull qwen3-vl:4b
ollama ls
ollama run qwen3-vl:4b
```

Then:

```julia
using PhluxAI
@model "qwen3-vl:4b"
```

## 12. Troubleshooting

### `ollama: command not found`

When Jupy is installed, rerun the PhluxAI build with visible output through
the default launcher:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.build(; verbose = true)'
```

If `jupy` itself is missing because this is the first bootstrap, use the
one-time Julia bootstrap command documented in `BUILD.md`.

See `BUILD.md` for the build flags and platform-specific installation behavior.

### The model is not found

Check the exact installed name:

```bash
ollama ls
```

Then use that exact string in `@model`, `@agent`, or `ModelSession`.

### Ollama is installed but PhluxAI cannot connect

Check whether the server is running:

```bash
ollama ps
```

If necessary, start it:

```bash
ollama serve
```

Then retry the PhluxAI session.

### Vision input fails

Confirm that the selected model is explicitly described as vision/image-input
capable in the Ollama model library and inspect its PhluxAI capabilities:

```julia
caps = model_capabilities("<model-name>")
supports(caps, :vision)
```

### The model does not fit in memory

Select a smaller model or a smaller quantized/tagged variant. Model size and
runtime memory requirements are model-specific; consult the model page before
downloading.

## 13. Repository hygiene

Ollama models are runtime data, not PhluxAI source. A PhluxAI Git repository
should contain package code, tests, documentation, and small examples—not
Ollama model weights.

The intended separation is:

```text
PhluxAI.jl repository
    source code
    tests
    documentation
    optional small example files

Ollama runtime
    downloaded text/vision model weights
    model manifests and caches
```

## Upstream references

- Ollama CLI reference: <https://docs.ollama.com/cli>
- Ollama pull API: <https://docs.ollama.com/api/pull>
- Ollama model library: <https://ollama.com/search>
- `gpt-oss:20b`: <https://ollama.com/library/gpt-oss:20b>
- `qwen3-vl:4b`: <https://ollama.com/library/qwen3-vl:4b>
