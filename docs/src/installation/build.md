# PhluxAI.jl Build Process

This document describes the external-runtime build performed by
`deps/build.jl`, the available environment flags, generated files, default
installation paths, validation commands, and common failure modes.

The build configures four components:

1. Ollama
2. `jupy-cli`
3. `comfy-cli`
4. A local ComfyUI installation

Model weights and user workflows are deliberately excluded. The user selects
and downloads image models after the software stack has been configured, and
supplies an API-format ComfyUI workflow at runtime. The build installs the
runtime; it does not decide which image architecture or workflow the user will
run.

## Build entry point

PhluxAI uses `jupy` as the normal Julia launcher. Once Jupy is installed, run
the build from the PhluxAI repository root with:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.build(; verbose = true)'
```

`jupy` resolves the nearest `Project.toml`, launches Julia with that project
active, and binds PythonCall to the matching project-local `.venv`. The
`--startup-file=no` option is passed through to Julia. `verbose = true` sends
build output directly to the terminal. Without that option, Julia normally
writes build output to:

```text
deps/build.log
```

### First-build bootstrap exception

`jupy-cli` is itself installed by `deps/build.jl`. On a clean machine where
`jupy` does not exist yet, the first build must therefore be bootstrapped once
with Julia:

```bash
julia --project=. --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.build(; verbose = true)
'
```

After that succeeds, use `jupy` for PhluxAI Julia execution and `jupip` for
project Python package operations. Bare `julia`, `python`, and `pip` are not
the normal PhluxAI workflow after bootstrap.

The package itself is the build target. A source file is not a valid package
identifier, so this remains incorrect regardless of launcher:

```bash
jupy -e 'using Pkg; Pkg.build("src/PhluxAI.jl")'
```

An explicit package-name build is valid when `PhluxAI` resolves in the active
environment:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.build("PhluxAI"; verbose = true)'
```

## Prerequisites

The build expects:

- Julia with the PhluxAI project activated
- Git
- Python 3.10 or newer
- Python virtual-environment support
- Bash on Linux and macOS, or PowerShell on Windows
- Network access during the first build
- Sufficient disk space for ComfyUI, PyTorch, accelerator libraries, and
  future model files

On Debian, Ubuntu, or Linux Mint, Python may be installed while the `venv`
module is packaged separately. When virtual-environment creation fails, install
an appropriate package such as:

```bash
sudo apt install python3-venv
```

The first GPU-enabled build can download several gigabytes. A model download
adds further storage requirements and is not part of `Pkg.build`.

## Source files

The build is divided into three Julia files:

```text
deps/
├── build.jl
├── install_jupy.jl
└── install_comfy.jl
```

`build.jl` is the package entry point. It retains Ollama installation logic and
calls the two dedicated installers.

`install_jupy.jl` installs or validates user-local `jupy` and `jupip`
commands.

`install_comfy.jl` uses `jupip` to create the project-local Python
environment, installs `comfy-cli`, installs ComfyUI, configures local launch
settings, creates a stable `comfy` launcher, and synchronizes
`requirements.txt`.

## Build sequence

### 1. Ollama

The build first checks whether `ollama` is already on `PATH`.

When it is absent, the platform-specific behavior is:

| Platform | Installation method |
|---|---|
| Linux | Official Ollama shell installer downloaded with `curl` |
| WSL | Linux installer, followed by a reminder that `ollama serve` may need to be started manually |
| macOS | `brew install --cask ollama` |
| Windows | `winget install --id Ollama.Ollama -e --silent` |
| Other | Warning with the manual download location |

Ollama installation failures are reported as warnings by the current build
script. The remaining hybrid-runtime installation can still proceed.

### 2. jupy-cli

The Jupy installer first looks for valid `jupy` and `jupip` commands. Both
commands must:

- be present;
- report the same Jupy tool version; and
- use a `jupy_core.py` containing `JUPY_SKIP_JULIA_SETUP` support.

When validation fails, the installer retrieves the configured Jupy repository
at a pinned Git reference and runs its platform installer.

Current defaults:

```text
repository: https://github.com/alt-f4-dev/jupy-cli.git
reference:  ac6e2339fa49c2b5d390d83d5fbc213fe614a74c
```

The default non-Windows command locations are:

```text
~/.local/bin/jupy
~/.local/bin/jupip
```

The default shared Jupy installation root is:

```text
~/.local/share/jupy
```

On Windows, the installer uses the user-local Jupy paths under
`%LOCALAPPDATA%`.

### 3. Project-local Python environment

The Comfy installer invokes `jupip` from the PhluxAI project root. Jupy creates
or reuses:

```text
<PhluxAI project root>/.venv
```

PhluxAI sets this environment variable internally for build-time `jupip`
operations:

```text
JUPY_SKIP_JULIA_SETUP=1
```

This prevents `jupip` from starting a nested Julia package operation while
`Pkg.build` is already active. It does not change normal interactive `jupy` or
`jupip` behavior outside the build.

PhluxAI does not invoke bare `python`, `python3`, `pip`, or `pip3` commands for
this setup. Python package operations are routed through `jupip` and the
project-local environment.

### 4. comfy-cli

The default package specification is pinned:

```text
comfy-cli==1.15.0
```

The corresponding executable is expected at:

```text
Linux/macOS: <project>/.venv/bin/comfy
Windows:     <project>/.venv/Scripts/comfy.exe
```

When the exact pinned version is already installed, the installer reuses it.
Otherwise, it runs an upgraded installation through `jupip`.

### 5. ComfyUI

The default persistent data root is platform-specific:

| Platform | Data root |
|---|---|
| Linux | `${XDG_DATA_HOME:-$HOME/.local/share}/PhluxAI/comfy` |
| macOS | `~/Library/Application Support/PhluxAI/comfy` |
| Windows | `%LOCALAPPDATA%\PhluxAI\comfy` |

The actual ComfyUI Git checkout is stored below that root:

```text
<data root>/ComfyUI
```

The build passes `--version latest` by default. `comfy-cli` resolves `latest`
to a concrete ComfyUI tag at build time. Pin `PHLUXAI_COMFYUI_VERSION` when a
stable, repeatable ComfyUI version is required.

ComfyUI-Manager is skipped by default. Set
`PHLUXAI_COMFY_INSTALL_MANAGER=1` to include it.

The installer writes a marker file at:

```text
<data root>/.phluxai-comfy-environment
```

The marker records the project virtual environment, selected device class,
requested ComfyUI version, and manager setting. A later build reuses the
installation when the marker matches. A changed configuration causes a
ComfyUI dependency restore.

### 6. Accelerator selection

The default device mode is `auto`.

Automatic detection uses this order:

1. Apple Silicon on macOS
2. NVIDIA when `nvidia-smi` exists
3. AMD when `rocm-smi` or `rocminfo` exists
4. Intel Arc when `xpu-smi` or `sycl-ls` exists
5. CPU fallback

The accepted explicit values are:

```text
auto
nvidia
amd
m-series
intel-arc
cpu
```

The NVIDIA and AMD modes can additionally pass explicit CUDA or ROCm version
values to `comfy-cli`.

### 7. Local ComfyUI configuration

After installation, the build disables `comfy-cli` tracking, selects the local
workspace, and stores these default ComfyUI launch arguments:

```text
--listen 127.0.0.1
--port 8188
--disable-api-nodes
--disable-auto-launch
```

This binds the server to loopback instead of all interfaces and prevents the
browser from opening automatically.

`--disable-api-nodes` disables ComfyUI nodes that use external API services. It
does not disable the local ComfyUI HTTP endpoints used by PhluxAI, including
`/prompt`, `/history`, and `/view`.

The generated launcher also exports:

```text
VIRTUAL_ENV=<PhluxAI project root>/.venv
COMFY_NO_TELEMETRY=1
DO_NOT_TRACK=1
COMFY_WHERE=local
COMFY_LOCAL_URL=http://127.0.0.1:8188
```

### 8. User-facing comfy launcher

The build creates a stable launcher in the Jupy command directory.

Default non-Windows path:

```text
~/.local/bin/comfy
```

Default Windows path:

```text
%LOCALAPPDATA%\Jupy\bin\comfy.cmd
```

When an unrelated file already occupies that path, the installer preserves it
with a `.pre-phluxai` suffix before writing the PhluxAI-managed launcher.

Ensure the launcher directory is on `PATH`. A new terminal may be required
after the first installation.

### 9. requirements.txt synchronization

At the end of the build, the installer runs `jupip freeze` and atomically
rewrites:

```text
<PhluxAI project root>/requirements.txt
```

This file is a snapshot of the complete local Python environment, including
platform- and accelerator-specific packages. CUDA, ROCm, CPU, operating-system,
and Python-version differences mean it should not be treated as a universal
cross-platform lock file without additional curation.

### 10. What the build deliberately does not install

The build stops after the local software runtime is ready. It does **not**
download:

- Ollama text or vision model weights;
- ComfyUI checkpoints;
- LoRAs, VAEs, ControlNet weights, CLIP encoders, or other model assets;
- user ComfyUI workflows; or
- generated images.

This separation is intentional. Model weights and workflows are usage/runtime
choices, while `deps/build.jl` is responsible only for installing the software
needed to execute them.

A maintained example workflow may be committed to the repository under a path
such as `examples/workflows/`, but the build does not depend on such a location.
PhluxAI accepts API-format workflows from arbitrary filesystem paths.

## Image-generation runtime model

PhluxAI separates three pieces of image generation:

1. **ComfyUI runtime** — installed and configured by the build.
2. **Image-model files** — selected and downloaded by the user into the ComfyUI
   model directories.
3. **API-format workflow** — selected by the user and supplied to PhluxAI with
   `ComfyBindings`.

A model filename alone does not define a complete ComfyUI graph. At runtime,
PhluxAI writes the selected filename into the workflow input identified by
`bindings.model`.

For example:

```julia
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

The node IDs and input names are workflow-specific. The example above assumes
node `4` is the model loader and node `6` is the positive-prompt encoder.

The fully explicit form does not require a process-wide default workflow:

```julia
backend = ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)

@model "qwen3-vl:4b" image=backend
```

## Environment flags

Boolean flags use the same parsing rule throughout the installers. The
following values are false:

```text
unset
empty string
0
false
no
off
```

Comparison is case-insensitive and surrounding whitespace is ignored. Other
nonempty values are treated as true.

### Top-level build flags

| Variable | Default | Effect |
|---|---:|---|
| `PHLUXAI_SKIP_OLLAMA_INSTALL` | false | Skips Ollama detection and installation |
| `PHLUXAI_SKIP_COMFY_INSTALL` | false | Skips the complete hybrid-runtime stage, including both Jupy and Comfy installation |

The second variable name is historical: it skips Jupy as well as Comfy because
both are installed by the same hybrid-runtime function.

### Jupy installer flags

| Variable | Default | Effect |
|---|---|---|
| `PHLUXAI_JUPY_REPOSITORY` | `https://github.com/alt-f4-dev/jupy-cli.git` | Overrides the Jupy Git repository |
| `PHLUXAI_JUPY_REF` | pinned commit | Overrides the branch, tag, or commit fetched for Jupy |
| `JUPY_BIN_DIR` | `~/.local/bin` on non-Windows systems | Overrides Jupy command discovery and the generated `comfy` launcher directory |
| `JUPY_INSTALL_ROOT` | `${XDG_DATA_HOME:-~/.local/share}/jupy` | Overrides the expected Jupy shared installation root |
| `JUPY_CORE` | derived from the Jupy installation root | Points validation directly at a `jupy_core.py` file |

`JUPY_SKIP_JULIA_SETUP` is set internally by PhluxAI when calling `jupip`
during `Pkg.build`. Users normally do not need to set it manually.

### Comfy installer flags

| Variable | Default | Effect |
|---|---|---|
| `PHLUXAI_COMFY_CLI_SPEC` | `comfy-cli==1.15.0` | Python package specification installed through `jupip` |
| `PHLUXAI_COMFYUI_VERSION` | `latest` | ComfyUI tag or version passed to `comfy install --version` |
| `PHLUXAI_COMFY_WORKSPACE` | platform data root | Overrides the persistent PhluxAI Comfy data root; the repository is placed in its `ComfyUI` child directory |
| `PHLUXAI_COMFY_DEVICE` | `auto` | Selects `auto`, `nvidia`, `amd`, `m-series`, `intel-arc`, or `cpu` |
| `PHLUXAI_COMFY_CUDA_VERSION` | empty | Passed to `comfy install --cuda-version` in NVIDIA mode |
| `PHLUXAI_COMFY_ROCM_VERSION` | empty | Passed to `comfy install --rocm-version` in AMD mode |
| `PHLUXAI_COMFY_FAST_DEPS` | false | Adds `--fast-deps` to the ComfyUI installation command |
| `PHLUXAI_COMFY_INSTALL_MANAGER` | false | Installs ComfyUI-Manager instead of passing `--skip-manager` |

The CUDA and ROCm version strings are passed unchanged to `comfy-cli`. Check
`comfy install --help` for the accepted syntax in the installed CLI version.

## Common build examples

The examples in this section assume the one-time bootstrap has completed and
`jupy` is available.

### Default build

```bash
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

### Skip Ollama installation

Use this when Ollama is managed separately:

```bash
PHLUXAI_SKIP_OLLAMA_INSTALL=1 \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

### Skip all external runtime installation

This is useful for lightweight syntax or package CI jobs:

```bash
PHLUXAI_SKIP_OLLAMA_INSTALL=1 \
PHLUXAI_SKIP_COMFY_INSTALL=1 \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

### Force CPU installation

```bash
PHLUXAI_COMFY_DEVICE=cpu \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

### Force NVIDIA installation

```bash
PHLUXAI_COMFY_DEVICE=nvidia \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

### Pin ComfyUI to a tag

```bash
PHLUXAI_COMFYUI_VERSION=v0.30.2 \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

Replace the example tag with the version required by the project.

### Use a custom persistent workspace

```bash
PHLUXAI_COMFY_WORKSPACE="$HOME/AI/PhluxAI-comfy" \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

The resulting ComfyUI repository will be:

```text
$HOME/AI/PhluxAI-comfy/ComfyUI
```

### Install ComfyUI-Manager

```bash
PHLUXAI_COMFY_INSTALL_MANAGER=1 \
jupy --startup-file=no -e '
using Pkg
Pkg.build(; verbose = true)
'
```

Manager and custom-node installation expand the code trusted by the local
ComfyUI process. Review third-party nodes before installing them.

## Generated layout

A default Linux development installation produces a layout similar to:

```text
PhluxAI.jl/
├── .venv/
│   └── bin/
│       └── comfy
├── requirements.txt
└── deps/
    ├── build.jl
    ├── install_jupy.jl
    └── install_comfy.jl

~/.local/share/jupy/
~/.local/bin/jupy
~/.local/bin/jupip
~/.local/bin/comfy

~/.local/share/PhluxAI/comfy/
├── .phluxai-comfy-environment
└── ComfyUI/
    ├── .git/
    ├── main.py
    ├── models/
    │   ├── checkpoints/
    │   ├── loras/
    │   ├── vae/
    │   └── ...
    ├── input/
    ├── output/
    └── user/

# Optional user-managed workflow location; not required by PhluxAI:
~/.local/share/PhluxAI/workflows/
```

The `.venv` is created beside the `Project.toml` used by the package build. In
a development checkout, that is the repository root. In an installed package,
it is the installed PhluxAI package root selected by Julia's package manager.

Recommended ignore entries for a development repository include:

```gitignore
.venv/
LocalPreferences.toml
.CondaPkg/
```

Jupy creates a `.gitignore` only when none exists. It does not merge these
entries into an existing file.

## Post-build validation

### Commands and versions

```bash
jupy --jupy-tool-version
jupip --jupy-tool-version
jupip show comfy-cli
comfy --version
```

The two Jupy versions should match. `jupip show comfy-cli` and `comfy --version`
should describe the Python package and executable selected by
`PHLUXAI_COMFY_CLI_SPEC`.

### Python environment

The `comfy` launcher should resolve to the project environment:

```bash
command -v comfy
head -n 12 "$(command -v comfy)"
```

The generated launcher contains the marker:

```text
Generated by PhluxAI.jl
```

### ComfyUI Git checkout

On a default Linux installation:

```bash
git -C ~/.local/share/PhluxAI/comfy/ComfyUI status --short
git -C ~/.local/share/PhluxAI/comfy/ComfyUI remote -v
```

### Local server

Start ComfyUI in the background:

```bash
comfy launch --background
```

Check the local endpoint:

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

On Linux, confirm that the listener is loopback-only:

```bash
ss -ltnp | grep ':8188'
```

The address should be `127.0.0.1:8188`, not `0.0.0.0:8188` or
`[::]:8188`.

Stop the server:

```bash
comfy stop
```

### Image-model and workflow readiness

A successful build does not imply that image generation is ready: the user
must still install a compatible image model and provide an API-format workflow.

Check the checkpoint directory on a default Linux installation:

```bash
ls -lh ~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/
```

After configuring a `ComfyBackend`, verify the local API from Julia:

```julia
comfy_available(backend)
```

`true` confirms that the configured ComfyUI server responds. It does not by
itself prove that the selected model and workflow are mutually compatible; the
first image generation is the integration test for that combination.

## Installing image models

The build installs no model weights. Install a selected model after reviewing
its source, license, required workflow, and expected directory.

A checkpoint download typically uses:

```bash
comfy model download \
  --url <MODEL_URL> \
  --relative-path models/checkpoints
```

With the default Linux workspace, that resolves beneath:

```text
~/.local/share/PhluxAI/comfy/ComfyUI/models/checkpoints/
```

Other model classes may belong in different subdirectories, such as `vae`,
`loras`, `controlnet`, `clip`, or architecture-specific locations. Follow the
model and workflow documentation rather than assuming every file is a
checkpoint.

Model files are persistent runtime/user data. They should not be copied into or
committed with the PhluxAI source repository.

List model files through `comfy-cli` when appropriate, or inspect the selected
ComfyUI model directory directly.

## Installing or storing workflows

PhluxAI does not download a workflow during `Pkg.build`. Workflows are
configuration and are supplied at runtime.

Use an API-format workflow exported from ComfyUI. The file may live anywhere:

```text
~/.local/share/PhluxAI/workflows/my_workflow_api.json
~/AI/workflows/my_workflow_api.json
/path/to/project/examples/workflows/reference_api.json
```

Only intentionally maintained reference workflows should be committed to the
package repository. Personal workflows and generated workflow experiments are
runtime/user data.

PhluxAI validates the workflow when constructing a `ComfyBackend` or when
calling `set_default_image_workflow!`.

A named model selection requires a model binding:

```julia
bindings = ComfyBindings(
    positive_prompt = WorkflowBinding("6", "text"),
    model = WorkflowBinding("4", "ckpt_name"),
)
```

Additional bindings such as `negative_prompt`, `seed`, `width`, `height`,
`batch_size`, `steps`, `cfg_scale`, `sampler_name`, `scheduler`, and
`filename_prefix` are optional. A runtime parameter can be changed only when
the workflow exposes the corresponding binding.

Configure a process-local model-neutral workflow:

```julia
set_default_image_workflow!(
    "/path/to/workflow_api.json",
    bindings,
)
```

Then select the image model for the next session:

```julia
@imagemodel "sd_xl_base_1.0.safetensors"
@model "qwen3-vl:4b"
```

or select it inline:

```julia
@model "qwen3-vl:4b" image="sd_xl_base_1.0.safetensors"
```

The fully explicit backend form is:

```julia
@model "qwen3-vl:4b" image=ComfyBackend(
    "/path/to/workflow_api.json",
    bindings;
    model = "sd_xl_base_1.0.safetensors",
)
```

`@imagemodel` is one-shot and is consumed by the next `@model` or `@agent`.
The default workflow remains configured only for the current Julia process.
Neither selection is a build artifact.

## Repository hygiene

The package repository should contain the API and reproducible source inputs,
not installed runtimes or large usage artifacts.

Recommended separation:

| Content | Source repository? |
|---|---|
| `src/`, `deps/`, `test/`, documentation | Yes |
| Small maintained example workflows | Optional |
| `.venv/` | No |
| Generated `requirements.txt` freeze | Usually no; platform-specific snapshot |
| ComfyUI Git checkout | No |
| ComfyUI model weights | No |
| Ollama model weights | No |
| Personal workflows | No |
| Generated images and session artifacts | No |

At minimum, keep `.venv/` ignored. Generated sessions, artifacts, and local
workflow directories should also be ignored when they are created beneath a
development checkout.

## Network and privacy behavior

The build itself uses network access to retrieve:

- the Ollama installer when Ollama is absent;
- the pinned Jupy checkout when Jupy is absent or invalid;
- `comfy-cli` and Python dependencies;
- the ComfyUI Git repository and selected tag; and
- PyTorch and accelerator packages.

After installation, PhluxAI configures ComfyUI for local routing and loopback
binding. It does not run `comfy setup`, request a cloud login, or download
models automatically. User-supplied model URLs are separate network actions.

Telemetry opt-out variables are set in both the installer environment and the
generated `comfy` launcher.

## Legacy ComfyUI repository URL in logs

`comfy-cli` 1.15.0 may print this legacy source URL while cloning:

```text
https://github.com/comfyanonymous/ComfyUI
```

GitHub redirects that transferred repository to the canonical Comfy Org
location:

```text
https://github.com/Comfy-Org/ComfyUI
```

The clone is still the official ComfyUI repository. To remove reliance on the
redirect and make the configured origin explicit, normalize it after the
build:

```bash
git -C ~/.local/share/PhluxAI/comfy/ComfyUI \
  remote set-url origin \
  https://github.com/Comfy-Org/ComfyUI.git
```

Then verify:

```bash
git -C ~/.local/share/PhluxAI/comfy/ComfyUI remote -v
```

Adjust the path when `PHLUXAI_COMFY_WORKSPACE` is set.

## Rebuild and recovery behavior

The build is intended to be repeatable:

- existing Ollama is reused;
- valid Jupy commands are reused;
- an exact matching `comfy-cli` version is reused;
- a valid ComfyUI checkout with a matching marker is reused;
- a configuration change triggers dependency restoration; and
- `requirements.txt` is refreshed at the end.

The installer does not automatically delete a nonempty directory that occupies
the expected ComfyUI checkout path but lacks `main.py`. This prevents accidental
data loss. Inspect such a directory manually before moving or deleting it.

An interrupted first clone may leave a partial checkout. Inspect Git state
before cleanup:

```bash
git -C ~/.local/share/PhluxAI/comfy/ComfyUI status
```

Do not remove the persistent data root without first backing up model files,
workflows, input files, output files, and user configuration.

## Troubleshooting

### Build output is hidden

Re-run with `verbose = true` through Jupy:

```bash
jupy --startup-file=no -e 'using Pkg; Pkg.build(; verbose = true)'
```

Or inspect:

```text
deps/build.log
```

### `jupy`, `jupip`, or `comfy` is not found

Confirm that the user-local command directory is on `PATH`:

```bash
printf '%s\n' "$PATH" | tr ':' '\n' | grep "$HOME/.local/bin"
```

For a POSIX shell, a typical configuration is:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add it to the appropriate shell startup file, then open a new terminal.

### Virtual-environment creation fails

Ask Jupy to resolve the project and create or reuse the project environment:

```bash
JUPY_VERBOSE=1 jupy -e 'println(Base.active_project())'
```

When the failure specifically reports that Python's `venv` module is missing,
install the operating-system package that provides it (for example
`python3-venv` on Debian, Ubuntu, or Linux Mint), then rerun `jupy`. Python
package installation for PhluxAI should continue to use `jupip`, not bare
`pip`.

### The wrong accelerator was selected

Set `PHLUXAI_COMFY_DEVICE` explicitly and rebuild. Changing the device setting
changes the installation marker and triggers a dependency restore.

### ComfyUI installation fails after a network interruption

Check the checkout directory before rerunning the build. An empty invalid
`ComfyUI` directory is removed automatically; a nonempty invalid directory is
preserved and causes an error for safety.

### Existing `comfy` command was replaced

The installer does not discard an unrelated launcher. Look for a backup such
as:

```text
~/.local/bin/comfy.pre-phluxai
~/.local/bin/comfy.pre-phluxai.2
```

### CI should not install system runtimes

For package parsing, unit tests, or documentation jobs that deliberately skip
the Jupy/Comfy installation stage, bare Julia is appropriate because `jupy`
may not exist in the CI image. Skip both external installation stages:

```bash
PHLUXAI_SKIP_OLLAMA_INSTALL=1 \
PHLUXAI_SKIP_COMFY_INSTALL=1 \
julia --project=. --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.build(; verbose = true)
Pkg.test()
'
```

Dedicated integration jobs can provision and test the external runtimes
separately.

## Maintenance notes

Before publishing a new PhluxAI release, review and update as needed:

- the pinned Jupy Git reference;
- `DEFAULT_COMFY_CLI_SPEC`;
- compatibility between the selected ComfyUI release and workflows;
- compatibility of documented `ComfyBindings` with maintained example
  workflows;
- the `@imagemodel`, `image=...`, and explicit `ComfyBackend` user API;
- Python and accelerator requirements;
- local launch flags;
- generated-path behavior on all supported operating systems; and
- CI skip flags and integration coverage.

A `latest` ComfyUI installation is convenient for development but is not fully
reproducible. Release validation should use an explicit
`PHLUXAI_COMFYUI_VERSION` or change the source default to a tested tag.

## Upstream references

- Julia Pkg API: <https://pkgdocs.julialang.org/v1/api/>
- jupy-cli: <https://github.com/alt-f4-dev/jupy-cli>
- comfy-cli: <https://github.com/Comfy-Org/comfy-cli>
- comfy-cli on PyPI: <https://pypi.org/project/comfy-cli/>
- ComfyUI: <https://github.com/Comfy-Org/ComfyUI>
- Comfy CLI documentation: <https://docs.comfy.org/comfy-cli/getting-started>
- Comfy CLI reference: <https://docs.comfy.org/comfy-cli/reference>
