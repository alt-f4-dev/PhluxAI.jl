# Documentation development

PhluxAI uses [Documenter.jl](https://documenter.juliadocs.org/stable/) to
build the manual under `docs/`.

## Build locally with Jupy

Run the docs build from the **repository root**:

```bash
PHLUXAI_SKIP_OLLAMA_INSTALL=1 \
PHLUXAI_SKIP_COMFY_INSTALL=1 \
PHLUXAI_BANNER=0 \
jupy docs/make.jl
```

Running from the repository root is important. Jupy discovers the root PhluxAI
project and reuses its existing `.venv`; running `jupy` from inside `docs/`
would create a second documentation-local `.venv`.

`docs/make.jl` activates and instantiates the documentation Julia environment
before loading Documenter and PhluxAI. Generated HTML is written to
`docs/build/`.

## GitHub Pages

`.github/workflows/documentation.yml` builds the manual on pull requests and
deploys from pushes to `main` and version tags. The documentation CI job
deliberately skips external inference runtime installation and therefore uses
the Julia executable supplied by GitHub Actions directly.

After the first successful deployment, configure GitHub Pages to serve the
`gh-pages` branch at `/ (root)` if GitHub has not selected it automatically.

Expected site URL:

```text
https://alt-f4-dev.github.io/PhluxAI.jl/
```

## Adding pages

Add Markdown under `docs/src/`, add the page to `pages` in `docs/make.jl`, and
build locally before pushing. Documenter validates local links, so renamed or
moved pages should have references updated in the same commit.

## API reference

`reference/api.md` uses `@autodocs` with `Private = false`, so public Julia
docstrings become part of the website automatically. Runtime-dependent examples
should remain ordinary fenced `julia` blocks unless they are deterministic and
suitable for doctesting.
