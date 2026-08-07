using Pkg

ENV["PHLUXAI_SKIP_OLLAMA_INSTALL"] = "1"
ENV["PHLUXAI_SKIP_COMFY_INSTALL"] = "1"
ENV["PHLUXAI_BANNER"] = "0"

if !haskey(ENV, "RESEAU_PRECOMPILE_ONLY")
    ENV["RESEAU_PRECOMPILE_ONLY"] = "eventloops,internal_poll,socket_ops,tcp,host_resolvers"
end

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using PhluxAI

const IS_CI = get(ENV, "CI", "false") == "true"
const GITHUB_REMOTE = Documenter.Remotes.GitHub("alt-f4-dev", "PhluxAI.jl")

DocMeta.setdocmeta!(
    PhluxAI,
    :DocTestSetup,
    :(using PhluxAI);
    recursive = true,
)

remote_options = if IS_CI
    (; repo = GITHUB_REMOTE)
else
    # A local checkout may not yet have a Git remote. Disabling remote links
    # keeps local documentation builds independent of GitHub configuration.
    (; repo = GITHUB_REMOTE, remotes = nothing)
end

makedocs(
    ;
    sitename = "PhluxAI.jl",
    modules = [PhluxAI],
    doctest = false,
    checkdocs = :none,
    linkcheck = false,
    format = Documenter.HTML(canonical = "https://alt-f4-dev.github.io/PhluxAI.jl/",
                             edit_link = IS_CI ? "main" : nothing),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started/basics.md",
        "Installation" => ["Build and external runtimes" => "installation/build.md",
                           "Ollama models" => "installation/ollama.md",
                           "ComfyUI models and workflows" => "installation/comfy.md"],
        "Reference" => "reference/api.md",
        "Development" => ["Documentation" => "development/documentation.md"],
    ],
    remote_options...,
)

if IS_CI
    deploydocs(repo = "github.com/alt-f4-dev/PhluxAI.jl.git", devbranch = "main")
end
