# PhluxAI.jl test profiles

The test runner inventories every test file matching `test/test_*.jl` through `suite_manifest.jl`. Each file is registered exactly once, and an inventory mismatch fails the run. The default profile is CPU-only; no live Ollama model or ComfyUI inference is intentionally started.

Run the default CPU and mock-HTTP suites from the repository root:

```bash
julia --project=. --startup-file=no test/runtests.jl
```

Set `PHLUXAI_TEST_PROFILE` to select another tier:

| Profile | Scope | Prerequisites |
|---|---|---|
| cpu | Unit, structural, and mock-HTTP tests | Julia and project dependencies |
| sandbox | Actual Bubblewrap filesystem/network confinement tests | Linux, bwrap, user namespaces, Python |
| live | Functional inference and image generation tests | Running Ollama/ComfyUI and the expected models |
| benchmark | Latency, context efficiency, VRAM/memory-policy measurements | Dedicated local inference hardware and model weights |
| all | Every suite, including live and benchmarks | All prerequisites; explicitly opt in |

For example, use `PHLUXAI_TEST_PROFILE=sandbox julia --project=. --startup-file=no test/runtests.jl` on a configured Linux workstation.

GitHub Actions runs the CPU profile on Julia 1.12 and 1.13. Sandbox, live, and benchmark profiles require separately provisioned environments and are not evidence of correctness merely because CPU CI passes.

When adding a `test_*.jl` file, add an entry in `suite_manifest.jl` and select the narrowest accurate tier. Never label a test CPU-only if it depends on a running model or a real GPU. Tests execute as independent Julia processes to avoid shared mutable state and module redefinition.
