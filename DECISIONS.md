# Implementation decisions

## DEC-2026-001: Provision self-contained parser wheels

**Date:** 2026-11-17
**Status:** Accepted
**Scope:** Parser runtime dependencies, CI provisioning, and source-analysis documentation.

### Context

The native parser tests passed, but the isolated HDF5 handlers failed. An import probe in the worker filesystem reproduced NumPy's `ImportError: libblas.so.3` on the Ubuntu 24.04 runner. The distribution library link depends on `/etc/alternatives`, outside the permitted runtime mounts.

### Decision

Provision the versions in `deps/analysis-requirements.txt` from binary wheels in the disposable system runtime. Install all dependencies into the visible `/usr` tree, ignoring host installations, and retain an isolated import probe before sandbox integration tests.

### Alternatives

Mounting `/etc/alternatives` would enlarge the worker filesystem. Binding selected host BLAS links would add distribution-specific library discovery to every worker launch. Self-contained wheels avoid both changes and preserve the existing authorization boundary.

### Evidence and reasoning

The isolated import probe in GitHub Actions run 44221771744 failed with the missing BLAS library. The local parser environment uses the selected NumPy, h5py, and pdfplumber versions and passes all ten parser scenarios.

### Consequences

Runtime images require compatible binary wheels. Provisioning fails if wheels are unavailable; CI does not fall back to a source build or broaden mounts. Direct parser versions are pinned; transitive packages remain resolved by pip. General-purpose host Python installations should not be overwritten.

### Validation

The parser scenarios pass locally. Hosted isolated-import and end-to-end worker validation are required on the resulting commit before closure.
