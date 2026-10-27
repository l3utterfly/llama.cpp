# Layla Windows runtime artifacts

The **Layla Windows runtime** workflow builds Windows x64 binaries from this fork's `layla-build` branch on GitHub-hosted `windows-2022` runners. It runs when runtime sources or packaging recipes change, and can also be started from **Actions -> Layla Windows runtime -> Run workflow** with `layla-build` selected. No self-hosted runner or GPU is required to compile the artifacts.

The six independent jobs build the CPU runtime, CUDA 12.4, CUDA 13.4, Vulkan, ROCm, and SYCL backends. CPU builds include portable instruction-set variants. CUDA uses GGML's portable architecture list rather than detecting the runner's GPU. SDK versions are pinned in `.github/workflows/layla-windows-runtime.yml` and follow the existing Windows release recipes.

## Download

Open a successful workflow run and download **layla-windows-runtime-x64-<commit SHA>** under **Artifacts**. Extract it as one directory. It contains:

```text
llama-server.exe
llama-server-impl.dll
llama-common.dll
llama.dll
ggml.dll
ggml-base.dll
mtmd.dll
vcomp140.dll
cudart64_*.dll, cublas*.dll, AMD and Intel runtime DLLs
LICENSE
licenses/
manifest.json
build-info/
backends/
  cpu/       ggml-cpu-*.dll
  cuda12/    ggml-cuda.dll
  cuda13/    ggml-cuda.dll
  vulkan/    ggml-vulkan.dll
  rocm/      ggml-hip.dll
  sycl/      ggml-sycl.dll
```

Each build also uploads **layla-windows-x64-<backend>** independently, even if a different backend fails. These component artifacts contain `build.json` and a `payload/` directory. Merge the payloads from the same run, preserving their relative paths. The CPU component supplies the shared runtime. CUDA 12 and CUDA 13 plugins must stay in separate directories because they have the same filename.

The combined package verifies source revisions, required files, SHA-256 inventories, and conflicting DLLs before assembly. `build-info/` records the SDK and CMake arguments for each component. `manifest.json` records the final file hashes. Artifacts are retained for 30 days; save the downloaded package with your app release.

## Use and bundle

Keep vendor runtime DLLs next to `llama-server.exe`, and preserve the backend folders. Install the Microsoft Visual C++ 2015-2022 x64 Redistributable, or provide it through the app's Windows packaging dependencies. GPU drivers remain a machine prerequisite; SDK driver DLLs such as `vulkan-1.dll` and `OpenCL.dll` are not bundled.

Run the server with the CPU directory as its working directory. To select a GPU plugin, set `GGML_BACKEND_PATH` to the absolute path of exactly one backend DLL:

```powershell
$runtime = (Resolve-Path './runtime').Path
$env:GGML_BACKEND_PATH = "$runtime/backends/cuda13/ggml-cuda.dll"
Push-Location "$runtime/backends/cpu"
try { & "$runtime/llama-server.exe" -m 'C:/models/model.gguf' -ngl 99 }
finally { Pop-Location }
```

For CPU-only use, clear `GGML_BACKEND_PATH` and pass `-ngl 0`. The server is built without TLS or an embedded web UI for use by Layla's local frontend. CI probes CPU backend discovery; GPU inference must be tested on the corresponding hardware.

These artifacts contain this fork's own runtime and backend binaries. Do not mix them with upstream core or plugin DLLs. The app's source-fingerprint packaging receipts are separate from this workflow's `manifest.json`; retain the app's package verification when importing these artifacts into a Layla release.
