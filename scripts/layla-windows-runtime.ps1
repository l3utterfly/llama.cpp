[CmdletBinding(DefaultParameterSetName = 'Build')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Build')]
    [ValidateSet('cpu', 'cuda12', 'cuda13', 'vulkan', 'rocm', 'sycl')]
    [string] $Backend,
    [Parameter(Mandatory, ParameterSetName = 'Assemble')]
    [switch] $Assemble
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$source = Split-Path $PSScriptRoot
Set-Location -LiteralPath $source
$revision = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine the source revision.' }
$plugins = @{ cpu = 'ggml-cpu-x64.dll'; cuda12 = 'ggml-cuda.dll'; cuda13 = 'ggml-cuda.dll'; vulkan = 'ggml-vulkan.dll'; rocm = 'ggml-hip.dll'; sycl = 'ggml-sycl.dll' }
$backends = @('cpu', 'cuda12', 'cuda13', 'vulkan', 'rocm', 'sycl')

function Copy-Required([string] $File, [string] $Directory) {
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing runtime file: $File" }
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    Copy-Item -LiteralPath $File -Destination $Directory
}

function Get-Inventory([string] $Root) {
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    @(Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName | ForEach-Object {
        @{ path = $_.FullName.Substring($prefix.Length).Replace('\', '/'); sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    })
}

function Copy-VendorDll([string[]] $Directories, [string] $Name, [string] $Destination) {
    foreach ($directory in $Directories) {
        $file = Join-Path $directory $Name
        if (Test-Path -LiteralPath $file -PathType Leaf) { Copy-Required $file $Destination; return }
    }
    throw "Vendor runtime DLL not found: $Name"
}

function Copy-SdkLicenses([string] $Root, [string] $Destination) {
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object { $_.Name -match '^(LICENSE|EULA|COPYING|THIRD.PARTY)' } | ForEach-Object {
        $relative = $_.FullName.Substring($prefix.Length)
        Copy-Required $_.FullName (Split-Path (Join-Path $Destination $relative))
    }
}

if ($Assemble) {
    $runtime = Join-Path $source 'build-windows/runtime'
    if (Test-Path -LiteralPath $runtime) { throw 'Use a fresh directory to assemble the runtime.' }
    New-Item -ItemType Directory -Path $runtime | Out-Null
    $metadata = @{}
    foreach ($name in $backends) {
        $component = Join-Path $source "build-windows/downloads/layla-windows-x64-$name"
        $manifest = Get-Content -LiteralPath "$component/build.json" -Raw | ConvertFrom-Json
        if ($manifest.revision -ne $revision -or $manifest.architecture -ne 'x64' -or $manifest.backend -ne $name) {
            throw "Mismatched runtime component: $name"
        }
        if (!$manifest.files -or $manifest.files.Count -eq 0) { throw "Empty runtime component: $name" }
        foreach ($entry in $manifest.files) {
            $root = [IO.Path]::GetFullPath("$component/payload").TrimEnd('\') + '\'
            $file = [IO.Path]::GetFullPath((Join-Path $root $entry.path))
            $destination = [IO.Path]::GetFullPath((Join-Path $runtime $entry.path))
            if (!$file.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or
                !$destination.StartsWith($runtime.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "Invalid artifact path: $($entry.path)"
            }
            if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Artifact hash mismatch: $file" }
            if (Test-Path -LiteralPath $destination) {
                if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Conflicting runtime file: $destination" }
            } else { Copy-Required $file (Split-Path $destination) }
        }
        $plugin = Join-Path $runtime "backends/$name/$($plugins[$name])"
        if (!(Test-Path -LiteralPath $plugin -PathType Leaf)) { throw "Missing backend: $plugin" }
        Copy-Required "$component/build.json" "$runtime/build-info"
        Move-Item -LiteralPath "$runtime/build-info/build.json" -Destination "$runtime/build-info/$name.json"
        $metadata[$name] = $manifest.sdk
    }
    foreach ($name in @('llama-server.exe', 'llama-server-impl.dll', 'llama-common.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll', 'mtmd.dll', 'vcomp140.dll')) {
        if (!(Test-Path -LiteralPath (Join-Path $runtime $name))) { throw "Missing core runtime file: $name" }
    }
    Copy-Required "$source/docs/layla-windows-runtime.md" $runtime
    @{ revision = $revision; architecture = 'x64'; backends = $metadata; files = @(Get-Inventory $runtime) } |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$runtime/manifest.json" -Encoding utf8
    Write-Host "Assembled runtime from $revision at $runtime"
    exit 0
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio 2022 C++ tools are required.' }
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -VsInstallationPath $vs -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$build = Join-Path $source "build-windows/$Backend"
$artifact = Join-Path $source "build-windows/artifacts/$Backend"
$payload = Join-Path $artifact 'payload'
if (Test-Path -LiteralPath $artifact) { throw "Use a fresh staging directory: $artifact" }
$argsCmake = @('-S', $source, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF', '-DGGML_CCACHE=OFF',
    '-DGGML_CPU=OFF', '-DGGML_OPENMP=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_BUILD_TOOLS=OFF',
    '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF',
    '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_OPENSSL=OFF')
$target = $plugins[$Backend].Replace('.dll', '')
$sdk = ''
switch ($Backend) {
    cpu {
        $argsCmake += @('-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', '-DGGML_CPU=ON', '-DGGML_CPU_ALL_VARIANTS=ON',
            '-DGGML_OPENMP=ON', '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON')
        $target = 'llama-server'
        $sdk = 'MSVC 2022'
    }
    { $_ -in 'cuda12', 'cuda13' } {
        if (!$env:CUDA_PATH) { throw 'CUDA_PATH is required.' }
        $argsCmake += @('-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', '-DGGML_CUDA=ON',
            "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe", "-DCUDAToolkit_ROOT=$env:CUDA_PATH", "-DGGML_CUDA_CCCL_VERSION=$env:CUDA_CCCL_VERSION")
        $sdk = $env:CUDA_PATH
    }
    vulkan {
        if (!$env:VULKAN_SDK) { throw 'VULKAN_SDK is required.' }
        $argsCmake += @('-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', '-DGGML_VULKAN=ON')
        $sdk = $env:VULKAN_SDK
    }
    rocm {
        if (!$env:HIP_PATH) { throw 'HIP_PATH is required.' }
        $clang = "$env:HIP_PATH/lib/llvm/bin/clang.exe"
        $argsCmake += @('-DGGML_HIP=ON', "-DCMAKE_PREFIX_PATH=$env:HIP_PATH", "-DHIP_PATH=$env:HIP_PATH",
            "-DCMAKE_C_COMPILER=$clang", "-DCMAKE_CXX_COMPILER=$env:HIP_PATH/lib/llvm/bin/clang++.exe",
            "-DCMAKE_HIP_COMPILER=$clang", '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types', "-DAMDGPU_TARGETS=$env:AMDGPU_TARGETS")
        $sdk = $env:ROCM_VERSION
    }
    sycl {
        $argsCmake += @('-DGGML_SYCL=ON', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=icx')
        $sdk = (& icx --version) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw 'Initialize the oneAPI environment before building SYCL.' }
    }
}
& cmake @argsCmake
if ($LASTEXITCODE -ne 0) { throw "$Backend configuration failed." }
& cmake --build $build --target $target --parallel ([Environment]::ProcessorCount)
if ($LASTEXITCODE -ne 0) { throw "$Backend compilation failed." }
$bin = Join-Path $build 'bin'
if ($Backend -eq 'cpu') {
    foreach ($name in @('llama-server.exe', 'llama-server-impl.dll', 'llama-common.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll', 'mtmd.dll')) {
        Copy-Required (Join-Path $bin $name) $payload
    }
    Get-ChildItem -LiteralPath $bin -Filter 'ggml-cpu*.dll' | ForEach-Object { Copy-Required $_.FullName "$payload/backends/cpu" }
    $openmp = @(Get-ChildItem -LiteralPath "$vs/VC/Redist/MSVC" -Recurse -Filter vcomp140.dll |
        Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.OpenMP\\' } | Sort-Object FullName -Descending)
    if (!$openmp.Count) { throw 'MSVC x64 OpenMP redistributable not found.' }
    Copy-Required $openmp[0].FullName $payload
    Copy-Required "$source/LICENSE" $payload
    Copy-Item -LiteralPath "$source/licenses" -Destination "$payload/licenses" -Recurse
} else {
    Copy-Required (Join-Path $bin $plugins[$Backend]) "$payload/backends/$Backend"
}
if ($Backend -in @('cuda12', 'cuda13')) {
    $major = if ($Backend -eq 'cuda12') { 12 } else { 13 }
    foreach ($name in @("cudart64_$major.dll", "cublas64_$major.dll", "cublasLt64_$major.dll")) {
        Copy-VendorDll @("$env:CUDA_PATH/bin", "$env:CUDA_PATH/bin/x64", "$env:CUDA_PATH/lib") $name $payload
    }
    Copy-SdkLicenses $env:CUDA_PATH "$payload/licenses/$Backend"
} elseif ($Backend -eq 'rocm') {
    foreach ($name in @('amdhip64_7.dll', 'rocm_kpack.dll', 'amd_comgr.dll', 'rocblas.dll', 'hipblas.dll')) {
        Copy-VendorDll @($env:HIP_BIN) $name $payload
    }
    if (Test-Path -LiteralPath "$env:HIP_BIN/hipblaslt.dll") { Copy-Required "$env:HIP_BIN/hipblaslt.dll" $payload }
    Get-ChildItem -LiteralPath $env:HIP_BIN -Filter '*.kpack' -File | ForEach-Object { Copy-Required $_.FullName $payload }
    foreach ($library in @('rocblas', 'hipblaslt')) {
        foreach ($directory in @("$env:HIP_BIN/$library", "$env:HIP_PATH/lib/$library")) {
            if (Test-Path -LiteralPath $directory -PathType Container) {
                Copy-Item -LiteralPath $directory -Destination "$payload/$library" -Recurse
                break
            }
        }
    }
    Copy-SdkLicenses $env:HIP_PATH "$payload/licenses/rocm"
} elseif ($Backend -eq 'sycl') {
    $components = @{
        mkl = @('mkl_sycl_blas.6.dll', 'mkl_core.3.dll', 'mkl_def.3.dll', 'mkl_avx2.3.dll', 'mkl_tbb_thread.3.dll')
        compiler = @('ur_adapter_level_zero.dll', 'ur_adapter_level_zero_v2.dll', 'ur_adapter_opencl.dll', 'ur_loader.dll', 'ur_win_proxy_loader.dll', 'sycl9.dll', 'svml_dispmd.dll', 'libmmd.dll', 'libiomp5md.dll')
        dnnl = @('dnnl.dll'); tbb = @('tbb12.dll'); tcm = @('tcm.dll', 'libhwloc-15.dll'); umf = @('umf.dll')
    }
    foreach ($component in $components.Keys) {
        foreach ($name in $components[$component]) { Copy-VendorDll @("$env:ONEAPI_ROOT/$component/latest/bin") $name $payload }
        Copy-SdkLicenses "$env:ONEAPI_ROOT/$component/latest" "$payload/licenses/sycl/$component"
    }
}
if (!(Test-Path -LiteralPath "$payload/backends/$Backend/$($plugins[$Backend])")) { throw "Backend DLL missing: $Backend" }
@{ revision = $revision; architecture = 'x64'; backend = $Backend; sdk = $sdk; cmake = $argsCmake; files = @(Get-Inventory $payload) } |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$artifact/build.json" -Encoding utf8
Write-Host "Staged $Backend from $revision at $artifact"
