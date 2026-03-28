param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cpu', 'cuda', 'rocm', 'vulkan')]
    [string]$Backend,

    [string]$RocmHipSdkFilename = 'AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe'
)

$ErrorActionPreference = 'Stop'

function Add-GitHubPath([string]$PathEntry) {
    if (-not $PathEntry) {
        return
    }

    if ($env:GITHUB_PATH) {
        $PathEntry | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
    } else {
        $env:PATH = "$PathEntry;$env:PATH"
    }
}

function Set-GitHubEnv([string]$Name, [string]$Value) {
    if (-not $Name) {
        return
    }

    if ($env:GITHUB_ENV) {
        "${Name}=${Value}" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    } else {
        Set-Item -Path "env:$Name" -Value $Value
    }
}

function Test-ExeHeader([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $false
    }

    $header = Get-Content -Path $Path -AsByteStream -TotalCount 2
    return $header.Length -eq 2 -and $header[0] -eq 0x4D -and $header[1] -eq 0x5A
}

switch ($Backend.ToLowerInvariant()) {
    'cpu' {
    }
    'cuda' {
        choco install cuda -y --no-progress

        $cudaRoot = $env:CUDA_PATH
        if (-not $cudaRoot -or -not (Test-Path $cudaRoot)) {
            $cudaRoot = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -ExpandProperty FullName -First 1
        }

        if (-not $cudaRoot) {
            throw 'CUDA toolkit install completed, but CUDA_PATH was not found.'
        }

        Add-GitHubPath "$cudaRoot\bin"
        Set-GitHubEnv 'CUDA_PATH' $cudaRoot
    }
    'rocm' {
        $candidateFilenames = @(
            $RocmHipSdkFilename,
            'AMD-Software-PRO-Edition-24.Q4-WinSvr2022-For-HIP.exe',
            'AMD-Software-PRO-Edition-24.Q3-WinSvr2022-For-HIP.exe'
        ) | Where-Object { $_ } | Select-Object -Unique

        $installer = Join-Path $env:RUNNER_TEMP 'hip-sdk-installer.exe'
        $downloadErrors = @()

        foreach ($filename in $candidateFilenames) {
            $url = "https://download.amd.com/developer/eula/rocm-hub/$filename"
            Write-Host "Trying HIP SDK download: $url"

            try {
                Invoke-WebRequest -Uri $url -OutFile $installer
                if (Test-ExeHeader $installer) {
                    Write-Host "Using HIP SDK installer $filename"
                    break
                }

                $downloadErrors += "Downloaded $filename, but the payload was not a Windows executable."
            } catch {
                $downloadErrors += "Failed to download $filename: $($_.Exception.Message)"
            }

            Remove-Item -Path $installer -ErrorAction SilentlyContinue
        }

        if (-not (Test-ExeHeader $installer)) {
            $details = $downloadErrors -join "`n"
            throw "Unable to download a Windows HIP SDK installer.`n$details"
        }

        $installLog = Join-Path $env:RUNNER_TEMP 'hip-sdk-install.log'
        $process = Start-Process $installer -ArgumentList '-install', '-log', $installLog -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "HIP SDK installer exited with code $($process.ExitCode). See $installLog"
        }

        $rocmRoot = @(
            (Join-Path $env:ProgramFiles 'AMD\ROCm'),
            (Join-Path $env:ProgramFiles 'AMD\HIP')
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1

        if (-not $rocmRoot) {
            throw "HIP SDK install completed, but ROCM_PATH was not found. See $installLog"
        }

        Add-GitHubPath "$rocmRoot\bin"
        Add-GitHubPath "$rocmRoot\llvm\bin"
        Set-GitHubEnv 'ROCM_PATH' $rocmRoot
        Set-GitHubEnv 'HIP_PATH' $rocmRoot
    }
    'vulkan' {
        choco install vulkan-sdk -y --no-progress

        $sdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if (-not $sdk) {
            throw 'Vulkan SDK install completed, but no C:\VulkanSDK directory was found.'
        }

        Add-GitHubPath "$($sdk.FullName)\Bin"
        Add-GitHubPath "$($sdk.FullName)\Bin32"
        Set-GitHubEnv 'VULKAN_SDK' $sdk.FullName
    }
}
