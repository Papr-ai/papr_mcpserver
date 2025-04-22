# Function to check if uv is installed
function Test-UvInstalled {
    try {
        $null = Get-Command uv -ErrorAction Stop
        Write-Host "uv is already installed"
        return $true
    }
    catch {
        return $false
    }
}

# Function to install uv
function Install-Uv {
    Write-Host "Installing uv..."
    try {
        # Check if winget is available
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Astral.uv
        }
        else {
            # Fallback to direct download and installation
            $tempDir = [System.IO.Path]::GetTempPath()
            $installerPath = Join-Path $tempDir "uv-installer.ps1"
            
            # Download and run the installer
            Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $installerPath
            & $installerPath
            
            # Clean up
            Remove-Item $installerPath -Force
        }
        
        # Add to PATH if needed
        $uvPath = "$env:LOCALAPPDATA\uv\bin"
        if ($env:Path -notlike "*$uvPath*") {
            [Environment]::SetEnvironmentVariable(
                "Path",
                [Environment]::GetEnvironmentVariable("Path", "User") + ";$uvPath",
                "User"
            )
            $env:Path = "$env:Path;$uvPath"
        }
        
        Write-Host "uv installation completed successfully!"
        return $true
    }
    catch {
        Write-Host "Failed to install uv: $_"
        return $false
    }
}

# Main script
if (-not (Test-UvInstalled)) {
    Write-Host "uv is not installed. Installing now..."
    if (Install-Uv) {
        # Verify installation
        if (Test-UvInstalled) {
            Write-Host "uv installation verified!"
            uv --version
        }
        else {
            Write-Host "Installation seemed to succeed but verification failed."
            exit 1
        }
    }
    else {
        Write-Host "Failed to install uv"
        exit 1
    }
}

Write-Host "Script completed successfully!" 