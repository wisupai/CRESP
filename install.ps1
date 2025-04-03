# PowerShell Installation Script for CRESP

# Admin check
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Installation directory
$installRoot = if ($isAdmin) { "$env:ProgramFiles\CRESP" } else { "$env:LOCALAPPDATA\CRESP" }
$installDir = "$installRoot\bin"

# Determine the latest version if not specified
$version = if ($args.Count -gt 0) { $args[0] } else {
    try {
        $latestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/wisupai/CRESP/releases/latest" -ErrorAction Stop
        $latestRelease.tag_name
    } catch {
        "v0.1.0-dev.1"  # Fallback version
    }
}

Write-Host "Downloading CRESP $version..." -ForegroundColor Blue

# Create installation directory if it doesn't exist
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Determine architecture
$arch = if ([Environment]::Is64BitOperatingSystem) { "x86_64" } else { "i686" }
$downloadUrl = "https://github.com/wisupai/CRESP/releases/download/$version/cresp-windows-$arch.zip"

# Create a temporary directory for downloading
$tempDir = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    $zipFile = "$tempDir\cresp.zip"
    
    # Download the binary
    Write-Host "Downloading from $downloadUrl..." -ForegroundColor Blue
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile
    
    # Extract the binary
    Write-Host "Extracting..." -ForegroundColor Blue
    Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force
    
    # Move the binary to the installation directory
    Write-Host "Installing CRESP to $installDir..." -ForegroundColor Blue
    Copy-Item "$tempDir\cresp.exe" -Destination "$installDir\cresp.exe" -Force
    
    # Add to PATH if it's not already there
    $paths = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User).Split(";")
    if (-not $paths.Contains($installDir)) {
        Write-Host "Adding $installDir to user PATH..." -ForegroundColor Yellow
        $newPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User) + ";$installDir"
        [Environment]::SetEnvironmentVariable("Path", $newPath, [EnvironmentVariableTarget]::User)
        
        # Update the current session's PATH
        $env:Path += ";$installDir"
    }
    
    Write-Host "CRESP $version has been installed to $installDir\cresp.exe" -ForegroundColor Green
    Write-Host "You can now use CRESP by running 'cresp' from a new terminal window." -ForegroundColor Green
    Write-Host "Run 'cresp --help' to see available commands." -ForegroundColor Green
    
} catch {
    Write-Host "Error during installation: $_" -ForegroundColor Red
} finally {
    # Clean up
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
} 