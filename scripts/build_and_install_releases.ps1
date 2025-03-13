# build_releases.ps1
# Script to build all release configurations for Arnold on Windows

Push-Location $PSScriptRoot/..

Write-Host "---------------------------------"
Write-Host "Building Release for Arnold-7.3.4"
Write-Host "---------------------------------"
cmake --preset windows-arnold-7.3.4
cmake --build . --target install --preset windows-arnold-7.3.4-release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build for windows-arnold-7.3.4-release failed!"
    Pop-Location
    exit 1
}

Write-Host "---------------------------------"
Write-Host "Building Release for Arnold-7.3.5"
Write-Host "---------------------------------"
cmake --preset windows-arnold-7.3.5
cmake --build . --target install --preset windows-arnold-7.3.5-release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build for windows-arnold-7.3.5-release failed!"
    Pop-Location
    exit 1
}

Pop-Location

Write-Host "-------------------------------"
Write-Host "All Release successfully built!"
Write-Host "-------------------------------"
