# check windows version and enable developer mode

$errorActionPreference = "SilentlyContinue"
Start-Transcript -Path ($env:TEMP+"/g_diffuser_installer.log") -Append -Force
Write-Host "Installer Custom Action - Setup Developer Mode"
$errorActionPreference = "Stop"

# verify windows version at least supports WSL2
$windows_version = [System.Environment]::OSVersion.Version
if ($windows_version.Major -lt 10)
{
    $outdated_windows = $true
}
else
{
    if ($windows_version.Build -lt 18362)
    {
        $outdated_windows = $true
    }
}
Write-Host "Found Windows version $windows_version..."
if ($outdated_windows)
{
    
    Write-Error "Windows 10 (build 18362) or later is required to run G-Diffuser Bot"
    Stop-Transcript
    exit 1
}

# Create AppModelUnlock registry key if it doesn't exist, required for enabling Developer Mode
$RegistryKeyPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock"
if ((Test-Path -Path $RegistryKeyPath) -eq $false) { New-Item -Path $RegistryKeyPath -ItemType Directory -Force }

$errorActionPreference = "SilentlyContinue"
$existing_value = Get-ItemPropertyValue -Path $RegistryKeyPath -Name AllowDevelopmentWithoutDevLicense
$errorActionPreference = "Stop"

if (($existing_value -eq $null) -or ($existing_value -eq 0))
{
    Write-Host "Enabling Developer Mode..."
    Set-ItemProperty -Path $RegistryKeyPath -Name AllowDevelopmentWithoutDevLicense -Value 1 -Force
    Write-Host "Developer Mode enabled."
}
else
{
    Write-Host "Developer Mode already enabled."
}

Stop-Transcript
exit 0