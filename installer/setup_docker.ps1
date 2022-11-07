# https://docs.docker.com/desktop/install/windows-install/

$errorActionPreference = "Stop"

$docker_install_reg_path = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Docker Desktop"
if (Test-Path $docker_install_reg_path)
{
    Write-Host "Docker Desktop is already installed."

    $errorActionPreference = "SilentlyContinue"
    $running_processes = Get-Process -Name "Docker Desktop"
    $errorActionPreference = "Continue"
    if ($running_processes)
    {
        Write-Host "Docker Desktop is already running."
    }
    else
    {
        $docker_desktop_path = (Get-ItemPropertyValue $docker_install_reg_path -Name "InstallLocation") + "/Docker Desktop.exe"
        Write-Host "Starting Docker Desktop..."
        Start-Process $docker_desktop_path
    }

    exit 0
}
else
{
    Write-Host "Could not find Docker Desktop installed."
}

$docker_installer_path = $env:TEMP + "/Docker Desktop Installer.exe"
if ((Test-Path $docker_installer_path) -eq $false)
{
    Write-Host "Downloading Docker Desktop for Windows, please wait..."
    $docker_installer_url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $docker_installer_download_path = $docker_installer_path + ".download"
    wget $docker_installer_url -OutFile $docker_installer_download_path
    mv $docker_installer_download_path $docker_installer_path
}
else
{
    Write-Host "Found Docker Desktop installer already downloaded at '$docker_installer_path'..."
}
Write-Host "Installing Docker Desktop for Windows..."
Start-Process $docker_installer_path -Wait -ArgumentList 'install','--quiet','--accept-license'

if (Test-Path $docker_install_reg_path)
{
    $docker_desktop_path = (Get-ItemPropertyValue $docker_install_reg_path -Name "InstallLocation") + "/Docker Desktop.exe"
    Write-Host "Starting Docker Desktop..."
    Start-Process $docker_desktop_path
}
else
{
    Write-Error "Error Installing Docker Desktop for Windows."
}