# https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

$errorActionPreference = "SilentlyContinue"
$existing_conda_command = Get-Command conda  # test if there is an available conda command on the PATH
$errorActionPreference = "Stop"

if ($existing_conda_command -eq $null)
{
    # download installer from internet
    $conda_installer_path = $env:TEMP + "/Miniconda3-latest-Windows-x86_64.exe"
    if ((Test-Path $conda_installer_path) -eq $false)
    {
        Write-Host "Downloading Miniconda3 for Windows, please wait..."
        $conda_installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        $conda_installer_download_path = $conda_installer_path + ".download"
        wget $conda_installer_url -OutFile $conda_installer_download_path
        mv $conda_installer_download_path $conda_installer_path
    }
    else
    {
        Write-Host "Found Miniconda3 installer already downloaded at '$conda_installer_path'..."
    }

    # run (hopefully silent) installer for current user only
    Write-Host "Installing Miniconda3 for Windows..."
    Start-Process $conda_installer_path -Wait -ArgumentList '/AddToPath=1','/RegisterPython=0', '/S'

    # TODO: may need to manually refresh the PATH here (also confirms installation)
}
else
{
    Write-Host "Conda is already installed."
}

Write-Host "Updating conda..."
Start-Process "conda" -Wait -ArgumentList "update","conda","-y" -NoNewWindow # ensure conda is up to date
Write-Host "Creating / Updating g_diffuser conda environment..."
Start-Process "conda" -Wait -ArgumentList "env","update","-f","./environment.yaml" -WorkingDirectory ("$env:LOCALAPPDATA\g-diffuser-bot") -NoNewWindow  # finally, create / update the g-diffuser environment