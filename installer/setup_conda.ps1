# https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

$errorActionPreference = "SilentlyContinue"
Start-Transcript -Path ($env:TEMP+"/g_diffuser_installer.log") -Append -Force
Write-Host "Installer Custom Action - Setup Conda"
$errorActionPreference = "Stop"

$conda_path = "$env:USERPROFILE\Miniconda3\Library\bin\conda.bat"

$existing_conda_command = Test-Path $conda_path
if ($existing_conda_command -eq $false)
{
    Write-Host "Conda installation not found at $conda_path..."

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
    Start-Process $conda_installer_path -Wait -ArgumentList '/AddToPath=1','/RegisterPython=0', '/S' -NoNewWindow

    # TODO: may need to manually refresh the PATH here (also confirms installation)
}
else
{
    Write-Host "Conda is already installed."
}

Write-Host "Updating conda..."
#& $conda_path update -y conda
Start-Process $conda_path -Wait -ArgumentList "update","conda","-y" -NoNewWindow # ensure conda is up to date
Write-Host "Creating / Updating g_diffuser conda environment..."
#& $conda_path env update -f ./environment.yaml
Start-Process $conda_path -Wait -ArgumentList "env","update","-f","./environment.yaml" -WorkingDirectory ("$env:LOCALAPPDATA\g-diffuser-bot") -NoNewWindow  # finally, create / update the g-diffuser environment

Stop-Transcript
exit 0