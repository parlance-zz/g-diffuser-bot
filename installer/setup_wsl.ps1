# https://learn.microsoft.com/en-us/windows/wsl/install

$errorActionPreference = "SilentlyContinue"
Start-Transcript -Path ($env:TEMP+"/g_diffuser_installer.log") -Append -Force
Write-Host "Installer Custom Action - Setup WSL"
$errorActionPreference = "Stop"

$wsl_path = "wsl.exe"
& $wsl_path --install
& $wsl_path --update
#Start-Process $wsl_path -ArgumentList "--install" -Wait -NoNewWindow
#Start-Process $wsl_path -ArgumentList "--update" -Wait -NoNewWindow

Stop-Transcript
exit 0