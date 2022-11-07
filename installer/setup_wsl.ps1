# https://learn.microsoft.com/en-us/windows/wsl/install

$errorActionPreference = "SilentlyContinue"
Start-Transcript -Path ($env:TEMP+"/g_diffuser_installer.log") -Append -Force
Write-Host "Installer Custom Action - Setup WSL"
$errorActionPreference = "Stop"

$wsl_path = "$env:WINDIR\sysnative\wsl.exe"
& $wsl_path --install
& $wsl_path --update
#Start-Process $wsl_path -ArgumentList "--install" -Wait -NoNewWindow # -Verb RunAs
#Start-Process $wsl_path -ArgumentList "--update" -Wait -NoNewWindow  #-Verb RunAs

Stop-Transcript
exit 0