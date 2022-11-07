# https://learn.microsoft.com/en-us/windows/wsl/install

$errorActionPreference = "Stop"

Start-Process "wsl.exe" -Wait -ArgumentList "--install"
Start-Process "wsl.exe" -Wait -ArgumentList "--update"