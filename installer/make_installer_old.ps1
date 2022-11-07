# Install-Module PowerShellProTools
# https://docs.poshtools.com/powershell-pro-tools-documentation/installers

$root_path = "D:/g-diffuser-bot"

$include_paths = @("*.py", "*.md", "*.yaml", "LICENSE", "*.bat", "*.ps1")
$exclude_paths = @(".gitattributes", ".gitignore", "*.sh", "*.txt", "*.png", "*.jpg", "*.json", "*.pyc", "*.bin")

pushd $root_path

New-Installer -ProductName "G_Diffuser Bot" -RequiresElevation -UpgradeCode '0a717b99-8de1-4ef6-92b8-dae69c81d62e' -Content {
    New-InstallerDirectory -PredefinedDirectoryName LocalAppDataFolder -Content {
        New-InstallerDirectory -DirectoryName "g-diffuser-bot" -Content {
            # root folder content
            Get-ChildItem ($root_path+"/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile

            # installer subfolder (with installer custom action scripts for setup and uninstall)
            New-InstallerDirectory -DirectoryName "installer" -Content {
                New-InstallerFile -Source .\installer\setup_developer_mode.ps1 -Id 'SetupDeveloperMode'
                New-InstallerFile -Source .\installer\setup_wsl.ps1 -Id 'SetupWSL'
                New-InstallerFile -Source .\installer\setup_docker.ps1 -Id 'SetupDocker'
                New-InstallerFile -Source .\installer\setup_conda.ps1 -Id 'SetupConda'
            }

            # root path subfolders
            New-InstallerDirectory -DirectoryName "backups" -Content {
                Get-ChildItem ($root_path+"/backups/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            }
            New-InstallerDirectory -DirectoryName "debug" -Content {
                Get-ChildItem ($root_path+"/debug/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            }
            New-InstallerDirectory -DirectoryName "inputs" -Content {
                Get-ChildItem ($root_path+"/inputs/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                New-InstallerDirectory -DirectoryName "scripts" -Content {
                    Get-ChildItem ($root_path+"/inputs/scripts/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                }
            }
            New-InstallerDirectory -DirectoryName "models" -Content {
                Get-ChildItem ($root_path+"/models/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            }
            New-InstallerDirectory -DirectoryName "outputs" -Content {
                Get-ChildItem ($root_path+"/outputs/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            }
            New-InstallerDirectory -DirectoryName "saved" -Content {
                Get-ChildItem ($root_path+"/saved/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            } 
            New-InstallerDirectory -DirectoryName "temp" -Content {
                Get-ChildItem ($root_path+"/temp/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
            }

            # discord bot sub-folders
            New-InstallerDirectory -DirectoryName "bot" -Content {
                Get-ChildItem ($root_path+"/bot/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                New-InstallerDirectory -DirectoryName "backups" -Content {
                    Get-ChildItem ($root_path+"/bot/backups/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                }
                New-InstallerDirectory -DirectoryName "inputs" -Content {
                    Get-ChildItem ($root_path+"/bot/inputs/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                    New-InstallerDirectory -DirectoryName "scripts" -Content {
                        Get-ChildItem ($root_path+"/bot/inputs/scripts/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                    }                    
                }
                New-InstallerDirectory -DirectoryName "outputs" -Content {
                    Get-ChildItem ($root_path+"/bot/outputs/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                }
                New-InstallerDirectory -DirectoryName "saved" -Content {
                    Get-ChildItem ($root_path+"/bot/saved/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                }                
            }

            # extensions / sdgrpcserver client sub-folders
            New-InstallerDirectory -DirectoryName "extensions" -Content {
                Get-ChildItem ($root_path+"/extensions/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                New-InstallerDirectory -DirectoryName "stable-diffusion-grpcserver" -Content {
                    Get-ChildItem ($root_path+"/extensions/stable-diffusion-grpcserver/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                    New-InstallerDirectory -DirectoryName "sdgrpcserver" -Content {
                        Get-ChildItem ($root_path+"/extensions/stable-diffusion-grpcserver/sdgrpcserver/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                        New-InstallerDirectory -DirectoryName "generated" -Content {
                            Get-ChildItem ($root_path+"/extensions/stable-diffusion-grpcserver/sdgrpcserver/generated/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile
                        }
                    }          
                }
            }
        }
    }
 } -OutputDirectory (Join-Path $PSScriptRoot ".") -Verbose -Platform x64 -AddRemoveProgramsIcon "installer/app_icon.ico" `
   -HelpLink "https://github.com/parlance-zz/g-diffuser-bot/blob/dev/README.md" `
   -AboutLink "https://github.com/parlance-zz/g-diffuser-bot" `
   -CustomAction @(
                New-InstallerCustomAction -FileId 'SetupDeveloperMode' -RunOnInstall -CheckReturnValue
                New-InstallerCustomAction -FileId 'SetupWSL' -RunOnInstall -CheckReturnValue
                New-InstallerCustomAction -FileId 'SetupDocker' -RunOnInstall -CheckReturnValue
                New-InstallerCustomAction -FileId 'SetupConda' -RunOnInstall -CheckReturnValue
   ) `
   -Version 1.0

 popd