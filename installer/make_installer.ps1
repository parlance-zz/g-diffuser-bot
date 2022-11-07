# Install-Module PowerShellProTools
# https://docs.poshtools.com/powershell-pro-tools-documentation/installers


$root_path = "D:/g-diffuser-bot"

$include_paths = @("*.py", "*.md", "*.yaml", "LICENSE", "*.bat")
$exclude_paths = @(".gitattributes", ".gitignore", "*.sh", "*.txt", "*.png", "*.jpg", "*.json", "*.pyc", "*.bin")

pushd $root_path

New-Installer -ProductName "G_Diffuser Bot" -RequiresElevation -UpgradeCode '0a717b99-8de1-4ef6-92b8-dae69c81d62e' -HelpLink "https://github.com/parlance-zz/g-diffuser-bot/blob/dev/README.md" -Content {
    New-InstallerDirectory -PredefinedDirectoryName LocalAppDataFolder -Content {
        New-InstallerDirectory -DirectoryName "g-diffuser-bot" -Content {
            # root folder content
            Get-ChildItem ($root_path+"/*") -File -Include $include_paths -Exclude $exclude_paths | New-InstallerFile

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

            # bot sub-folders
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
 } -OutputDirectory (Join-Path $PSScriptRoot ".") -Platform x64 -AddRemoveProgramsIcon "installer/app_icon.ico" -Version 1.0

 popd