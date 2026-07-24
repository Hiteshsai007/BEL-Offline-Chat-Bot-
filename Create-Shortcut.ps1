$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\BEL AI Assistant.lnk")
$Shortcut.TargetPath = "E:\BEL\Launch.bat"
$Shortcut.WorkingDirectory = "E:\BEL"
$Shortcut.Description = "BEL Offline Fault Code Assistant"
$Shortcut.IconLocation = "shell32.dll,22" 
$Shortcut.Save()
Write-Host "Desktop shortcut created successfully!"
