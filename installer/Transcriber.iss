; Inno Setup script for Transcriber (DevCon Productions)
; Build the app first:  .venv\Scripts\python.exe -E -m PyInstaller Transcriber.spec --noconfirm
; Then compile this:     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\Transcriber.iss
; Output installer:      installer\Output\Transcriber-Setup-1.3.exe

#define MyAppName "Transcriber"
#define MyAppVersion "1.3"
#define MyAppPublisher "DevCon Productions"
#define MyAppExeName "Transcriber.exe"

[Setup]
AppId={{8F3C1A22-7B4E-4E2A-9F1D-TRANSCRIBER01}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Per-machine install (Program Files) needs admin; that's why writable state
; (config/credentials/logs) is redirected to %APPDATA%\Transcriber at runtime.
PrivilegesRequired=admin
OutputDir=Output
OutputBaseFilename=Transcriber-Setup-{#MyAppVersion}
SetupIconFile=..\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; The entire PyInstaller one-folder build.
Source: "..\dist\Transcriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Leave the user's config/credentials/logs in %APPDATA%\Transcriber alone on
; uninstall (don't delete personal data). Uncomment to also remove them:
; Type: filesandordirs; Name: "{userappdata}\Transcriber"
