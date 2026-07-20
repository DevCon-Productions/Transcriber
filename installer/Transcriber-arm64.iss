; Inno Setup script for Transcriber — Windows-on-ARM64 build (DevCon Productions)
; Build the app first:
;   .venv-arm64\Scripts\python.exe -E -m PyInstaller Transcriber-arm64.spec --noconfirm
; Then compile this:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" installer\Transcriber-arm64.iss
; Output installer:  installer\Output\Transcriber-ARM64-Setup-1.3-arm64.exe
;
; Distinct from the x64 installer (Transcriber.iss): its OWN AppId (so x64 and ARM
; are separate products and either can be installed independently), arm64-only
; architecture, and a filename containing "ARM64" — which the in-app self-updater's
; arch-aware asset picker keys off (it downloads only the *ARM64* asset on ARM).

#define MyAppName "Transcriber"
#define MyAppNameFull "Transcriber (ARM64)"
#define MyAppVersion "1.4-arm64"
#define MyAppPublisher "DevCon Productions"
#define MyAppExeName "Transcriber.exe"

[Setup]
; NOTE: AppId is DELIBERATELY different from the x64 build's
; {{8F3C1A22-7B4E-4E2A-9F1D-TRANSCRIBER01}} so the two architectures are separate
; installed products and the ARM self-updater upgrades the ARM install in place.
AppId={{8F3C1A22-7B4E-4E2A-9F1D-TRANSCRIBERARM64}}
; Display name, install folder, and Start-menu group are ALL distinct from the x64
; build so the two can coexist on one machine and never overwrite each other. (A
; shared "Transcriber" folder was letting an x64 install block the ARM one.) The
; app's writable state also uses a separate %APPDATA%\Transcriber-ARM64 dir.
AppName={#MyAppNameFull}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/DevCon-Productions/Transcriber
DefaultDirName={autopf}\Transcriber ARM64
DefaultGroupName={#MyAppNameFull}
DisableProgramGroupPage=yes
; Per-machine install (Program Files) needs admin; writable state (config/
; credentials/logs) is redirected to %APPDATA%\Transcriber at runtime.
PrivilegesRequired=admin
OutputDir=Output
OutputBaseFilename=Transcriber-ARM64-Setup-{#MyAppVersion}
SetupIconFile=..\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; ARM64 Windows only. Installs into the native (64-bit) Program Files.
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; The entire PyInstaller one-folder ARM64 build.
Source: "..\dist\Transcriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppNameFull}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppNameFull}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppNameFull}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Launch after an interactive install. The self-updater runs the installer
; non-silently (os.startfile -> UAC -> wizard), so this relaunch fires and the
; app restarts on the new version. (skipifsilent only suppresses it for truly
; silent installs, which the updater doesn't use.)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Leave the user's config/credentials/logs in %APPDATA%\Transcriber alone on
; uninstall (don't delete personal data). Uncomment to also remove them:
; Type: filesandordirs; Name: "{userappdata}\Transcriber"
