#define AppVersion GetEnv("OGDD_VERSION")

#if AppVersion == ""
  #error OGDD_VERSION must be defined before compiling the installer.
#endif

[Setup]
AppId={{A90A65B8-B8AA-4B03-9E90-114C6DCEBA5B}
AppName=OGDD
AppVersion={#AppVersion}
AppVerName=OGDD {#AppVersion}
AppPublisher=Armando Prado
AppPublisherURL=https://github.com/armandoprado73-coder/OGDD
AppSupportURL=https://github.com/armandoprado73-coder/OGDD/issues
AppUpdatesURL=https://github.com/armandoprado73-coder/OGDD/releases
DefaultDirName={localappdata}\Programs\OGDD
DefaultGroupName=OGDD
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist\installers
OutputBaseFilename=OGDD-{#AppVersion}-Windows-x64-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\OGDD.exe
CloseApplications=yes

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos adicionales:"; Flags: unchecked

[Files]
Source: "..\..\dist\pyinstaller\OGDD\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\OGDD"; Filename: "{app}\OGDD.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\OGDD"; Filename: "{app}\OGDD.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\OGDD.exe"; Description: "Iniciar OGDD"; Flags: nowait postinstall skipifsilent
