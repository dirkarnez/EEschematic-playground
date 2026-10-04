@REM run as Administrator
@echo off

set DOWNLOADS_DIR=%USERPROFILE%\Downloads

@REM set PREFIX=D:\Softwares
set PREFIX=%DOWNLOADS_DIR%


set SEVENZIP=C:\"Program Files"\7-Zip\7z.exe
set PYTHON_DIR=%PREFIX%\python-3.13.9-amd64-portable
set PYTHON_EXE=%PYTHON_DIR%\python.exe

if not exist %PYTHON_EXE% (
cd /d "%TEMP%" &&^
%SystemRoot%\System32\curl.exe "https://github.com/dirkarnez/python-portable/releases/download/v3.13.9/python-3.13.9-amd64-portable.zip" -L -O  &&^
%SEVENZIP% x python-3.13.9-amd64-portable.zip -o"%PYTHON_DIR%"  &&^
del python-3.13.9-amd64-portable.zip
)

if exist %PYTHON_EXE% (
    echo python %PYTHON_EXE% found
)


set GIT_EXE=%PREFIX%\PortableGit\bin\git.exe
if not exist %GIT_EXE% (
cd /d "%TEMP%" &&^
%SystemRoot%\System32\curl.exe "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/PortableGit-2.42.0.2-64-bit.7z.exe" -L -O  &&^
PortableGit-2.42.0.2-64-bit.7z.exe -o%DOWNLOADS_DIR%\PortableGit -y &&^
del PortableGit-2.42.0.2-64-bit.7z.exe
)

if exist %GIT_EXE% (
    echo git %GIT_EXE% found
)

set NGSPICE_EXE=%PREFIX%\ngspice-47_64\Spice64\bin\ngspice.exe
if not exist %NGSPICE_EXE% (
cd /d "%TEMP%" &&^
%SystemRoot%\System32\curl.exe "https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/47/ngspice-47_64.7z?ts=gAAAAABqwiQ34dbpFN2c2zJ9I_ft4EXqZJrsCWi6Gr3xSAB5V1j6e6u9nGqrKah_fywSOj5-Jry9Nx-8OSHHlJEHRlPLFgabtA%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fngspice%2Ffiles%2Fng-spice-rework%2F47%2Fngspice-47_64.7z%2Fdownload" -L -O  &&^
%SEVENZIP% x ngspice-47_64.7z -o%DOWNLOADS_DIR%\ngspice-47_64 &&^
del ngspice-47_64.7z
)

if exist %NGSPICE_EXE% (
    echo ngspice %NGSPICE_EXE% found
)


