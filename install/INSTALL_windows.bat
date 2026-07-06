@echo off
REM ==========================================================================
REM  AI in Healthcare -- complete one-file installer for Windows
REM
REM  You do NOT need Python, conda, git, or Jupyter first. This downloads the
REM  course, installs its own private Python + all libraries, and opens the
REM  notebooks in your browser. First run takes a few minutes.
REM
REM  HOW TO USE: double-click this file. If SmartScreen warns you,
REM  click "More info" -> "Run anyway". You only do that once.
REM ==========================================================================
setlocal
set "DEST=%USERPROFILE%\Desktop\outset-ai-healthcare"

echo.
echo   AI in Healthcare -- installing everything (first run: a few minutes)
echo   -------------------------------------------------------------------

REM 1. Download the course (as a zip -- no git needed) to the Desktop.
echo   Downloading the course...
powershell -Command "Invoke-WebRequest 'https://github.com/jinchiwei/outset-ai-healthcare/archive/refs/heads/main.zip' -OutFile \"$env:TEMP\outset.zip\""
if exist "%DEST%" rmdir /s /q "%DEST%"
if exist "%TEMP%\outset-ai-healthcare-main" rmdir /s /q "%TEMP%\outset-ai-healthcare-main"
powershell -Command "Expand-Archive -Force \"$env:TEMP\outset.zip\" \"$env:TEMP\""
move "%TEMP%\outset-ai-healthcare-main" "%DEST%" >nul
cd /d "%DEST%"

REM 2. Install uv (a tiny tool that supplies Python + installs libraries).
where uv >nul 2>nul
if errorlevel 1 (
  echo   Installing the setup helper ^(uv^)...
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

REM 3. Private Python + course libraries.
echo   Preparing Python 3.11 and installing libraries ^(torch, etc.^)...
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-student.txt

echo.
echo   Done! The course is on your Desktop in 'outset-ai-healthcare'.
echo   Opening the notebooks in your browser now...
echo   Next time, just double-click START_HERE_windows.bat inside that folder.
echo.
uv run --python .venv jupyter lab notebooks

pause
