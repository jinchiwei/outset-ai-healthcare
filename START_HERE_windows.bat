@echo off
REM ==========================================================================
REM  AI in Healthcare -- one-click local setup for Windows
REM  Double-click this file. It installs everything and opens the notebooks
REM  in your browser. First run takes a few minutes (downloading Python +
REM  libraries); after that it's instant.
REM
REM  If Windows SmartScreen warns you: click "More info" -> "Run anyway".
REM ==========================================================================
setlocal
cd /d "%~dp0"

echo.
echo   AI in Healthcare -- setting up (first run: a few minutes)
echo   ---------------------------------------------------------

REM 1. uv is a tiny tool that installs Python and all the libraries for us.
where uv >nul 2>nul
if errorlevel 1 (
  echo   Installing the setup helper ^(uv^)...
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

REM 2. Create an isolated Python and install the course libraries.
echo   Preparing Python 3.11...
uv venv --python 3.11 .venv
echo   Installing course libraries ^(torch, etc.^)...
uv pip install --python .venv -r requirements-student.txt

REM 3. Launch JupyterLab pointed at the notebooks.
echo.
echo   Done. Opening the notebooks in your browser...
echo   (Leave this window open while you work. Close it to stop.)
echo.
uv run --python .venv jupyter lab notebooks

pause
