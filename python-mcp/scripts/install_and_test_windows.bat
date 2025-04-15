@echo off
setlocal enabledelayedexpansion

:: Check if uv exists
where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo uv is already installed
) else (
    echo Installing uv...
    powershell -Command "Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile 'install.ps1'; .\install.ps1"
)

:: Add uv to PATH if not already there
echo %PATH% | find /i "%USERPROFILE%\.cargo\bin" >nul
if %ERRORLEVEL% NEQ 0 (
    set PATH=%USERPROFILE%\.cargo\bin;%PATH%
)

echo Creating virtual environment...
uv venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
uv pip install -r requirements.txt
uv pip install -r requirements-test.txt

echo Running tests...
pytest tests/ -v

echo Running coverage report...
pytest tests/ --cov=paprmcp --cov-report=term-missing

echo Tests completed successfully! 