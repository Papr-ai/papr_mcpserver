@echo off
REM Copyright 2024 Papr AI
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

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
if exist requirements-test.txt (
    uv pip install -r requirements-test.txt
) else (
    echo requirements-test.txt not found, skipping test dependencies
)

echo Running tests...
if exist tests\ (
    pytest tests/ -v
    if %ERRORLEVEL% NEQ 0 (
        echo Tests failed!
        exit /b 1
    )
    
    echo Running coverage report...
    pytest tests/ --cov=paprmcp --cov-report=term-missing
    if %ERRORLEVEL% NEQ 0 (
        echo Coverage report failed!
        exit /b 1
    )
    
    echo Tests completed successfully!
) else (
    echo No tests directory found, skipping tests
) 