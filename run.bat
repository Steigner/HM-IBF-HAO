@echo off
setlocal EnableDelayedExpansion

REM Windows-native entry point for this repository, equivalent to run.sh (see that file
REM for the canonical Bash version). Both target the same image/container names, so it
REM does not matter which of the two you use.
REM
REM   run.bat                       interactive shell in the container
REM   run.bat verify                the full verification gate
REM   run.bat smoke                 preprocess -> train -> evaluate, end to end
REM   run.bat cargo test --workspace
REM
REM Set HM_IBF_NIX=1 to use the dev-nix image instead (adds Nix/R/IRACE, needed for
REM `hm-ibf train`/`pipeline`). It runs as a separate container so both variants can be
REM up at the same time:
REM
REM   set HM_IBF_NIX=1 ^& run.bat     shell in the Nix-enabled container

set "TARGET_VOLUME=hm-ibf-hao-target"
if "%HM_IBF_NIX%"=="1" (
    set "IMAGE=hm-ibf-hao:dev-nix"
    set "CONTAINER=hm-ibf-hao-nix"
    set "BUILD_TARGET=dev-nix"
) else (
    set "IMAGE=hm-ibf-hao:dev"
    set "CONTAINER=hm-ibf-hao"
    set "BUILD_TARGET=dev"
)

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
cd /d "%REPO_ROOT%" || exit /b 1

docker image inspect "%IMAGE%" >nul 2>&1
if errorlevel 1 (
    echo ==^> Building %IMAGE%
    set "DOCKER_BUILDKIT=1"
    docker build --target %BUILD_TARGET% -t "%IMAGE%" . || exit /b 1
)

set "RUNNING="
for /f "delims=" %%i in ('docker ps -q -f "name=^%CONTAINER%$"') do set "RUNNING=%%i"
if not defined RUNNING (
    docker rm -f "%CONTAINER%" >nul 2>&1
    docker volume create "%TARGET_VOLUME%" >nul || exit /b 1
    echo ==^> Starting %CONTAINER%
    REM The target directory lives in a named volume: bind-mounting it would make cargo
    REM builds crawl on Docker Desktop's shared filesystem. Both image variants share it,
    REM since they use the same Rust toolchain version.
    docker run -d --name "%CONTAINER%" ^
        -v "%REPO_ROOT%":/app ^
        -v "%TARGET_VOLUME%":/app/target ^
        -w /app ^
        "%IMAGE%" sleep infinity >nul || exit /b 1
)

if "%~1"=="" (
    docker exec -it "%CONTAINER%" bash
    exit /b %errorlevel%
)

if "%~1"=="verify" (
    docker exec "%CONTAINER%" bash /app/scripts/verify.sh
    exit /b %errorlevel%
)

if "%~1"=="smoke" (
    docker exec "%CONTAINER%" bash /app/scripts/smoke.sh %2
    exit /b %errorlevel%
)

docker exec "%CONTAINER%" bash -c "cd /app && %*"
exit /b %errorlevel%
