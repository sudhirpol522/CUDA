@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if "%~1"=="" (
    echo Usage:
    echo   compile.bat filename.cu              - compile and run
    echo   compile.bat filename.cu profile      - compile + Nsight profile  ^(saved to profiles\^)
    echo   compile.bat filename.cu analysis     - compile + Nsight full analysis  ^(saved to analysis\^)
    exit /b 1
)

set BASENAME=%~n1
set INPUT=src\%~nx1
set OUTPUT=%BASENAME%.exe

if not exist "%INPUT%" (
    echo Error: src\%~nx1 not found. Place .cu files in the src\ folder.
    exit /b 1
)

nvcc -allow-unsupported-compiler "%INPUT%" -o "%OUTPUT%"
if not %errorlevel%==0 (
    echo.
    echo Compilation failed!
    exit /b 1
)

echo.
echo Compiled successfully: %OUTPUT%

if "%~2"=="profile" (
    echo Running Nsight Compute profiler...
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k cd /d D:\CUDA && ncu -o profiles\%BASENAME%-profile %OUTPUT% && echo. && echo Saved to profiles\%BASENAME%-profile.ncu-rep && pause'"
) else if "%~2"=="analysis" (
    echo Running Nsight Compute full analysis...
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k cd /d D:\CUDA && ncu --set full -o analysis\%BASENAME%-analysis %OUTPUT% && echo. && echo Saved to analysis\%BASENAME%-analysis.ncu-rep && pause'"
) else (
    echo Running...
    echo.
    "%OUTPUT%"
)
