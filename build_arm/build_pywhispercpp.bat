@echo off
REM ============================================================================
REM Build pywhispercpp (whisper.cpp Python binding) natively on Windows-on-ARM64.
REM
REM There is no prebuilt win_arm64 wheel, and whisper.cpp/ggml refuses to build
REM with MSVC on ARM ("MSVC is not supported for ARM, use clang"). This builds it
REM with the rtools45 MinGW-w64 clang (LLVM 19, native arm64), via the Ninja
REM generator so pywhispercpp's setup.py drops its hardcoded "-A ARM64" (which
REM would force MSVC).
REM
REM Prereqs:
REM   * Native ARM64 Python venv at %VENV% (see BUILD_ARM.md).
REM   * rtools45-aarch64 installed at %RTOOLS% (provides clang/clang++/cmake).
REM   * ninja on PATH (this uses the one bundled with VS Build Tools; any works).
REM
REM Usage:  build_arm\build_pywhispercpp.bat
REM ============================================================================
setlocal
set "VENV=C:\VYSERIX\ClaudeCode\Transcriber\Transcriber\.venv-arm64"
set "RTOOLS=C:\rtools45-aarch64\aarch64-w64-mingw32.static.posix\bin"
set "NINJADIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "SHIM=%~dp0pt_shim.h"

set "PATH=%RTOOLS%;%NINJADIR%;%PATH%"
set "CMAKE_GENERATOR=Ninja"
set "CC=clang"
set "CXX=clang++"
REM Shim the missing THREAD_POWER_THROTTLING_STATE (MinGW header gap), and disable
REM pybind11's hidden-visibility namespace (clang-mingw rejects hidden+dllexport).
set "CFLAGS=-include %SHIM%"
set "CXXFLAGS=-include %SHIM% -DPYBIND11_NAMESPACE=pybind11"

echo === toolchain ===
where clang & where ninja & where cmake

echo === building + installing pywhispercpp into the ARM venv ===
"%VENV%\Scripts\python.exe" -m pip install pywhispercpp --no-cache-dir
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )

REM The MinGW build tags the extension "win_amd64" even though it is ARM64 code;
REM rename it to the suffix this interpreter expects so it can be imported.
for /f "delims=" %%S in ('"%VENV%\Scripts\python.exe" -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set "EXTSUFFIX=%%S"
set "SITE=%VENV%\Lib\site-packages"
if exist "%SITE%\_pywhispercpp.cp313-win_amd64.pyd" (
    echo Renaming misnamed extension -> _pywhispercpp%EXTSUFFIX%
    move /Y "%SITE%\_pywhispercpp.cp313-win_amd64.pyd" "%SITE%\_pywhispercpp%EXTSUFFIX%"
)

echo === verify import ===
"%VENV%\Scripts\python.exe" -c "import _pywhispercpp; from pywhispercpp.model import Model; print('pywhispercpp import OK (native ARM64)')"
endlocal
