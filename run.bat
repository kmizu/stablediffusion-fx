@echo off
rem Stable Diffusion NUI launcher. Prints nothing to stdout itself (MCP stdio mode uses stdout).
setlocal EnableExtensions
cd /d "%~dp0"

set "SDFX_JAVA="
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" set "SDFX_JAVA=%JAVA_HOME%\bin\java.exe"
if not defined SDFX_JAVA for /f "delims=" %%J in ('where java 2^>nul') do if not defined SDFX_JAVA set "SDFX_JAVA=%%J"
if not defined SDFX_JAVA (
  >&2 echo [sdfx] Java 17 or newer was not found. Install a JDK 17+ and set JAVA_HOME or add java to PATH.
  exit /b 1
)

set "SDFX_VER="
for /f "tokens=1,2,3" %%A in ('"%SDFX_JAVA%" -version 2^>^&1') do if /i "%%B"=="version" if not defined SDFX_VER set "SDFX_VER=%%~C"
set "SDFX_MAJOR="
for /f "tokens=1 delims=." %%M in ("%SDFX_VER%") do set "SDFX_MAJOR=%%M"
set /a SDFX_MAJORN=0
if defined SDFX_MAJOR set /a SDFX_MAJORN=%SDFX_MAJOR% 2>nul
if %SDFX_MAJORN% LSS 17 (
  >&2 echo [sdfx] Java 17 or newer is required ^(found "%SDFX_VER%" at %SDFX_JAVA%^).
  exit /b 1
)

set "SDFX_JAR=%~dp0target\stablediffusion-nui.jar"
if not exist "%SDFX_JAR%" (
  >&2 echo [sdfx] target\stablediffusion-nui.jar is missing, building it with sbt ^(or run scripts\build.bat^)...
  where sbt >nul 2>nul
  if errorlevel 1 (
    >&2 echo [sdfx] sbt was not found. Install sbt, then run scripts\build.bat.
    exit /b 1
  )
  call sbt -batch ui/assembly 1>&2
  if errorlevel 1 (
    >&2 echo [sdfx] build failed.
    exit /b 1
  )
)

set "SDFX_MCP="
for %%A in (%*) do if /i "%%~A"=="--mcp" set "SDFX_MCP=1"

if defined SDFX_MCP (
  "%SDFX_JAVA%" -Xlog:disable -Xlog:all=warning:stderr -Djava.awt.headless=true -Dfile.encoding=UTF-8 -Xmx2g -Dsdfx.home="%~dp0." -cp "%SDFX_JAR%" sdfx.mcp.McpMain %*
) else (
  "%SDFX_JAVA%" -Dprism.allowhidpi=true -Dfile.encoding=UTF-8 -Xmx2g -Dsdfx.home="%~dp0." -jar "%SDFX_JAR%" %*
)
exit /b %ERRORLEVEL%
