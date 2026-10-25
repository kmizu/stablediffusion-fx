@echo off
setlocal
cd /d "%~dp0.."
call sbt -batch ui/assembly
exit /b %ERRORLEVEL%
