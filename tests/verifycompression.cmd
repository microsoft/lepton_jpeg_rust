@echo off
..\target\release\lepton_jpeg_util.exe %1 nul > error.txt
if errorlevel 1 (
echo %1 failed %errorlevel%
echo ------------ >> failedlog.txt
echo %1 failed %errorlevel% >> failedlog.txt
type error.txt >> failedlog.txt
)