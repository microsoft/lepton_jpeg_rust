@echo off
setlocal enabledelayedexpansion
for  /r %1 %%f in (*.jpg)  do call verifycompression.cmd ^"%%f"^