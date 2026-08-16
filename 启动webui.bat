@echo off
chcp 65001

title FireRedTTS3 小红书TTS3 整合包 0817 --by pyvideotrans.com

rem set https_proxy=http://127.0.0.1:10808
set NO_PROXY=localhost,127.0.0.1,api.gradio.app
set GRADIO_ANALYTICS_ENABLED=False

set "MY_RUNTIME=%~dp0runtime"
set "PATH=%MY_RUNTIME%;%PATH%"

echo.


call runtime\python webui.py

pause