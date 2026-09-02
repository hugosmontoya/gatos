@echo off
cd /d "%~dp0"
echo ========================================================
echo   Iniciando servidor local para Gesture Meme Detector
echo ========================================================
echo.
echo Abriendo en tu navegador: http://localhost:8000
echo Recuerda presionar "Permitir" cuando te pida acceso a la camara.
echo.
echo (Para cerrar el servidor, solo cierra esta ventana)
echo.

start http://localhost:8000
python -m http.server 8000

pause
