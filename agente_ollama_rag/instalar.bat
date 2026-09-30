@echo off
cd /d "%~dp0"

echo Verificando Python...
python --version

if errorlevel 1 (
    echo.
    echo Python nao foi encontrado no PATH.
    echo Instale o Python ou configure o PATH do Windows.
    pause
    exit /b 1
)

echo.
echo Atualizando pip...
python -m pip install --upgrade pip

echo.
echo Instalando dependencias...
python -m pip install -r requirements.txt

echo.
echo Instalacao concluida.
pause