@echo off
cd /d "%~dp0"
echo Cleaning previous build...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del PPLDetector.spec 2>nul

echo Building exe...
pyinstaller ^
  --onefile ^
  --windowed ^
  --name PPLDetector ^
  --collect-all torch ^
  --collect-all transformers ^
  --collect-all tokenizers ^
  --hidden-import huggingface_hub ^
  --hidden-import safetensors ^
  --hidden-import regex ^
  --hidden-import filelock ^
  --hidden-import numpy ^
  main.py

echo.
echo Done. Output: dist\PPLDetector.exe
pause
