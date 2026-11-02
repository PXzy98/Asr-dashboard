@echo off
rem ASR Dashboard launcher (Windows). Requires uv: https://docs.astral.sh/uv/
chcp 65001 >nul
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv not found. Install it with:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [WARN] FFmpeg not found in PATH. Install with: winget install Gyan.FFmpeg
)
if not exist ".venv" (
  echo First run: installing dependencies ^(CUDA 12.8 PyTorch^). Use "uv sync --extra cpu" for CPU only.
  uv sync --extra cu128
  if errorlevel 1 ( pause & exit /b 1 )
)
uv run --no-sync python main.py %*
if errorlevel 1 pause
