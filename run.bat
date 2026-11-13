@echo off
title AI Multi-Modal Interaction System
echo ============================================================
echo   AI-Based Multi-Modal Human Interaction System
echo   B.Tech Data Science Major Project
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start Flask application
echo Starting server at http://localhost:5000 ...
python app.py

pause
