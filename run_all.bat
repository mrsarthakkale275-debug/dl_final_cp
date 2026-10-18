@echo off
title Launch AdaptiveInspect AI System
echo ========================================================
echo Launching AdaptiveInspect AI (Backend + Frontend)
echo ========================================================
echo Starting Backend in separate window...
start "AdaptiveInspect AI - Backend (Port 8000)" cmd /k "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting Frontend in separate window...
start "AdaptiveInspect AI - Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo Both servers launched successfully!
echo Web Application: http://localhost:5173
echo Backend API Docs: http://127.0.0.1:8000/docs
echo ========================================================
timeout /t 5
