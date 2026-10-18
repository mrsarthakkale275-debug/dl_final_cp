@echo off
title AdaptiveInspect AI - Backend
echo ========================================================
echo Starting AdaptiveInspect AI Backend on http://127.0.0.1:8000
echo ========================================================
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
pause
