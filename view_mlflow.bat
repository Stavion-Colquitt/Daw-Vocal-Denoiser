@echo off
rem Opens the MLflow dashboard for this project at http://127.0.0.1:5003 (experiment "daw-vocal-denoiser").
cd /d "%~dp0"
start "" http://127.0.0.1:5003
py -3.12 -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5003
