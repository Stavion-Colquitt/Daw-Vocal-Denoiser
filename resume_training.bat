@echo off
rem Continue an interrupted run from checkpoints\latest.pth (same total length, same MLflow run).
cd /d "%~dp0"
powershell -NoProfile -Command "Start-Process py -ArgumentList '-3.12','train.py','--resume','--hours','9' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput train_resume.log -RedirectStandardError train_resume.err"
echo Resumed in the background. Progress: train_resume.log
pause
