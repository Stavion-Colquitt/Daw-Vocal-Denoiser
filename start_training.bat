@echo off
rem Double-click to start the all-day training run (about 9 hours), then export + test automatically.
rem Training runs hidden in the background: you can close this window. Progress: train.log, or MLflow (see README).
rem If the PC restarts mid-run, double-click resume_training.bat to continue where it left off.
cd /d "%~dp0"
if exist train.log ren train.log "train_%RANDOM%.log.old"
powershell -NoProfile -Command "Start-Process py -ArgumentList '-3.12','train.py','--hours','9' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput train.log -RedirectStandardError train.err; Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','after_training.ps1' -WorkingDirectory '%~dp0' -WindowStyle Hidden"
echo Training started in the background (about 9 hours). You can close this window.
pause
