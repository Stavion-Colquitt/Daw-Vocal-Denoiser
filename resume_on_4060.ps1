# Resume the DAW denoiser from checkpoints\latest.pth on the RTX 4060 only (2060 is busy with call v4),
# then export + parity + held-out eval. 4 data workers so it shares the CPU with the v4 run.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # bus 2B = RTX 4060 = index 1
$env:CUDA_VISIBLE_DEVICES = '1'
if (Test-Path train_resume.log) { Move-Item train_resume.log "train_resume_2060_$(Get-Date -Format HHmm).log.old" -Force }
$log = Join-Path $here 'after_training.log'
"resume on 4060 started $(Get-Date)" | Out-File $log
py -3.12 train.py --resume --hours 9 --workers 4 1> train_resume.log 2> train_resume.err
"training finished $(Get-Date)" | Out-File $log -Append
py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
py -3.12 eval.py 200 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
"ALL DONE $(Get-Date)" | Out-File $log -Append
