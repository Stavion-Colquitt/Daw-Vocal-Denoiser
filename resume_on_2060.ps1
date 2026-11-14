# Resume the interrupted run from checkpoints\latest.pth on the RTX 2060 only, then export + parity + held-out eval.
# Pins by PCI bus order (2060 = bus 04 = index 0) so the job can never land on the 4060.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
$log = Join-Path $here 'after_training.log'
"resume on 2060 started $(Get-Date)" | Out-File $log
py -3.12 train.py --resume --hours 9 1> train_resume.log 2> train_resume.err
"training finished $(Get-Date)" | Out-File $log -Append
py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
py -3.12 eval.py 200 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
"ALL DONE $(Get-Date)" | Out-File $log -Append
