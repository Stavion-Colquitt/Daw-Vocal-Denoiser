# Waits for train.py to finish, then: ONNX export -> streaming parity/speed check -> held-out singing eval.
# Everything logs to after_training.log in this folder.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$log = Join-Path $here 'after_training.log'
"started $(Get-Date)" | Out-File $log
while (-not (Select-String -Path train.log -Pattern '^done' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
"training finished $(Get-Date)" | Out-File $log -Append
py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
py -3.12 eval.py 200 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
"ALL DONE $(Get-Date)" | Out-File $log -Append
