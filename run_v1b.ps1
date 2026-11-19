# Train DAW Vocal Denoiser v1b from scratch: v1c's recipe (same data, 108,779 steps) with bigger GRU (384 hidden units).
# Data locations come from DAW_DATA / DNS_DATA (see README: Data paths).
#   .\run_v1b.ps1            first start
#   .\run_v1b.ps1 -Resume    resume after a pause or crash (nothing lost)
param([switch]$Resume)
Set-Location $PSScriptRoot
$env:DAW_HIDDEN = '384'; $env:DAW_LOOKAHEAD = '0'; $env:DAW_OWN_EXTRA = 'own_vocals_extra'
if ($Resume) { py -3.12 train.py --resume --workers 2 } else { py -3.12 train.py --steps 108779 --workers 2 }
py -3.12 export_onnx.py     # one-frame streaming ONNX model
py -3.12 check_parity.py    # streaming engine vs PyTorch, CPU time per block