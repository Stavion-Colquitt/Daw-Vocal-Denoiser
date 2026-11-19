# Train DAW Vocal Denoiser v1b with the shared code in denoiser/. Checkpoints, MLflow db and the ONNX export land in
# this folder (DAW_RUN_DIR). Data locations: DAW_DATA / DNS_DATA (see the main README, Data paths).
# v1b = v1c's data and steps with a wider GRU (384).
#   .\run.ps1           first start
#   .\run.ps1 -Resume   continue after a pause or crash (nothing lost)
param([switch]$Resume)
Set-Location $PSScriptRoot
$env:DAW_RUN_DIR = $PSScriptRoot; $env:DAW_HIDDEN = '384'; $env:DAW_LOOKAHEAD = '0'; $env:DAW_OWN_EXTRA = 'own_vocals_extra'
if ($Resume) { py -3.12 ..\..\denoiser\train.py --resume --workers 4 }
else {
    py -3.12 ..\..\denoiser\train.py --steps 108779 --workers 4
}
py -3.12 ..\..\denoiser\export_onnx.py     # -> daw_denoiser.onnx in this folder
py -3.12 ..\..\denoiser\check_parity.py    # streaming engine vs PyTorch, CPU time per block
