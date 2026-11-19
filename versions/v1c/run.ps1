# Train DAW Vocal Denoiser v1c with the shared code in denoiser/. Checkpoints, MLflow db and the ONNX export land in
# this folder (DAW_RUN_DIR). Data locations: DAW_DATA / DNS_DATA (see the main README, Data paths).
# v1c = a 47,079-step main run, then a 61,700-step extension (re-warmed to 3e-4, cosine to 1e-5): 108,779 steps in total.
#   .\run.ps1           first start
#   .\run.ps1 -Resume   continue after a pause or crash (nothing lost)
param([switch]$Resume)
Set-Location $PSScriptRoot
$env:DAW_RUN_DIR = $PSScriptRoot; $env:DAW_HIDDEN = '256'; $env:DAW_LOOKAHEAD = '0'; $env:DAW_OWN_EXTRA = 'own_vocals_extra'
if ($Resume) { py -3.12 ..\..\denoiser\train.py --resume --workers 4 }
else {
    py -3.12 ..\..\denoiser\train.py --steps 47079 --workers 4
    py -3.12 ..\..\denoiser\train.py --resume --extend_steps 61700 --workers 4
}
py -3.12 ..\..\denoiser\export_onnx.py     # -> daw_denoiser.onnx in this folder
py -3.12 ..\..\denoiser\check_parity.py    # streaming engine vs PyTorch, CPU time per block
