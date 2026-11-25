# Train DAW Vocal Denoiser v1_clean: v1 with the 2026-11-25 leakage fix.
# Same model, same hyperparameters, same 108,773-step budget as v1; the ONLY difference is the data fix in
# ..\..\denoiser\dataset.py (VocalSet copies filtered out of the DNS speech branch, see
# denoiser-report\LEAKAGE_AUDIT_2026-11-25.md). Pinned to GPU 1 (RTX 5070): GPU 0 runs the TTS training and is
# never used here. Checkpoints, MLflow db and the ONNX export land in this folder (DAW_RUN_DIR).
#   .\run.ps1           first start
#   .\run.ps1 -Resume   continue after a pause or crash (nothing lost)
param([switch]$Resume)
Set-Location $PSScriptRoot
$env:DAW_RUN_DIR = $PSScriptRoot; $env:DAW_HIDDEN = '256'; $env:DAW_LOOKAHEAD = '0'; $env:DAW_OWN_EXTRA = ''
$env:DNS_DATA = 'C:\Users\stavi\Projects\audio-ml-pipeline\data'      # same data roots v1 used
$env:DAW_DATA = 'C:\Users\stavi\Projects\datasets\daw-denoiser'
$env:CUDA_VISIBLE_DEVICES = '1'                                       # GPU 1 (RTX 5070) only, GPU 0 is the TTS card
if ($Resume) { py -3.12 ..\..\denoiser\train.py --resume --workers 4 }
else {
    py -3.12 ..\..\denoiser\train.py --steps 108773 --workers 4       # v1 was --hours 9 = 108,773 steps
}
py -3.12 ..\..\denoiser\export_onnx.py     # -> daw_denoiser.onnx in this folder
py -3.12 ..\..\denoiser\check_parity.py    # streaming engine vs PyTorch, CPU time per block
