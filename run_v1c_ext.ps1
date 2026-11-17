# (Nov 16, Stavion: "when LM 2 is done on the 2060, have v1c train again") Continue v1c on the RTX 2060 from its
# finished checkpoint: +61,700 steps (v1c's 47,079 + this = 108,779, the same total as v1), re-warm to 3e-4, cosine to
# 1e-5. The GPU router starts it after LM 2 finishes and pauses it (pause.flag, nothing lost) while TTS needs the 2060.
#   First start:  .\run_v1c_ext.ps1            Resume after a pause/crash:  .\run_v1c_ext.ps1 -Resume
# Writes ext_done.txt when the extension, ONNX export and parity check are finished. eval.py is NOT run (Stavion decides).
param([switch]$Resume, [int]$ExtendSteps = 61700)
$run = 'C:\Users\stavi\Projects\daw-runs\v1c'; Set-Location $run
$log = Join-Path $run 'run_v1c.log'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'; $env:CUDA_VISIBLE_DEVICES = '0'; $env:DAW_OWN_EXTRA = 'own_vocals_extra'; $env:DAW_DATA = 'C:\Users\stavi\Projects\datasets\daw-denoiser'   # Nov 17: NVMe copy (D: USB HDD was the bottleneck)
Remove-Item pause.flag -ErrorAction SilentlyContinue
if (-not $Resume) {
    "extension started $(Get-Date) (+$ExtendSteps steps, RTX 2060)" | Out-File $log -Append
    if (Test-Path train.log) { Move-Item train.log "train_main_$(Get-Date -Format MMdd_HHmm).log" -Force }
    if (Test-Path checkpoints\best.pth) { Copy-Item checkpoints\best.pth checkpoints\best_main_run.pth -Force }   # keep the original v1c best
    py -3.12 train.py --resume --extend_steps $ExtendSteps 1>> train.log 2>> train.err
} else {
    "extension resumed $(Get-Date)" | Out-File $log -Append
    py -3.12 train.py --resume --workers 2 1>> train.log 2>> train.err
}
if (Select-String train.log -Pattern '^done\.' -Quiet) {
    "extension training finished $(Get-Date)" | Out-File $log -Append
    py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
    py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
    "EXTENSION DONE $(Get-Date)" | Out-File $log -Append
    "done $(Get-Date)" | Out-File ext_done.txt
}
