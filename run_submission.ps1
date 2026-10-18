# Single run script: raw test mcap -> results/test_poses.csv (causal, camera + car-state speed + track map).
# Usage: .\run_submission.ps1 -Mcap TEST.mcap -Map map.mcap [-Python python]
param([Parameter(Mandatory)][string]$Mcap, [Parameter(Mandatory)][string]$Map, [string]$Python = "python")
$ErrorActionPreference = "Stop"; Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force cache, results | Out-Null
if (!(Test-Path models/vit7750/inference-best.pt)) { throw "Download inference-best.pt and gallery-laps1-5.npz from the GitHub release into models/vit7750/" }
& $Python localizer/kiro_localize.py test --mcap $Mcap --query-cache cache/test.npz --output results/test_poses_retrieval_only.csv
& $Python localizer/extract_twist.py $Mcap cache/test_twist.npy
& $Python -c "import numpy as np,sys; sys.path.insert(0,'localizer'); import track_odometry as to; L=to.Line(to.racing_line(sys.argv[1])); g=np.load('models/vit7750/gallery-laps1-5.npz'); np.save('cache/gallery_s.npy', L.project(g['poses'][:,:2]))" $Map
& $Python localizer/run_test_track.py $Map
Write-Host "Wrote results/test_poses.csv"
