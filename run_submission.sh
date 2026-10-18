#!/usr/bin/env bash
# Usage: ./run_submission.sh TEST.mcap map.mcap
set -euo pipefail; cd "$(dirname "$0")"; PY="${PYTHON:-python3}"; mkdir -p cache results
"$PY" localizer/kiro_localize.py test --mcap "$1" --query-cache cache/test.npz --output results/test_poses_retrieval_only.csv
"$PY" localizer/extract_twist.py "$1" cache/test_twist.npy
"$PY" -c "import numpy as np,sys; sys.path.insert(0,'localizer'); import track_odometry as to; L=to.Line(to.racing_line(sys.argv[1])); g=np.load('models/vit7750/gallery-laps1-5.npz'); np.save('cache/gallery_s.npy', L.project(g['poses'][:,:2]))" "$2"
"$PY" localizer/run_test_track.py "$2"
