# SchubKI 2026: camera-only race-car localization (Constructor Labs challenge)

Causal 10 Hz pose estimation (x, y, yaw) on Yas Marina from the front camera, the track map and car-state speed.
No GNSS, LiDAR, IMU or ground truth at inference.

- Slides (6, PDF): [docs/slides.pdf](docs/slides.pdf)
- Video walkthrough (3:09): [docs/demo_video.mp4](docs/demo_video.mp4)
- Hidden test poses: [results/test_poses.csv](results/test_poses.csv) (`timestamp_ns,x,y,yaw`, 739 poses, run metadata in `results/test_poses.run.json`)
- Validation results (official `evaluate.py`, laps 6-7): `results/validation_lap*_score*.txt`, poses `results/validation_lap*_poses*.csv`
- Trajectory plots vs ground truth on the track map: `results/trajectory_lap6.png`, `results/trajectory_lap7.png`; test: `results/trajectory_test.png`

## Approach

1. **Place descriptors.** torchvision ViT-B/16 (ImageNet-1k init), last two encoder blocks + final norm fine-tuned with
   triplet loss on training laps 1-5 (cross-lap positives <=5 m / 15 deg, negatives >=25 m, static hard-negative mining).
   Training code: `training/backend/race_localization/vit_hour_finetune.py`, `vit_laptop_finetune.py`, `place_model.py`.
   Submitted weights: step 7750 (`models/inference-best.card.json`).
2. **Gallery.** Descriptors of 8,269 front-left training images (laps 1-5) with their training poses.
3. **Causal track odometry (submitted, `localizer/track_odometry.py`).** Arc length `s(t) = s0 + integral(speed dt)` from
   `/constructor0/validation/twist`; the start offset `s0` is a running, decaying, hysteresis-protected vote of
   `s(top-k retrieval) - travelled`. Pose = racing line of the track map at `s(t)`, yaw = line tangent. No first pose needed.
4. Alternatives in the repo: `motion_tracker.py` (multi-hypothesis retrieval + twist), `kiro_localize.py` retrieval filter.

## Results (validation laps 6-7, official scorer, coverage 100 %)

| Method | Lap | Position RMSE | Lateral RMSE | Yaw RMSE | <1 m |
|---|---|---:|---:|---:|---:|
| Track odometry (submitted) | 6 / 7 | 2.64 / 1.97 m | 1.35 / 1.26 m | 2.72 / 2.29 deg | 19 / 14 % |
| Multi-hypothesis tracker | 6 / 7 | 1.79 / 1.83 m | 0.63 / 0.83 m | 0.79 / 1.02 deg | 69 / 63 % |
| Retrieval filter (no motion, first pose) | 6 / 7 | 2.45 / 2.91 m | 0.82 / 0.84 m | 1.07 / 1.31 deg | 53 / 56 % |

The training run has no twist topic, so validation speed is a ground-truth-derived proxy (|d position / dt|); the hidden test
uses the real car-state twist. Laps 6-7 were seen in earlier team baselines, so they are reserved validation, not pristine data.
The multi-hypothesis tracker is better at night but jumped between straights on the daylight test run; track odometry
was selected for the test because it stays continuous (see `results/trajectory_test.png`).

**Runtime:** 75 ms/frame ViT encode (batch 12) + 0.6 ms/frame tracking = ~13 Hz on AMD Ryzen AI 7 350, Radeon 860M iGPU
(DirectML), 32 GB RAM, Windows 11. Faster than the 10 Hz camera.

## Run

```powershell
python -m pip install -r requirements.txt
# put inference-best.pt (release asset) and gallery-laps1-5.npz into models/vit7750/
python localizer/build_gallery.py TRAIN.mcap models/vit7750/gallery-laps1-5.npz   # rebuild gallery from the raw training run
.\run_submission.ps1 -Mcap TEST.mcap -Map map.mcap        # Linux: ./run_submission.sh TEST.mcap map.mcap
```

Validation: `localizer/kiro_localize.py validate` encodes laps 6-7, `localizer/validate_track.py` / `validate_motion_tracker.py`
run the trackers, `localizer/score_and_plot.py` calls the official scorer and draws the plots.

## Limitations and failure cases

- Night-only training; daylight test retrieval is weak. The track constraint assumes the car is near the racing line,
  so the true lateral offset is not observed.
- Repetitive straights dominate longitudinal error. One correction jump in the first seconds of the test run.
- `build_gallery.py` was written for reproducibility at the deadline and has not been executed end to end; the submitted
  gallery was built from the same images by the training pipeline (descriptor equality verified, cosine 1.0).
- A final 1-hour retraining attempt produced 0 new updates (time budget spent on baseline encoding and pair mining);
  the submitted model is the earlier step-7750 fine-tune.

## Declarations

Pre-trained: torchvision ViT-B/16 ImageNet-1k weights (BSD-3). Libraries: PyTorch, torchvision, torch-directml, NumPy,
Pillow, mcap, matplotlib. Data: Constructor Labs hackathon dataset only (not redistributed here). AI tools: OpenAI Codex,
Kiro (Claude) for code generation and review; the team is responsible for the result. Code: MIT (see LICENSE).
