# Wide-view ViT retrieval: 5 crops with shift-aligned matching

Inference-only change on top of the frozen step-7750 ViT-B/16 release. No new weights.

## Method

The single-crop descriptor (Resize 256, CenterCrop 224) sees about 42% of the 1506 px image width. Here each frame gives 5 overlapping 224x224 crops at the same Resize-256 scale, spanning the full width (x = 0, 76, 152, 228, 305 at 529 px; the middle crop is exactly the old CenterCrop). Each crop yields its normalized CLS embedding.

Similarity between a query and a gallery frame is the maximum over horizontal shifts s in {-2..2} of the mean cosine between query crop k and gallery crop k+s. The shift absorbs yaw and lateral offset. Only the current query image and the frozen gallery are used.

## Lap-3 retrieval (gallery laps 1-2, 4497 frames; 1236 queries; same 5 m/15 deg positive and 25 m distant rules as `vit_finetune.retrieval_metrics`)

| Descriptor | R1 | R5 | Distant top-1 |
|---|---|---|---|
| Step 7750 centre crop (reproduced) | 822 | 901 | 53 |
| 5-crop mean | 867 | 924 | 42 |
| 5-crop aligned, shifts -1..1 | 873 | 939 | 41 |
| 5-crop aligned, shifts -2..2 | 888 | 956 | 40 |

These are retrieval proxies, not official pose, coverage or runtime scores.

## Caveats

- About 16 variants were compared on lap 3, so the selected result is optimistic. Confirm on development laps 4-5 through the coordinator before adoption. Laps 4-7 were not used.
- Learned heads on the frozen features (linear projection and per-crop weighting, multi-similarity and InfoNCE losses, laps 1-2 gradients) did not beat the training-free crop mean: the backbone already separates laps 1-2, so training loss collapses to about 0 without lap-3 gain.
- Cost: 5 ViT forwards per frame and 25 crop dot products per gallery pair instead of 1. Frame runtime is not yet measured. The place map must store 5x768 per keyframe.
- Not yet wired into `vit_place_localization`.

## Reproduce

```powershell
$Py = '<venv>\Scripts\python.exe'
& $Py backend\tools\eval_vit_multicrop.py extract --dataset-dir <prepared>\dataset --dataset-sha256 <sha> --cache-root <prepared>\cache-fl --weights <inference-best.pt> --work <private work dir>
& $Py backend\tools\eval_vit_multicrop.py evaluate --dataset-dir <prepared>\dataset --dataset-sha256 <sha> --cache-root <prepared>\cache-fl --work <private work dir>
```

Extraction took 276 s for 5733 frames on an RTX 5070 Ti Laptop (JPEG decode bound). Feature caches and metric files stay in the private work directory, never in Git.
