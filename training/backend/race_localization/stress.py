"""Predetermined calibrated RGB racing stresses, with evaluator-only query labels.

The scene and poses generate images, never feature correspondences. Reference
mapping and query localization use the ordinary image-derived vision pipeline.
This is a fabricated circuit segment, not evidence of real racing performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .synthetic import demo_calibration, render_image


SEED = 20261007
REFERENCE_COUNT = 48
TRAIN_COUNT = 36
DEVELOPMENT_COUNT = 35
AUDIT_COUNT = 31
CAPTURE_INTERVAL_S = 0.12
ROAD_HALF_WIDTH_M = 4.5


def _centerline(s: float) -> tuple[np.ndarray, float]:
    """Arc-length parameterized straight, left corner, then right corner."""
    s = float(s)
    if s <= 10.0:
        return np.array([s, 0.0, 0.0]), 0.0
    # A 20 m radius left turn spans 1.2 rad, followed by a gentler right turn.
    if s <= 34.0:
        angle = (s - 10.0) / 20.0
        return np.array([10.0 + 20.0 * np.sin(angle),
                         20.0 * (1.0 - np.cos(angle)), 0.0]), angle
    entry = np.array([10.0 + 20.0 * np.sin(1.2),
                      20.0 * (1.0 - np.cos(1.2)), 0.0])
    length = min(s - 34.0, 24.0)
    angle = 1.2 - length / 30.0
    end = entry + [30.0 * (np.sin(1.2) - np.sin(angle)),
                   30.0 * (np.cos(angle) - np.cos(1.2)), 0.0]
    if s > 58.0:
        end += (s - 58.0) * np.array([np.cos(0.4), np.sin(0.4), 0.0])
    return end, angle


def _point(s: float, lateral: float, height: float = 0.0) -> np.ndarray:
    center, yaw = _centerline(s)
    return center + [-np.sin(yaw) * lateral, np.cos(yaw) * lateral, height]


def _vehicle_pose(s: float, lateral: float, yaw_offset: float = 0.0) -> np.ndarray:
    center, yaw = _centerline(s)
    yaw += yaw_offset
    c, sn = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]
    # Lateral displacement uses the track normal, not the perturbed camera yaw.
    pose[:3, 3] = _point(s, lateral)
    return pose


def _asphalt(index: int, low_texture: bool) -> np.ndarray:
    if low_texture:
        return np.full((240, 320, 3), (56, 57, 59), np.uint8)
    rng = np.random.default_rng(SEED + 101 * index)
    fine = rng.normal(0, 8, (240, 320)).astype(np.float32)
    coarse = cv2.resize(rng.normal(0, 9, (24, 32)).astype(np.float32), (320, 240))
    gray = np.clip(57 + fine + coarse, 15, 108).astype(np.uint8)
    texture = np.repeat(gray[:, :, None], 3, axis=2)
    for _ in range(24):
        a = tuple(int(v) for v in rng.integers([0, 0], [320, 240]))
        b = (min(319, a[0] + int(rng.integers(5, 48))),
             min(239, a[1] + int(rng.integers(3, 18))))
        shade = int(rng.integers(26, 106))
        cv2.line(texture, a, b, (shade, shade, shade), 1, cv2.LINE_AA)
    return texture


def _barrier(index: int, repeated: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED + 9000 + (0 if repeated else index * 53))
    texture = np.full((240, 320, 3), (157, 164, 165), np.uint8)
    noise = rng.integers(-17, 18, (240, 320, 1), dtype=np.int16)
    texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Every panel shares the same structural red/white safety pattern.
    for j in range(8):
        color = (36, 44, 175) if j % 2 else (224, 226, 225)
        cv2.rectangle(texture, (40 * j, 148), (40 * j + 39, 216), color, -1)
    cv2.rectangle(texture, (0, 0), (319, 13), (76, 80, 82), -1)
    if not repeated:
        for _ in range(30):
            x, y = (int(v) for v in rng.integers([0, 20], [310, 140]))
            color = tuple(int(v) for v in rng.integers(45, 215, 3))
            cv2.rectangle(texture, (x, y), (min(319, x + 13), y + 8), color, -1)
    return texture


def _sign(index: int, repeated: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED + 17000 + (0 if repeated else index * 149))
    texture = np.full((220, 300, 3), (30, 56, 35), np.uint8)
    cv2.rectangle(texture, (3, 3), (296, 216), (225, 234, 229), 7)
    label = "RACE" if repeated else ("APEX", "PIT", "S-BEND", "EXIT")[index % 4]
    cv2.putText(texture, label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (235, 240, 232), 3, cv2.LINE_AA)
    for _ in range(24):
        center = tuple(int(v) for v in rng.integers([15, 98], [285, 200]))
        color = tuple(int(v) for v in rng.integers(40, 245, 3))
        cv2.circle(texture, center, int(rng.integers(3, 11)), color, -1)
    return texture


def _scene(low_texture: bool = False, repeated: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed world surfaces, including noncoplanar barriers and signs."""
    surfaces = []
    samples = np.arange(-12.0, 85.0, 2.8)
    for index, (a, b) in enumerate(zip(samples[:-1], samples[1:])):
        for lane in range(3):
            left = -ROAD_HALF_WIDTH_M + lane * 3.0
            right = left + 3.0
            corners = np.array([_point(a, left), _point(b, left),
                                _point(b, right), _point(a, right)])
            surfaces.append((corners, _asphalt(index * 3 + lane, low_texture)))
        for side in (-1, 1):
            inner, outer = side * 4.5, side * 4.85
            curb = np.full((100, 320, 3), (235, 236, 235), np.uint8)
            for j in range(8):
                if j % 2:
                    cv2.rectangle(curb, (40 * j, 0), (40 * j + 39, 99), (32, 39, 188), -1)
            surfaces.append((np.array([_point(a, inner), _point(b, inner),
                                       _point(b, outer), _point(a, outer)]), curb))
            lateral = side * 5.45
            surfaces.append((np.array([_point(a, lateral), _point(b, lateral),
                                       _point(b, lateral, 2.8), _point(a, lateral, 2.8)]),
                             _barrier(index * 2 + (side == 1), repeated)))
        if index % 3 == 1:
            # Boards face along the segment, yielding a second surface direction.
            for side in (-1, 1):
                lateral = side * 4.95
                surfaces.append((np.array([_point(b, lateral, 1.3),
                                           _point(b, lateral - side * 1.2, 1.3),
                                           _point(b, lateral - side * 1.2, 3.5),
                                           _point(b, lateral, 3.5)]),
                                 _sign(index * 2 + (side == 1), repeated)))
    return surfaces


def _track() -> dict:
    samples = np.linspace(-5.0, 62.0, 150)
    return {"centerline": [_point(s, 0).tolist() for s in samples],
            "left_boundary": [_point(s, ROAD_HALF_WIDTH_M).tolist() for s in samples],
            "right_boundary": [_point(s, -ROAD_HALF_WIDTH_M).tolist() for s in samples]}


def _intervention(kind: str, start: int, end: int, **parameters) -> dict:
    return {"kind": kind, "start_frame": start, "end_frame": end, "parameters": parameters}


def condition_definitions() -> list[dict]:
    """Prespecified development conditions; bounds are inclusive capture indices."""
    last = DEVELOPMENT_COUNT - 1
    return [
        {"id": "turns", "label": "Textured S-bend baseline", "interventions": []},
        {"id": "texture-loss", "label": "Low-texture asphalt", "interventions": [
            _intervention("asphalt_texture_loss", 0, last, surface="road", gray_bgr=[56, 57, 59])]},
        {"id": "repeated-barriers", "label": "Repeated barriers and signs", "interventions": [
            _intervention("repeated_world_texture", 0, last, surfaces=["barriers", "signs"])]},
        {"id": "motion-blur", "label": "Motion blur through the corners", "interventions": [
            _intervention("horizontal_motion_blur", 8, 27, kernel_px=19)]},
        {"id": "illumination", "label": "Exposure and side shadow variation", "interventions": [
            _intervention("illumination_variation", 0, last, gain_range=[0.43, 1.10], shadow_gain=0.48)]},
        {"id": "combined", "label": "Combined texture, repetition, blur and light", "interventions": [
            _intervention("asphalt_texture_loss", 0, last, surface="road", gray_bgr=[56, 57, 59]),
            _intervention("repeated_world_texture", 0, last, surfaces=["barriers", "signs"]),
            _intervention("horizontal_motion_blur", 8, 27, kernel_px=19),
            _intervention("illumination_variation", 0, last, gain_range=[0.43, 1.10], shadow_gain=0.48)]},
        {"id": "outage-recovery", "label": "Camera occlusion and visual recovery", "interventions": [
            _intervention("camera_occlusion", 12, 20, mean_bgr=[9, 11, 13], noise_sigma=0.35,
                          recovery_frame=21)]},
    ]


def _blur(image: np.ndarray, kernel_px: int = 19) -> np.ndarray:
    kernel = np.zeros((kernel_px, kernel_px), np.float32)
    kernel[kernel_px // 2, :] = 1.0 / kernel_px
    return cv2.filter2D(image, -1, kernel)


def _illuminate(image: np.ndarray, index: int) -> np.ndarray:
    gain = 0.765 + 0.335 * np.sin(index * 0.25 + 0.5)
    width = image.shape[1]
    horizontal = np.linspace(0, 1, width)[None, :, None]
    shadow_edge = 0.5 + 0.20 * np.sin(index * 0.18)
    shadow = 0.48 + 0.52 * np.clip((horizontal - shadow_edge + 0.10) / 0.20, 0, 1)
    return np.clip(image.astype(float) * gain * shadow + 4, 0, 255).astype(np.uint8)


def _occlude(image: np.ndarray, index: int, seed_offset: int) -> np.ndarray:
    # Sensor noise provides independent content without a visible ID or pose code.
    rng = np.random.default_rng(SEED + seed_offset + index * 307)
    noise = rng.normal(0, 0.35, image.shape)
    return np.clip(np.array([9, 11, 13]) + noise, 0, 255).astype(np.uint8)


def _trajectory(split: str) -> list[np.ndarray]:
    definitions = {
        "reference": (REFERENCE_COUNT, -2.0, 52.0, 0.0, 0.0),
        "train": (TRAIN_COUNT, 0.17, 47.63, -0.38, -0.013),
        "development": (DEVELOPMENT_COUNT, 0.61, 48.17, 0.28, 0.009),
        "audit": (AUDIT_COUNT, 1.11, 48.71, -0.68, 0.023),
    }
    count, start, end, lateral, yaw_offset = definitions[split]
    poses = []
    for index, s in enumerate(np.linspace(start, end, count)):
        wobble = 0 if split == "reference" else 0.10 * np.sin(index / 5.0)
        phase = 0 if split == "reference" else 0.004 * np.sin(index / 7.0)
        poses.append(_vehicle_pose(s, lateral + wobble, yaw_offset + phase))
    return poses


def generate_stress_suite(output_dir: Path) -> dict:
    """Write 360 canonical RGB frames and independent metric vehicle labels.

    Split complete trajectories first. All seven variants of the development
    trajectory retain its group and split. Audit is a separate untouched path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = demo_calibration()
    vehicle_from_camera = np.linalg.inv(np.asarray(calibration["T_CV"]))
    scenes = {"normal": _scene(), "texture-loss": _scene(low_texture=True),
              "repeated-barriers": _scene(repeated=True),
              "combined": _scene(low_texture=True, repeated=True)}
    track = _track()
    conditions = condition_definitions()
    sessions, ground_truth = [], {}
    definitions = [
        ("reference", "reference", "Metric reference route", []),
        ("train", "training", "Distinct training route", [
            _intervention("illumination_variation", 0, TRAIN_COUNT - 1,
                          gain_range=[0.75, 0.97]),
            _intervention("camera_occlusion", 14, 19, mean_bgr=[9, 11, 13], noise_sigma=0.35),
            _intervention("horizontal_motion_blur", 8, 11, kernel_px=11)]),
        *(("development", c["id"], c["label"], c["interventions"]) for c in conditions),
        ("audit", "held-out", "Held-out independent route", [
            _intervention("illumination_variation", 0, AUDIT_COUNT - 1,
                          gain_range=[0.62, 0.94]),
            _intervention("horizontal_motion_blur", 9, 13, kernel_px=15),
            _intervention("camera_occlusion", 19, 25, mean_bgr=[9, 11, 13], noise_sigma=0.35)]),
    ]
    development_base = []
    for split, condition, label, interventions in definitions:
        sid = f"stress-{split}-{condition}"
        folder = output_dir / "images" / sid
        folder.mkdir(parents=True, exist_ok=True)
        frames, labels = [], []
        poses = _trajectory(split)
        for index, vehicle_pose in enumerate(poses):
            camera_pose = vehicle_pose @ vehicle_from_camera
            if split == "development" and condition in {"motion-blur", "illumination", "outage-recovery"}:
                image = development_base[index].copy()
            else:
                scene_key = condition if condition in scenes else "normal"
                image = render_image(camera_pose, calibration, scenes[scene_key])
            if split == "development" and condition == "turns":
                development_base.append(image.copy())
            if split == "development":
                if condition in {"motion-blur", "combined"} and 8 <= index <= 27:
                    image = _blur(image)
                if condition in {"illumination", "combined"}:
                    image = _illuminate(image, index)
                if condition == "outage-recovery" and 12 <= index <= 20:
                    image = _occlude(image, index, 40000)
            if split == "train":
                image = np.clip(image.astype(float) * (0.86 + 0.11 * np.sin(index / 6.0)), 0, 255).astype(np.uint8)
                if 8 <= index <= 11:
                    image = _blur(image, 11)
                if 14 <= index <= 19:
                    image = _occlude(image, index, 70000)
            if split == "audit":
                image = np.clip(image.astype(float) * (0.78 + 0.16 * np.cos(index / 8.0)) + 5, 0, 255).astype(np.uint8)
                if 9 <= index <= 13:
                    image = _blur(image, 15)
                if 19 <= index <= 25:
                    image = _occlude(image, index, 100000)
            fid = f"{index:05d}"
            relative = f"images/{sid}/{fid}.png"
            if not cv2.imwrite(str(output_dir / relative), image):
                raise RuntimeError(f"Could not write stress image {relative}")
            timestamp = round(index * CAPTURE_INTERVAL_S, 8)
            frame = {"frame_id": fid, "timestamp_s": timestamp, "image": relative}
            if split == "reference":
                frame["T_WC"] = camera_pose.tolist()
            else:
                labels.append({"frame_id": fid, "timestamp_s": timestamp,
                               "position_m": vehicle_pose[:3, 3].tolist(),
                               "yaw_rad": float(np.arctan2(vehicle_pose[1, 0], vehicle_pose[0, 0]))})
            frames.append(frame)
        sessions.append({"id": sid, "name": f"Synthetic stress: {label}", "split": split,
                         "group_id": f"stress-original-{split}-route-v1", "frames": frames,
                         "track": track,
                         "scenario": {"id": condition, "label": label,
                                      "trajectory_id": f"stress-original-{split}-route-v1",
                                      "interventions": interventions}})
        if labels:
            ground_truth[sid] = labels
    benchmark = {"kind": "synthetic_stress", "version": 1, "seed": SEED,
                 "description": "Fabricated calibrated S-bend circuit segment. No real racing accuracy claim.",
                 "conditions": conditions,
                 "split_policy": {"unit": "original_trajectory", "development_variants_grouped": True,
                                  "audit_usage": "held_out", "reference_covers_query_route": True},
                 "scene": {"units": "meters", "world_frame": "synthetic_metric_circuit",
                           "road_width_m": 9.0, "left_turn_radius_m": 20.0,
                           "right_turn_radius_m": 30.0, "query_arc_length_range_m": [0.17, 48.71],
                           "reference_arc_length_range_m": [-2.0, 52.0],
                           "maximum_heading_rad": 1.2}}
    manifest = {"version": 1, "name": "Camera Lab synthetic racing stress suite v1",
                "source": "synthetic", "calibration": calibration, "evaluation": {},
                "benchmark": benchmark, "sessions": sessions, "ground_truth": ground_truth}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    scene_metadata = {"renderer": "synthetic.render_image projected world surfaces", "seed": SEED,
                      "benchmark": benchmark,
                      "limitations": ["Approximate painter-order occlusion and whole-surface near-plane rejection",
                                      "Piecewise planar raster textures, no photorealistic material or lighting model",
                                      "Image-space blur, exposure and sensor occlusion are controlled interventions",
                                      "No rolling shutter or continuous shutter exposure integration",
                                      "Synthetic results do not establish real racing accuracy"],
                      "manifest_sha256": hashlib.sha256((output_dir / "manifest.json").read_bytes()).hexdigest()}
    (output_dir / "scene.json").write_text(json.dumps(scene_metadata, indent=2), encoding="utf-8")
    return manifest
