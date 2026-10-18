"""Small calibrated RGB scene for end-to-end verification, not race realism.

The renderer knows its poses. The estimator receives only rendered images and
allowed reference poses. Query labels are written separately for the evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _vehicle_pose(x: float, lateral: float, phase: float = 0.0) -> np.ndarray:
    y = 0.55 * np.sin(x / 8.0) + lateral
    yaw = np.arctan2(0.55 / 8.0 * np.cos(x / 8.0), 1.0) + phase
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = [x, y, 0]
    return pose


def demo_calibration() -> dict:
    r_cv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = r_cv
    transform[:3, 3] = -r_cv @ np.array([0.65, 0.0, 1.25])
    return {
        "image_width": 640, "image_height": 360,
        "K": [[440.0, 0.0, 320.0], [0.0, 440.0, 180.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0], "T_CV": transform.tolist(),
    }


def _texture(index: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(1471 + index * 31)
    height, width = (280, 320) if kind == "wall" else (240, 320)
    base = rng.integers(75, 130, (height, width, 1), dtype=np.uint8)
    texture = np.repeat(base, 3, axis=2)
    if kind == "road":
        texture = (texture * 0.43).astype(np.uint8)
    # Irregular fixed texture is world geometry, not image-space pose markers.
    for _ in range(75 if kind == "wall" else 30):
        x, y = rng.integers(8, width - 20), rng.integers(8, height - 20)
        w, h = rng.integers(4, 23), rng.integers(4, 23)
        shade = int(rng.integers(10, 245 if kind == "wall" else 110))
        color = (shade, shade, shade)
        cv2.rectangle(texture, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
        if rng.random() < 0.45:
            cv2.line(texture, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)
    if kind == "wall":
        cv2.rectangle(texture, (8, 8), (width - 9, height - 9), (205, 215, 220), 4)
        cv2.putText(texture, f"SECTOR {index:02d}", (18, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(texture, "CAMERA LAB", (18, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 30, 40), 2, cv2.LINE_AA)
    return texture


def _scene() -> list[tuple[np.ndarray, np.ndarray]]:
    surfaces = []
    for i in range(20):
        a, b = -6.0 + i * 3.0, -3.0 + i * 3.0
        for side in (-1, 1):
            y = side * 6.0
            corners = np.array([[a, y, 0], [b, y, 0], [b, y, 5.2], [a, y, 5.2]])
            surfaces.append((corners, _texture(i * 2 + (side == 1), "wall")))
        for j in range(4):
            y0, y1 = -6.0 + j * 3.0, -3.0 + j * 3.0
            corners = np.array([[a, y0, 0], [b, y0, 0], [b, y1, 0], [a, y1, 0]])
            tex = _texture(100 + 4 * i + j, "road")
            if j in (0, 3):
                # Curbs add static texture away from the largely plain roadway.
                for k in range(8):
                    cv2.rectangle(tex, (0, k * 30), (35, k * 30 + 27),
                                  (35, 45, 185) if k % 2 else (215, 215, 215), -1)
            surfaces.append((corners, tex))
    # Different heights and orientations prevent a purely planar PnP exercise.
    for i, x in enumerate((3.0, 11.0, 19.0, 27.0, 35.0, 43.0)):
        corners = np.array([[x, -5.2, 2.0], [x, -3.5, 2.0],
                            [x, -3.5, 4.7], [x, -5.2, 4.7]])
        surfaces.append((corners, _texture(300 + i, "wall")))
    return surfaces


def render_image(camera_pose: np.ndarray, calibration: dict,
                 surfaces: list[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    """Project fixed textured surfaces, with no query landmark injection."""
    surfaces = _scene() if surfaces is None else surfaces
    width, height = calibration["image_width"], calibration["image_height"]
    sky = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        sky[y, :] = [min(220, 150 + y // 8), min(170, 105 + y // 10), min(145, 75 + y // 12)]
    image = sky
    camera_from_world = np.linalg.inv(camera_pose)
    intrinsic = np.asarray(calibration["K"], dtype=float)
    visible = []
    for corners, texture in surfaces:
        coordinates = (camera_from_world[:3, :3] @ corners.T).T + camera_from_world[:3, 3]
        if np.min(coordinates[:, 2]) <= 0.3 or np.mean(coordinates[:, 2]) > 45:
            continue
        uvw = coordinates @ intrinsic.T
        pixels = uvw[:, :2] / uvw[:, 2:3]
        if (pixels[:, 0].max() < 0 or pixels[:, 0].min() >= width or
                pixels[:, 1].max() < 0 or pixels[:, 1].min() >= height):
            continue
        visible.append((float(np.mean(coordinates[:, 2])), pixels, texture))
    for _, pixels, texture in sorted(visible, key=lambda item: item[0], reverse=True):
        x0 = max(0, int(np.floor(pixels[:, 0].min())))
        y0 = max(0, int(np.floor(pixels[:, 1].min())))
        x1 = min(width, int(np.ceil(pixels[:, 0].max())) + 1)
        y1 = min(height, int(np.ceil(pixels[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        th, tw = texture.shape[:2]
        source = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
        destination = (pixels - [x0, y0]).astype(np.float32)
        homography = cv2.getPerspectiveTransform(source, destination)
        rendered = cv2.warpPerspective(texture, homography, (x1 - x0, y1 - y0),
                                      flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), homography,
                                  (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
        roi = image[y0:y1, x0:x1]
        roi[mask > 0] = rendered[mask > 0]
    # Camera hood is static in the image, so a small lower region is excluded by
    # the detector. It is presentation, not a localization landmark.
    cv2.fillConvexPoly(image, np.int32([[200, height], [245, height - 25],
                                      [395, height - 25], [440, height]]), (24, 26, 29))
    return image


def generate_demo(output_dir: Path) -> dict:
    """Write canonical reference/train/development/audit RGB packages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = demo_calibration()
    vehicle_from_camera = np.linalg.inv(np.asarray(calibration["T_CV"]))
    scene = _scene()
    x_track = np.linspace(-3, 45, 100)
    centerline = np.c_[x_track, 0.55 * np.sin(x_track / 8.0), np.zeros_like(x_track)]
    track = {"centerline": centerline.tolist(),
             "left_boundary": (centerline + [0, 4.5, 0]).tolist(),
             "right_boundary": (centerline - [0, 4.5, 0]).tolist()}
    definitions = [
        ("reference", 19, -2.0, 30.4, 0.0, 0.0, 0.32),
        ("train", 46, 0.13, 26.43, -0.26, -0.012, 0.12),
        ("development", 60, 0.47, 27.07, 0.26, 0.008, 0.12),
        ("audit", 43, 1.09, 28.79, -0.47, 0.019, 0.15),
    ]
    sessions, ground_truth = [], {}
    for session_index, (split, count, start, end, lateral, phase, dt) in enumerate(definitions):
        session_id = f"demo-{split}"
        frames, labels = [], []
        folder = output_dir / "images" / session_id
        folder.mkdir(parents=True, exist_ok=True)
        for index, x in enumerate(np.linspace(start, end, count)):
            # Distinct whole trajectories and groups, not random frame splits.
            dynamic_lateral = lateral + (0.09 * np.sin(index / 7.0) if split != "reference" else 0)
            vehicle_pose = _vehicle_pose(float(x), float(dynamic_lateral), phase)
            camera_pose = vehicle_pose @ vehicle_from_camera
            image = render_image(camera_pose, calibration, scene)
            if split == "train":
                image = np.clip(image.astype(float) * (0.8 + 0.17 * np.sin(index / 8)), 0, 255).astype(np.uint8)
                if 14 <= index <= 16:
                    image = cv2.GaussianBlur(image, (9, 9), 2.2)
                if 25 <= index <= 31:
                    # A training-only occlusion yields genuine successful and
                    # failed outcomes for the optional diagnostic risk model.
                    # Different RGB content keeps it distinct from dev blanks.
                    image[:] = (7, 9, 11)
                    image[0, 0] = (index, 9, 11)
            if split == "development" and 14 <= index <= 16:
                kernel = np.zeros((13, 13), np.float32)
                kernel[6, :] = 1 / 13
                image = cv2.filter2D(image, -1, kernel)
            if split == "development" and 23 <= index <= 29:
                image[:] = 0
                # Tiny distinct margin marks preserve frame identity without
                # providing usable features or a pose encoding to inference.
                image[0, 0] = [index, 0, 0]
            if split == "audit":
                image = np.clip(image.astype(float) * 0.74 + 12, 0, 255).astype(np.uint8)
                if 18 <= index <= 22:
                    image = cv2.GaussianBlur(image, (15, 15), 4.0)
                if 30 <= index <= 32:
                    image[:, :260] = (32, 30, 28)
            frame_id = f"{index:05d}"
            relative = f"images/{session_id}/{frame_id}.png"
            if not cv2.imwrite(str(output_dir / relative), image):
                raise RuntimeError(f"Could not write synthetic image {relative}")
            timestamp = round(index * dt, 8)
            frame = {"frame_id": frame_id, "timestamp_s": timestamp, "image": relative}
            if split == "reference":
                frame["T_WC"] = camera_pose.tolist()
            else:
                labels.append({"frame_id": frame_id, "timestamp_s": timestamp,
                               "position_m": vehicle_pose[:3, 3].tolist(),
                               "yaw_rad": float(np.arctan2(vehicle_pose[1, 0], vehicle_pose[0, 0]))})
            frames.append(frame)
        sessions.append({"id": session_id, "name": f"Synthetic {split} trajectory",
                         "split": split, "group_id": f"synthetic-session-{session_index}",
                         "frames": frames, "track": track})
        if labels:
            ground_truth[session_id] = labels
    manifest = {"version": 1, "name": "Camera Lab calibrated synthetic circuit",
                "source": "synthetic", "calibration": calibration,
                "sessions": sessions, "ground_truth": ground_truth}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "scene.json").write_text(json.dumps({
        "renderer": "projected textured surfaces v1", "seed": 1471,
        "description": "Fabricated short circuit corridor, not Yas Marina or a racing simulator.",
        "limitations": ["Approximate surface occlusion", "No rolling shutter or exposure integration",
                         "Synthetic results do not establish real racing accuracy"],
    }, indent=2), encoding="utf-8")
    return manifest
