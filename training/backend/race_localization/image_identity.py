"""One camera decoder and pixel identity for runtime/import/upgrade boundaries."""
import hashlib

import cv2
import numpy as np


DECODER_ID = f"opencv-{cv2.__version__}-imread-color-rgb-dimensions-v1"


def decode_color(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Camera image cannot be decoded.")
    return image


def rgb_hash(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(f"{width}x{height}\0".encode() + rgb.tobytes()).hexdigest()
