""" This script prepares a single character for CNN model """

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

# This function makes uniform input
def _load_grayscale(source):
    """
    ---------------------------------------------
    This function allows three types of input:
    1) Path file
    2) PIL image
    3) NumPy array
    ---------------------------------------------
    """
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("L")

    if isinstance(source, Image.Image):
        return source.convert("L")

    if isinstance(source, np.ndarray):
        if source.ndim == 2:                                    # gray scale image
            gray = source
        elif source.ndim == 3 and source.shape[2] == 4:         # BGRA image (4 channels)
            gray = cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
        elif source.ndim == 3 and source.shape[2] == 3:         # BGR image (3 channels)
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported NumPy image shape: {source.shape}")

        return Image.fromarray(gray)                            # Return PIL image in gray scale

    raise TypeError(f"Unsupported image type: {type(source)}")

# This function cuts possible lateral connections
def _trim_connectors(foreground):
    mask = foreground > 0               # each non-zero pixel is ink
    height, width = mask.shape

    # If not to wide component, return foreground
    if width <= height * 1.3:
        return foreground

    column_ink = mask.sum(axis=0)
    minimum_ink = max(2, round(height * 0.12))                  # minimum ink
    core_columns = np.flatnonzero(column_ink >= minimum_ink)    # important column

    if len(core_columns) < 2:
        return foreground

    # First and Final important column with a small margin
    left = max(0, int(core_columns[0]) - 2)
    right = min(width, int(core_columns[-1]) + 3)

    # If too aggressive cut, cancel it
    if right - left < width * 0.35:
        return foreground

    return foreground[:, left:right]    # return useful image part


def preprocess_character(
    source,                             # path image
    target_size=20,                     # ch will be scaled to stay in 20x20 pixel square
    trim_connectors=False,
):
    image = _load_grayscale(source)

    # small borders removal
    margin_x = max(2, round(image.width * 0.02))
    margin_y = max(2, round(image.height * 0.02))

    if image.width > margin_x * 2 and image.height > margin_y * 2:
        image = image.crop(
            (
                margin_x,
                margin_y,
                image.width - margin_x,
                image.height - margin_y,
            )
        )

    # increase writing-background contrast
    image = ImageOps.autocontrast(image)
    pixels = np.asarray(image)

    # pixel inversion: black to white and viceversa
    if pixels.mean() > 127:
        pixels = 255 - pixels

    # ink mask creation
    threshold = max(20, int(pixels.max() * 0.15))
    mask = pixels > threshold  # only relevant pixels

    if not mask.any():
        raise ValueError("No character found in the image")

    foreground = np.where(mask, pixels, 0).astype(np.uint8)  # foreground isolation

    # crop on real content, empty space around ch is cut
    rows, columns = np.nonzero(mask)
    foreground = foreground[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]

    if trim_connectors:
        foreground = _trim_connectors(foreground)

    character = Image.fromarray(foreground)  # back to PIL
    width, height = character.size

    scale = target_size / max(width, height)

    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    character = character.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    character = ImageOps.autocontrast(character)

    character = character.filter(ImageFilter.MaxFilter(3))  # MaxFilter(3) makes ch more readable for model

    canvas = Image.new("L", (28, 28), color=0)  # black 28x28 image             

    # The ch is paste at the center of black canvas
    canvas.paste(
        character,
        (
            (28 - resized_width) // 2,
            (28 - resized_height) // 2,
        ),
    )

    # Conversion to tensor with shape (1, 28, 28)
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5  # normalization (0 to -1, 0.5 to 0, 1 to 1)

    return tensor.unsqueeze(0), canvas
