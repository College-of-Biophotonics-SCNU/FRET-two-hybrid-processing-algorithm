import numpy as np


def normalize_image(image):
    """Normalize the image to 0-255 range for 8-bit color channels."""
    if image.max() == image.min():
        # Handle case where image has no variation (all pixels have the same value)
        return np.zeros_like(image, dtype=np.uint8) if image.max() == 0 else np.ones_like(image, dtype=np.uint8) * 255

    # Normalize image to [0, 1] range first, then scale to [0, 255]
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return (normalized_image * 255).astype(np.uint8)