import numpy as np
from PIL import Image


def prepare_sample(image: Image) -> np.array:
    """Convert `image` to 1D array concatenating rows of pixel intensities"""
    return np.asarray(image).flatten()
