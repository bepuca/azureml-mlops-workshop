import numpy as np
from PIL import Image
from transform import prepare_sample


def test_prepare_sample():
    pixels = np.random.rand(28, 28) * 255
    pixels = pixels.astype("uint8")
    image = Image.fromarray(pixels)

    expected_sample = pixels.flatten()

    sample = prepare_sample(image)

    assert (sample == expected_sample).all()
