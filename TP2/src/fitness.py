import numpy as np
from PIL import Image


def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def compute_triangle_fitness(individual, target_image):
    """
    Compare the rendered triangle image to the target image in RGB space.
    Assumes target_image is already resized to match canvas.
    """
    rendered_rgba = individual.render()

    # Flatten transparency onto white
    white_bg = Image.new("RGBA", rendered_rgba.size, (255, 255, 255, 255))
    rendered_rgb = Image.alpha_composite(white_bg, rendered_rgba).convert("RGB")
    target_rgb = target_image.convert("RGB")

    rendered_array = np.array(rendered_rgb, dtype=np.float32) / 255.0
    target_array = np.array(target_rgb, dtype=np.float32) / 255.0

    mse = compute_mse(rendered_array, target_array)
    return 1 / (1 + mse)
