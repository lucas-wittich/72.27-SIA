from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(path):
    return Image.open(path).convert("RGB")


def resize_image(image, width, height):
    return image.resize((width, height), Image.LANCZOS)  # Use high-quality resampling


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def plot_fitness_curve(fitness_history, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
