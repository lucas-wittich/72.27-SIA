# main.py
# Runs the genetic algorithm for image approximation using translucent triangles

import json
import os
from utils import load_image, resize_image, save_image, plot_fitness_curve
from ga_engine import GAEngine
from PIL import Image


def main():
    # Get number of triangles from user input
    try:
        num_triangles = int(input("Enter number of triangles to use: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    # Get number of generations from user input
    try:
        num_generations = int(input("Enter number of generations: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    # Load configuration from JSON
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Could not find {config_path}. Please make sure it exists.")
        return

    with open(config_path, "r") as file:
        config = json.load(file)

    # Override number of triangles from user input
    config["num_triangles"] = num_triangles
    config["num_generations"] = num_generations  # Set number of generations from input

    # Load and resize target image
    target_image = load_image(config["input_image"])
    canvas_size = config["canvas_size"]
    target_image = resize_image(target_image, *canvas_size)

    # Create output folder if not exists
    os.makedirs("data/outputs", exist_ok=True)

    # Initialize GA Engine
    engine = GAEngine(
        target_image=target_image,
        canvas_size=canvas_size,
        num_triangles=num_triangles,
        population_size=config["population_size"],
        num_generations=num_generations,  # Use input value for generations
        mutation_rate=config["mutation_rate"],
        crossover_rate=config["crossover_rate"],
        selection_method=config.get("selection_method", "tournament"),
        selection_params=config.get("selection_params", {})
    )

    # Run evolution and track fitness
    best_individual, fitness_history = engine.evolve()

    # Print fitness progression and save the best result
    for generation, fitness in enumerate(fitness_history, 1):
        print(f"Generation {generation}: Best Fitness = {fitness:.6f}")

    # Save the best individual image
    rendered = best_individual.render()
    save_image(rendered, "data/outputs/best_individual.png")

    # Save side-by-side comparison image
    input_resized = target_image.convert("RGB")
    output_img = rendered.convert("RGB")
    combined = Image.new("RGB", (input_resized.width * 2, input_resized.height))
    combined.paste(input_resized, (0, 0))
    combined.paste(output_img, (input_resized.width, 0))
    combined.save("data/outputs/side_by_side.png")

    # Plot fitness curve and save
    plot_fitness_curve(fitness_history, "data/outputs/fitness_curve.png")

    print(f"\nBest fitness: {best_individual.fitness:.6f}")
    print("Results saved to data/outputs/")


if __name__ == "__main__":
    main()
