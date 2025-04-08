import random
from individual import TriangleIndividual
from fitness import compute_triangle_fitness
from selection import get_selection_method


class GAEngine:
    def __init__(self, target_image, canvas_size, num_triangles, population_size, num_generations,
                 mutation_rate, crossover_rate, selection_method="tournament", selection_params=None):
        self.target_image = target_image
        self.canvas_size = canvas_size
        self.num_triangles = num_triangles
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if selection_params is None:
            selection_params = {}
        self.selection_func = get_selection_method(selection_method, **selection_params)

        self.population = []
        self.best_individual = None
        self.fitness_history = []

    def initialize_population(self):
        """Creates the initial population of triangle individuals."""
        self.population = [
            TriangleIndividual.random_initialize(self.num_triangles, self.canvas_size)
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self):
        """Assigns fitness scores to all individuals in the population."""
        for individual in self.population:
            individual.fitness = compute_triangle_fitness(individual, self.target_image)
            assert individual.fitness is not None, "Fitness not set"


    def select_parents(self):
        """Selects parents using the configured selection method."""
        return self.selection_func(self.population, self.population_size)

    def crossover(self, parent1, parent2):
        """Performs one-point crossover or returns clones if crossover is skipped."""
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        point = random.randint(1, self.num_triangles - 1)
        child1_triangles = parent1.triangles[:point] + parent2.triangles[point:]
        child2_triangles = parent2.triangles[:point] + parent1.triangles[point:]

        child1 = TriangleIndividual(child1_triangles, self.canvas_size)
        child2 = TriangleIndividual(child2_triangles, self.canvas_size)
        return child1, child2

    def mutate(self, individual):
        """Mutates an individual."""
        individual.mutate(mutation_rate=self.mutation_rate)
    

    def evolve(self):
        """Runs the full genetic algorithm loop and returns the best individual and fitness history."""
        self.initialize_population()
        self.evaluate_fitness()

        self.best_individual = max(self.population, key=lambda ind: ind.fitness).clone()
        self.fitness_history.append(self.best_individual.fitness)

        for gen in range(self.num_generations):
            parents = self.select_parents()
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i + 1 < self.population_size else parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])

            self.population = next_generation[:self.population_size]
            self.evaluate_fitness()

            current_best = max(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.clone()

            self.fitness_history.append(self.best_individual.fitness)
            print(f"Generation {gen+1}: Best Fitness = {self.best_individual.fitness:.6f}")

        return self.best_individual, self.fitness_history
