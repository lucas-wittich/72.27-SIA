import random
import copy
from PIL import Image, ImageDraw


class TriangleIndividual:
    def __init__(self, triangles, canvas_size):
        self.triangles = triangles
        self.canvas_size = canvas_size
        self.fitness = None

    @classmethod
    def random_initialize(cls, num_triangles, canvas_size):
        """
        Create a TriangleIndividual with a random genotype of smaller, clustered triangles.
        """
        width, height = canvas_size
        triangles = []

        for _ in range(num_triangles):
            # Choose a random center point within the canvas
            cx = random.randint(0, width)
            cy = random.randint(0, height)

            # Limit triangle size to 10–40 pixels for better control
            max_size = 40
            size = random.randint(10, max_size)

            # Generate points for each triangle
            triangle = {
                "x1": cx + random.randint(-size, size),
                "y1": cy + random.randint(-size, size),
                "x2": cx + random.randint(-size, size),
                "y2": cy + random.randint(-size, size),
                "x3": cx + random.randint(-size, size),
                "y3": cy + random.randint(-size, size),
                "color": (
                    random.randint(0, 255),    # R
                    random.randint(0, 255),    # G
                    random.randint(0, 255),    # B
                    random.randint(30, 180)    # Alpha (transparency)
                )
            }

            # Ensure the points are within canvas bounds
            for key in ["x1", "x2", "x3"]:
                triangle[key] = max(0, min(triangle[key], width))
            for key in ["y1", "y2", "y3"]:
                triangle[key] = max(0, min(triangle[key], height))

            triangles.append(triangle)

        return cls(triangles, canvas_size)

    def clone(self):
        clone = TriangleIndividual(copy.deepcopy(self.triangles), self.canvas_size)
        clone.fitness = self.fitness
        return clone

    def mutate(self, mutation_rate=0.05, delta=10):
        width, height = self.canvas_size
        for triangle in self.triangles:
            if random.random() < mutation_rate:
                point = random.choice(["x1", "y1", "x2", "y2", "x3", "y3"])
                triangle[point] += random.randint(-delta, delta)
                triangle[point] = max(0, min(triangle[point], width if 'x' in point else height))

            if random.random() < mutation_rate:
                c_idx = random.randint(0, 3)
                color = list(triangle["color"])
                color[c_idx] += random.randint(-delta, delta)
                color[c_idx] = max(0, min(color[c_idx], 255))
                triangle["color"] = tuple(color)

    def render(self):
        img = Image.new("RGBA", self.canvas_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        for triangle in self.triangles:
            points = [
                (triangle["x1"], triangle["y1"]),
                (triangle["x2"], triangle["y2"]),
                (triangle["x3"], triangle["y3"]),
            ]
            draw.polygon(points, fill=triangle["color"])
        return img

