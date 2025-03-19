# README - Practical Assignment 1 (TP1)

## Artificial Intelligence Systems - ITBA

This repository contains the resolution of Practical Assignment 1 for the Artificial Intelligence Systems course at the Instituto Tecnológico de Buenos Aires.

## Assignment Description

The objective of this assignment is to implement and analyze different search methods applied to classic problems.

### Exercise 1: 8-Puzzle (Theoretical Analysis)

The 8-puzzle consists of a 3x3 board with shuffled numbers and an empty space. The goal is to arrange the numbers by moving adjacent ones into the empty space until reaching the target configuration.

#### Considerations to Analyze:
- Define the state structure used.
- Identify at least two non-trivial admissible heuristics.
- Explain which search methods would be suitable, with which heuristic, and why.

### Exercise 2: Implementation of a Search Engine

You must choose and implement a search engine for one of the following games:

1. **Sokoban** ([Description](https://en.wikipedia.org/wiki/Sokoban))
   - There is no restriction on the number of moves.
   - The goal is to optimize the number of moves.
   - Complexity varies depending on the board size and the number of boxes and goals.

2. **Grid World (multi-agent)**
   - Contains multiple agents and goals in a grid environment.
   - Each board configuration has specific characteristics.

### Implementation

A search engine must be implemented using the following algorithms:
- **BFS** (Breadth-First Search)
- **DFS** (Depth-First Search)
- **Greedy Search**
- **A*** (A-star)
- **IDDFS** (Iterative Deepening Depth-First Search, optional)

Additionally, the following should be evaluated:
- **Admissible heuristics** (at least 2)

### Evaluation of Results

At the end of the execution, the following data must be recorded:
- Result (success/failure)
- Solution cost
- Number of expanded nodes
- Number of frontier nodes
- Solution (sequence of moves from initial to final state)
- Processing time

## Requirements

To run the code, you need:
- Python 3.x
- Necessary libraries (install with `pip install -r requirements.txt` if a dependencies file is included)

## Execution

1. Clone this repository:
   ```sh
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_NAME>
   ```

2. Run the code according to the chosen problem:
   - For Sokoban:
     ```sh
     python sokoban_solver.py
     ```
   - For Grid World:
     ```sh
     python gridworld_solver.py
     ```

3. Review the generated results in the console output or corresponding log files.

## Deliverables

The practical assignment must include:
- Source code
- Explanatory presentation
- This `README.md` file with execution instructions

## Authors
Include the names of the team members.

---

This assignment is part of the Artificial Intelligence Systems course at ITBA.

