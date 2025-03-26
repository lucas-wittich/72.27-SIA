# README - Practical Assignment 1 (TP1)

## Artificial Intelligence Systems - ITBA

Implementation of a **Sokoban** game in Python with different search algorithms:

- Uninformed search: **DFS**, **BFS**
- Informed search: **Greedy**, **A\*** with two different heuristics

## Requirements

To run the code, you need:
- Python 3.12

## Execution

1. Clone this repository:
   ```sh
   git clone https://github.com/lucas-wittich/72.27-SIA
   cd 72.27-SIA/TP1/Sokoban
   ```

2. Run the code:

   ```sh
   python <file-to-run.py>
   ```

   - [`sokoban.py`](Sokoban/sokoban.py)  
   Play Sokoban manually using the **W A S D** keys
   Optional: Select a difficulty (easy, medium, hard)
   ```sh
   python sokoban.py --difficulty <desired-difficulty>
   ```

   - [`dfs.py`](Sokoban/dfs.py)  
   Solve Sokoban using the **Depth-First Search (DFS)** algorithm

   - [`bfs.py`](Sokoban/bfs.py)  
   Solve Sokoban using the **Breadth-First Search (BFS)** algorithm

   - [`greedy.py`](Sokoban/greedy.py)  
   Solve Sokoban with **Greedy Search**  
   Heuristics: **Manhattan Distance** and **Misplaced Pieces**

   - [`astar.py`](Sokoban/astar.py)  
   Solve Sokoban using the **A\*** algorithm  
   Heuristics: **Manhattan Distance** and **Misplaced Pieces**

   - [`main.py`](Sokoban/main.py)  
   Runs several games with all the different algorithms and yields both plots and summaries of their performances. 

3. Review the generated results in the console output.

## Notes

- The game uses **three fixed boards**.


## Authors
- Mona Helness
- Lilian Michalak
- Camilla Adriazola Johannessen
- Lukas Wittich

---


