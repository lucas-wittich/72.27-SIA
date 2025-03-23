import heapq
import time
from sokoban import initial_state, goal_state, actions, action_function, goal_test, print_state

# Find goal positions for heuristic calculations
goal_positions = [(i, j) for i in range(len(goal_state))
                  for j in range(len(goal_state[i])) if goal_state[i][j] == '.']

# Heuristic 1: Manhattan Distance
def manhattan_heuristic(state):
    boxes = [(i, j) for i in range(len(state)) for j in range(len(state[i])) if state[i][j] == '*']
    total_dist = 0
    for box in boxes:
        dists = [abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goal_positions]
        if dists:
            total_dist += min(dists)
    return total_dist

# Heuristic 2: Misplaced Boxes
def misplaced_heuristic(state):
    misplaced = 0
    for i, j in goal_positions:
        if state[i][j] != '*':
            misplaced += 1
    return misplaced

# Node class for Greedy Search
class Node:
    def __init__(self, state, parent=None, action=None, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state

# Greedy Best-First Search implementation
def greedy_search(heuristic_func):
    start_node = Node(state=[row[:] for row in initial_state], heuristic=heuristic_func(initial_state))
    open_list = []
    heapq.heappush(open_list, start_node)
    visited = set()

    while open_list:
        current_node = heapq.heappop(open_list)

        if goal_test(current_node.state, goal_state):
            return reconstruct_path(current_node)

        visited.add(hash(current_node))

        for action in actions:
            next_state = action_function(current_node.state, action)
            next_node = Node(state=next_state,
                             parent=current_node,
                             action=action,
                             heuristic=heuristic_func(next_state))

            if hash(next_node) not in visited:
                heapq.heappush(open_list, next_node)

    return None

# Reconstruct the solution path from goal to start
def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path

# Run Greedy Search with reporting
def run_greedy(heuristic_func, heuristic_name):
    print(f"\nRunning Greedy Best-First Search with heuristic: {heuristic_name}")
    start_time = time.time()
    solution = greedy_search(heuristic_func)
    end_time = time.time()
    duration = end_time - start_time

    if solution:
        print(f"Solution found in {len(solution)} moves. Showing step-by-step progression:\n")
        state = [row[:] for row in initial_state]
        print_state(state)
        for i, (action, new_state) in enumerate(solution, 1):
            print(f"Move {i}: {action}")
            print_state(new_state)
        return {
            'algorithm': f"Greedy ({heuristic_name})",
            'steps': len(solution),
            'time': duration
        }
    else:
        print("No solution found.")
        return {
            'algorithm': f"Greedy ({heuristic_name})",
            'steps': None,
            'time': duration
        }

# Main function to run both heuristics
def run_all():
    results = []
    results.append(run_greedy(manhattan_heuristic, 'Manhattan Distance'))
    results.append(run_greedy(misplaced_heuristic, 'Misplaced Boxes'))

    print("\nSummary:")
    for res in results:
        if res['steps'] is not None:
            print(f"{res['algorithm']}: {res['steps']} moves, time: {res['time']:.4f} seconds")
        else:
            print(f"{res['algorithm']}: No solution, time: {res['time']:.4f} seconds")

if __name__ == "__main__":
    run_all()
