import time
from collections import deque
from sokoban import initial_state, goal_state, actions, action_function, goal_test, print_state

# Node class for BFS, similar to A*
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state

# Breadth-First Search implementation
def bfs_search():
    start_node = Node(state=[row[:] for row in initial_state])
    queue = deque()
    queue.append(start_node)
    visited = set()
    visited.add(hash(start_node))

    while queue:
        current_node = queue.popleft()

        if goal_test(current_node.state, goal_state):
            return reconstruct_path(current_node)

        for action in actions:
            next_state = action_function(current_node.state, action)
            next_node = Node(state=next_state, parent=current_node, action=action)

            if hash(next_node) not in visited:
                visited.add(hash(next_node))
                queue.append(next_node)

    return None

# Reconstruct path from goal to start
def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path

# Run BFS, display moves, and return performance data
def run_bfs():
    print("\nRunning Breadth-First Search (BFS)...")
    start_time = time.time()
    solution = bfs_search()
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
            'algorithm': 'BFS',
            'steps': len(solution),
            'time': duration
        }
    else:
        print("No solution found.")
        return {
            'algorithm': 'BFS',
            'steps': None,
            'time': duration
        }

# Main function to run BFS and print summary
if __name__ == "__main__":
    result = run_bfs()
    print("\nSummary:")
    if result['steps'] is not None:
        print(f"{result['algorithm']}: {result['steps']} moves, time: {result['time']:.4f} seconds")
    else:
        print(f"{result['algorithm']}: No solution, time: {result['time']:.4f} seconds")
