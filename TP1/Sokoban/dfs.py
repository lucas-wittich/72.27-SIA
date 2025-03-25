import time
from sokoban import initial_state, goal_state, actions, action_function, goal_test, print_state

# Node class for DFS
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state

# Depth-First Search implementation (with optional depth limit)
def dfs_search(depth_limit=None):
    start_node = Node(state=[row[:] for row in initial_state])
    stack = [start_node]
    visited = set()
    visited.add(hash(start_node))

    while stack:
        current_node = stack.pop()

        if goal_test(current_node.state, goal_state):
            return reconstruct_path(current_node)

        if depth_limit is not None and get_depth(current_node) >= depth_limit:
            continue

        for action in actions:
            next_state = action_function(current_node.state, action)
            next_node = Node(state=next_state, parent=current_node, action=action)

            if hash(next_node) not in visited:
                visited.add(hash(next_node))
                stack.append(next_node)

    return None

# Reconstruct path from goal to start
def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path

# Get depth of a node (number of moves from start)
def get_depth(node):
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth

# Run DFS, display moves, and return performance data
def run_dfs(depth_limit=None):
    print("\nRunning Depth-First Search (DFS)...")
    if depth_limit is not None:
        print(f"Depth limit: {depth_limit}")
    start_time = time.time()
    solution = dfs_search(depth_limit=depth_limit)
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
            'algorithm': 'DFS',
            'steps': len(solution),
            'time': duration
        }
    else:
        print("No solution found.")
        return {
            'algorithm': 'DFS',
            'steps': None,
            'time': duration
        }

# Main function to run DFS and print summary
if __name__ == "__main__":
    # Optional: set a depth limit to avoid infinite loops
    depth_limit = None  # e.g., 100
    result = run_dfs(depth_limit=depth_limit)
    print("\nSummary:")
    if result['steps'] is not None:
        print(f"{result['algorithm']}: {result['steps']} moves, time: {result['time']:.4f} seconds")
    else:
        print(f"{result['algorithm']}: No solution, time: {result['time']:.4f} seconds")
