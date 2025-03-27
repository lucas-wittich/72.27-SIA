import time
from sokoban import actions, action_function, goal_test, print_state, puzzles, get_puzzle


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state


def dfs_search(init_state, goal, depth_limit=None):
    nodes_expanded = 0
    max_frontier = 0
    start_node = Node(state=[row[:] for row in init_state])
    stack = [start_node]
    visited = set()
    visited.add(hash(start_node))
    while stack:
        max_frontier = max(max_frontier, len(stack))
        current_node = stack.pop()
        nodes_expanded += 1
        if goal_test(current_node.state, goal):
            solution_path = reconstruct_path(current_node)
            return solution_path, nodes_expanded, max_frontier, True
        if depth_limit is not None and get_depth(current_node) >= depth_limit:
            continue
        for action in actions:
            next_state = action_function(current_node.state, action, goal)
            next_node = Node(state=next_state, parent=current_node, action=action)
            if hash(next_node) not in visited:
                visited.add(hash(next_node))
                stack.append(next_node)
    return None, nodes_expanded, max_frontier, False


def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def get_depth(node):
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def run_dfs(init_state, goal, depth_limit=None):
    if depth_limit is not None:
        print(f"Depth limit: {depth_limit}")
    start_time = time.time()
    solution, nodes_expanded, max_frontier, success = dfs_search(init_state, goal, depth_limit=depth_limit)
    end_time = time.time()
    duration = end_time - start_time
    if success:
        print("\nRunning Depth-First Search (DFS)...")
        print(f"Solution found in {len(solution)} moves. Showing step-by-step progression:\n")
        state = [row[:] for row in init_state]
        print_state(state)
        for i, (action, new_state) in enumerate(solution, 1):
            print(f"Move {i}: {action}")
            print_state(new_state)
        return {
            'algorithm': 'DFS',
            'steps': len(solution),
            'time': duration,
            'nodes_expanded': nodes_expanded,
            'max_frontier': max_frontier,
            'solution_path': solution,
            'success': True
        }
    else:
        print("DFS did not find a solution.")
        return {
            'algorithm': 'DFS',
            'steps': None,
            'time': duration,
            'nodes_expanded': nodes_expanded,
            'max_frontier': max_frontier,
            'solution_path': None,
            'success': False
        }


if __name__ == "__main__":
    results = []
    for puzzle in puzzles:
        init, goal = get_puzzle(puzzle)
        res = run_dfs(init, goal)
        res['puzzle'] = puzzle
        results.append(res)
    print("\nSummary:")
    for res in results:
        if res['success']:
            print(f"{res['algorithm']}: Difficulty: {res['puzzle']}, {res['steps']} moves, time: {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")
        else:
            print(f"{res['algorithm']}: Difficulty: {res['puzzle']}, No solution, time: {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")
