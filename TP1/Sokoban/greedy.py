import heapq
import time
from sokoban import actions, action_function, goal_test, print_state, puzzles, get_puzzle


def manhattan_heuristic(state, goal):
    goal_positions = [(i, j) for i in range(len(goal))
                      for j in range(len(goal[i])) if goal[i][j] == '.']
    boxes = [(i, j) for i in range(len(state)) for j in range(len(state[i])) if state[i][j] == '*']
    total_dist = 0
    for box in boxes:
        dists = [abs(box[0]-g[0]) + abs(box[1]-g[1]) for g in goal_positions]
        if dists:
            total_dist += min(dists)
    return total_dist


def misplaced_heuristic(state, goal):
    goal_positions = [(i, j) for i in range(len(goal))
                      for j in range(len(goal[i])) if goal[i][j] == '.']
    misplaced = 0
    for i, j in goal_positions:
        if state[i][j] != '*':
            misplaced += 1
    return misplaced


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


def greedy_search(heuristic_func, init_state, goal):
    nodes_expanded = 0
    max_frontier = 0
    start_node = Node(state=[row[:] for row in init_state],
                      heuristic=heuristic_func(init_state, goal))
    open_list = []
    heapq.heappush(open_list, start_node)
    visited = set()
    while open_list:
        max_frontier = max(max_frontier, len(open_list))
        current_node = heapq.heappop(open_list)
        nodes_expanded += 1
        if goal_test(current_node.state, goal):
            solution_path = reconstruct_path(current_node)
            return solution_path, nodes_expanded, max_frontier, True
        visited.add(hash(current_node))
        for action in actions:
            next_state = action_function(current_node.state, action, goal)
            next_node = Node(state=next_state,
                             parent=current_node,
                             action=action,
                             heuristic=heuristic_func(next_state, goal))
            if hash(next_node) not in visited:
                heapq.heappush(open_list, next_node)
    return None, nodes_expanded, max_frontier, False


def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def run_greedy(init_state, goal, heuristic_func, heuristic_name, difficulty=""):
    start_time = time.time()
    solution, nodes_expanded, max_frontier, success = greedy_search(heuristic_func, init_state, goal)
    end_time = time.time()
    duration = end_time - start_time
    if success:
        print(f"\nRunning Greedy Best-First Search with heuristic: {heuristic_name} on difficulty: {difficulty}")
        print(f"Solution found in {len(solution)} moves. Showing step-by-step progression:\n")
        state = [row[:] for row in init_state]
        print_state(state)
        for i, (action, new_state) in enumerate(solution, 1):
            print(f"Move {i}: {action}")
            print_state(new_state)
        return {
            'algorithm': f"Greedy ({heuristic_name})",
            'steps': len(solution),
            'time': duration,
            'nodes_expanded': nodes_expanded,
            'max_frontier': max_frontier,
            'solution_path': solution,
            'success': True
        }
    else:
        print(f"Greedy ({heuristic_name}) on difficulty: {difficulty} did not find a solution.")
        return {
            'algorithm': f"Greedy ({heuristic_name})",
            'steps': None,
            'time': duration,
            'nodes_expanded': nodes_expanded,
            'max_frontier': max_frontier,
            'solution_path': None,
            'success': False
        }


def run_all():
    results = []
    for puzzle in puzzles:
        init, goal = get_puzzle(puzzle)
        man_res = run_greedy(init, goal, manhattan_heuristic, 'Manhattan Distance', puzzle)
        man_res['puzzle'] = puzzle
        mis_res = run_greedy(init, goal, misplaced_heuristic, 'Misplaced Boxes', puzzle)
        mis_res['puzzle'] = puzzle
        results.append(man_res)
        results.append(mis_res)
    print("\nSummary:")
    for res in results:
        if res['success']:
            print(f"{res['algorithm']}: Difficulty: {res['puzzle']}, {res['steps']} moves, time: {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")
        else:
            print(f"{res['algorithm']}: Difficulty: {res['puzzle']}, No solution, time: {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")


if __name__ == "__main__":
    run_all()
