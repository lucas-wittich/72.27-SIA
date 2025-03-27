import time
import matplotlib.pyplot as plt
import os

from bfs import run_bfs
from dfs import run_dfs
from greedy import run_greedy, manhattan_heuristic, misplaced_heuristic
from astar import run_astar_with_report, manhattan_heuristic as astar_manhattan, misplaced_heuristic as astar_misplaced
from sokoban import puzzles, get_puzzle

# List of algorithm/heuristic types to test.
ALGORITHM_TYPES = [
    "BFS",
    "DFS",
    "A* (Manhattan)",
    "A* (Misplaced)",
    "Greedy (Manhattan)",
    "Greedy (Misplaced)"
]


def run_single_test(difficulty, alg_type):
    init_state, goal_state = get_puzzle(difficulty)

    if alg_type == "BFS":
        result = run_bfs(init_state, goal_state)
    elif alg_type == "DFS":
        result = run_dfs(init_state, goal_state, depth_limit=None)
    elif alg_type == "A* (Manhattan)":
        result = run_astar_with_report(init_state, goal_state, astar_manhattan, "A* Manhattan", difficulty)
    elif alg_type == "A* (Misplaced)":
        result = run_astar_with_report(init_state, goal_state, astar_misplaced, "A* Misplaced", difficulty)
    elif alg_type == "Greedy (Manhattan)":
        result = run_greedy(init_state, goal_state, manhattan_heuristic, "Greedy Manhattan", difficulty)
    elif alg_type == "Greedy (Misplaced)":
        result = run_greedy(init_state, goal_state, misplaced_heuristic, "Greedy Misplaced", difficulty)
    else:
        raise ValueError(f"Unknown algorithm type: {alg_type}")

    result["puzzle"] = difficulty
    result["algorithm"] = alg_type
    return result


def run_all_tests():
    results = []
    for diff in puzzles:
        for alg in ALGORITHM_TYPES:
            result = run_single_test(diff, alg)
            results.append(result)
    return results


def normalize_results(results):
    normalized = []
    for res in results:
        normalized.append({
            'puzzle': res.get('puzzle'),
            'algorithm': res.get('algorithm'),
            'steps': res['steps'],
            'time': res['time'],
            'nodes_expanded': res['nodes_expanded'],
            'max_frontier': res['max_frontier'],
            'success': res['success'],
            'solution_path': res.get('solution_path')
        })
    return normalized


def print_summary(results):
    print("\n--- Summary ---")
    for res in results:
        if res['success']:
            print(f"{res['puzzle']} - {res['algorithm']}: {res['steps']} moves, {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")
        else:
            print(f"{res['puzzle']} - {res['algorithm']}: No solution, {res['time']:.4f} s, "
                  f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}")


def add_avg_time_per_move(normalized_results):
    for res in normalized_results:
        if res['steps'] is not None and res['steps'] > 0:
            res['avg_time_per_move'] = res['time'] / res['steps']
        else:
            res['avg_time_per_move'] = 0
    return normalized_results


def aggregate_by_algorithm(normalized_results):
    agg = {}
    count = {}
    for res in normalized_results:
        alg = res['algorithm']
        if alg not in agg:
            agg[alg] = {'total_time': 0, 'total_steps': 0,
                        'total_nodes_expanded': 0, 'total_max_frontier': 0}
            count[alg] = 0
        if res['steps'] is not None and res['steps'] > 0:
            agg[alg]['total_time'] += res['time']
            agg[alg]['total_steps'] += res['steps']
        agg[alg]['total_nodes_expanded'] += res['nodes_expanded']
        agg[alg]['total_max_frontier'] += res['max_frontier']
        count[alg] += 1
    aggregated = []
    for alg, data in agg.items():
        avg_time = data['total_time'] / data['total_steps'] if data['total_steps'] > 0 else 0
        avg_nodes = data['total_nodes_expanded'] / count[alg]
        avg_frontier = data['total_max_frontier'] / count[alg]
        aggregated.append({
            'algorithm': alg,
            'aggregated_avg_time': avg_time,
            'aggregated_avg_nodes_expanded': avg_nodes,
            'aggregated_avg_max_frontier': avg_frontier
        })
    return aggregated


def sort_results(normalized_results):
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    return sorted(normalized_results, key=lambda r: (difficulty_order.get(r['puzzle'], 99), r['algorithm']))


def sort_aggregated(aggregated_results):
    return sorted(aggregated_results, key=lambda r: r['algorithm'])


def plot_results(results, aggregated):
    names = [f"{res['puzzle']} - {res['algorithm']}" for res in results]
    steps = [res['steps'] if res['steps'] is not None else 0 for res in results]
    times = [res['time'] for res in results]
    avg_times = [res.get('avg_time_per_move', 0) for res in results]
    nodes_expanded = [res['nodes_expanded'] for res in results]
    max_frontier = [res['max_frontier'] for res in results]

    # Figure 1: Moves and Runtime
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))
    axs1[0].bar(names, steps)
    axs1[0].set_title("Moves")
    axs1[0].tick_params(axis='x', rotation=45)
    axs1[1].bar(names, times)
    axs1[1].set_title("Runtime (s)")
    axs1[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # Figure 2: Average Time per Move, Nodes Expanded, and Max Frontier
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    axs2[0].bar(names, avg_times)
    axs2[0].set_title("Avg Time per Move (s)")
    axs2[0].tick_params(axis='x', rotation=45)
    axs2[1].bar(names, nodes_expanded)
    axs2[1].set_title("Nodes Expanded")
    axs2[1].tick_params(axis='x', rotation=45)
    axs2[2].bar(names, max_frontier)
    axs2[2].set_title("Max Frontier")
    axs2[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # Figure 3: Aggregated statistics per algorithm.
    algs = [item['algorithm'] for item in aggregated]
    agg_avg_time = [item['aggregated_avg_time'] for item in aggregated]
    agg_avg_nodes = [item['aggregated_avg_nodes_expanded'] for item in aggregated]
    agg_avg_frontier = [item['aggregated_avg_max_frontier'] for item in aggregated]

    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 6))
    axs3[0].bar(algs, agg_avg_time)
    axs3[0].set_title("Aggregated Avg Time per Move (s)")
    axs3[0].tick_params(axis='x', rotation=45)
    axs3[1].bar(algs, agg_avg_nodes)
    axs3[1].set_title("Aggregated Avg Nodes Expanded")
    axs3[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # Figure 4: Aggregated Max Frontier
    plt.figure(figsize=(6, 5))
    plt.bar(algs, agg_avg_frontier)
    plt.title("Aggregated Avg Max Frontier")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_solution_paths_to_files(results):
    os.makedirs("results", exist_ok=True)

    for res in results:
        puzzle = res.get('puzzle', 'Unknown_Difficulty')
        alg = res.get('algorithm', 'Unknown_Algorithm')

        safe_puzzle = puzzle.replace(" ", "_")
        safe_alg = alg.replace(" ", "_").replace("(", "").replace(")", "")
        filename = os.path.join("results", f"{safe_puzzle}_{safe_alg}.txt")

        with open(filename, "w") as f:
            f.write(f"Difficulty: {puzzle}, Algorithm: {alg}\n")
            if res.get("success") and res.get("solution_path"):
                f.write(f"Solution found in {res['steps']} moves, in {res['time']:.4f} s\n")
                f.write(
                    f"Nodes expanded: {res['nodes_expanded']}, Max frontier: {res['max_frontier']}\n Full solution path:\n")
                for i, (action, state) in enumerate(res["solution_path"], 1):
                    f.write(f"  Move {i}: {action}\n")
                    for row in state:
                        f.write("    " + " ".join(row) + "\n")
                    f.write("\n")
            else:
                f.write("  No solution found.\n")
            f.write("-" * 40 + "\n")


if __name__ == "__main__":
    raw_results = run_all_tests()
    normalized_results = normalize_results(raw_results)
    normalized_results = add_avg_time_per_move(normalized_results)
    normalized_results = sort_results(normalized_results)
    aggregated_results = aggregate_by_algorithm(normalized_results)
    aggregated_results = sort_aggregated(aggregated_results)

    print_summary(normalized_results)
    print_solution_paths_to_files(normalized_results)
    plot_results(normalized_results, aggregated_results)
