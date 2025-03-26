import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from bfs import run_bfs
from dfs import run_dfs
from greedy import run_greedy, manhattan_heuristic, misplaced_heuristic
from astar import run_astar_with_report, manhattan_heuristic as astar_manhattan, misplaced_heuristic as astar_misplaced
from sokoban import puzzles, get_puzzle


def run_algorithms_for_difficulty(difficulty):
    results = []
    init_state, goal_state = get_puzzle(difficulty)

    # Run BFS
    bfs_result = run_bfs(init_state, goal_state)
    bfs_result['puzzle'] = difficulty
    results.append(bfs_result)

    # Run DFS
    dfs_result = run_dfs(init_state, goal_state, depth_limit=None)
    dfs_result['puzzle'] = difficulty
    results.append(dfs_result)

    # Run A* with Manhattan heuristic
    astar_m_result = run_astar_with_report(init_state, goal_state, astar_manhattan, 'A* Manhattan', difficulty)
    astar_m_result['puzzle'] = difficulty
    results.append(astar_m_result)

    # Run A* with Misplaced Boxes heuristic
    astar_mis_result = run_astar_with_report(init_state, goal_state, astar_misplaced, 'A* Misplaced', difficulty)
    astar_mis_result['puzzle'] = difficulty
    results.append(astar_mis_result)

    # Run Greedy with Manhattan heuristic
    greedy_m_result = run_greedy(init_state, goal_state, manhattan_heuristic, 'Greedy Manhattan', difficulty)
    greedy_m_result['puzzle'] = difficulty
    results.append(greedy_m_result)

    # Run Greedy with Misplaced Boxes heuristic
    greedy_mis_result = run_greedy(init_state, goal_state, misplaced_heuristic, 'Greedy Misplaced', difficulty)
    greedy_mis_result['puzzle'] = difficulty
    results.append(greedy_mis_result)

    return results


def run_all_algorithms():
    results = []
    with ThreadPoolExecutor(max_workers=(len(puzzles))) as executor:
        future_to_diff = {executor.submit(run_algorithms_for_difficulty, diff): diff for diff in puzzles}
        for future in as_completed(future_to_diff):
            diff = future_to_diff[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                print(f"{diff} generated an exception: {exc}")
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
            'success': res['success']
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
    # Define a difficulty ordering; puzzles not in the map get a high order value.
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    return sorted(normalized_results, key=lambda r: (difficulty_order.get(r['puzzle'], 99), r['algorithm']))


def sort_aggregated(aggregated_results):
    return sorted(aggregated_results, key=lambda r: r['algorithm'])


def plot_results(results, aggregated):
    # Per-run metrics
    names = [f"{res['puzzle']} - {res['algorithm']}" for res in results]
    steps = [res['steps'] if res['steps'] is not None else 0 for res in results]
    times = [res['time'] for res in results]
    avg_times = [res.get('avg_time_per_move', 0) for res in results]
    nodes_expanded = [res['nodes_expanded'] for res in results]
    max_frontier = [res['max_frontier'] for res in results]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].bar(names, steps)
    axs[0, 0].set_title("Moves")
    axs[0, 0].set_ylabel("Moves")
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].bar(names, times)
    axs[0, 1].set_title("Runtime (s)")
    axs[0, 1].set_ylabel("Time (s)")
    axs[0, 1].tick_params(axis='x', rotation=45)

    axs[0, 2].bar(names, avg_times)
    axs[0, 2].set_title("Avg Time per Move (s)")
    axs[0, 2].set_ylabel("Avg Time (s)")
    axs[0, 2].tick_params(axis='x', rotation=45)

    axs[1, 0].bar(names, nodes_expanded)
    axs[1, 0].set_title("Nodes Expanded")
    axs[1, 0].set_ylabel("Nodes Expanded")
    axs[1, 0].tick_params(axis='x', rotation=45)

    axs[1, 1].bar(names, max_frontier)
    axs[1, 1].set_title("Max Frontier")
    axs[1, 1].set_ylabel("Max Frontier")
    axs[1, 1].tick_params(axis='x', rotation=45)

    # Leave the last subplot empty
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_aggregated_stats(aggregated):
    # Aggregated metrics are computed per algorithm over all puzzles.
    algs = [item['algorithm'] for item in aggregated]
    agg_avg_time = [item['aggregated_avg_time'] for item in aggregated]
    agg_avg_nodes = [item['aggregated_avg_nodes_expanded'] for item in aggregated]
    agg_avg_frontier = [item['aggregated_avg_max_frontier'] for item in aggregated]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].bar(algs, agg_avg_time)
    axs[0].set_title("Aggregated Avg Time per Move (s)")
    axs[0].set_ylabel("Avg Time (s)")
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].bar(algs, agg_avg_nodes)
    axs[1].set_title("Aggregated Avg Nodes Expanded")
    axs[1].set_ylabel("Avg Nodes Expanded")
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].bar(algs, agg_avg_frontier)
    axs[2].set_title("Aggregated Avg Max Frontier")
    axs[2].set_ylabel("Avg Max Frontier")
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raw_results = run_all_algorithms()
    normalized_results = normalize_results(raw_results)
    normalized_results = add_avg_time_per_move(normalized_results)
    normalized_results = sort_results(normalized_results)
    aggregated_results = aggregate_by_algorithm(normalized_results)
    aggregated_results = sort_aggregated(aggregated_results)
    print_summary(normalized_results)
    plot_results(normalized_results, aggregated_results)
    plot_aggregated_stats(aggregated_results)
