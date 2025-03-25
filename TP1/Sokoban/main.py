import time
import matplotlib.pyplot as plt
from bfs import run_bfs
from dfs import run_dfs
from greedy import run_greedy, manhattan_heuristic, misplaced_heuristic
from astar import run_astar_with_report, manhattan_heuristic as astar_manhattan, misplaced_heuristic as astar_misplaced

# Run all algorithms and store results
def run_all_algorithms():
    results = []

    bfs_result = run_bfs()
    results.append(bfs_result)

    dfs_result = run_dfs(depth_limit=None)
    results.append(dfs_result)

    astar_m_result = run_astar_with_report(astar_manhattan, 'A* Manhattan Distance')
    results.append(astar_m_result)

    astar_mis_result = run_astar_with_report(astar_misplaced, 'A* Misplaced Boxes')
    results.append(astar_mis_result)

    greedy_m_result = run_greedy(manhattan_heuristic, 'Greedy Manhattan Distance')
    results.append(greedy_m_result)

    greedy_mis_result = run_greedy(misplaced_heuristic, 'Greedy Misplaced Boxes')
    results.append(greedy_mis_result)

    return results

# Normalize results: unify 'heuristic' and 'algorithm' under 'name'
def normalize_results(results):
    normalized = []
    for res in results:
        name = res.get('algorithm') or res.get('heuristic') or 'Unnamed'
        normalized.append({
            'name': name,
            'steps': res['steps'],
            'time': res['time']
        })
    return normalized

# Plot results
def plot_results(results):
    names = [res['name'] for res in results]
    steps = [res['steps'] if res['steps'] is not None else 0 for res in results]
    times = [res['time'] for res in results]

    # Plot moves
    plt.figure(figsize=(10, 5))
    plt.bar(names, steps)
    plt.title("Number of Moves per Algorithm")
    plt.ylabel("Moves")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot runtime
    plt.figure(figsize=(10, 5))
    plt.bar(names, times)
    plt.title("Runtime per Algorithm (seconds)")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Print summary
def print_summary(results):
    print("\n--- Summary ---")
    for res in results:
        if res['steps'] is not None:
            print(f"{res['name']}: {res['steps']} moves, {res['time']:.4f} seconds")
        else:
            print(f"{res['name']}: No solution, {res['time']:.4f} seconds")

if __name__ == "__main__":
    raw_results = run_all_algorithms()
    all_results = normalize_results(raw_results)
    print_summary(all_results)
    plot_results(all_results)
