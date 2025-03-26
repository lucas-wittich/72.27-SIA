import argparse
import platform
import sys
from collections import deque

# Initial and goal state (unchanged)
initial_state = [
    ['#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', '#', '#', ' ', ' ', ' ', '#', '#'],
    ['#', '.', '@', '*', ' ', ' ', '#', '#'],
    ['#', '#', '#', ' ', '*', '.', '#', '#'],
    ['#', '.', '#', '#', '*', ' ', '#', '#'],
    ['#', ' ', '#', ' ', '.', ' ', '#', '#'],
    ['#', '*', ' ', '*', '*', '*', '.', '#'],
    ['#', ' ', ' ', ' ', '.', ' ', ' ', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#'],
]

goal_state = [
    ['#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', '#', '#', ' ', ' ', ' ', '#', '#'],
    ['#', '.', ' ', ' ', ' ', ' ', '#', '#'],
    ['#', '#', '#', ' ', ' ', '.', '#', '#'],
    ['#', '.', '#', '#', ' ', ' ', '#', '#'],
    ['#', ' ', '#', ' ', '.', ' ', '#', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', '.', '#'],
    ['#', ' ', ' ', ' ', '.', ' ', ' ', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#'],
]

initial_state_easy = [
    ['#', '#', '#', '#', '#', '#', '#'],
    ['#', ' ', ' ', ' ', '#', '#', '#'],
    ['#', ' ', '*', '@', '#', '#', '#'],
    ['#', ' ', '#', '*', '#', '#', '#'],
    ['#', ' ', '.', ' ', '.', ' ', '#'],
    ['#', '#', ' ', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#', '#']
]

goal_state_easy = [
    ['#', '#', '#', '#', '#', '#', '#'],
    ['#', ' ', ' ', ' ', '#', '#', '#'],
    ['#', ' ', ' ', ' ', '#', '#', '#'],
    ['#', ' ', '#', ' ', '#', '#', '#'],
    ['#', ' ', '.', ' ', '.', ' ', '#'],
    ['#', '#', ' ', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#', '#']
]

initial_state_hard = [
    ['#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '.', '*', '#', '#', ' ', '#'],
    ['#', ' ', '.', '*', '.', '#', ' ', '#'],
    ['#', ' ', ' ', '*', '@', '#', ' ', '#'],
    ['#', '#', '#', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '#', ' ', '*', '#', '#', '#'],
    ['#', '#', '#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#']
]

goal_state_hard = [
    ['#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', '#', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '.', ' ', '#', '#', ' ', '#'],
    ['#', ' ', '.', ' ', '.', '#', ' ', '#'],
    ['#', ' ', ' ', ' ', '.', '#', ' ', '#'],
    ['#', '#', '#', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#']
]

puzzles = {
    'easy': (initial_state_easy, goal_state_easy),
    'medium': (initial_state, goal_state),
    'hard': (initial_state_hard, goal_state_hard),
}

actions = ['up', 'down', 'left', 'right']


def generate_next_states(state):
    next_states = []
    for action in actions:
        new_state = action_function(state, action)
        if new_state not in next_states:
            next_states.append(new_state)
    return next_states


def action_function(state, action):
    new_state = [row[:] for row in state]
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == '@':
                player_pos = (i, j)

    new_player_pos = None
    if action == 'up':
        new_player_pos = (player_pos[0] - 1, player_pos[1])
    elif action == 'down':
        new_player_pos = (player_pos[0] + 1, player_pos[1])
    elif action == 'left':
        new_player_pos = (player_pos[0], player_pos[1] - 1)
    elif action == 'right':
        new_player_pos = (player_pos[0], player_pos[1] + 1)

    if new_player_pos and new_state[new_player_pos[0]][new_player_pos[1]] != '#':
        target = new_state[new_player_pos[0]][new_player_pos[1]]
        if target in [' ', '.']:
            new_state[new_player_pos[0]][new_player_pos[1]] = '@'
            new_state[player_pos[0]][player_pos[1]] = '.' if goal_state[player_pos[0]][player_pos[1]] == '.' else ' '
        elif target == '*':
            move_offsets = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
            offset = move_offsets[action]
            new_box_pos = (new_player_pos[0] + offset[0], new_player_pos[1] + offset[1])
            if state[new_box_pos[0]][new_box_pos[1]] not in ['#', '*']:
                new_state[new_box_pos[0]][new_box_pos[1]] = '*'
                new_state[new_player_pos[0]][new_player_pos[1]] = '@'
                new_state[player_pos[0]][player_pos[1]] = '.' if goal_state[player_pos[0]][player_pos[1]] == '.' else ' '

    return new_state


def goal_test(state, goal_state):
    for i in range(len(goal_state)):
        for j in range(len(goal_state[i])):
            if goal_state[i][j] == '.' and state[i][j] != '*':
                return False
    return True


def print_state(state):
    for row in state:
        print(' '.join(row))
    print()


def get_user_action():
    valid_actions = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
    try:
        while True:
            user_input = input("Move (WASD or Q to restart): ").strip().lower()
            if user_input == 'q':
                return 'restart'
            elif user_input in valid_actions:
                return valid_actions[user_input]
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
        sys.exit(0)


def compute_reachable_box_positions():
    rows = len(goal_state)
    cols = len(goal_state[0])
    reachable = set()
    for i in range(rows):
        for j in range(cols):
            if goal_state[i][j] == '.':
                reachable.add((i, j))
    changed = True
    while changed:
        changed = False
        for i in range(rows):
            for j in range(cols):
                if goal_state[i][j] != '#' and (i, j) not in reachable:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_i, next_j = i + dx, j + dy
                        prev_i, prev_j = i - dx, j - dy
                        if (0 <= next_i < rows and 0 <= next_j < cols and
                                0 <= prev_i < rows and 0 <= prev_j < cols):
                            if (next_i, next_j) in reachable and goal_state[prev_i][prev_j] != '#':
                                reachable.add((i, j))
                                changed = True
                                break
    return reachable


def all_boxes_stuck(state):
    reachable = compute_reachable_box_positions()
    for i in range(len(state)):
        for j in range(len(state[i])):
            # Only consider boxes that are not already on a goal.
            if state[i][j] == '*' and (i, j) not in reachable:
                return True
    return False


def play_game():
    state = [row[:] for row in initial_state]
    print("Use W A S D to move. Press 'q' to restart. Press Ctrl+C to quit.\n")
    print_state(state)

    while True:
        if goal_test(state, goal_state):
            print("Puzzle Solved! All boxes are on goals!")
            return False  # Do not restart automatically

        if all_boxes_stuck(state):
            print("GAME OVER. A box is stuck and cannot reach a goal!")
            return False

        action = get_user_action()
        if action == 'restart':
            print("Game restarting...\n")
            return True  # Signal to restart
        else:
            state = action_function(state, action)
            print_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sokoban Game")
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                        help='Select difficulty level: easy, medium, or hard')
    args = parser.parse_args()

    selected_initial, selected_goal = puzzles[args.difficulty]
    initial_state = selected_initial
    goal_state = selected_goal

    while True:
        restart = play_game()
        if not restart:
            break
