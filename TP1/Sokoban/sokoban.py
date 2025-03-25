import platform
import sys

# Detect OS for keyboard input
o_system = platform.system()
if o_system == 'Windows':
    import keyboard
else:
    import tty
    import termios

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# Initial and goal state (unchanged)
initial_state = [
    ['#','#', '#', '#', '#', '#', '#','#'],
    ['#','#', '#', ' ', ' ', ' ', '#','#'],
    ['#','.', '@', '*', ' ', ' ', '#','#'],
    ['#','#', '#', ' ', '*', '.', '#','#'],
    ['#','.', '#', '#', '*', ' ', '#','#'],
    ['#',' ', '#', ' ', '.', ' ', '#','#'],
    ['#','*', ' ', '*', '*', '*', '.','#'],
    ['#',' ', ' ', ' ', '.', ' ', ' ','#'],
    ['#','#', '#', '#', '#', '#', '#','#'],
]

goal_state = [
    ['#','#', '#', '#', '#', '#', '#','#'],
    ['#','#', '#', ' ', ' ', ' ', '#','#'],
    ['#','.', ' ', ' ', ' ', ' ', '#','#'],
    ['#','#', '#', ' ', ' ', '.', '#','#'],
    ['#','.', '#', '#', ' ', ' ', '#','#'],
    ['#',' ', '#', ' ', '.', ' ', '#','#'],
    ['#',' ', ' ', ' ', ' ', ' ', '.', '#'],
    ['#',' ', ' ', ' ', '.', ' ', ' ','#'],
    ['#','#', '#', '#', '#', '#', '#','#'],
]

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

# Extended to detect 'q' for restart
def get_user_action():
    valid_actions = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
    while True:
        if o_system == "Windows":
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name in valid_actions:
                    return valid_actions[event.name]
                elif event.name == 'q':
                    return 'restart'
        else:
            user_input = getch()
            if user_input in valid_actions:
                return valid_actions[user_input]
            elif user_input == 'q':
                return 'restart'

def all_boxes_stuck(state):
    for i in range(1, len(state)-1):
        for j in range(1, len(state[i])-1):
            if state[i][j] == '*':
                if goal_state[i][j] == '.':
                    continue
                if (state[i-1][j] == '#' and state[i][j-1] == '#') or \
                   (state[i-1][j] == '#' and state[i][j+1] == '#') or \
                   (state[i+1][j] == '#' and state[i][j-1] == '#') or \
                   (state[i+1][j] == '#' and state[i][j+1] == '#'):
                    continue
                else:
                    return False
    return True

# Play game once
def play_game():
    state = [row[:] for row in initial_state]
    print("Use W A S D to move. Press 'q' to restart. Press Ctrl+C to quit.\n")
    print_state(state)

    while True:
        if goal_test(state, goal_state):
            print("Puzzle Solved! All boxes are on goals!")
            return False  # Do not restart automatically

        if all_boxes_stuck(state):
            print("GAME OVER. All boxes are stuck!")
            return False

        action = get_user_action()
        if action == 'restart':
            print("Game restarting...\n")
            return True  # Signal to restart
        else:
            state = action_function(state, action)
            print_state(state)

# Run with restart loop
if __name__ == "__main__":
    while True:
        restart = play_game()
        if not restart:
            break  # Exit if not restarting
