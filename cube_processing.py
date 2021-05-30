import numpy as np
import random
from PIL import Image

solved_state = ["oooooooo", "gggggggg", "rrrrrrrr", "bbbbbbbb", "wwwwwwww", "yyyyyyyy"]
moves_qtm = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
moves_htm = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'", "R2", "L2", "F2", "B2", "U2", "D2"]
number_of_moves_qtm = 12
groups = {"R": ["R'", "R", "R2"],
             "L": ["L'", "L", "L2"],
             "U": ["U'", "U", "U2"],
             "D": ["D'", "D", "D2"],
             "F": ["F'", "F", "F2"],
             "B": ["B'", "B", "B2"]}

not_changing = {"R" : "L",
                "L" : "R",
                "U": "D",
                "D": "U",
                "F": "B",
                "B": "F"}

map_colors_to_numbers = {"w": 0,
                         "y": 1,
                         "b": 2,
                         "g": 3,
                         "r": 4,
                         "o": 5}

class Cube_3x3:

    def __init__(self):
        self.states = []
        self.solved_states = []
        self.paths = []

        self.index_in_current_depth = 0
        self.split_rate = 2
        self.current_depth = 0

        self.solutions = []
        self.scramble = []


        self.nodes = {}
        self.node_paths = {}
        self.current_batch_info = []
        self.K = 1
        self.min_solution_length = 30
        self.solution_length_update_needed = False
        self.max_index = 1
        self.weights = []
        self.number_of_states = 20
        self.max_number_of_states_kept = 5000
        self.sorted_values = {}
        self.max_branching_factor = 9
        self.visited_states = {}

    def get_prediction_inputs_search(self):
        batch = []
        for state in self.states:
            for move in moves_qtm:
                new_state = state[:]
                map_moves[move](new_state)
                batch.append(new_state)
        return batch

    def set_solve_test_scramble(self, length, scramble=[], print_scramble=True):
        if scramble == []:
            scramble = generate_random_sequence_qtm(length)
        self.scramble = scramble
        if print_scramble:
            print("Scramble: " + " ".join(scramble))
        state = solved_state[:]
        apply_sequence(state, scramble)
        self.states = [state]
        self.paths = [[]]
        self.solutions = []
        self.visited_states = []

    def update_states_search_with_limited_branching(self, predictions, number_of_states, branching_factor):
        new_states = []
        new_paths = []
        predictions = np.reshape(predictions, np.shape(predictions)[0])
        reduced_predictions = []
        reduced_moves = []
        for i in range(len(self.states)):
            current_values = predictions[i*number_of_moves_qtm:i*number_of_moves_qtm+number_of_moves_qtm]
            sorted = np.sort(current_values)
            for j in range(0, branching_factor):
                index, = np.where(current_values == sorted[j])
                reduced_predictions.append(current_values[index[0]])
                reduced_moves.append(moves_qtm[index[0]])
        predictions = reduced_predictions
        sorted = np.sort(predictions)
        value = sorted[np.min([number_of_states, len(predictions)-1])]
        min = sorted[0]
        for i in range(len(self.states)):
            for j in range(branching_factor):
                index = i*branching_factor + j
                if predictions[index] <= value:
                    new_state = self.states[i][:]
                    map_moves[reduced_moves[index]](new_state)
                    if new_state not in new_states and new_state not in self.visited_states:
                        new_path = self.paths[i][:]
                        new_path.append(reduced_moves[index])
                        new_paths.append(new_path)
                        new_states.append(new_state)
                        if new_state == solved_state:
                            self.solutions.append(new_path)
                        #if predictions[i*number_of_moves_qtm + j] == min:
                        #    print(new_state)
        self.visited_states = self.visited_states + self.states
        self.states = new_states
        self.paths = new_paths
        if len(self.solutions) > 0:
            return True, min
        return False, min

    def generate_new_training_set(self, number_of_states, scramble_range, weights):
        X = []
        Y = []
        lengths = random.choices(range(scramble_range[0], scramble_range[1]+1), weights=weights, k=number_of_states)

        for length in lengths:
            scramble = generate_random_sequence_qtm(length)
            state = solved_state[:]
            apply_sequence(state, scramble)
            X.append(state)
            Y.append(float(length))
        return X, np.asarray(Y)

    def get_prediction_inputs_Bellman(self, number_of_states, scramble_range, weights=[]):
        self.states = []
        self.solved_states = []
        batch = []
        if weights == []:
            lengths = random.choices(range(scramble_range[0], scramble_range[1] + 1), k=number_of_states)
        else:
            lengths = random.choices(range(scramble_range[0], scramble_range[1] + 1), weights=weights, k=number_of_states)
        for length in lengths:
            scramble = generate_random_sequence_qtm(length)
            state = solved_state[:]
            apply_sequence(state, scramble)
            self.states.append(state)
            for move in moves_qtm:
                changed_state = state[:]
                map_moves[move](changed_state)
                batch.append(changed_state)
            if length == 1:
                if len(scramble[0]) > 1:
                    reverse_move = scramble[0][0]
                else:
                    reverse_move = scramble[0] + "'"
                index = moves_qtm.index(reverse_move)
                self.solved_states.append(len(batch)-(number_of_moves_qtm-index))
        return batch


    def convert_strings_to_arrays(self, states_array):
        array = []
        for state in states_array:
            current_state = []
            for string in state:
                for char in string:
                    current_state.append(map_colors_to_numbers[char])
            array.append(current_state)
        return np.asarray(array)


    def get_updated_values_Bellman(self, predictions):
        update_x = []
        update_y = []
        for i in range(0, len(self.states)):
            if len(self.solved_states) > 0 and self.solved_states[0] < (i+1)*(number_of_moves_qtm):
                min = 0
                del self.solved_states[0]
            else:
                min = np.min([x[-1] for x in predictions[(i*number_of_moves_qtm):(i+1)*number_of_moves_qtm]])
            value = 1+min
            update_x.append(self.states[i])
            update_y.append(value)
        return update_x, np.asarray(update_y)

    def get_test_batch(self, number_of_states, scramble_range, weights):
        X = []
        Y = []
        lengths = random.choices(range(scramble_range[0], scramble_range[1]+1), weights=weights, k=number_of_states)
        for length in lengths:
            scramble = generate_random_sequence_qtm(length)
            state = solved_state[:]
            apply_sequence(state, scramble)
            X.append(state)
            Y.append(length)
        return X, np.asarray(Y)


def do_R(state):
    state[2] = rotate_side_clockwise(state[2])
    buffer1 = state[1][2:5]
    state[1] = state[1][:2] + state[5][2:5] + state[1][5:]
    buffer2 = state[4][2:5]
    state[4] = state[4][:2] + buffer1 + state[4][5:]
    buffer1 = state[3][0] + (state[3][6:8])[::-1]
    state[3] = buffer2[2] + state[3][1:6] + buffer2[:2]
    state[5] = state[5][:2] + buffer1[::-1] + state[5][5:]

def do_R_prime(state):
    state[2] = rotate_side_counter_clockwise(state[2])
    buffer1 = state[1][2:5]
    state[1] = state[1][:2] + state[4][2:5] + state[1][5:]
    buffer2 = state[5][2:5]
    state[5] = state[5][:2] + buffer1 + state[5][5:]
    buffer1 = state[3][0] + (state[3][6:8])[::-1]
    state[3] = buffer2[2] + state[3][1:6] + buffer2[0] + buffer2[1]
    state[4] = state[4][:2] + buffer1[::-1] + state[4][5:]

def do_U(state):
    state[4] = rotate_side_clockwise(state[4])
    buffer1 = state[1][:3]
    state[1] = state[2][:3] + state[1][3:]
    buffer2 = state[0][:3]
    state[0] = buffer1 + state[0][3:]
    buffer1 = state[3][:3]
    state[3] = buffer2 + state[3][3:]
    state[2] = buffer1 + state[2][3:]

def do_U_prime(state):
    state[4] = rotate_side_counter_clockwise(state[4])
    buffer1 = state[1][:3]
    state[1] = state[0][:3] + state[1][3:]
    buffer2 = state[2][:3]
    state[2] = buffer1 + state[2][3:]
    buffer1 = state[3][:3]
    state[3] = buffer2 + state[3][3:]
    state[0] = buffer1 + state[0][3:]

def do_L(state):
    state[0] = rotate_side_clockwise(state[0])
    buffer1 = state[1]
    state[1] = state[4][0] + state[1][1:6] + state[4][6:]
    buffer2 = state[5][0] + state[5][6:8][::-1]
    state[5] = buffer1[0] + state[5][1:6] + buffer1[6:]
    buffer1 = state[3]
    state[3] = state[3][0:2] + buffer2[::-1] + state[3][5:]
    state[4] = buffer1[4] + state[4][1:6] + buffer1[2:4]

def do_L_prime(state):
    state[0] = rotate_side_counter_clockwise(state[0])
    buffer1 = state[1]
    state[1] = state[5][0] + state[1][1:6] + state[5][6:]
    buffer2 = state[4]
    state[4] = buffer1[0] + state[4][1:6] + buffer1[6:]
    buffer1 = state[3]
    state[3] = state[3][0:2] + buffer2[6:] + buffer2[0] + state[3][5:]
    state[5] = buffer1[4] + state[5][1:6] + buffer1[2:4]


def do_F(state):
    state[1] = rotate_side_clockwise(state[1])
    buffer1 = state[4]
    state[4] = state[4][:4] + state[0][2:5] + state[4][7]
    buffer2 = state[2]
    state[2] = buffer1[6] + state[2][1:6] + buffer1[4:6]
    buffer1 = state[5]
    state[5] = buffer2[6:] + buffer2[0] + state[5][3:]
    state[0] = state[0][:2] + buffer1[:3] + state[0][5:]

def do_F_prime(state):
    state[1] = rotate_side_counter_clockwise(state[1])
    buffer1 = state[4]
    state[4] = state[4][:4] + state[2][6:] + state[2][0] + state[4][7]
    buffer2 = state[0]
    state[0] = state[0][:2] + buffer1[4:7] + state[0][5:]
    buffer1 = state[5]
    state[5] = buffer2[2:5] + state[5][3:]
    state[2] = buffer1[2] + state[2][1:6] + buffer1[:2]

def do_D(state):
    state[5] = rotate_side_clockwise(state[5])
    buffer1 = state[1]
    state[1] = state[1][:4] + state[0][4:7] + state[1][7]
    buffer2 = state[2]
    state[2] = state[2][:4] + buffer1[4:7] + state[2][7]
    buffer1 = state[3]
    state[3] = state[3][:4] + buffer2[4:7] + state[3][7]
    state[0] = state[0][:4] + buffer1[4:7] + state[0][7]

def do_D_prime(state):
    state[5] = rotate_side_counter_clockwise(state[5])
    buffer1 = state[1]
    state[1] = state[1][:4] + state[2][4:7] + state[1][7]
    buffer2 = state[0]
    state[0] = state[0][:4] + buffer1[4:7] + state[0][7]
    buffer1 = state[3]
    state[3] = state[3][:4] + buffer2[4:7] + state[3][7]
    state[2] = state[2][:4] + buffer1[4:7] + state[2][7]

def do_B(state):
    state[3] = rotate_side_clockwise(state[3])
    buffer1 = state[4]
    state[4] = state[2][2:5] + state[4][3:]
    buffer2 = state[0]
    state[0] = buffer1[2] + state[0][1:6] + buffer1[:2]
    buffer1 = state[5]
    state[5] = state[5][:4] + buffer2[6:] + buffer2[0] + state[5][7]
    state[2] = state[2][:2] + buffer1[4:7] + state[2][5:]

def do_B_prime(state):
    state[3] = rotate_side_counter_clockwise(state[3])
    buffer1 = state[4]
    state[4] = state[0][6:] + state[0][0] + state[4][3:]
    buffer2 = state[2]
    state[2] = state[2][:2] + buffer1[:3] + state[2][5:]
    buffer1 = state[5]
    state[5] = state[5][:4] + buffer2[2:5] +state[5][7]
    state[0] = buffer1[6] + state[0][1:6] + buffer1[4:6]

def do_y(state):
    state[4] = rotate_side_clockwise(state[4])
    state[5] = rotate_side_counter_clockwise(state[5])
    buffer1 = state[0]
    state[0] = state[1]
    state[1] = state[2]
    state[2] = state[3]
    state[3] = buffer1

def do_y_prime(state):
    state[4] = rotate_side_counter_clockwise(state[4])
    state[5] = rotate_side_clockwise(state[5])
    buffer1 = state[0]
    state[0] = state[3]
    state[3] = state[2]
    state[2] = state[1]
    state[1] = buffer1

def do_x(state):
    state[2] = rotate_side_clockwise(state[2])
    state[0] = rotate_side_counter_clockwise(state[0])
    buffer1 = state[1]
    state[1] = state[5]
    state[5] = state[3]
    state[5] = rotate_side_clockwise(state[5])
    state[5] = rotate_side_clockwise(state[5])
    state[3] = state[4]
    state[3] = rotate_side_clockwise(state[3])
    state[3] = rotate_side_clockwise(state[3])
    state[4] = buffer1

def do_x_prime(state):
    state[2] = rotate_side_counter_clockwise(state[2])
    state[0] = rotate_side_clockwise(state[0])
    buffer1 = state[1]
    state[1] = state[4]
    state[4] = state[3]
    state[4] = rotate_side_clockwise(state[4])
    state[4] = rotate_side_clockwise(state[4])
    state[3] = state[5]
    state[3] = rotate_side_clockwise(state[3])
    state[3] = rotate_side_clockwise(state[3])
    state[5] = buffer1

def do_z(state):
    state[1] = rotate_side_clockwise(state[1])
    state[3] = rotate_side_counter_clockwise(state[3])
    buffer1 = state[4]
    state[4] = state[0]
    state[4] = rotate_side_clockwise(state[4])
    state[0] = state[5]
    state[0] = rotate_side_clockwise(state[0])
    state[5] = state[2]
    state[5] = rotate_side_clockwise(state[5])
    state[2] = buffer1
    state[2] = rotate_side_clockwise(state[2])

def do_z_prime(state):
    state[1] = rotate_side_counter_clockwise(state[1])
    state[3] = rotate_side_clockwise(state[3])
    buffer1 = state[4]
    state[4] = state[2]
    state[4] = rotate_side_counter_clockwise(state[4])
    state[2] = state[5]
    state[2] = rotate_side_counter_clockwise(state[2])
    state[5] = state[0]
    state[5] = rotate_side_counter_clockwise(state[5])
    state[0] = buffer1
    state[0] = rotate_side_counter_clockwise(state[0])

def reverse_move(move):
    if len(move) == 1:
        return move + "'"
    else:
        return move[0]

def rotate_side_clockwise(str):
    return str[-2:] + str[:-2]

def rotate_side_counter_clockwise(str):
    return str[2:] + str[:2]

map_moves = {"R": do_R,
             "R'": do_R_prime,
             "L": do_L,
             "L'": do_L_prime,
             "U": do_U,
             "U'": do_U_prime,
             "D": do_D,
             "D'": do_D_prime,
             "F": do_F,
             "F'": do_F_prime,
             "B": do_B,
             "B'": do_B_prime,
             "y": do_y,
             "y'": do_y_prime,
             "x": do_x,
             "x'": do_x_prime,
             "z": do_z,
             "z'": do_z_prime}

def apply_sequence(state, sequence):
    for move in sequence:
        if move[-1] == "2":
            map_moves[move[0]](state)
            map_moves[move[0]](state)
        else:
            map_moves[move](state)

def generate_random_sequence_htm(length):
    sequence = generate_random_sequence_qtm(int(length*1.4))
    final_sequence = []
    i = 0
    while len(final_sequence) < length and i < len(sequence):
        if sequence[i] == sequence[i+1]:
            final_sequence.append(sequence[i][0]+"2")
            i += 2
        else:
            final_sequence.append(sequence[i])
            i += 1
    return final_sequence


def generate_random_sequence_qtm(length):
    sequence = []
    prev_move = random.choice(moves_qtm)
    sequence.append(prev_move)
    if len(prev_move) == 1:
        locked_moves = [prev_move+ "'"]
    else:
        locked_moves = [prev_move[0]]

    for i in range(1, length):
        possible_moves = [x for x in moves_qtm if x not in locked_moves]
        new_move = random.choice(possible_moves)
        sequence.append(new_move)
        if prev_move == new_move:
            locked_moves.append(new_move)
        elif prev_move[0] == not_changing[new_move[0]]:
            locked_moves = locked_moves + groups[prev_move[0]]
        else:
            locked_moves = []

        if len(new_move) == 1:
            locked_moves.append(new_move + "'")
        else:
            locked_moves.append(new_move[0])
        prev_move = new_move
    return sequence

colors = {"r": [178,34,34],
"b": [0, 0, 139],
"g": [0, 128, 0],
"y": [255, 255, 0],
"w": [255, 255, 255],
"o": [255, 140, 0],
"x": [0, 0, 0]}

def display_cube(state):
    image = np.ones((11, 14, 3))*255
    image[2][5] = colors["w"]
    image[5][5] = colors["g"]
    image[8][5] = colors["y"]
    image[5][2] = colors["o"]
    image[5][8] = colors["r"]
    image[5][11] = colors["b"]

    map_side(image, state[0], 1, 4)
    map_side(image, state[1], 4, 4)
    map_side(image, state[2], 7, 4)
    map_side(image, state[3], 10, 4)
    map_side(image, state[4], 4, 1)
    map_side(image, state[5], 4, 7)

    image = Image.fromarray(image.astype('uint8'))
    image = image.resize((1400, 1100), Image.NEAREST)
    image = np.array(image)

    draw_vertical_line(image, 100, 400, 700, 2, colors["x"])
    draw_vertical_line(image, 400, 100, 1000, 2, colors["x"])
    draw_vertical_line(image, 700, 100, 1000, 2, colors["x"])
    draw_vertical_line(image, 1000, 400, 700, 2, colors["x"])
    draw_vertical_line(image, 1300, 400, 700, 2, colors["x"])#

    draw_vertical_line(image, 200, 400, 700, 1, colors["x"])
    draw_vertical_line(image, 300, 400, 700, 1, colors["x"])
    draw_vertical_line(image, 500, 100, 1000, 1, colors["x"])
    draw_vertical_line(image, 600, 100, 1000, 1, colors["x"])
    draw_vertical_line(image, 800, 400, 700, 1, colors["x"])
    draw_vertical_line(image, 900, 400, 700, 1, colors["x"])
    draw_vertical_line(image, 1100, 400, 700, 1, colors["x"])
    draw_vertical_line(image, 1200, 400, 700, 1, colors["x"])

    draw_horizantal_line(image, 400, 700, 100, 2, colors["x"])
    draw_horizantal_line(image, 100, 1300, 400, 2, colors["x"])
    draw_horizantal_line(image, 100, 1300, 700, 2, colors["x"])
    draw_horizantal_line(image, 400, 700, 1000, 2, colors["x"])

    draw_horizantal_line(image, 400, 700, 200, 1, colors["x"])
    draw_horizantal_line(image, 400, 700, 300, 1, colors["x"])
    draw_horizantal_line(image, 100, 1300, 500, 1, colors["x"])
    draw_horizantal_line(image, 100, 1300, 600, 1, colors["x"])
    draw_horizantal_line(image, 400, 700, 800, 1, colors["x"])
    draw_horizantal_line(image, 400, 700, 900, 1, colors["x"])

    image = Image.fromarray(image.astype('uint8'))
    Image._show(image)

def draw_vertical_line(image, x, y, y_end, width, color):
    for i in range(y, y_end):
        for j in range(x-width, x+width+1):
            image[i][j] = color

def draw_horizantal_line(image, x, x_end, y, width, color):
    for i in range(x, x_end):
        for j in range(y-width, y+width+1):
            image[j][i] = color


def map_side(array, side, x, y):
    array[y][x] = colors[side[0]]
    array[y][x+1] = colors[side[1]]
    array[y][x+2] = colors[side[2]]
    array[y+1][x] = colors[side[7]]
    array[y+1][x + 2] = colors[side[3]]
    array[y+2][x] = colors[side[6]]
    array[y+2][x + 1] = colors[side[5]]
    array[y+2][x + 2] = colors[side[4]]