import cube_processing
import numpy as np
import models

scramble_range = (1, 18)
load_index = 0
save_index = 0
test_set = []

indices = {"Stacked_Attention_Curriculum": 433,
           "DeepCubeA": 1185,
           "Recurrent": 2169,
           "DeepCubeA_Curriculum": 2502}

def generate_weights():
    weights = np.ones(scramble_range[1]-scramble_range[0]+1)*((scramble_range[1]+scramble_range[0])/2)
    for i in range(len(weights)):
        weights[i] = weights[i] + i*2.5
    weights = weights/np.sum(weights)
    print(weights)
    return list(weights)

def print_performance_evaluation(model, cube, weights=None):
    global test_set
    if test_set == []:
        X, Y = cube.get_test_batch(3000, (1,20), weights)
        X = cube.convert_strings_to_arrays(X)
        test_set = (X, Y)
    prediction = model.predict(test_set[0], verbose=1, batch_size=128)
    errors = np.abs(np.subtract(test_set[1], np.reshape([x[-1] for x in prediction], (3000,))))
    loss = np.sum(errors)/3000
    loss2 = np.dot(errors, errors)/3000

    for i in range(20):
        print(str(test_set[1][i]) + "  " + str(prediction[i]))
    print("MAE: " + str(loss))
    print("MSE: " + str(loss2))

def solve_cube(model, number_of_states, length=50, scramble=[], print_progress=True):
    cube = cube_processing.Cube_3x3()
    cube.set_solve_test_scramble(length, scramble=scramble, print_scramble=print_progress)
    go_on = False
    i = 0
    while not go_on:
        i += 1
        inputs = cube.convert_strings_to_arrays(cube.get_prediction_inputs_search())
        predictions = model.predict(inputs, batch_size=1024)
        go_on, min_value = cube.update_states_search_with_limited_branching([x[-1] for x in predictions], number_of_states, 9)
        if print_progress:
            print("Current movecount: " + str(i) + "   Predicted moves left: " + str(min_value))

    if print_progress:
        for solution in cube.solutions:
            print("Solutions: " + " ".join(solution))

    return len(cube.solutions[0])

def load_weights(model, index=load_index):
    try:
        model.load_weights("checkpoint/weights-" + str(index) + ".ckpt")
        print("Weights loaded successfully!")
    except:
        print("No weights loaded.")
        pass

def save_weights(model, index=save_index):
    try:
        model.save_weights("checkpoint/weights-" + str(index) + ".ckpt")
    except:
        f = open("checkpoint/weights-" + str(index) + ".ckpt", "w+")
        model.save_weights("checkpoint/weights-" + str(index) + ".ckpt")
        f.close()

def load_scrambles(filename):
    f = open(filename, "r")
    scrambles = []
    lengths = []
    for line in f.readlines():
        scramble = line[0:-10].split(" ")
        lengths.append(line[-7:-5])
        scrambles.append(scramble)
    return scrambles, lengths

def print_values_for_scramble(model, scramble):
    states = []
    cube = cube_processing.Cube_3x3()
    for i in range(len(scramble)):
        current_state = cube_processing.solved_state[:]
        cube_processing.apply_sequence(current_state, scramble[:i+1])
        states.append(current_state)
    print(model.predict(cube.convert_strings_to_arrays(states)))
    prediction = list(np.reshape([x[-1] for x in model.predict(cube.convert_strings_to_arrays(states))], (len(scramble),)))
    prediction.reverse()
    print(list(prediction))

def train_model(model):
    global scramble_range
    cube = cube_processing.Cube_3x3()
    weights = generate_weights()

    print_performance_evaluation(model, cube)

    while True:
        num_splits = 10
        for i in range(num_splits):
            X, Y = cube.generate_new_training_set(200000, scramble_range, weights)
        X_new = cube.convert_strings_to_arrays(X)
        model.fit(X_new, Y, batch_size=128, verbose=1, shuffle=False)
        print_performance_evaluation(model, cube)
        save_weights(model)

        inputs = cube.convert_strings_to_arrays(
                cube.get_prediction_inputs_Bellman(20000, scramble_range, weights=weights))
        predictions = model.predict(inputs, batch_size=128, verbose=1)

        X, Y = cube.get_updated_values_Bellman(predictions)
        X = cube.convert_strings_to_arrays(X)
        model.fit(X, Y, batch_size=128, verbose=1, shuffle=True)
        print_performance_evaluation(model, cube)
        save_weights(model)


def evaluate_model(model, n_states=250):
    scrambles, lengths = load_scrambles("evaluation_set.txt")
    accumulated_differences = 0
    diffs = []
    for scramble_index in range(0, len(scrambles)):
        print("Optimal solution: " + str(lengths[scramble_index]))
        solution_length = solve_cube(model, n_states, scramble=scrambles[scramble_index])
        accumulated_differences += solution_length - int(lengths[scramble_index])
        diffs.append(solution_length - int(lengths[scramble_index]))
        print("\n")

    diffs = np.sort(diffs)
    print(accumulated_differences)
    print(accumulated_differences/len(scrambles))
    print(diffs)

def load_stacked_attention_model(load_best_parameters=True):
    model = models.stacked_attention_model()
    if load_best_parameters: load_weights(model, index=indices["Stacked_Attention_Curriculum"])
    return model

def load_recurrent_model(load_best_parameters=True):
    model = models.recurrent_attention_network()
    if load_best_parameters: load_weights(model, index=indices["Recurrent"])
    return model

def load_DeepCubeA(load_best_parameters=True):
    model = models.deepCubeA()
    if load_best_parameters: load_weights(model, index=indices["DeepCubeA"])
    return model

if __name__ == '__main__':
    model = load_stacked_attention_model(True)
    evaluate_model(model, n_states=1000)