import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, optimizers, layers, activations, regularizers, initializers
import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision
np.set_printoptions(threshold=sys.maxsize)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

class Print_Tensor(layers.Layer):
    def __init__(self, **kwargs):
        super(Print_Tensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Print_Tensor, self).build(input_shape)

    def call(self, input_data):
        print(input_data)
        return input_data

    def compute_output_shape(self, input_shape):
        return input_shape

class Concat_Const_One_Hot(layers.Layer):
    def __init__(self, number, **kwargs):
        self.number = number
        super(Concat_Const_One_Hot, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = tf.one_hot(list(range(self.number)), self.number, dtype=tf.float16)
        super(Concat_Const_One_Hot, self).build(input_shape)

    def call(self, input_data):
        return K.concatenate([input_data, tf.broadcast_to(self.kernel, (tf.shape(input_data)[0], tf.shape(input_data)[1], self.number))], axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]+self.dimension2)

class Concat_Const_Numbers(layers.Layer):
    def __init__(self, **kwargs):
        super(Concat_Const_Numbers, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = tf.constant([0,1,2,3,4,5], dtype=tf.uint8)
        super(Concat_Const_Numbers, self).build(input_shape)

    def call(self, input_data):
        return K.concatenate([input_data, tf.broadcast_to(self.kernel, (tf.shape(input_data)[0], 6))], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+6)

class Queuery_Variables(layers.Layer):
    def __init__(self, number_of_queueries, dimension, **kwargs):
        self.output_shape2 = (number_of_queueries, dimension)
        super(Queuery_Variables, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=self.output_shape2, trainable=True, initializer=initializers.RandomUniform(minval=0.001, maxval=1))
        super(Queuery_Variables, self).build(input_shape)

    def call(self, input_data):
        return tf.broadcast_to(self.kernel, (tf.shape(input_data)[0], self.output_shape2[0], self.output_shape2[1]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_shape2[0], self.output_shape2[1])

def attention_block(values, number_of_queries):
    queries = Queuery_Variables(number_of_queries, 60)(values)
    multi_head_attention_out = layers.MultiHeadAttention(num_heads=6, key_dim=60)(queries, values)
    add_out = layers.Add()([values, multi_head_attention_out])

    dense_out_1 = layers.Dense(60, activation=None)(add_out)
    elu_out = layers.ELU()(dense_out_1)
    dense_out_2 = layers.Dense(60, activation=None)(elu_out)
    out = layers.LayerNormalization()(dense_out_2)
    return out

def stacked_attention_model():
    input = layers.Input(shape=(48), name='inputs_one', dtype=tf.uint8)
    constant_added = Concat_Const_Numbers()(input)
    embedding = tf.one_hot(constant_added, 6)
    position_encoding_added = Concat_Const_One_Hot(54)(embedding)

    upsample = keras.layers.UpSampling1D(size=(4))(position_encoding_added)

    block_1_out = attention_block(upsample, 216)
    block_2_out = attention_block(block_1_out, 216)
    block_3_out = attention_block(block_2_out, 216)
    block_4_out = attention_block(block_3_out, 216)

    dense_out = layers.Dense(60, activation=None)(layers.Flatten()(block_4_out))
    elu_out = layers.ELU()(dense_out)
    output = layers.Dense(1, activation=None)(elu_out)

    model = Model(inputs=input, outputs=output)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=5e-5, beta_1=0.6, beta_2=0.9), metrics=['acc'], run_eagerly=False)
    model.summary()
    return model

def residual_block(inputs):
    dense_out_1 = layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l2(0.00001))(inputs)
    bn_out_1 = layers.BatchNormalization()(dense_out_1)
    dense_out_2 = layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l2(0.00001))(bn_out_1)
    bn_out_2 = layers.BatchNormalization()(dense_out_2)

    add_out = layers.Add()([inputs, bn_out_2])
    return add_out

def deepCubeA():
    input = layers.Input(shape=(48), name='inputs_one', dtype=tf.uint8)
    constant_added = Concat_Const_Numbers()(input)

    embedding = layers.Flatten()(tf.one_hot(constant_added, 6))

    dense_out_1 = layers.Dense(5000, activation="relu", kernel_regularizer=regularizers.l2(0.00001))(embedding)
    bn_out_1 = layers.BatchNormalization()(dense_out_1)
    dense_out_2 = layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l2(0.00001))(bn_out_1)
    bn_out_2 = layers.BatchNormalization()(dense_out_2)

    res_out_1 = residual_block(bn_out_2)
    res_out_2 = residual_block(res_out_1)
    res_out_3 = residual_block(res_out_2)
    res_out_4 = residual_block(res_out_3)

    output = layers.Dense(1, activation=None)(res_out_4)

    model = Model(inputs=input, outputs=output)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=1e-5), metrics=['acc'], run_eagerly=False)
    model.summary()
    return model

def recurrent_step(cube, query, LSTM_cell_state, previous_LSTM_output, Dense_Layer_1, Dense_Layer_Query,
                   Dense_Layer_Information, Dense_Layer_Output, Forget_Gate_Layer, Input_Gate_Layer, Update_Layer, Output_Gate_Layer):

    attention_out = layers.Attention()([query, cube])
    dense_out_1 = layers.ELU()(Dense_Layer_1(layers.Flatten()(attention_out)))

    LSTM_input = layers.Concatenate()([dense_out_1, previous_LSTM_output])
    forget_gate_values = Forget_Gate_Layer(LSTM_input)

    new_LSTM_cell_state = layers.Multiply()([forget_gate_values, LSTM_cell_state])

    input_gate_values = Input_Gate_Layer(LSTM_input)
    update_candidate_values = Update_Layer(LSTM_input)

    new_LSTM_cell_state = layers.Add()([layers.Multiply()([input_gate_values, update_candidate_values]), new_LSTM_cell_state])

    output_1_candidates = activations.tanh(new_LSTM_cell_state)
    output_1_gate_values = Output_Gate_Layer(LSTM_input)
    LSTM_output = layers.Multiply()([output_1_candidates, output_1_gate_values])

    information_output = layers.ELU()(Dense_Layer_Information(LSTM_output))
    value_output = Dense_Layer_Output(information_output)

    new_query = Dense_Layer_Query(LSTM_output)

    return tf.reshape(new_query, tf.shape(query)), new_LSTM_cell_state, LSTM_output, value_output

def recurrent_attention_network():
    input = layers.Input(shape=(48), name='inputs_one', dtype=tf.uint8)
    constant_added = Concat_Const_Numbers()(input)
    embedding = tf.one_hot(constant_added, 6)
    position_encoding_added = Concat_Const_One_Hot(54)(embedding)

    starting_query = Queuery_Variables(1, 60)(position_encoding_added)
    starting_cell_state = layers.Flatten()(Queuery_Variables(1, 600)(position_encoding_added))
    starting_output_state = layers.Flatten()(Queuery_Variables(1, 600)(position_encoding_added))

    Dense_Layer_1 = layers.Dense(60, activation=None)
    Forget_Gate_Layer = layers.Dense(600, activation="sigmoid")
    Input_Gate_Layer = layers.Dense(600, activation="sigmoid")
    Update_Layer = layers.Dense(600, activation="tanh")
    Output_Gate_Layer = layers.Dense(600, activation="sigmoid")
    Dense_Layer_Query = layers.Dense(60, activation="sigmoid")
    Dense_Layer_Information = layers.Dense(400, activation=None)
    Dense_Layer_Output = layers.Dense(1, activation=None)

    query_1, cell_state_1, output_state_1, output_1 = recurrent_step(position_encoding_added, starting_query, starting_cell_state,
                                                                     starting_output_state, Dense_Layer_1,
                                                                     Dense_Layer_Query, Dense_Layer_Information,
                                                                     Dense_Layer_Output, Forget_Gate_Layer,
                                                                     Input_Gate_Layer, Update_Layer, Output_Gate_Layer)
    query_2, cell_state_2, output_state_2, output_2 = recurrent_step(position_encoding_added, query_1, cell_state_1, output_state_1,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_3, cell_state_3, output_state_3, output_3 = recurrent_step(position_encoding_added, query_2, cell_state_2, output_state_2,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_4, cell_state_4, output_state_4, output_4 = recurrent_step(position_encoding_added, query_3, cell_state_3, output_state_3,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_5, cell_state_5, output_state_5, output_5 = recurrent_step(position_encoding_added, query_4, cell_state_4, output_state_4,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_6, cell_state_6, output_state_6, output_6 = recurrent_step(position_encoding_added, query_5, cell_state_5, output_state_5,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_7, cell_state_7, output_state_7, output_7 = recurrent_step(position_encoding_added, query_6, cell_state_6, output_state_6,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_8, cell_state_8, output_state_8, output_8 = recurrent_step(position_encoding_added, query_7, cell_state_7, output_state_7,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_9, cell_state_9, output_state_9, output_9 = recurrent_step(position_encoding_added, query_8, cell_state_8, output_state_8,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)
    query_10, cell_state_10, output_state_10, output_10 = recurrent_step(position_encoding_added, query_9, cell_state_9, output_state_9,
                                                                     Dense_Layer_1, Dense_Layer_Query,
                                                                     Dense_Layer_Information, Dense_Layer_Output,
                                                                     Forget_Gate_Layer, Input_Gate_Layer, Update_Layer,
                                                                     Output_Gate_Layer)

    output = layers.Concatenate()([output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10])

    model = Model(inputs=input, outputs=output)
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(learning_rate=1e-5), metrics=[],
                  run_eagerly=False)
    model.summary()
    return model

def custom_loss(y_actual, y_predicted):
    difference = tf.subtract(y_actual, y_predicted)
    #weights = tf.constant(np.asarray([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 000.1, 000.1, 1.0]), dtype="float16")
    weights = tf.constant(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), dtype="float16")

    difference = tf.multiply(difference, weights)
    multiply_result = tf.multiply(difference, difference)
    error = tf.reduce_sum(multiply_result, axis=1)
    return error
