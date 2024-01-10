"""

This module implements a large convolutional neural network for reinforcement learning.

"""

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


def build_model(input_shape, num_nodes_policy, num_nodes_value = 512):
    nn_input = Input(shape = input_shape)

    # Build hidden layers 1-3.
    nn_hidden = Conv2D(32, (3, 3), data_format = "channels_first", padding = "same")(nn_input)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(32, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(32, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)
    nn_hidden = MaxPooling2D((2, 2))(nn_hidden)
    nn_hidden = Dropout(0.2)(nn_hidden)

    # Build hidden layers 4-6.
    nn_hidden = Conv2D(64, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(64, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(64, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)
    nn_hidden = MaxPooling2D((2, 2))(nn_hidden)
    nn_hidden = Dropout(0.2)(nn_hidden)

    # Build hidden layers 7-9.
    nn_hidden = Conv2D(128, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(128, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(128, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)
    nn_hidden = MaxPooling2D((2, 2))(nn_hidden)
    nn_hidden = Dropout(0.2)(nn_hidden)

    # Build hidden layers 10-12.
    nn_hidden = Conv2D(256, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(256, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)

    nn_hidden = Conv2D(256, (3, 3), data_format = "channels_first", padding = "same")(nn_hidden)
    nn_hidden = BatchNormalization(axis = 1)(nn_hidden)
    nn_hidden = Activation("relu")(nn_hidden)
    nn_hidden = MaxPooling2D((2, 2))(nn_hidden)
    nn_hidden = Dropout(0.2)(nn_hidden)

    # Build policy hidden and output layers to return probability distribution over moves.
    nn_policy = Conv2D(2, (1, 1), data_format = "channels_first")(nn_hidden)
    nn_policy = BatchNormalization(axis = 1)(nn_policy)
    nn_policy = Activation("relu")(nn_policy)
    nn_policy = Flatten()(nn_policy)
    nn_policy = Dense(num_nodes_policy, activation = "softmax")(nn_policy)

    # Build value hidden and output layers to indicate which player is winning.
    nn_value = Conv2D(1, (1, 1), data_format = "channels_first")(nn_hidden)
    nn_value = BatchNormalization(axis = 1)(nn_value)
    nn_value = Activation("relu")(nn_value)
    nn_value = Flatten()(nn_value)
    nn_value = Dense(num_nodes_value, activation = "relu")(nn_value)
    nn_value = Dropout(0.2)(nn_value)
    nn_value = Dense(1, activation = "tanh")(nn_value)

    model = Model(inputs = [nn_input], outputs = [nn_policy, nn_value])

    return model
