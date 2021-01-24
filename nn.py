# from keras.layers import Dense, Conv2D, Flatten
# import tensorflow.keras as keras
# import tensorflow as tf
#
# class BaseDeepQNet(keras.Model):
#     def __init__(self, input_dims, n_actions):
#         super(BaseDeepQNet, self).__init__()
#         self.conv1 = Conv2D(16, (1, 3), activation='relu', input_shape=input_dims)
#         self.conv2 = Conv2D(16, (3, 1), activation='relu')
#         self.flat = Flatten()
#         self.dense1 = Dense(16, activation='linear')
#         self.dense2 = Dense(n_actions, activation='linear')
#
#     def call(self, state):
#         x = self.conv1(state)
#         x = self.conv2(x)
#         x = self.flat(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#
#         return x
#
# class DuelingDeepQNet(keras.Model):
#     def __init__(self, input_dims, n_actions):
#         super(DuelingDeepQNet, self).__init__()
#         self.conv1 = Conv2D(16, (1, 3), activation='relu', input_shape=input_dims)
#         self.conv2 = Conv2D(16, (3, 1), activation='relu')
#         self.dense1 = Dense(16, activation='linear')
#         self.dense2 = Dense(n_actions, activation='linear')
#         self.V = Dense(1, activation=None)
#         self.A = Dense(n_actions, activation=None)
#
#     def call(self, state):
#         x = self.conv1(state)
#         x = self.conv2(x)
#         x = x.flatten()
#         x = self.dense1(x)
#         x = self.dense2(x)
#         V = self.V(x)
#         A = self.A(x)
#         Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
#
#         return A, Q

from keras.layers import Dense, Activation, Conv2D, Flatten
import tensorflow.keras as keras
import tensorflow as tf

class BaseDeepQNet(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(BaseDeepQNet, self).__init__()
        self.conv1 = Conv2D(16, (1, 3), activation='relu', input_shape=input_dims)
        self.conv2 = Conv2D(16, (3, 1), activation='relu')
        self.flat = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(n_actions, activation='linear')

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

class DuelingDeepQNet(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(DuelingDeepQNet, self).__init__()
        self.conv1 = Conv2D(16, (1, 3), activation='relu', input_shape=input_dims)
        self.conv2 = Conv2D(16, (3, 1), activation='relu')
        self.flat = Flatten()
        self.dense1 = Dense(64, activation='linear')
        self.dense2 = Dense(n_actions, activation='linear')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)

        return A