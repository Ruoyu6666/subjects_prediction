import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model



class graph_convolution(layers.Layer):
    def __init__(self, num_features, num_classes):
        super(graph_convolution, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, adj = inputs
        x = tf.matmul(adj, x)
        x = self.dense1(x)
        x = tf.matmul(adj, x)
        x = self.dense2(x)
        return x


class NodeConv(keras.layers.Layer):
    def __init__(self, num_classes, n_output_nodes, activation="relu", **kwargs):
        """
        :param num_classes: number of target classes
        :param n_output_nodes: number of output nodes
        :param activation: activation function
        :param kwargs: NA
        """
        super(NodeConv, self).__init__(**kwargs)
        self.n_output_nodes = n_output_nodes
        self.num_classes = num_classes
        self.activation = keras.layers.Activation(activation)

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        """
        a_shape, x_shape = input_shape
        self.num_vertices = a_shape[1]
        w_init = keras.initializers.HeNormal(seed=None)
        self.weight = tf.Variable(
            initial_value=w_init(shape=(x_shape[2], self.n_output_nodes),
                                 dtype='float32'), trainable=True)

    def call(self, inputs):
        """
        :param inputs: input tensor shape
        :return: output of a particular node
        """
        a_tilda, x = inputs[0], inputs[1]
        y = tf.tensordot(tf.matmul(a_tilda, x), self.weight, axes=[2, 0])
        x_next = self.activation(y)
        return x_next
