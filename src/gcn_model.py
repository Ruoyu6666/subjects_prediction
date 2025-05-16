import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model


class GraphConvolution(layers.Layer):
    def __init__(self, output_dim, activation="relu", **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = layers.Activation(activation)

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        """
        self.num_vertices = input_shape[0][1]
        self.weight = self.add_weight(shape=(self.num_vertices, self.output_dim), initializer='glorot_uniform')

    def call(self, inputs):
        """
        :param inputs: input tensor
        """
        a, x = inputs[0], inputs[1]
        y = tf.tensordot(tf.matmul(a, x), self.weight, axes=[2, 0])
        return self.activation(y)




def build_gcn_model(input_shape, output_dim, num_classes):
    """
    Build the GCN model.
    :param input_shape: shape of the input tensor
    :param output_dim: output dimension
    :param num_classes: number of classes
    :return: GCN model
    """
    a_input = tf.keras.Input(shape=(None,), sparse=True)
    x_input = tf.keras.Input(shape=(input_shape,))

    gcn1 = GraphConvolution(output_dim=output_dim, activation='relu')([a_input, x_input])
    gcn2 = GraphConvolution(output_dim=num_classes, activation='softmax')([a_input, gcn1])

    return Model(inputs=[a_input, x_input], outputs=gcn2)