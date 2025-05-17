import tensorflow as tf
from tensorflow.keras import layers, Model


class GraphConvolution(layers.Layer):
    def __init__(self, A, output_dim, activation="relu", rate=0.0):
        '''
        :param A: adjacency matrix  
        :param output_dim: output dimension
        :param activation: activation function  
        '''
        super(GraphConvolution, self).__init__()
        self.A = A
        self.output_dim = output_dim
        self.activation = layers.Activation(activation)
        self.rate = rate

    def build(self, input_shape):
        """
        :param input_shape: input tensor shape
        """
        self.weight = self.add_weight(shape=(input_shape[1], self.output_dim), initializer='glorot_uniform', dtype=tf.float32, trainable=True)

    def call(self, X):
        """
        :param inputs: input tensor
        """
        X = tf.nn.dropout(X, rate=self.rate)
        X = self.A @ X @ self.weight
        return self.activation(X)