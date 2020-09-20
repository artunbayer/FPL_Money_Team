import pandas as pd
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import layer
import metrics
flags = tf.app.flags
FLAGS = flags.FLAGS



class fantasyPL:
    def __init__(self, placeholders,layers, input_dim, **kwargs):

        self.vars = {}
        self.placeholders = placeholders
        self.layers = []
        self.activations = []
        self.num_layers = layers
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = 1

        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = None

        self.build()

    def _build(self):

        self.layers.append(layer.Dense(input_dim=self.input_dim,
                                       output_dim=FLAGS.hidden1,
                                       placeholders=self.placeholders,
                                       act=tf.nn.relu,
                                       dropout=True))

        for _ in range(self.num_layers -2):
            self.layers.append(layer.Dense(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden1,
                                           placeholders=self.placeholders,
                                           act=tf.nn.relu,
                                           dropout=True))

        self.layers.append(layer.Dense(input_dim=FLAGS.hidden1,
                                       output_dim=self.output_dim,
                                       placeholders=self.placeholders,
                                       act=tf.nn.relu,
                                       dropout=True))



    def build(self):
        """ Wrapper for _build() """

        self._build()

        # Build sequential layer model
        # Feed the values from the previous layer to the next layer
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # self.vars = {var.name: var for var in variables}
        print("variables =", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        # Build metrics
        self._loss()
        self._accuracy()

        # matrix1 = variables[0]
        # matrix2 = variables[1]

        # attempt at stop_gradients which does not retain graph connections
        # masked_matrix1 = entry_stop_gradients_column(matrix1, tf.expand_dims(self.weight_mask,0))
        # masked_matrix2 = entry_stop_gradients_row(matrix2, tf.expand_dims(self.weight_mask, 1))

        self.opt_op2 = self.optimizer.compute_gradients(self.loss, variables)[1]
        # self.opt_op2 = self.optimizer.compute_gradients(self.loss, ga)[0][0]
        # self.deneme = self.optimizer.compute_gradients(self.loss, variables[0])[0][0]
        # print(self.opt_op2)
        # print(self.deneme)
        # self.opt_op2 = tf.equal(matrix2, masked_matrix2)

        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):

        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += metrics.masked_mean_squared_error(self.outputs, self.placeholders['labels'],
                                                          self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = metrics.masked_mean_squared_error(self.outputs, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])

    def predict(self):
        return self.outputs
