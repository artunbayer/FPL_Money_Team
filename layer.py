import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import inits


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Dense:

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=True):

        self.vars = {}
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act

        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout

        # len(self.support) is asserted to be 1 since there is only one weight matrix per layer
        # initialize the weight matrices
            # the variables are found under the key tf.GlobalKeys.GLOBAL_VARIABLES
        self.vars['weights_' + str(0)] = inits.glorot([input_dim, output_dim], name='weights_' + str(0))

        # initialize the biases as 0 matrices of correct shapes
        if self.bias:
            self.vars['bias'] = inits.zeros([output_dim], name='bias')

    def __call__(self, inputs, sample_mask=None):
        # the graph convolution operation

        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)

        # for matrix do pre_sup = HW

        output = tf.matmul(x, self.vars['weights_' + str(0)])
        # we actually have only one support

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


