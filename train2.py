from __future__ import division
from __future__ import print_function
import model
import requests
import pandas as pd
import model
import numpy as np
import time
import data
import utils
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt

###
###  PREPROCESS THE DATA
###########################
#x, y = data.preprocces_data()
###########################






# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
#flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

# Load data
training_inputs, training_data_values, test_inputs, test_data_values = data.load_big_data()
print(training_inputs.shape)
print(training_data_values.shape)


train_mask = utils.sample_mask(np.arange(training_inputs.shape[0]), training_inputs.shape[0])
test_mask = utils.sample_mask(np.arange(test_inputs.shape[0]), test_inputs.shape[0])
#val_mask = utils.sample_mask(val_indexes, y.shape[0])


y_train = np.zeros(training_data_values.shape)
#y_val = np.zeros(y.shape)
y_test = np.zeros(test_data_values.shape)
y_train[train_mask, :] = training_data_values[train_mask, :]
#y_val[val_mask, :] = y[val_mask, :]
y_test[test_mask, :] = test_data_values[test_mask, :]

# Some preprocessing
#features = utils.preprocess_features(training_inputs)
#features_test = utils.preprocess_features(test_inputs)

model_func = model.fantasyPL

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None,training_inputs.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

# Create model
model = model_func(placeholders, 4, input_dim=training_inputs.shape[1])

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = utils.construct_feed_dict(features, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
t_0 = time.time()

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = utils.construct_feed_dict(training_inputs, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    #cost, acc, duration = evaluate(features, y_val, val_mask, placeholders)
    #cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          #"val_loss=", "{:.5f}".format(cost),
           #"val_acc=", "{:.5f}".format(acc),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

    if epoch > 500 and cost_val[-1] > np.mean(cost_val[-(1000+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(test_inputs, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print('Total time:', time.time() - t_0)




feed_dict = utils.construct_feed_dict(test_inputs, y_test, test_mask, placeholders)
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
prediction = sess.run(model.predict(), feed_dict=feed_dict)


#vector1 = np.resize(y[test_indexes], (1, test_indexes.size))
#vector2 = np.resize(prediction[test_indexes], (1, test_indexes.size))
#vector3 = vector2 - vector1
#print(np.dot(vector3, vector3.transpose())/test_indexes.size)

print(np.hstack((test_data_values, prediction)))

plottable = np.resize(test_data_values, (1,test_data_values.shape[0]))[0]
xs = np.resize(np.arange(test_data_values.shape[0]),(1, test_data_values.shape[0]))
plottable2 = np.resize(prediction, (1, test_data_values.shape[0]))[0]

non_zeros = plottable.nonzero()
plt.scatter(plottable2[non_zeros], plottable[non_zeros],s=5)
plt.plot([0, 300], [0, 300], 'k-', color = 'r')
plt.xlabel('Predicted Score')
plt.ylabel('Actual Score')
print(np.mean(np.abs(plottable2[non_zeros]-plottable[non_zeros])/plottable[non_zeros]))

plt.show()
