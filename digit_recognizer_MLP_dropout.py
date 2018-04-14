import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import seaborn as sns
#get_ipython().magic('matplotlib inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
import math


def forward_propagation(X, keep_prob):
    """
    Implements the forward propagation for 5 layer Neural Network.

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)

    Returns:
    A5 -- the output of the last LINEAR unit
    """

    # Weight and Bias initialization
    W1 = tf.get_variable("W1", [784, 200], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1", shape=[200], initializer=tf.constant_initializer(0.0))

    W2 = tf.get_variable("W2", [200, 100], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", shape=[100], initializer=tf.constant_initializer(0.0))

    W3 = tf.get_variable("W3", [100, 60], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3", shape=[60], initializer=tf.constant_initializer(0.0))

    W4 = tf.get_variable("W4", [60, 30], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4", shape=[30], initializer=tf.constant_initializer(0.0))

    W5 = tf.get_variable("W5", [30, 10], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable("b5", shape=[10], initializer=tf.constant_initializer(0.0))

    A1 = tf.matmul(X, W1) + b1
    Z1 = tf.nn.relu(A1)
    D1 = tf.nn.dropout(Z1, keep_prob)
    print("Layer 1: ", Z1.shape)

    A2 = tf.matmul(D1, W2) + b2
    Z2 = tf.nn.relu(A2)
    D2 = tf.nn.dropout(Z2, keep_prob)
    print("Layer 2: ", Z2.shape)

    A3 = tf.matmul(D2, W3) + b3
    Z3 = tf.nn.relu(A3)
    D3 = tf.nn.dropout(Z3, keep_prob)
    print("Layer 3: ", Z3.shape)

    A4 = tf.matmul(D3, W4) + b4
    Z4 = tf.nn.relu(A4)
    D4 = tf.nn.dropout(Z4, keep_prob)
    print("Layer 4: ", Z4.shape)

    A5 = tf.matmul(D4, W5) + b5

    print("Layer 5: ", A5.shape)

    return A5


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


if __name__ == '__main__':

    # Load the data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Pandas dataframe to numpy nd array by taking dataframe values.
    X_train = X_train.values
    test = test.values

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)

    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)


    # Placeholders
    #tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Forward Propagation
    Z = forward_propagation(X, keep_prob)

    # Cost and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)

    # Calculate the correct predictions
    predict_op = tf.argmax(Z, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    minibatch_size = 32
    num_epochs = 100
    m = len(X_train)
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    print_cost = True
    costs = []

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):

                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                    ### END CODE HERE ###

                    minibatch_cost += temp_cost / num_minibatches


                print("\nEpoch: ", epoch)
                print("Cost: ", minibatch_cost)
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                dev_accuracy = accuracy.eval({X: X_val, Y: Y_val})
                print("Train Accuracy: ", train_accuracy)
                print("Dev Accuracy: ", dev_accuracy)

            '''
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            '''






