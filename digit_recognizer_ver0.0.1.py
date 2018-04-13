import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns
get_ipython().magic('matplotlib inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
import math



def forward_propagation(X):
    """
    Implements the forward propagation for the model:
    (CONV2D -> RELU)*2 -> MAXPOOL -> Dropout -> (CONV2D -> RELU)*2 -> MAXPOOL -> Dropout -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Weights initialization 
    W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    
    # CONV2D: stride of 1, padding 'SAME'
    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    
    # Dropout layer
    D1 = tf.nn.dropout(P1, keep_prob=0.75)
    
    
    # CONV2D: stride of 1, padding 'SAME'
    Z3 = tf.nn.conv2d(D1, W3, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    
    # CONV2D: stride of 1, padding 'SAME'
    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P2 = tf.nn.max_pool(A4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    
    # Dropout layer
    D2 = tf.nn.dropout(P2, keep_prob=0.75)
    
    
    # FLATTEN
    P = tf.contrib.layers.flatten(D2)
    
    # Dropout layer
    D3 = tf.nn.dropout(P, keep_prob=0.5)
    
    # FULLY-CONNECTED without non-linear activation function.
    # 10 neurons in output layer.  
    Z5 = tf.contrib.layers.fully_connected(D3, 10, activation_fn=None)
   


    print(X.shape)
    print(Z1.shape)
    print(Z2.shape)
    print(P1.shape)
    
    print(Z3.shape)
    print(Z4.shape)
    print(P2.shape)
    print(P.shape)
    print(Z5.shape)

    return Z5


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
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
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


if __name__ == '__main__':

    # Load the data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1)

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 10)

    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


    # Placeholders
    #tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    # Forward Propagation
    Z5 = forward_propagation(X)

    # Cost and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)

    # Calculate the correct predictions
    predict_op = tf.argmax(Z5, 1)
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


                Print("\nEpoch: ", epoch)
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






