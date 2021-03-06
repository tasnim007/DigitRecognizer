import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import math

def forward_propagation(X, mode):
    """
    Implements the forward propagation for the model:
    """
    # Inner function for batch normalization
    def batch_norm_cnv(inputs):
        return tf.layers.batch_normalization(inputs, axis=3, momentum=0.99,
        epsilon=1e-5, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def batch_norm(inputs):
        return tf.layers.batch_normalization(inputs, axis=1, momentum=0.99,
        epsilon=1e-5, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    W1 = tf.get_variable("W1", [6, 6, 1, 32],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1", shape=[32],
                         initializer=tf.constant_initializer(0.0))
    W2 = tf.get_variable("W2", [5, 5, 32, 32],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", shape=[32],
                         initializer=tf.constant_initializer(0.0))
    W3 = tf.get_variable("W3", [4, 4, 32, 64],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3", shape=[64],
                         initializer=tf.constant_initializer(0.0))
    W4 = tf.get_variable("W4", [3, 3, 64, 64],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4", shape=[64],
                         initializer=tf.constant_initializer(0.0))
    # CONV2D: stride of 1, padding 'SAME'
    A1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    # Batch Normalization
    BN1 = batch_norm_cnv(A1)
    # RELU
    Z1 = tf.nn.relu(BN1)
    # CONV2D: stride of 1, padding 'SAME'
    A2 = tf.nn.conv2d(Z1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
    # Batch Normalization
    BN2 = batch_norm_cnv(A2)
    # RELU
    Z2 = tf.nn.relu(BN2)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P1 = tf.nn.max_pool(Z2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    # CONV2D: stride of 1, padding 'SAME'
    A3 = tf.nn.conv2d(P1, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
    # Batch Normalization
    BN3 = batch_norm_cnv(A3)
    # RELU
    Z3 = tf.nn.relu(BN3) 
    # CONV2D: stride of 1, padding 'SAME'
    A4 = tf.nn.conv2d(Z3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
    # Batch Normalization
    BN4 = batch_norm_cnv(A4)
    # RELU
    Z4 = tf.nn.relu(BN4)
    # MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(Z4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    # FLATTEN
    P = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED with relu activation function.
    # 200 neurons in output layer.
    Z5 = tf.contrib.layers.fully_connected(P, 200)
    # Batch Normalization
    BN5 = batch_norm(Z5)
    # FULLY-CONNECTED without non-linear activation function.
    # 10 neurons in output layer.  
    Z6 = tf.contrib.layers.fully_connected(BN5, 10, activation_fn=None)
    print(X.shape)
    print(Z1.shape)
    print(Z2.shape)
    print(P1.shape)
    print(Z3.shape)
    print(Z4.shape)
    print(P2.shape)
    print(P.shape)
    print(Z5.shape)
    print(Z6.shape)
    return Z6

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    """
    m = X.shape[0]   
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:
                                  k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:
                                  k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
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
    # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
    # Encode labels to one hot vectors
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 10)
    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size = 0.1, random_state=random_seed)
    # Placeholders
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    mode = tf.placeholder(tf.bool, name="mode")
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
    tf.set_random_seed(1)                            
    seed = 3                                     
    print_cost = True
    costs = []
    with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) 
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X,
                                                                           Y:minibatch_Y,
                                                                           mode: True})
                    minibatch_cost += temp_cost / num_minibatches
                print("\nEpoch: ", epoch)
                print("Cost: ", minibatch_cost)
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, mode: False})
                dev_accuracy = accuracy.eval({X: X_val, Y: Y_val, mode: False})
                print("Train Accuracy: ", train_accuracy)
                print("Dev Accuracy: ", dev_accuracy)
            predicted_labels = predict_op.eval({X: test, mode: False})
            np.savetxt('result_CNN_bn.csv',
                       np.c_[range(1, len(test)+1), predicted_labels],
                       delimiter=',',
                       header='ImageId,Label',
                       comments='',
                       fmt='%d')	