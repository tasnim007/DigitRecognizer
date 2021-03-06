import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import math

def forward_propagation(X):
    """
    Implements the forward propagation for Single layer Neural Network.
    """
    # Weight and Bias initialization
    W = tf.get_variable("W", [784, 10], 
			initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b = tf.get_variable("b", shape=[10], 
			initializer=tf.constant_initializer(0.0))
    A = tf.matmul(X, W) + b
    print(A.shape)
    return A

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    """
    m = X.shape[0]  
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    num_complete_minibatches = math.floor(m / mini_batch_size)  
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: 
		k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: 
		k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
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
    # Encode labels to one hot vectors 
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
				 test_size=0.1, random_state=random_seed)
    # Placeholders
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    # Forward Propagation
    Z = forward_propagation(X)
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
									   Y:minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches
                print("\nEpoch: ", epoch)
                print("Cost: ", minibatch_cost)
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                dev_accuracy = accuracy.eval({X: X_val, Y: Y_val})
                print("Train Accuracy: ", train_accuracy)
                print("Dev Accuracy: ", dev_accuracy)
            predicted_labels = predict_op.eval({X: test})
            np.savetxt('result_slnn.csv',
                       np.c_[range(1, len(test)+1), predicted_labels],
                       delimiter=',',
                       header='ImageId,Label',
                       comments='',
                       fmt='%d')







