'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import tensorflow as tf
import numpy as np
import pickle
import time


# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron():
    # Network Parameters
    # n_hidden_1 = 256  # 1st layer number of features
    n_hidden_1 = 64  # 1st layer number of features

    # n_hidden_2 = 256  # 2nd layer number of features
    n_hidden_2 = 64  # 2nd layer number of features

    n_input = 2376  # data input
    n_classes = 2

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer,x,y

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# Parameters

training_epochs = 100
batch_size = 100

# Construct model
pred,x,y = create_multilayer_perceptron()

# load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = preprocess()

# for lambdaval in lambda_range: # set the regularization hyper-parameter
learning_rate = 0.0001
lambda_hidden_save=np.zeros((1,4))

learning_range=np.array([0,0.0001,0.001,0.01,0.1,0.2,0.3,0.5])

for learning_rate in learning_range:

    start_time = time.time()
    # Define loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    #Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_features.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_features[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

        training_time= time.time() - start_time
        print("Optimization Finished! training time: ",training_time,'learning_rate: ',learning_rate)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracy_test = accuracy.eval({x: test_features, y: test_labels})
        print("Accuracy test:", accuracy_test)

        accuracy_valid=accuracy.eval({x: valid_features, y: valid_labels})
        print("Accuracy validation:", accuracy_valid)


        # ww = np.array([[n_hidden], [lambdaval], [100 * np.mean(error2 == 0)], [100 * np.mean(error3 == 0)]])
        ww = np.array([[learning_rate], [accuracy_valid], [accuracy_test], [training_time]])
        lambda_hidden_save = np.concatenate((lambda_hidden_save, ww.T), axis=0)

lambda_hidden_save= np.delete(lambda_hidden_save, 0, 0)

ind_optimal = np.argmax(lambda_hidden_save[:,1])
learning_rate_optimal=lambda_hidden_save[ind_optimal,0] #optimal value of regularization parameter


parameters={'save':[learning_rate_optimal,lambda_hidden_save]}

open_file = open('deepNN.pickle','wb')
pickle.dump(parameters,open_file)
open_file.close()

load_file = open('deepNN.pickle','rb')
parameter1= pickle.load(load_file)

print('optimal learning rate in the considered range: ',parameter1['save'][0])
print(parameter1['save'][1])
