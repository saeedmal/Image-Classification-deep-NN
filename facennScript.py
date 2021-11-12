'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import time
import pickle
from math import *
from scipy.optimize import minimize

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # add offset to all training data
    a_1 = np.concatenate((training_data, np.ones((training_data.shape[0], 1))), axis=1).T  # 785*50000

    z_2 = np.dot(w1, a_1)  # 50*50000
    a_2 = sigmoid(z_2)
    a_2 = np.concatenate((a_2, np.ones((1, a_2.shape[1]))), axis=0)  # 51*50000

    z_3 = np.dot(w2, a_2)
    a_3 = sigmoid(z_3)  # 10*50000

    N = a_3.shape[1]  # No. of training examples
    y = np.zeros((n_class, N))
    out = a_3  # 10*50000 output

    for i in range(0, N):
        y[np.int(training_label[i]), i] += 1  # construct y for all examples

    J_i = np.sum(y * np.log(out) + (1 - y) * np.log(1 - out), axis=0)  # (50000,)
    J = 1 / N * (-np.sum(J_i) + 0.5 * lambdaval * (sum(sum(w1 * w1)) + sum(sum(w2 * w2))))  # single scalar

    delta_3 = a_3 - y  # a_3-y for all training examples 10*50000
    delJ_w2 = 1 / N * (np.dot(delta_3, a_2.T) + lambdaval * w2)  # del(J)/del(w_2)=(a_2 * delta_3+lambda*w2) /N

    w2_modified = np.delete(w2, -1, 1)
    delta_2 = np.dot(w2_modified.T, delta_3) * sigmoid(z_2) * (1 - sigmoid(z_2))  # 51*50000
    delJ_w1 = 1 / N * (np.dot(delta_2, a_1.T) + lambdaval * w1)  # del(J)/del(w_1)=(a_1 * delta_2+lambda*w1) /N

    obj_grad = np.concatenate((delJ_w1.flatten(), delJ_w2.flatten()), 0)
    return (J, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
        % Network.

        % Input:
        % w1: matrix of weights of connections from input layer to hidden layers.
        %     w1(i, j) represents the weight of connection from unit j in input
        %     layer to unit j in hidden layer.
        % w2: matrix of weights of connections from hidden layer to output layers.
        %     w2(i, j) represents the weight of connection from unit j in input
        %     layer to unit j in hidden layer.
        % data: matrix of data. Each row of this matrix represents the feature
        %       vector of a particular image

        % Output:
        % label: a column vector of predicted labels"""
    # add offset to all training data
    a_1 = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1).T  # 785*50000

    z_2 = np.dot(w1, a_1)  # 50*50000
    a_2 = sigmoid(z_2)
    a_2 = np.concatenate((a_2, np.ones((1, a_2.shape[1]))), axis=0)  # 51*50000

    z_3 = np.dot(w2, a_2)
    a_3 = sigmoid(z_3)  # 10*50000

    labels = np.argmax(a_3, 0)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

# lambda_range=range(0,65,5)

# for lambdaval in lambda_range: # set the regularization hyper-parameter
lambdaval=0.0
lambda_hidden_save=np.zeros((1,5))
w_save={}

for lambdaval in range(0,65,5):

    start_time = time.time() #training starts here	
    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    # set the regularization hyper-parameter
    print('n_hidden: ', n_hidden, ',lambdaval: ', lambdaval)
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    training_time=time.time()-start_time #training ends here

    # # Test the computed parameters
    # predicted_label1 = nnPredict(w1, w2, train_data)
    # error1 = predicted_label1 - np.double(train_label)
    # accuracy_train = 100 * np.mean(error1 == 0)
    # print(accuracy_train)

    # Test the computed parameters for cross-validation data
    predicted_label2 = nnPredict(w1, w2, validation_data)
    error2 = predicted_label2 - np.double(validation_label)
    accuracy_valid = 100 * np.mean(error2 == 0)
    print(' accuracy validation: ',accuracy_valid)

    # Test the computed parameters for test data
    predicted_label3 = nnPredict(w1, w2, test_data)
    error3 = predicted_label3 - np.double(test_label)
    accuracy_test = 100 * np.mean(error3 == 0)
    print(' accuracy test: ',accuracy_test)

    # ww = np.array([[n_hidden], [lambdaval], [100 * np.mean(error2 == 0)], [100 * np.mean(error3 == 0)]])
    ww = np.array([ [n_hidden], [lambdaval], [accuracy_valid], [accuracy_test], [training_time] ])
    lambda_hidden_save = np.concatenate((lambda_hidden_save, ww.T), axis=0)

    dict2={str(n_hidden)+str(lambdaval):[w1,w2]}
    w_save.update(dict2)


# Finding the optimal hyper-parameters
lambda_hidden_save= np.delete(lambda_hidden_save, 0, 0)
# print(lambda_hidden_save)
ind_optimal = np.argmax(lambda_hidden_save[:,2])
hidden_opt = lambda_hidden_save[ind_optimal,0]
lambda_opt = lambda_hidden_save[ind_optimal,1] #optimal value of regularization parameter
test_accu = lambda_hidden_save[ind_optimal,3] #accuracy of test

print('lambda optimal:',lambda_opt)
print('Test accuracy:', test_accu)

