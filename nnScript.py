import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time as time
import matplotlib.pyplot as plt
import pickle
# % matplotlib inline


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess():

    global selected_features
    mat = loadmat('mnist_all.mat')

    train_data = np.zeros(
        (1, mat.get('train0').shape[1]))  # consider an initial value for train_data. this row will be removed later on.
    train_label = np.array([15])

    validation_data = np.zeros((1, mat.get('train0').shape[
        1]))  # consider an initial value for validation_data. this row will be removed later on.
    validation_label = np.array([15])

    test_data = np.zeros((1, mat.get('test0').shape[1]))
    test_label = np.array([15])

    for key in mat:

        if "train" in key:

            value_row = np.random.permutation(mat.get(key).shape[0])
            train_data_index = value_row[1000:value_row.shape[0]]
            validation_data_index = value_row[0:1000]
            train_data = np.concatenate((train_data, mat.get(key)[train_data_index]), axis=0)
            train_label = np.concatenate((train_label, np.repeat(key[-1], train_data_index.shape[0])), axis=0)
            validation_data = np.concatenate((validation_data, mat.get(key)[validation_data_index]), axis=0)
            validation_label = np.concatenate((validation_label, np.repeat(key[-1], validation_data_index.shape[0])),
                                              axis=0)

        elif "test" in key:

            value_row_test = np.random.permutation(mat.get(key).shape[0])
            test_data = np.concatenate((test_data, mat.get(key)[value_row_test]), axis=0)
            test_label = np.concatenate((test_label, np.repeat(key[-1], value_row_test.shape[0])), axis=0)

    train_data = np.delete(train_data, 0, 0)  # eliminate the first row of zeros
    train_label = np.delete(train_label, 0, 0)  # eliminate the first label
    validation_data = np.delete(validation_data, 0, 0)
    validation_label = np.delete(validation_label, 0, 0)
    test_data = np.delete(test_data, 0, 0)
    test_label = np.delete(test_label, 0, 0)

    perm_train = np.random.permutation(train_data.shape[0])
    train_data = np.double(train_data[perm_train]) / 255.0
    train_label = train_label[perm_train]

    perm_valid = np.random.permutation(validation_data.shape[0])
    validation_data = np.double(validation_data[perm_valid]) / 255.0
    validation_label = validation_label[perm_valid]

    perm_test = np.random.permutation(test_data.shape[0])
    test_data = np.double(test_data[perm_test]) / 255.0
    test_label = test_label[perm_test]

    # Feature selection: Data set has lots of zeroes that should be decrease
    STD = np.std(train_data, 0)  # Criterion for selecting features
    criteria = np.where(STD > 0.001)
    selected_features=criteria[0]
    train_data = train_data[:, criteria[0]]
    validation_data = validation_data[:, criteria[0]]
    test_data = test_data[:, criteria[0]]

    print('----preprocess-----')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    # obj_val = 0

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

    h = np.dot(w2.T, delta_3)  # 51*50000
    delta_2 = h * a_2 * (1 - a_2)  # 51*50000
    h2 = np.delete(delta_2, -1, 0)  # 50*50000

    delJ_w1 = 1 / N * (np.dot(h2, a_1.T) + lambdaval * w1)  # del(J)/del(w_1)=(a_1 * delta_2+lambda*w1) /N

    # w2_modified = np.delete(w2, -1, 1)
    # delta_2 = np.dot(w2_modified.T, delta_3) * sigmoid(z_2) * (1 - sigmoid(z_2))  # 51*50000
    # delJ_w1 = 1 / N * (np.dot(delta_2, a_1.T) + lambdaval * w1)  # del(J)/del(w_1)=(a_1 * delta_2+lambda*w1) /N

    obj_grad = np.concatenate((delJ_w1.flatten(), delJ_w2.flatten()), 0)
    return (J, obj_grad)


def nnPredict(w1, w2, data):
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

#################### Train Neural Network############################################################################

global selected_features

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# print(selected_features)
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit) examples 4,8,12,16,20
n_hidden = 4

# set the number of nodes in output unit
n_class = 10

lambdaval = 0.0
opts = {'maxiter': 50}  # Preferred value.
# accuracy = np.zeros((3, 13))

lambda_hidden_save = np.zeros((1, 5))
w_save = {}
n_hidden_range=range(4, 24, 4)
lambda_range=range(0,65,5)

for n_hidden in n_hidden_range:

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    for lambdaval in lambda_range:

        start_time = time.time() #training starts here
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
accu = lambda_hidden_save[ind_optimal,3] #accuracy of test
w1_opt=w_save[str(int(hidden_opt))+str(int(lambda_opt))][0]
w2_opt=w_save[str(int(hidden_opt))+str(int(lambda_opt))][1]

#saving the results of the optimization
open_file1=open('save.pickle','wb')
save={'w':w_save,'n_l':lambda_hidden_save}
pickle.dump(save,open_file1)
open_file1.close()

# Saving optimals as pickle file
optimal_set = {'selected features':selected_features, 'optimal_hidden_unit':hidden_opt, 'optimal_lambda':lambda_opt, 'w1':w1_opt, 'w2':w2_opt}

open_file = open('params.pickle','wb')
pickle.dump(optimal_set,open_file)
open_file.close()

load_file = open('params.pickle','rb')
opt_set = pickle.load(load_file)
print(opt_set)

ind_optimal = np.argmax(lambda_hidden_save[:,2])
hidden_opt = lambda_hidden_save[ind_optimal,0]
lambda_opt = lambda_hidden_save[ind_optimal,1] #optimal value of regularization parameter
accu = lambda_hidden_save[ind_optimal,3] #accuracy of test
w1_opt=w_save[str(int(hidden_opt))+str(int(lambda_opt))][0]
w2_opt=w_save[str(int(hidden_opt))+str(int(lambda_opt))][1]
print(lambda_hidden_save[ind_optimal,2])
print('Optimal hidden unit:',hidden_opt)
print('Optimal lambda:',lambda_opt)
print('Test set accuracy:',accu)

# figures showing the performance of normal neural network for different hyper-parameters

#effect of lambdaval
plt.figure(1,figsize=(12,6),dpi=1200) #n_hidden=4
plt.plot(lambda_range, lambda_hidden_save[0:len(lambda_range), 4])
plt.xlabel("lambda, regularization parameter")
plt.ylabel("accuracy of validation data")
plt.savefig('effect_lambda_4.jpg', format='jpg')
plt.show()

plt.figure(2,figsize=(12,6),dpi=1200) #varying n_hidden in 4,8,12,16,20
plt.plot(lambda_range, lambda_hidden_save[0:len(lambda_range), 2])
plt.plot(lambda_range, lambda_hidden_save[len(lambda_range):2*len(lambda_range), 2])
plt.plot(lambda_range, lambda_hidden_save[2*len(lambda_range):3*len(lambda_range), 2])
plt.plot(lambda_range, lambda_hidden_save[3*len(lambda_range):4*len(lambda_range), 2])
plt.plot(lambda_range, lambda_hidden_save[4*len(lambda_range):5*len(lambda_range), 2])
plt.legend(['hidden units=4','=8','=12','=16','=20'],ncol=4,loc=4)
plt.xlabel("lambda, regularization parameter")
plt.ylabel("accuracy of validation data")
plt.savefig('effect_lambda.jpg', format='jpg')
plt.show()

#effect of hidden units on training time
plt.figure(3,figsize=(12,6),dpi=1200)
plt.plot(n_hidden_range,lambda_hidden_save[np.array([0,13,26,39,52]), 4])
plt.xlabel("Number of hidden units")
plt.ylabel("training time")
plt.savefig('effect_hidden_unit.jpg', format='jpg')
plt.show()
