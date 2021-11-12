# Image-Classification-deep-NN

This repository is a very simple exampple of the multiple category classification on a RGB or gray scale image data sets.

The methods are written from scratch and built-in python methods are utilized the least.

The First part of the code in nnScript.py uses a neural network with one hidden layer of 4 to 24 nodes and a softmax activation function for multiclass classification of hand-written digits. The dataset for this experiment is the MNIST hand-written digits collection with 70000 photos with a size of 28 by 28 for each.

The second code in in deepnnScript.py contains a binary classification(whether the person in the picture wears glasses or not) on CelebFaces Attributes Dataset with around 200K images which are flattened to vectors to apply the deep neural network. Different hyper-parameters and hidden node numbers are compared in terms of the cost function and accuracy measures.

The third part in facennScript.py is a comparison between the user-developed code and the tensorflow libraries for deep neural networks applied on the CelebA data set.
