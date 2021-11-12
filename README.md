# Image-Classification-deep-NN for MNIST and CelebA

This repository is a simple example of the multiple category classification on an RGB or grayscale image data set.

The methods are written from scratch, and built-in python methods are utilized the least.

The first part of the code in nnScript.py uses a neural network with one hidden layer of 4 to 24 nodes and a softmax activation function for multiclass classification of hand-written digits. The dataset for this experiment is the MNIST hand-written digits collection with 70000 photos of size 28 by 28 for each.

The second code in deepnnScript.py contains a binary classification(whether the person in the picture wears glasses or not) on CelebFaces Attributes Dataset with around 200K images which are flattened to vectors to apply the deep neural network. Different hyper-parameters and hidden node numbers are compared in terms of the cost function and accuracy measures.

The third part in facennScript.py compares the user-developed code and the TensorFlow libraries for deep neural networks applied on the Celebi data set.
