# CZ4042 Neural Network - Assignment 1

Part A: Convolutional Neural Network

## Getting Started

### Prerequisite

- Python 2.7 or above (with pip)
- Jupyter

### Running

- Using python `python start.py`:
  - To run question 1: `python q1.py`
  - To run question 2: `python q2.py`
  - To run question 3: `python q3.py`
- Using jupyter:
  - execute `jupyter notebook` on the current directory.
  - execute code in [Autoencoder](Auto Encoder.ipynb).

## Content

### Part 1

To recognize MNIST digits, design a convolutional neural network consisting of

- An Input layer of 28x28 dimensions
- A convolution layer 𝐶1 of 15 feature maps and filters of window size 9x9. A max pooling layer 𝑆1 with a pooling window of size 2x2.
- A convolution layer 𝐶2 of 20 feature maps and filters of window size 5x5. A max pooling layer 𝑆2 with a pooling window of size 2x2.
- A fully connected layer 𝐹3 of size 100.
- A softmax layer 𝐹4 of size 10.

Train the network using ReLu activation functions for neurons and mini batch gradient descent learning. Set batch size 128, learning rate 𝛼 = 0.05 and decay parameter 𝛽 = 10−4

- Plot the training cost and test accuracy with learning epochs.
- For two representative test patterns, plot the feature maps at the convolution and pooling layers.

### Part 2

Repeat part 1 by adding the momentum term to mini batch gradient descent learning with momentum parameter 𝛾 = 0.1.

### Part 3

Repeat part 1 by using RMSProp algorithm for learning. Use 𝛼 = 0.001, 𝛽 = 1𝑒−4, 𝜌 = 0.9, and 𝜖 = 10−6 for RMSProp.
