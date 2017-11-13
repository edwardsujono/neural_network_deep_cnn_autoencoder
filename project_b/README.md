# CZ4042 Neural Network - Assignment 1

Part B: Approximation Problem

## Getting Started

### Prerequisite

- Python 2.7 or above (with pip)
- Jupyter

### Running

- Using python:
  - To run question 1,2,3: `python questions.py`
- Using jupyter:
  - execute `jupyter notebook` on the current directory.
  - execute code in [Autoencoder](Auto Encoder.ipynb).

## Content

### Question 1
Design a stacked denoising autoencoder consisting of three hidden-layers; 900 neurons in the
first hidden-layer, 625 neurons in the second hidden-layer, and 400 neurons in the third
hidden-layer. To train the network:

- Use the training dataset of MNIST digits
- Corrupt the input data using a binomial distribution at 10% corruption level.
- Use cross-entropy as the cost function

Plot
- learning curves (i.e., reconstruction errors on training data) for training each layer
- Plot 100 samples of weights (as images) learned at each layer
- For 100 representative test images plot
  - reconstructed images by the network.
  - Hidden layer activation
### Question 2

Train a five-layer feedforward neural network to recognize MNIST data, initialized by the three
hidden layers learned in part (1) and by adding a softmax layer as the output layer. Plot the
training errors and test accuracies during training.

### Question 3

Repeat part (1) and (2) by introducing the momentum term for gradient descent learning and
the sparsity constraint to the cost function. Choose momentum parameter 𝛾 = 0.1, penalty
parameter 𝛽 = 0.5, and sparsity parameter 𝜌 = 0.05. Compare the results with those of part
(1) and (2)

