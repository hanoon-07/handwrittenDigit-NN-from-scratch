# Simple Feedforward Neural Network from Scratch to detect Handwritten digits

This project implements a basic feedforward neural network in Python using NumPy, trained using stochastic gradient descent (SGD). The network is trained on the MNIST dataset of handwritten digits.

---

##  Phase 1

#### 1. **Feedforward Neural Network**
A feedforward network architecture with fully connected layers was built using:
- Randomly initialized weights and biases
- The sigmoid activation function

#### 2. **Stochastic Gradient Descent (SGD)**
The training algorithm used is stochastic gradient descent, where:
- The training data is shuffled every epoch
- Data is split into **mini-batches**
- Gradients are computed via **backpropagation**
- Weights and biases are updated after each mini-batch

#### 3. **Mini-Batch Gradient Descent**
Instead of updating weights on each training example, we use mini-batches of data (e.g., 10 samples per batch). This stabilizes training and accelerates convergence.

---

## 📊 Training and Results

Trained the model on 50,000 MNIST training images and evaluated it on 10,000 test images. The architecture used was:

### ✅ Network Configuration
- **Input layer**: 784 neurons (28x28 pixels flattened)
- **Hidden layer**: 30 neurons
- **Output layer**: 10 neurons (one for each digit 0–9)

### 🔧 Training Parameters
#### The following hyperparameters were **carefully** chosen to maximize accuracy on the MNIST dataset:
- **Epochs**: 30
- **Mini-batch size**: 10
- **Learning rate (eta)**: 3.0

### 📈 Sample Performance
The following are the evaluation results from the test set after selected epochs:
Epoch 25: 9498 / 10000
Epoch 26: 9517 / 10000
Epoch 27: 9518 / 10000

Peak performance was reached around epoch 27, with the network correctly classifying **9518 out of 10,000** test images — i.e., a **95.18% accuracy**.

---

## 🧩 File Overview

- `network.py`: Implements the neural network class, including forward pass, backpropagation, SGD, and evaluation methods.
- `mnist_loader.py`: Loads and preprocesses the MNIST dataset (requires `mnist.pkl.gz`).
- `data/mnist.pkl.gz`: Compressed and pickled dataset file (must be placed correctly).

---

## 💡 Requirements

- Python 3.x
- NumPy
- `mnist.pkl.gz` dataset (placed under `./data/` directory)

---

## 📦 How to Run
- Python main.py

