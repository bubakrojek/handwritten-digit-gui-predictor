# Drawn Digit Predictor

Handwritten digit recognition project built from scratch in NumPy.  
The model is trained on the MNIST dataset and integrated with a PyQt GUI that lets you draw digits on a canvas and immediately see the predicted digit.

---

## Table of Contents

- [Description](#description)
- [Demo](#demo)
- [Features](#features)
- [Technologies](#technologies)
- [Usage](#usage)
- [Model and Theory](#model-and-theory)
- [GUI – How to Use](#gui--how-to-use)
- [License](#license)

---

## Description

The goal of this project is to build a digit classifier (0–9) from scratch, without using deep learning frameworks such as PyTorch or TensorFlow. The model:

- is trained on the classic MNIST dataset,
- can also classify digits drawn by the user in a GUI window,
- demonstrates how to implement forward pass, backpropagation, softmax, and cross-entropy in pure NumPy. [file:1]

---

## Demo

Digit prediction based on user drawing.

![img](docs/prediction.png)

Top 3 digit probabilities.

![probs](docs/probs.png)

Correcting network

![correcting](docs/correcting_network.png)

---

## Features

- various network sizes, activations such as sigmoid or softmax and loss functions,
- training network on MNIST Dataset,
- input data normalization and one hot encoding,
- PyQt GUI with a drawing canvas and a button that triggers model prediction,

---

## Technologies
- Python 3.13
- NumPy
- PyQt6

---

## Usage

- Training own network:
```
nn = neural_network.NeuralNetwork([28 * 28, 256, 128,64, 10], activations=['relu', 'relu','relu',  'softmax'], learning_rate=0.01)
nn.train(x_train, y_train_hot_encoded, epochs=3000, iterations=10)
```
- Saving network:
```
nn.save_network(f'model_relu_3k_epoch_bigger_hidden_3_hidden')
```
- Reading network:
```
path = 'models/model_relu_3k_epoch_bigger_hidden_3_hidden.pkl'
nn = neural_network.load_network(path)
```

Mam narazie tyle, ma to sens?
