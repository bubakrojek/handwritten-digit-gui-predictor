import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def bce(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def ce(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred),axis=0))

def sigmoid(val):
    return 1 / (1 + np.exp(-np.clip(val, -500, 500)))


def sigmoid_derivative(val):
    return val * (1 - val)


def soft_max(values):
    shifted = values - np.max(values, axis=0, keepdims=True)

    exp_values = np.exp(shifted)
    sum_exp = np.sum(exp_values, axis=0, keepdims=True)

    return exp_values / sum_exp


def example_net():
    X_train = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])

    Y_train = np.array([[0, 1, 1, 0]])

    nn = NeuralNetwork(layer_sizes=[2, 4, 4, 1], learning_rate=1)

    print("=== Trening sieci na problemie XOR ===")
    print(f"Dane wejściowe:\n{X_train}")
    print(f"Oczekiwane wyjścia:\n{Y_train}\n")

    output = nn.train(X_train, Y_train, epochs=10000)

    print("\n=== Wyniki po treningu ===")
    print(f"Predykcje: {output.round(3)}")
    print(f"Oczekiwane: {Y_train}")
    print(f"Błąd: {np.abs(output - Y_train).round(3)}")
    return output


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

        for i,activation in zip(range(len(layer_sizes) - 1), activations):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1],activation)
            self.layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y):
        dA = self.layers[-1].output - Y

        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA,Y)

            layer.W -= self.learning_rate * dW
            layer.b -= self.learning_rate * db

    def train(self, X, y, epochs):
        for epoch in range(epochs):

            output = self.forward(X)
            self.backward(y)
            if epoch % 10 == 0:
                preds = np.argmax(output, axis=0)
                true = np.argmax(y, axis=0)
                acc = np.mean(preds == true)
                loss = bce(y, output) if self.layers[-1].b.shape[0]==1 else ce(y,output)
                print(f"Epoch: {epoch+10} - loss: {loss:.4f} - acc: {acc:.4f}")

        return output


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros((output_size, 1))
        self.input = None
        self.output = None
        self.activation_function = sigmoid if activation_function == 'sigmoid' else soft_max

    def forward(self, A_prev):
        self.input = A_prev
        Z = self.W @ self.input + self.b
        self.output = self.activation_function(Z)
        return self.output

    def backward(self, dA,y_true):
        m = self.input.shape[1]
        dZ = (dA * sigmoid_derivative(self.output)) if self.activation_function==sigmoid else self.output-y_true
        dW = (1 / m) * dZ @ self.input.T
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = self.W.T @ dZ

        return dA_prev, dW, db


