import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QApplication

import neural_network
from app import MainWindow
from mnist_loader import MnistDataloader

def one_hot_encode(data):
    data_hot_encoded = np.zeros((data.max() + 1, data.shape[0]))
    data_hot_encoded[data, np.arange(data.size)] = 1
    return data_hot_encoded

def get_data():
    path = 'data/'
    training_images_path = join(path, 'train-images.idx3-ubyte')
    training_labels_path = join(path, 'train-labels.idx1-ubyte')
    test_images_path = join(path, 't10k-images.idx3-ubyte')
    test_labels_path = join(path, 't10k-labels.idx1-ubyte')

    loader = MnistDataloader(training_images_path, training_labels_path, test_images_path, test_labels_path)
    return loader.load_data()

def main():
    path='model_relu_3k_epoch_bigger_hidden_3_hidden.pkl'
    nn = neural_network.load_network(path)

    app = QApplication(sys.argv)

    window = MainWindow(nn)

    window.resize(800, 600)
    window.show()

    nn.save_network(path)

    '''
    (x_train, y_train), (x_test, y_test) = get_data()
    print(x_train.shape)
    x_train = x_train / 255
    x_test = x_test / 25

    y_train_hot_encoded = one_hot_encode(y_train)
    y_test_hot_encoded = one_hot_encode(y_test)

    predictions=nn.predict_probs(x_test)
    print(neural_network.accuracy(y_test_hot_encoded,predictions))

    
    nn= neural_network.NeuralNetwork([28 * 28, 256, 128,64, 10], activations=['relu', 'relu','relu',  'softmax'], learning_rate=0.01)
    nn.train(x_train, y_train_hot_encoded, epochs=3000, iterations=10)
    nn.save_network(f'model_relu_3k_epoch_bigger_hidden_3_hidden')'''


    # for i,x in enumerate(output):
    # print(i,x[0])

    #nn = neural_network.load_network('model_relu_2k_epoch.pkl')
    #test_predictions=nn.predict_probs(x_test)
    #print(neural_network.accuracy(y_test_hot_encoded,test_predictions))

    #test_predictions = nn.predict(x_test)
    #print(neural_network.accuracy(y_test, test_predictions))

    #plt.imshow(x_test[:,1].reshape(28,28))
    #plt.show()
    #print(nn.predict(x_test[:,1]))



    sys.exit(app.exec())


if __name__ == '__main__':
    main()
