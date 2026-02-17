from os.path import join

import numpy as np

from mnist_loader import MnistDataloader
from neural_network import NeuralNetwork

import pickle

def main():
    path = 'data/'
    training_images_path = join(path, 'train-images.idx3-ubyte')
    training_labels_path = join(path, 'train-labels.idx1-ubyte')
    test_images_path = join(path, 't10k-images.idx3-ubyte')
    test_labels_path = join(path, 't10k-labels.idx1-ubyte')

    loader = MnistDataloader(training_images_path, training_labels_path, test_images_path, test_labels_path)
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    #print(x_train.shape[0])
    x_train = x_train / 255
    # y_train=y_train.T
    #x_train=np.array([row[:10000] for row in x_train])
    #y_train=y_train[:10000]

    y_train_hot_encoded=np.zeros((y_train.max()+1,y_train.shape[0]))
    y_train_hot_encoded[y_train,np.arange(y_train.size)]=1
    print(x_train.shape)
    nn=NeuralNetwork([28*28,128,64,10],activations=['sigmoid','sigmoid','softmax'],learning_rate=0.01)
    output = nn.train(x_train, y_train_hot_encoded, epochs=10000)
    print(y_train[-1])
    print(output.size)
    for i,x in enumerate(output):
        print(i,x[0])

    with open("model.pkl", "wb") as f:
        pickle.dump(nn,f)

if __name__ == '__main__':
    main()
