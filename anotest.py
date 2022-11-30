from juice import *
import numpy as np
import tensorflow as tf

(trainx, trainy), (testx, testy) = tf.keras.datasets.mnist.load_data()

jarvis = NeuralNetwork(
    layers = [
        Flatten(inputsize = (28, 28)),
        Layer(cells = 16, activation = actrelu),
        Layer(cells = 16, activation = actrelu),
        Layer(cells = 10, activation = actsigmoid)
    ],
    name = 'Jarvis'
)

print(jarvis.compile())

bruh = jarvis.train(trainx, trainy)
print(len(bruh))