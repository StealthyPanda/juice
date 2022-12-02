from juice import *
import numpy as np
import tensorflow as tf

(trainx, trainy), (testx, testy) = tf.keras.datasets.mnist.load_data()

newtrainy = np.zeros(shape = (60000, 10))
for each in range(60000):
    newtrainy[each][trainy[each]] = 1

# print(newtrainy[:2])

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

bruh = jarvis.train(trainx, newtrainy)
print(len(bruh))
for each in bruh:
    print(each.shape)