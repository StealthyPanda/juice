import pickle
import tensorflow as tf

data = tf.keras.datasets.mnist.load_data()

with open('mnistdatasetpickle', 'wb') as file:
    pickle.dump(data, file)
