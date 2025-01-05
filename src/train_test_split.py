import numpy as np
import tensorflow as tf
#from tensorflow import keras 
from keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
# Load MNIST dataset

def train_test_Minst_Dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the data in the form of 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0  
    #  Reshapes the training and testing data for compatibility with convolutional layers in a neural network:
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    # Define the 7-segment representation
    # 1: for the segment is ON.
    # 0: for the segment is OFF.
    digit_to_segments = {
    0: [1, 1, 1, 1, 1, 1, 0],
    1: [0, 1, 1, 0, 0, 0, 0],
    2: [1, 1, 0, 1, 1, 0, 1],
    3: [1, 1, 1, 1, 0, 0, 1],
    4: [0, 1, 1, 0, 0, 1, 1],
    5: [1, 0, 1, 1, 0, 1, 1],
    6: [1, 0, 1, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 0, 0],
    8: [1, 1, 1, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1],
    }
    # converting Y values into their corresponding 7-segment LED display representations.
    y_train_segments = np.array([digit_to_segments[d] for d in y_train])
    print("y_train_segments :")
    print(y_train_segments)
    y_test_segments = np.array([digit_to_segments[d] for d in y_test])
    print()
    print("y_test_segments :")
    print(y_train_segments)