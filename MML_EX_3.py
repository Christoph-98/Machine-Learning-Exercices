# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 02:55:59 2024

@author: chris
"""

# Exercise: Handwritten Digit Classification with a Neural Network (MNIST)

# Objective:
# Build, train, and evaluate a feedforward neural network for handwritten digit
# classification using the MNIST dataset.

# Tasks:
# 1. Load the MNIST dataset.
# 2. Preprocess the data by:
#    - Normalizing the pixel values to the range [0, 1].
#    - Converting the class labels to one-hot encoded vectors.
#    - Reshaping each 28×28 image into a vector of length 784.
# 3. Construct a neural network consisting of:
#    - An input layer for 784 features.
#    - One hidden Dense layer with 128 neurons and ReLU activation.
#    - An output Dense layer with 10 neurons and Softmax activation.
# 4. Train the model.
# 5. Use the trained model to predict the labels of the test images.

from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the first 5 training images
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i in range(10):
    sample = x_train[y_train == i][0]
    axes[i].imshow(sample)
    axes[i].set_title('Lable: {}'.format(i))
    
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)

model = Sequential()

model.add(Flatten(input_shape=(28* 28,)))
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
#model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x=x_train, y=y_train, batch_size=512, epochs=10)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)


fig, axes = plt.subplots(1, 10, figsize=(15, 5))


for i in range(10):
    x_sample = x_test[i]
    axes[i].imshow(x_sample.reshape(28,28))
    y_true = np.argmax(y_test, axis=1)
    axes[i].set_title('Lable: {}'.format(y_true[i]))

for ax in axes.flatten():
    ax.axis('off') 
    
plt.show()