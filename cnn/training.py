import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from detection_modules import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dropout
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Run this program to train detection model

path = r'D:\code\GroomingDetection\cnn\TrainingData'

training, labels = load_data(path)  #Load Training set and Label set

# Random disorder of data sets
# Through get_ State() save state, set_ State() reloads the state,
# which enables training and labels to complete random scrambling while keeping the corresponding relationship unchanged.
state = np.random.get_state()
np.random.shuffle(training)
np.random.set_state(state)
np.random.shuffle(labels)

# Convert to one-hot matrix and set the value according to the number of label types
labels_hot = to_categorical(labels, 7)

# training = training.reshape(training.shape[0], 100, 100 ,3)
training = training.astype('float32')
training /= 255

print('Loading finished!')

model = Sequential()

# vgg16
# model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(256, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(256, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(256, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))

# Recommend network
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu' ,input_shape=(100, 100, 3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Modified on vgg16
# model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(256, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(256, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(Conv2D(512, kernel_size=(3,3),padding='same', activation='relu'))
# model.add(MaxPool2D((2,2), strides=(2,2)))
#
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.15))
# model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training, labels_hot, batch_size=256, epochs=10)

model.save(r'D:\code\GroomingDetection\cnn\TrainingData\model')