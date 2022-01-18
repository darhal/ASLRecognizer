import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
from utils import *
from keras_visualizer import visualizer 

# print("Keras:", keras.__version__)
# print("Tensorflow:", tf.__version__)

# Pre-processing :
(train_X, train_Y) = get_data_xy('sign_mnist_train.csv')
(test_X, test_Y) = get_data_xy('sign_mnist_test.csv')

# save_img(test_X[131-2], "131P")
# save_img(test_X[376-2], "376O")
# save_img(test_X[361-2], "361U")
# save_img(test_X[331-2], "331T")
# save_img(test_X[303-2], "303N")

modelname = input("Please enter model name (Keep this empty and press enter if you want to train) : ")
if (modelname != ""):
    model = load_model("model")
    # print(model.evaluate(test_X, test_Y))
    # save_img(test_X[131], "131P")
    # save_img(test_X[376], "376O")
    # demo(model, train_X, [5, 38, 50, 66, 0, 1])
    while True:
        imgname = input("Enter image name : ")
        if (imgname != ""):
            img = load_image("demo/"+imgname+".png")
            predict(model, img)
        else:
            break
    exit(0)


NB_OUTPUT = 25

# Model definition :
model = tf.keras.Sequential([
    tf.keras.layers.Convolution2D(filters =32, kernel_size=5, activation='relu',input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='same'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Convolution2D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='same'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

'''data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.06),
    ]
)

model = tf.keras.Sequential([
    # data_augmentation,
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NB_OUTPUT, activation="softmax")
])'''

visualizer(model, format='png', view=True)

# Compile :
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train :
early_stopping_monitor = EarlyStopping(monitor='loss', patience=3)
model.fit(train_X, train_Y, epochs=20, batch_size=1000, verbose=1, callbacks=[early_stopping_monitor], 
        validation_data=(test_X, test_Y))

# Evalution with the test case
print(model.evaluate(test_X, test_Y))
# plt.use('Agg')
modelname = input("Please enter model name to save to the disk (Keep this empty to abort) : ")
if (modelname != ""):
    save_model(model, modelname)

demo(model, train_X, [5, 38, 50, 66, 0, 1])

while True:
    imgname = input("Enter image name : ")
    if (imgname != ""):
        img = load_image("demo/"+imgname+".png")
        predict(model, img)
    else:
        break