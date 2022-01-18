import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
from PIL import Image
from skimage import color, io
import cv2
from PIL import Image

"""train_X = train_df.iloc[:,1:]
train_Y = train_df.iloc[:,0]
train_Y = tf.keras.utils.to_categorical(train_Y)
train_X = train_X.values.reshape(-1,28,28,1)"""

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def get_data_xy(filename):
    df = pd.read_csv(filename)
    x = df.drop(columns=['label'])
    y = df[['label']]
    nb_col = x.shape[0]
    img_flat_arr = np.array([v for i in range(nb_col) for _,v in enumerate(x.iloc[i])])
    x = img_flat_arr.reshape(nb_col, 28, 28, 1) / 255.0
    y = y.transpose().to_numpy()[0]
    y = tf.keras.utils.to_categorical(y)
    return (x,y)

def save_model(model, filename):
    tf.keras.models.save_model(model, filename)

def load_model(filename):
    loaded_model = tf.keras.models.load_model(filename, compile = True)
    return loaded_model

def predict(model, img_mat):
    prediction = model.predict(img_mat)
    n = np.argmax(prediction)
    print(f"Prediction : {ALPHABET[n]} Letter number : {n}")
    # display_labeled_image(img_mat, ALPHABET[predicted])

def display_labeled_image(img, str):
    plt.imshow(img)
    plt.title(str)
    plt.savefig(str+".png")

def load_image(path):
    img = io.imread(path)
    re = cv2.resize(img, (28, 28))
    arr = np.array(re) / 255.0
    img = arr.reshape((-1, 28, 28, 1))
    return img

def demo(model, train_X, use_samples):
    # A few random samples
    samples_to_predict = []
    
    # Generate plots for samples
    for sample in use_samples:
        # Generate a plot
        reshaped_image = train_X[sample].reshape((28, 28))
        plt.imshow(reshaped_image, cmap='gray')
        # plt.show()
        plt.savefig(f"{sample}.png")
        # Add sample to array for prediction
        samples_to_predict.append(train_X[sample])

    # Convert into Numpy array
    samples_to_predict = np.array(samples_to_predict)
    # print(samples_to_predict.shape)

    # Generate predictions for samples
    predictions = model.predict(samples_to_predict)
    # print(predictions)

    # Generate arg maxes for predictions
    classes = np.argmax(predictions, axis = 1)
    print([ALPHABET[i] for i in classes])

def save_img(arr, lettre):
    nArray = arr.reshape((28, 28)) * 255.0
    cv2.imwrite(lettre+'.png', nArray)
