## import packages

import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import load_model

# Create data

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
img_size = 224


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# Prepare data
train = get_data('dataset')

# Splitting dataset
x_train = []
y_train = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

# Normalize the data
x_train = np.array(x_train) / 255                         #dividing by 255 is standardize value for normalization

# Reshape the array
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

datagen = ImageDataGenerator(
    featurewise_center=False,                            # set input mean to 0 over the dataset
    samplewise_center=False,                             # set each sample mean to 0
    featurewise_std_normalization=False,                 # divide inputs by std of the dataset
    samplewise_std_normalization=False,                  # divide each input by its std
    zca_whitening=False,                                 # apply ZCA whitening
    rotation_range=30,                                   # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,                                      # Randomly zoom image
    width_shift_range=0.1,                               # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,                              # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,                                # randomly flip images
    vertical_flip=False)                                 # randomly flip images

datagen.fit(x_train)

# Build model

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile Model
opt = Adam(lr=0.000001)                     #best optimiser available
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=500)

model.save('my_model_final.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model_final.h5')
