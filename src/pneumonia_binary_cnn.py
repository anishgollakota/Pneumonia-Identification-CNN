import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers

img_width = 300
img_height = 300

datagen = ImageDataGenerator(rescale=1./255.0, validation_split=0.2)

train_data_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(img_width, img_height),
    class_mode="binary",
    batch_size=20,
    subset="training"
)

validation_data_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(img_width, img_height),
    class_mode="binary",
    batch_size=10,
    subset="validation"
)

#Creating the CNN

#Convolution Layers
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation=tf.nn.relu))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation=tf.nn.relu))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))

#Finish Convolutions, Flatten Output

model.add(Flatten())

#Add Dense Layers

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dense(64, activation=tf.nn.relu))

model.add(Dense(16, activation=tf.nn.relu))

model.add(Dense(1, activation=tf.nn.sigmoid))

#Summarizing the model

model.summary()

#Compiling model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Fitting the model to the training set

history = model.fit_generator(
    generator=train_data_generator,
    steps_per_epoch=len(train_data_generator)//5,
    epochs=20,
    validation_data=validation_data_generator,
    validation_steps=len(validation_data_generator)//5
)

model.save("pnemonia_trained.h5")
