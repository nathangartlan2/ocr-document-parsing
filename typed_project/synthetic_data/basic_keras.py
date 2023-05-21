import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sklearn
import bidict as bd
import numpy as np
import cv2


class basic_keras():

    def compile(self):
        inputs = keras.Input(shape=(self.width, self.height, 1))
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss="binary_crossentropy",
                      optimizer="rmsprop",
                      metrics=["accuracy"])

        return model

    def numeric_labels(self):
        dict = {}
        numeric_labels = []
        for i in range(0, len(self.labels)):
            dict[self.labels[i]] = i
            numeric_labels.append(i)

        self.translation = bd.bidict(dict)

        return numeric_labels

    def __init__(self,
                 labels,
                 images,
                 width,
                 height,
                 demo: bool) -> None:
        self.labels = labels
        self.images = images
        self.width = width
        self.height = height
        self.numeric_labels = self.numeric_labels()
        self.model = self.compile()
        self.demo = demo

    def format_inputs(self):
        resized_images = []
        formated_labels = []

        for image in self.images:

            # change image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.demo:
                cv2.imshow("input", image)
                cv2.waitKey(1000)

            image_tensor = tf.keras.preprocessing.image.img_to_array(
                image)

            # resize the image with padding
            resized_image = tf.image.resize_with_pad(
                image_tensor, target_height=self.height, target_width=self.width)

            # resized_image = tf.keras.preprocessing.image.array_to_img(
            #     resized_image)
            resized_images.append(resized_image)

        for label in self.numeric_labels:
            formated_labels.append(label)

        return np.array(resized_images), np.array(formated_labels)

    def train(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="convnet_from_scratch.keras",
                save_best_only=True,
                monitor="val_loss")
        ]

        X, Y = self.format_inputs()

        x_train, x_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.2, random_state=42)

        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        x_val = tf.convert_to_tensor(x_val)
        y_val = tf.convert_to_tensor(y_val)

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=100,
            callbacks=callbacks,
            validation_data=(x_val, y_val)
        )

    def plot_training(self):

        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history["val_accuracy"]
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()
