import keras
from keras import layers
import matplotlib.pyplot as plt
import model_base
import tensorflow as tf


class basic_keras(model_base):

    def compile(self, width, height):
        inputs = keras.Input(shape=(width, height, 1))
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=1, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=1, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=1, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=1, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=1, activation="relu")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss="binary_crossentropy",
                      optimizer="rmsprop",
                      metrics=["accuracy"])

        return model

    def __init__(self,
                 labels,
                 images,
                 width,
                 height) -> None:
        self.labels = labels
        self.images = images
        self.width = width
        self.height = height
        self.model = compile(self)

    def resize_images(self):
        resized_images = []

        for image in self.images:
            image_tensor = tf.keras.preprocessing.image.img_to_array(image)
            resized_image = tf.image.resize_with_pad(
                image_tensor, target_height=self.height, target_width=self.width)

            resized_image = tf.keras.preprocessing.image.array_to_img(
                resized_image)
            resized_images.append(resized_image)

        return resized_images

    def train(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="convnet_from_scratch.keras",
                save_best_only=True,
                monitor="val_loss")
        ]

        x_resized = self.resize_images()

        self.history = self.model.fit(
            x_resized,
            self.labels,
            epochs=10,
            callbacks=callbacks,
            validation_split=0.2)

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
