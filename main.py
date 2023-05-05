import read_input as rd
import data_load as dl

import keras
import tensorflow
from keras import layers
import random
import matplotlib.pyplot as plt


def main():
    root_directory = "/Users/nathangartlan1/Documents/AA-Nathan/uchicago-mpcs-work/AppliedDataAnalysis/Project/Technical/iam_datasets"

    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    word_labels = dl.read_word_labels(root_directory, True)
    full_dataset = dl.load_and_label_word_images(word_labels)

    full_dataset = full_dataset.shuffle(buffer_size=1000).batch(32)

    # Define the sizes of the train, validation, and test sets
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Shuffle the dataset and split it into train, validation, and test sets
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size).take(val_size)
    test_dataset = full_dataset.skip(train_size + val_size).take(test_size)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet_from_scratch.keras",
            save_best_only=True,
            monitor="val_loss")
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=val_dataset,
        callbacks=callbacks)

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
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

    test_model = keras.models.load_model("convnet_from_scratch.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
