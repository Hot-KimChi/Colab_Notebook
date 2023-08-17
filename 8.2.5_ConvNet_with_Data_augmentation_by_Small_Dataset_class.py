import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from keras.utils import image_dataset_from_directory
import zipfile

import os, shutil, pathlib, datetime

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class ResidualCNN:
    def __init__(self):
        self.original_dir = pathlib.Path("extracted_files/train")
        self.new_base_dir = pathlib.Path("cats_vs_dogs_small")
        self.path_name = "cats_vs_dogs_small"

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = None

        self.model = None
        self.history = None

        self.AUTOTUNE = tf.data.AUTOTUNE

    def download_and_extract(self):
        file_name = 'dogs-vs-cats.zip'
        if os.path.exists(file_name):
            pass
        else:
            import gdown
            gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output=file_name)

            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall('extracted_files/')
                with zipfile.ZipFile('extracted_files/train.zip', 'r') as zip_2nd:
                    zip_2nd.extractall('extracted_files/')


    def make_subset(self, subset_name, start_index, end_index):
        for category in ("cat", "dog"):
            dir = self.new_base_dir / subset_name / category
            os.makedirs(dir)
            fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src=self.original_dir / fname, dst=dir / fname)


    def prepare_data(self):
        if os.path.exists(self.path_name):
            pass
        else:
            self.make_subset("train", start_index=0, end_index=1000)
            self.make_subset("validation", start_index=1000, end_index=1500)
            self.make_subset("test", start_index=1500, end_index=2500)

        self.train_ds = image_dataset_from_directory(
            self.new_base_dir / "train",
            image_size=(180, 180),
            batch_size=32)
        self.val_ds = image_dataset_from_directory(
            self.new_base_dir / "validation",
            image_size=(180, 180),
            batch_size=32)
        self.test_ds = image_dataset_from_directory(
            self.new_base_dir / "test",
            image_size=(180, 180),
            batch_size=32)

        self.class_names = self.train_ds.class_names


    def visualize_data(self, check=False):
        if check == 'Yes':
            plt.figure(figsize=(10, 10))
            for images, labels in self.train_ds.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(self.class_names[labels[i]])
                    plt.axis("off")
                plt.show()
        else:
            pass


    def build_model(self):
        inputs = keras.Input(shape=(180, 180, 3))

        x = layers.Rescaling(1. / 255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

        x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
        x = layers.Flatten()(x)

        # def residual_block(x, filters, pooling=False):
        #     residual = x
        #     x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        #     x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        #
        #     if pooling:
        #         x = layers.MaxPooling2D(2, padding='same')(x)
        #         residual = layers.Conv2D(filters, 1, strides=2)(residual)
        #     elif filters != residual.shape[-1]:
        #         residual = layers.Conv2D(filters, 1)(residual)
        #
        #     x = layers.add([x, residual])
        #
        #     return x
        #
        # x = residual_block(x, filters=32, pooling=True)
        # x = residual_block(x, filters=64, pooling=True)
        # x = residual_block(x, filters=128, pooling=False)
        #
        # x = layers.GlobalAveragePooling2D()(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)


    def train_model(self, epochs=10):
        current_date = datetime.datetime.now()
        self.today_date = current_date.date()

        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        callbacks = [keras.callbacks.ModelCheckpoint(
            filepath=f'{self.today_date}_computer_vision.keras',
            save_best_only=True,
            monitor='val_loss')
        ]

        self.history = self.model.fit(self.train_ds,
                                      epochs=epochs,
                                      callbacks=callbacks,
                                      batch_size=32,
                                      validation_data=self.val_ds)


    def plot_model_summary(self):
        self.model.summary()


    def plot_training_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


    def test_evaluation(self):
        test_model = keras.models.load_model(f'{self.today_date}_computer_vision.keras')
        test_loss, test_acc = test_model.evaluate(self.test_ds)
        print(f"테스트 정확도: {test_acc:.3f}")


# Create an instance of the ResidualCNN class
residual_cnn = ResidualCNN()

# Execute the steps
residual_cnn.download_and_extract()
residual_cnn.prepare_data()
residual_cnn.visualize_data(check=False)
residual_cnn.build_model()
residual_cnn.train_model(epochs=50)
residual_cnn.plot_model_summary()
residual_cnn.plot_training_history()
residual_cnn.test_evaluation()
