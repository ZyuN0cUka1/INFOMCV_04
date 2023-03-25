from string import Template
import keras.models
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import os.path
import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_generator = ImageDataGenerator(validation_split=0.2)

dataset_path = "data_choice/"
df = pd.read_csv(dataset_path + "styles.csv", nrows=5000)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.sample(frac=1).reset_index(drop=True)
batch_size = 32

training_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory=dataset_path + "images",
    x_col="image",
    y_col="subCategory",
    target_size=(96, 96),
    batch_size=batch_size,
    subset="training"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory=dataset_path + "images",
    x_col="image",
    y_col="subCategory",
    target_size=(96, 96),
    batch_size=batch_size,
    subset="test"
)

test_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory=dataset_path + "images",
    x_col="image",
    y_col="subCategory",
    target_size=(96, 96),
    batch_size=batch_size,
    subset="validation"
)

classes = len(training_generator.class_indices)

dense = [
    tf.keras.Input(shape=(96, 96, 3)),
    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
]

model_path = Template('model/ntr_${name}.h5')
fig_path = Template('figure/ntr_${name}_${addition}.png')


class ntrTrainer:
    def __init__(self, _training_generator=training_generator, _validation_generator=validation_generator,
                 _test_generator=test_generator, _dense=None, _optimizer='adam', _loss=None, _loss_weight=None,
                 _model=None, _epochs=15, _classes=classes):
        self.test_generator = _test_generator
        self.training_generator = _training_generator
        self.validation_generator = _validation_generator
        self.classes = _classes

        if _loss is None:
            _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if (_model is not None) and os.path.isfile(model_path.substitute(name=_model)):
            self.model = keras.models.load_model(model_path.substitute(name=_model))
            self.name = _model
        else:
            if _dense is None:
                _dense = [
                    tf.keras.Input(shape=(96, 96, 3)),
                    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation="softmax")
                ]
            self.model = tf.keras.Sequential(_dense)
            self.model.compile(optimizer=_optimizer, loss=_loss, loss_weights=_loss_weight, metrics=['accuracy'])
            self.history = self.model.fit(self.training_generator, epochs=_epochs, validation_data=self.validation_generator)
            self.model_save(_name='temp')
            self.name = 'temp'

        self.model.summary()
        try:
            self.plot_training_figure()
        finally:
            pass
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        self.predictions = self.probability_model.predict(self.test_generator)

    def predict(self, _test_images=None):
        if _test_images is None:
            _test_images = self.test_generator
        self.predictions = self.probability_model.predict(_test_images)

    def model_save(self, _name):
        self.name = _name
        self.model.save(model_path.substitute(name=_name))
        print(f"file {model_path.substitute(name=_name)} was saved!")
        self.plot_training_figure()
        if _name != 'temp':
            if os.path.isfile(model_path.substitute(name='temp')):
                os.remove(model_path.substitute(name='temp'))
                print(f"file {model_path.substitute(name='temp')} was removed!")

    def evaluate(self, _verbose=2, _test_images=None, _test_labels=None):
        if _test_images is None:
            _test_images = self.test_generator
        if _test_labels is None:
            _test_labels = self.test_generator

        test_loss, test_acc = self.model.evaluate(_test_images, _test_labels, verbose=_verbose)
        print('\nTest accuracy:', test_acc)

    def plot_image(self, _ind, _predictions_array, _true_label, _img):
        _true_label, _img = _true_label[_ind], _img[_ind]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(_img, cmap=plt.cm.binary)

        predicted_label = np.argmax(_predictions_array)
        if predicted_label == _true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.classes[predicted_label],
                                             100 * np.max(_predictions_array),
                                             self.classes[_true_label]),
                   color=color)

    def plot_training_figure(self):
        plt.figure()
        plt.plot(self.history.history['accuracy'], 'b', label='Training accuracy')
        plt.plot(self.history.history['val_accuracy'], 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(fig_path.substitute(name=self.name, addition='acc'))
        print(f"file {fig_path.substitute(name=self.name, addition='acc')} was saved!")
        plt.figure()
        plt.plot(self.history.history['loss'], 'b', label='Training loss')
        plt.plot(self.history.history['val_loss'], 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(fig_path.substitute(name=self.name, addition='loss'))
        print(f"file {fig_path.substitute(name=self.name, addition='loss')} was saved!")
