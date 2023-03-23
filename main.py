import os.path
import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import Template

model_path = Template('model/${name}.h5')

fashion_mnist = tf.keras.datasets.fashion_mnist
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = fashion_mnist.load_data()
mnist_class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                    'Ankle boot']
mnist_shape = mnist_test_images.shape[1:3]


class tfMnistTrainer:
    def __init__(self, _train_images=mnist_train_images, _test_images=mnist_test_images, _train_shape=mnist_shape,
                 _train_labels=mnist_train_labels, _test_labels=mnist_test_labels, _class_names=None,
                 _dense=None, _optimizer='adam', _loss=None, _loss_weight=None, _model=None, _epochs=10):

        self.train_images = _train_images / 255.0
        self.test_images = _test_images / 255.0
        self.train_labels = _train_labels
        self.test_labels = _test_labels

        if _loss is None:
            _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if _class_names is None:
            self.class_names = mnist_class_name

        if (_model is not None) and os.path.isfile(model_path.substitute(name=_model)):
            self.model = keras.models.load_model(model_path.substitute(name=_model))
        else:
            if _dense is None:
                _dense = [
                    tf.keras.layers.Flatten(input_shape=_train_shape),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(10)
                ]
            self.model = tf.keras.Sequential(_dense)
            self.model.compile(optimizer=_optimizer, loss=_loss, loss_weights=_loss_weight, metrics=['accuracy'])
            self.model.fit(self.train_images, self.train_labels, epochs=_epochs)
            self.model_save(_name='temp')

        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        self.predictions = self.probability_model.predict(self.test_images)

    def predict(self, _test_images=None):
        if _test_images is None:
            _test_images = self.test_images
        self.predictions = self.probability_model.predict(_test_images)

    def model_save(self, _name):
        self.model.save(model_path.substitute(name=_name))
        print(f"file {model_path.substitute(name=_name)} was saved!")
        if _name != 'temp':
            if os.path.isfile(model_path.substitute(name='temp')):
                os.remove(model_path.substitute(name='temp'))
                print(f"file {model_path.substitute(name='temp')} was removed!")

    def evaluate(self, _verbose=2, _test_images=None, _test_labels=None):
        if _test_images is None:
            _test_images = self.test_images
        if _test_labels is None:
            _test_labels = self.test_labels

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

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(_predictions_array),
                                             self.class_names[_true_label]),
                   color=color)


def plot_value_array(ind, predictions_array, true_label):
    true_label = true_label[ind]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
if __name__ == '__main__':
    model = None  # load the existing model

    # dense = None # the custom dense of the model if model is None
    dense = [
        tf.keras.Input(shape=(mnist_shape[0], mnist_shape[1], 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]

    optimizer = 'adam'
    # optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)   # RNN
    # optimizer = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    # optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # optimizer = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # optimizer = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    loss = None
    # loss = {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    #         'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
    #         'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
    #         'kullback_leibler_divergence', 'poisson', 'cosine_proximity'}

    tc = tfMnistTrainer(_dense=dense, _model=model, _optimizer=optimizer)
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        tc.plot_image(i, tc.predictions[i], tc.test_labels, tc.test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, tc.predictions[i], tc.test_labels)
    plt.tight_layout()
    plt.show()
