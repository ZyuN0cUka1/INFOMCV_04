import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = fashion_mnist.load_data()
mnist_class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                    'Ankle boot']


class tfMnistTrainer:
    def __init__(self, train_images=mnist_train_images, test_images=mnist_test_images,
                 train_labels=mnist_train_labels, test_labels=mnist_test_labels, class_names=mnist_class_name,
                 dense=None, optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 loss_weight=None):
        self.class_names = class_names
        if dense is None:
            dense = [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
            ]

        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.model = tf.keras.Sequential(dense)
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weight, metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, epochs=10)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        self.predictions = self.probability_model.predict(self.test_images)

    def predict(self, test_images=None):
        if test_images is None:
            test_images = self.test_images
        self.predictions = self.probability_model.predict(test_images)

    def evaluate(self, verbose=2, test_images=None, test_labels=None):
        if test_images is None:
            test_images = self.test_images
        if test_labels is None:
            test_labels = self.test_labels

        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=verbose)
        print('\nTest accuracy:', test_acc)

    def plot_image(self, i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
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
    tc = tfMnistTrainer()
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
