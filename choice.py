from string import Template
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import keras.models
import tensorflow as tf
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
    subset="validation"
)

classes = len(training_generator.class_indices)

model_path = Template('model/ntr_${name}.h5')
fig_path = Template('figure/ntr_${name}_${addition}.png')


def plot_training_figure(history, name):
    plt.figure()
    plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(fig_path.substitute(name=name, addition='acc'))
    print(f"file {fig_path.substitute(name=name, addition='acc')} was saved!")
    plt.figure()
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(fig_path.substitute(name=name, addition='loss'))
    print(f"file {fig_path.substitute(name=name, addition='loss')} was saved!")


if __name__ == '__main__':
    inputs = tf.keras.Input(shape=(96, 96, 3)),
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss_weights=None, metrics=['accuracy'])
    his = model.fit(training_generator, epochs=15, validation_data=validation_generator)

    plot_training_figure(history=his, name='choice')
