import keras.models
import tensorflow as tf
import pandas as pd
import main
from keras_preprocessing.image import ImageDataGenerator

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

keras.models.Model()
# tc = main.tfMnistTrainer(_dense=dense)
