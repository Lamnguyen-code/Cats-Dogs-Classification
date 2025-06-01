import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

TRAIN_PATH = "dataset/training_set"
VALIDATION_PATH = "dataset/validation_set"
TEST_PATH = "dataset/test_set"
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 5
IMG_DIR = "test_images/"
LABELS = ["cat", "dog"]
TEST_SET_PATH = "test_set"

def create_dataset():
    train_datagen = ImageDataGenerator(
    rotation_range=15,           # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,       # Randomly shift images horizontally by up to 15% of width
    height_shift_range=0.1,      # Randomly shift images vertically by up to 15% of height
    shear_range=0.1,             # Randomly shear images
    zoom_range=0.1,              # Randomly zoom in/out by up to 15%
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest'          # Fill new pixels with the nearest value
)
    validation_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_dataset = train_datagen.flow_from_directory(
        TRAIN_PATH, 
        target_size = (IMAGE_SIZE, IMAGE_SIZE), 
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )

    validation_dataset = validation_datagen.flow_from_directory(
        VALIDATION_PATH,
        target_size = (IMAGE_SIZE, IMAGE_SIZE), 
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )

    test_dataset = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size = (IMAGE_SIZE, IMAGE_SIZE), 
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )

    return train_dataset, validation_dataset, test_dataset

def build_and_train_model():
    (train_dataset, validation_dataset, test_dataset) = create_dataset()

    # build model
    # model = keras.Sequential([
    #     Input(shape = (150, 150, 3)), 
    #     Rescaling(1./255),

    #     Conv2D(32, (3, 3), activation = "relu", kernel_initializer = "he_uniform",padding = "same"), 
    #     MaxPooling2D(pool_size = (2, 2)),
    #     Dropout(rate = 0.2),

    #     Conv2D(64, (3, 3), activation = "relu", kernel_initializer = "he_uniform", padding = "same"),
    #     MaxPooling2D(pool_size = (2, 2)),
    #     Dropout(rate = 0.3), 

    #     Conv2D(128, (3, 3), activation = "relu", kernel_initializer = "he_uniform", padding = "same"),
    #     MaxPooling2D(pool_size = (2, 2)),
    #     Dropout(rate = 0.4),

    #     Flatten(),
    #     Dense(128, activation = "relu", kernel_initializer = "he_uniform"), 
    #     Dense(64, activation = "relu", kernel_initializer = "he_uniform"),
    #     Dense(1, activation = "sigmoid")
    # ])

    base_model = VGG16(weights = "imagenet", include_top = False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation = "relu")(x)
    x = Dense(216, activation = "relu")(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = base_model.input, outputs = x)

    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    H =  model.fit(
        train_dataset, validation_data = validation_dataset,  
        epochs = EPOCHS, verbose = 1,
    )
    _, accuracy = model.evaluate(test_dataset)
    print(f"Accuracy {accuracy}")
    
    fig = plt.figure()
    numOfEpochs = EPOCHS
    plt.plot(np.arange(0, numOfEpochs), H.history['loss'], label='training loss')
    plt.plot(np.arange(0, numOfEpochs), H.history['val_loss'], label='validation loss')
    plt.plot(np.arange(0, numOfEpochs), H.history['accuracy'], label='accuracy')
    plt.plot(np.arange(0, numOfEpochs), H.history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss|Accuracy')
    plt.legend()
    plt.show()

    model.save("model.keras")

def test_model():
    model = keras.models.load_model("model.keras")
    test_gen = ImageDataGenerator()
    test_set = test_gen.flow_from_directory(
        TEST_SET_PATH,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )
    _, accuracy = model.evaluate(test_set)
    print(f"Accuracy is {accuracy}")

# build_and_train_model()
test_model()

