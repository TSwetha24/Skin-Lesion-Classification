import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  
TRAIN_DIR = r"C:\Users\Sweth\Documents\seg1\train"
VALID_DIR = r"C:\Users\Sweth\Documents\seg1\validation"

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# MobileNet-like Scratch Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", input_shape=(224, 224, 3)),
        BatchNormalization(),
        ReLU(),
        DepthwiseConv2D((3, 3), padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, (1, 1)),
        BatchNormalization(),
        ReLU(),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Create & Train Model
model = build_model()
model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)

# Save Model
model.save("skin_lesion_classifier_scratch.keras")
