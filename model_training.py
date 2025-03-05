# model_training.py - Train the Eye Disease Detection Model

import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

# Split the dataset into training and validation sets.
# Ensure your dataset is organized with one folder per class in the 'dataset' folder.
splitfolders.ratio('C:\\Users\\lashi\\OneDrive\\Desktop\\eye\\content\\dataset', output='output', seed=1337, ratio=(0.8, 0.2))

# Define image size
IMAGE_SIZE = [224, 224]

# Data augmentation for the training set and rescaling for validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen  = ImageDataGenerator(rescale=1./255)

# Load the dataset from the split folders
training_set = train_datagen.flow_from_directory(
    'output/train',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'output/val',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

# Load the pre-trained VGG19 model without its top layers
base_model = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of VGG19
x = Flatten()(base_model.output)
prediction = Dense(4, activation='softmax')(x)  # Adjust the number of classes if needed
model = Model(inputs=base_model.input, outputs=prediction)
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the trained model
model.save('evgg.h5')
