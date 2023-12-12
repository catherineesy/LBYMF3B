# LBYMF3B
Term End Project 

Training: 

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

input_shape = (224, 224, 3)
num_classes = 5
batch_size = 36
epochs = 20

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Blood_cells/Train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    classes=[str(i) for i in range(1, num_classes + 1)],
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    'Blood_cells/Validation',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    classes=[str(i) for i in range(1, num_classes + 1)],
    shuffle=False
)

steps_per_epoch = 3600 // batch_size
validation_steps = 1800 // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

model.save('5.20epoch.white_blood_cell_model.keras')


Testing: 

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

model = tf.keras.models.load_model('5.20epoch.white_blood_cell_model.keras')

test_dir = 'Blood_cells/Test/'

test_images = os.listdir(test_dir)

output_file = '5.30epoch.test_results.txt'
with open(output_file, 'w') as file:
    for test_image in test_images:
        img_path = os.path.join(test_dir, test_image)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)

        class_labels = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        confidence = predictions[0][predicted_class]

        file.write(f"{predicted_label}")
        #file.write(f"Image: {test_image}\n")
        #file.write(f"Predicted class: {predicted_label}\n")
        #file.write(f"Confidence: {confidence}\n")
        file.write("\n")

print(f"Results saved to {output_file}")
