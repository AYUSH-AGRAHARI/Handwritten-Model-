import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape data for CNN
x_train = np.expand_dims(tf.keras.utils.normalize(x_train, axis=1), axis=-1)
x_test = np.expand_dims(tf.keras.utils.normalize(x_test, axis=1), axis=-1)


# Create a data augmentation function using tf.image
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
    return image, label


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
validation_dataset = validation_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Define a Convolutional Neural Network model with more layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 ),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model with training and validation data
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# Save the model with a .h5 extension
model.save('handwritten_cnn_augmented.h5')

# Load the model
model = load_model('handwritten_cnn_augmented.h5')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Predict on custom images
image_number = 1
while True:
    image_path = f"digits/digit{image_number}.png"

    # Break the loop if the image does not exist
    if not os.path.isfile(image_path):
        print(f"No more images found at {image_path}")
        break

    try:
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            image_number += 1
            continue

        # Resize the image to 28x28 pixels
        img = cv2.resize(img, (28, 28))

        # Invert the image (if your model was trained with inverted images)
        img = 255 - img  # Invert the image pixel values

        # Normalize the image
        img = tf.keras.utils.normalize(img, axis=1)

        # Reshape for CNN input
        img = np.expand_dims(img, axis=-1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        print(f"This Digit is Probably a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0].reshape(28, 28), cmap=plt.cm.binary)  # Ensure the image is reshaped correctly
        plt.show()

        # Move to the next image number
        image_number += 1

    except Exception as e:
        print(f"Error: {e}")
        imag