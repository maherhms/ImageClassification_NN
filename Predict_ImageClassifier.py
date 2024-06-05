import tensorflow as tf
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    resize = tf.image.resize(img, (256, 256))
    resize = resize / 255.0  # Normalize the image to [0, 1]
    return np.expand_dims(resize, axis=0)  # Add batch dimension

# Load the trained model
model = tf.keras.models.load_model('./models/imageclassifier.keras')

# Path to the test image
test_image_path = './test_data/test02.jpg'

# Preprocess the image
input_image = load_and_preprocess_image(test_image_path)

# Make a prediction
prediction = model.predict(input_image)

# Interpret the result
if prediction > 0.5:
    result = "Sad"
else:
    result = "Happy"

# Print the result
print(f'Predicted class is {result}')

# Display the image
plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title(f'Predicted class: {result}')
plt.show()