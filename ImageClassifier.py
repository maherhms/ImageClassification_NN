import tensorflow as tf
import os
import cv2
from PIL import Image
from PIL import ImageFile
import numpy as np
from matplotlib import pyplot as plt

#1. Install Dependencies and Setup environment variables
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_growth=True)])

# Verify GPU devices
print(tf.config.list_physical_devices('GPU'))

#2. Clean wrong data
# Data directory
data_dir = './data'
# Accepted image extensions
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Clean up images
ImageFile.LOAD_TRUNCATED_IMAGES = True
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print('Failed to read image: {}'.format(image_path))
                    os.remove(image_path)
                    continue
                with Image.open(image_path) as img:
                    if img.format.lower() not in image_exts:
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
                        print(img.format)
            except Exception as e:
                print('Issue with image {}: {}'.format(image_path, e))
                # os.remove(image_path)

# Verify directory structure
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if not os.path.isdir(class_path):
        print('Found a non-directory entry in data_dir: {}'.format(class_path))

#3. Load Datasets/batches
# Create a dataset from the directory
data = tf.keras.utils.image_dataset_from_directory("data")

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()


