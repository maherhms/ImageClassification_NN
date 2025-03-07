import tensorflow as tf
import os
import cv2
from PIL import Image
from PIL import ImageFile
import numpy as np
from matplotlib import pyplot as plt

# 1. Install Dependencies and Setup environment variables
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Verify GPU devices
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 2. Clean wrong data
data_dir = './data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
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

# Verify directory structure
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if not os.path.isdir(class_path):
        print('Found a non-directory entry in data_dir: {}'.format(class_path))

# 3. Load Datasets/batches
data = tf.keras.utils.image_dataset_from_directory("data", image_size=(256, 256), batch_size=32)
data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# 4. Scale down data
data = data.map(lambda x, y: (x / 255, y))

# 5. Split Data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

print(f"train size: {train_size}")

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# 6. Build Deep Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# 7. Train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('./models/imageclassifier.keras', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

hist = model.fit(train, epochs=20, validation_data=val,
                 callbacks=[tensorboard_callback, checkpoint_cb, early_stopping_cb])

# 8. Plot Performance
# LOSS
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
fig.show()

# ACCURACY
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
fig.show()

# 9. Evaluate model
model_precision = tf.keras.metrics.Precision()
model_recall = tf.keras.metrics.Recall()
model_accuracy = tf.keras.metrics.BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    model_precision.update_state(y, yhat)
    model_recall.update_state(y, yhat)
    model_accuracy.update_state(y, yhat)
print(model_precision.result().numpy(), model_recall.result().numpy(), model_accuracy.result().numpy())

# 10. Test model
img = cv2.imread('./test_data/154006829.jpg')
# plt.imshow(img)
# plt.show()

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
# plt.show()

yhat = model.predict(np.expand_dims(resize / 255, 0))
if yhat > 0.5:
    print(f'Predicted class is dog')
else:
    print(f'Predicted class is cat')

# 11. Save model
model.save(os.path.join('models', 'imageclassifier.keras'))
new_model = tf.keras.models.load_model('./models/imageclassifier.keras')
new_model.predict(np.expand_dims(resize / 255, 0))
