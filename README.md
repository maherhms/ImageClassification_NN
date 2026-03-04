# ImageClassification_NN

A **Convolutional Neural Network (CNN)** built with **TensorFlow / Keras** that classifies images into categories. This project covers the full ML workflow — from raw data cleaning and preprocessing, through model training and evaluation, to inference on new images.

Trained on a **cats vs dogs dataset** with results tracked via TensorBoard and model checkpointing.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| Test Precision | logged via `model_precision` |
| Test Recall | logged via `model_recall` |
| Test Accuracy | logged via `model_accuracy` |

> Training loss and accuracy curves are plotted automatically at the end of training using Matplotlib.

---

## 📦 Project Structure

```
ImageClassification_NN/
├── models/                      # Folder where trained models are saved
├── data/                        # Dataset directory — images organized into class folders
├── test_data/                   # Sample images for prediction/testing
├── Train_ImageClassifier.py     # Full training pipeline
├── Predict_ImageClassifier.py   # Loads saved model and runs inference
├── README.md                    # This file
├── pyproject.toml               # Python project config
└── .gitignore
```

---

## 🧠 How It Works

The training pipeline (`Train_ImageClassifier.py`):
1. Scans and cleans the dataset — removes corrupt or unsupported images
2. Loads images into batches using `tf.keras.utils.image_dataset_from_directory`
3. Normalizes pixel values to [0, 1]
4. Splits data into 70% train / 20% validation / 10% test
5. Builds and trains a CNN with 3 convolutional blocks
6. Uses callbacks: TensorBoard logging, model checkpointing (best model only), early stopping
7. Plots training/validation loss and accuracy curves
8. Evaluates precision, recall, and accuracy on the test set
9. Saves the final model to `./models/imageclassifier.keras`

The inference script (`Predict_ImageClassifier.py`):
- Loads the saved model
- Preprocesses a new image (resize to 256x256, normalize)
- Outputs the predicted class with a matplotlib visualization

---

## 🏗️ Model Architecture

```
Input (256x256x3)
→ Conv2D(16, 3x3, relu) → MaxPooling
→ Conv2D(32, 3x3, relu) → MaxPooling
→ Conv2D(16, 3x3, relu) → MaxPooling
→ Flatten
→ Dense(256, relu)
→ Dense(1, sigmoid)  ← binary output
```

Compiled with:
- Optimizer: `Adam`
- Loss: `BinaryCrossentropy`
- Metrics: `Accuracy`

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV — image reading and preprocessing
- NumPy
- Matplotlib — training visualization
- Pillow — image format validation

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/maherhms/ImageClassification_NN.git
cd ImageClassification_NN
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

pip install tensorflow opencv-python matplotlib pillow
```

---

## 📁 Prepare Your Dataset

Organize training images by class:

```
data/
├── cat/
│   ├── img1.jpg
│   └── ...
├── dog/
│   ├── img1.jpg
│   └── ...
```

Place test images in `test_data/`.

---

## 📈 Train the Model

```bash
python Train_ImageClassifier.py
```

This will clean the dataset, train the CNN, save the best model, and plot performance curves automatically.

---

## 🔍 Run Inference

```bash
python Predict_ImageClassifier.py
```

Modify the `test_image_path` variable inside the script to point to any image you want to classify.

---

## 📡 Monitor Training with TensorBoard

```bash
tensorboard --logdir logs
```

Open the URL shown in the terminal to view live loss and accuracy curves during or after training.

---

## 🔧 Possible Improvements

- Multi-class classification support
- Data augmentation (flipping, rotation, zoom)
- Transfer learning with pretrained models (MobileNet, ResNet, EfficientNet)
- Confusion matrix and per-class metrics
- Simple web interface for drag-and-drop prediction

---

## 📄 License

Open for learning and modification. See `LICENSE` for details.
