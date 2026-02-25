# ImageClassification_NN

A simple **image classification neural network** built with **TensorFlow / Keras** that trains a CNN to classify images and provides tools for training and inference.

This project was built as a university assignment and demonstrates the full workflow of preparing image data, training a model, evaluating it, and using the model to make predictions on new images.

---

## 📦 Project Structure

```
ImageClassification_NN/
├── models/                      # Folder where trained models are saved
├── data/                        # Dataset directory — images organized into class folders
├── test_data/                   # Sample images for prediction/testing
├── Train_ImageClassifier.py     # Script for training the image classifier
├── Predict_ImageClassifier.py   # Script to load model and make predictions
├── README.md                    # This file
├── pyproject.toml               # Python project config
└── .gitignore
```

---

## 🧠 Project Overview

This project implements a Convolutional Neural Network (CNN) using **TensorFlow/Keras** to classify images. It:

- Cleans and validates the dataset
- Loads images and creates training, validation, and test splits
- Builds a CNN model
- Trains the model with callbacks (TensorBoard, checkpoints, early stopping)
- Visualizes training results (loss vs accuracy)
- Evaluates model performance on test data
- Saves and loads the trained model
- Predicts new images

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV (for image preprocessing)
- NumPy
- Matplotlib (for visualization)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/maherhms/ImageClassification_NN.git
cd ImageClassification_NN
```

### 2. Install Dependencies

Create a virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate     # macOS / Linux
# OR
venv\Scripts\activate        # Windows

pip install tensorflow opencv-python matplotlib pillow
```

---

## 📁 Prepare Your Dataset

Organize your training images in a folder structure like:

```
data/
├── class1/
│   ├── img1.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
```

Place any test images you want to classify in the `test_data/` folder.

---

## 📈 Training the Model

Run the training script:

```bash
python Train_ImageClassifier.py
```

This script:

- Loads and preprocesses images from the `data/` directory
- Builds and trains a CNN model
- Saves the best model in the `models/` folder
- Plots training/validation loss and accuracy
- Evaluates on a test split

---

## 🔍 Making Predictions

After training, run:

```bash
python Predict_ImageClassifier.py
```

This loads the saved model and predicts the class of a new image from the `test_data/` directory. You can modify the image path inside the script to test different images.

---

## 📊 Monitoring Training (Optional)

If TensorBoard logging is enabled in the training script, run:

```bash
tensorboard --logdir logs
```

Then open the provided local URL in your browser to monitor training metrics.

---

## 🧾 Possible Improvements

- Add support for multi-class classification
- Implement data augmentation
- Use transfer learning (e.g., MobileNet, ResNet)
- Add confusion matrix and detailed evaluation metrics
- Create a simple web interface for predictions

---

## 📄 License

This project was created for academic purposes. You are free to modify and use it for learning.