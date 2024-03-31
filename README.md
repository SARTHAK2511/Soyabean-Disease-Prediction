# Soybean Leaf Disease Classification using DINOv2 Embeddings and SVM

This repository contains code for classifying soybean leaf diseases using DINOv2 embeddings and Support Vector Machine (SVM) classification.

## Dependencies

- numpy
- torch
- torchvision
- PIL
- opencv-python (cv2)
- json
- tqdm
- roboflow
- supervision

## Usage

1. Clone the repository containing the Soybean Leaf Dataset for Disease Classification.
2. Ensure you have the necessary dependencies installed.
3. Run the provided code cells sequentially.

## Steps

### Step 1: Data Preparation

Ensure that you have the Soybean Leaf Dataset for Disease Classification. You can obtain the dataset from [here](https://github.com/makhan010385/Soybean-/tree/main/Soybean%20Leaf%20Dataset%20for%20Disease%20Classification). The dataset should be organized in folders, where each folder represents a class, and contains images in JPG format.

### Step 2: Compute Embeddings

The code utilizes the DINOv2 model to compute embeddings of the images. The embeddings are then saved to a JSON file named `all_embeddings.json`.

### Step 3: Train SVM Classifier

A Support Vector Machine (SVM) classifier is trained using the computed embeddings and corresponding class labels.

### Step 4: Predictions

You can make predictions using the trained SVM classifier on new images. Provide the path to the image in the `input_file` variable and run the code cell. The predicted class label will be displayed along with the input image.

## Note

- Make sure to adjust file paths and directories according to your setup.
- Experiment with different pre-trained models and hyperparameters for better classification performance.
