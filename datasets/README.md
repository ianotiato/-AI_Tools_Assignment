# Datasets Documentation

## Overview

This folder contains all datasets used in the AI Tools Assignment.

## Dataset Details

### 1. Iris Species Dataset

- **File**: `raw/iris.csv`
- **Source**: Scikit-learn built-in dataset
- **Size**: 150 samples
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Target**: 3 species (setosa, versicolor, virginica)
- **Use**: Task 1 - Classical ML with Scikit-learn

### 2. MNIST Handwritten Digits

- **Source**: TensorFlow/Keras built-in dataset
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28x28 grayscale images
- **Classes**: 10 digits (0-9)
- **Use**: Task 2 - Deep Learning with CNN
- **Note**: Sample images saved in `processed/mnist_samples/`

### 3. Amazon Product Reviews

- **File**: `raw/amazon_reviews_sample.csv`
- **Source**: Sample data created for demonstration
- **Size**: 8 sample reviews
- **Fields**: review_text, rating, product
- **Use**: Task 3 - NLP with spaCy

## Data Sources

- Iris: Built into scikit-learn
- MNIST: Built into TensorFlow
- Amazon Reviews: Sample data for demonstration purposes

## Preprocessing

All preprocessing steps are documented in the respective Jupyter notebooks in the `code/` folder.
