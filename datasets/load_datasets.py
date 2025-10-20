"""
Dataset loading utilities for AI Tools Assignment
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist


def load_iris_data():
    """Load and return Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df


def load_mnist_data():
    """Load and return MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def load_amazon_reviews():
    """Load sample Amazon reviews"""
    return pd.read_csv('datasets/raw/amazon_reviews_sample.csv')


if __name__ == "__main__":
    # Test dataset loading
    iris_df = load_iris_data()
    print(f"Iris data shape: {iris_df.shape}")

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"MNIST train: {x_train.shape}, test: {x_test.shape}")

    reviews_df = load_amazon_reviews()
    print(f"Amazon reviews: {reviews_df.shape}")
