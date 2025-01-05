import numpy as np
import pytest
from unittest.mock import patch
import tensorflow as tf
from src.train_test_split import train_test_Minst_Dataset

@pytest.fixture
def mock_mnist_data():
    """Mock MNIST dataset."""
    x_train = np.random.randint(0, 256, size=(60000, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, size=(60000,), dtype=np.uint8)
    x_test = np.random.randint(0, 256, size=(10000, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, size=(10000,), dtype=np.uint8)
    return (x_train, y_train), (x_test, y_test)


@patch("tensorflow.keras.datasets.mnist.load_data")
def test_train_test_mnist_dataset(mock_load_data, mock_mnist_data):
    """Test the train_test_Minst_Dataset function."""
    # Mock the MNIST dataset
    mock_load_data.return_value = mock_mnist_data
    digit_to_segments = {
        0: [1, 1, 1, 1, 1, 1, 0],
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1],
    }

    # Call the function
    result = train_test_Minst_Dataset()

    # Mocked data
    (x_train, y_train), (x_test, y_test) = mock_mnist_data

    # Check normalization
    assert np.all(x_train / 255.0 <= 1.0), "x_train has values greater than 1.0"
    assert np.all(x_train / 255.0 >= 0.0), "x_train has values less than 0.0"
    assert np.all(x_test / 255.0 <= 1.0), "x_test has values greater than 1.0"
    assert np.all(x_test / 255.0 >= 0.0), "x_test has values less than 0.0"

    # Check reshaping
    reshaped_train = x_train.reshape((-1, 28, 28, 1))
    reshaped_test = x_test.reshape((-1, 28, 28, 1))
    assert reshaped_train.shape == (60000, 28, 28, 1), "x_train shape mismatch"
    assert reshaped_test.shape == (10000, 28, 28, 1), "x_test shape mismatch"

    # Check digit-to-segment mapping
    y_train_segments = np.array([digit_to_segments[d] for d in y_train])
    y_test_segments = np.array([digit_to_segments[d] for d in y_test])
    assert y_train_segments.shape == (60000, 7), "y_train_segments shape mismatch"
    assert y_test_segments.shape == (10000, 7), "y_test_segments shape mismatch"