# Standard scientific Python imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval, tune_hparams,generate_param_combinations
from sklearn import svm, metrics
from itertools import product

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
# Create a combination of all parameters
all_param_combinations = generate_param_combinations(gamma_ranges, C_ranges)

# 1. Get the dataset
X, y = read_digits()

# 2. Splitting
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

        # 4. Data preprocessing
        X_train = preprocess_data(X_train)
        X_dev = preprocess_data(X_dev)
        X_test = preprocess_data(X_test)

        # 5. Model training
        model_params = {'gamma': 0.001}
        best_hparams, best_model, best_accuracy_dev = tune_hparams(X_train, y_train, X_dev, y_dev, all_param_combinations)

        # 6. Getting model predictions on test set
        best_accuracy_test = best_model.score(X_test, y_test)

        # 7. Output the results
        print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size} train_acc={best_accuracy_dev} dev_acc={best_accuracy_dev} test_acc={best_accuracy_test}")
        print(f"Best Hyperparameters: {best_hparams}")


# 7. Output the results
print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size} train_acc={best_accuracy_dev} dev_acc={best_accuracy_dev} test_acc={best_accuracy_test}")
print(f"Best Hyperparameters: {best_hparams}")

# Add these lines to print the number of total samples and the size of the images
n_samples = len(X)
image_height, image_width = X[0].shape
print(f"Total Samples in the Dataset: {n_samples}")
print(f"Image Size - Height: {image_height}, Width: {image_width}")



# ... (previous code)

def resize_images(images, new_size):
    resized_images = [cv2.resize(img, (new_size, new_size)) for img in images]
    return np.array(resized_images)

# Load the dataset
X, y = read_digits()

# Define different image sizes
image_sizes = [4, 6, 8]

# Define data split sizes
train_size = 0.7
dev_size = 0.1
test_size = 0.2

for size in image_sizes:
    # Resize images
    X_resized = resize_images(X, size)
    
    # Split data
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X_resized, y, test_size, dev_size)
    
    # Data preprocessing
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)
    
    # Model training (you can use your existing code for this)
    model_params = {'gamma': 0.001}
    best_hparams, best_model, best_accuracy_dev = tune_hparams(X_train, y_train, X_dev, y_dev, all_param_combinations)
    
    # Model evaluation
    best_accuracy_train = best_model.score(X_train, y_train)
    best_accuracy_dev = best_model.score(X_dev, y_dev)
    best_accuracy_test = best_model.score(X_test, y_test)
    
    # Print results
    print(f"image size: {size}x{size} train_size: {train_size} dev_size: {dev_size} test_size: {test_size} train_acc: {best_accuracy_train:.2f} dev_acc: {best_accuracy_dev:.2f} test_acc: {best_accuracy_test:.2f}")

