# Standard scientific Python imports
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

