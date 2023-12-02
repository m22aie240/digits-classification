"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd

roll_no = "m22aie240"  # Your roll number

num_runs = 1
# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
classifier_param_dict = {}

# 2.1 SVM
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params_svm = {'gamma': gamma_list, 'C': C_list}
h_params_svm_combinations = get_hyperparameter_combinations(h_params_svm)
classifier_param_dict['svm'] = h_params_svm_combinations

# 2.2 Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {'max_depth': max_depth_list}
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations

# 2.3 Logistic Regression
solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
h_params_lr = {'solver': solver_list}
h_params_lr_combinations = get_hyperparameter_combinations(h_params_lr)
classifier_param_dict['lr'] = h_params_lr_combinations

results = []
test_sizes = [0.2]
dev_sizes = [0.2]
for cur_run_i in range(num_runs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1 - test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy, best_model = tune_hparams(X_train, y_train, X_dev, y_dev, current_hparams, model_type)

                # Construct the filename based on the model type
                if model_type == 'lr':
                    # Only for logistic regression, include the solver in the filename
                    model_filename = f"{roll_no}_lr_{best_hparams['solver']}.joblib"
                else:
                    # For other models, use a generic naming scheme
                    model_filename = f"{roll_no}_{model_type}.joblib"

                dump(best_model, model_filename)

                # Loading of model
                best_model = load(best_model_path)

                # Model evaluation
                test_acc, test_f1, predicted_y = predict_and_eval(best_model, X_test, y_test)
                train_acc, train_f1, _ = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                # Print model performance
                print(f"{model_type}\ttest_size={test_size:.2f} dev_size={dev_size:.2f} train_size={train_size:.2f} train_acc={train_acc:.2f} dev_acc={dev_acc:.2f} test_acc={test_acc:.2f}, test_f1={test_f1:.2f}")

                # Cross-validation for Logistic Regression
                if model_type == 'lr':
                    scores = cross_val_score(LogisticRegression(**best_hparams), X_train, y_train, cv=5)
                    print(f"LR Solver: {best_hparams['solver']} - Mean CV Score: {scores.mean():.2f}, Std: {scores.std():.2f}")

                # Append results
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc': train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)

# Additional code for model saving and GitHub integration as needed

