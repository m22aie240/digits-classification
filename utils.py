# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Utilities

def preprocess_data(data):
        # flatten the images
        n_samples = len(data)
        data = data.reshape((n_samples, -1))
        return data

def split_data(x, y, test_size, random_state=1):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=random_state)
        return X_train, X_test, y_train, y_test

def train_model(x,y,model_params, model_type="svm"):
        if model_type == "svm":
                clf = svm.SVC
        model = clf(**model_params)
        model.fit(x, y)
        return model

def read_digits():
        digits = datasets.load_digits()
        X = digits.images
        y = digits.target
        y = digits.target
        return X, y

def split_train_dev_test(X, y, test_size, dev_size):
	X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	relative_dev_size = dev_size / (1 - test_size)
	X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=relative_dev_size, random_state=42)
	return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
	predicted = model.predict(X_test)
	report = metrics.classification_report(y_test, predicted)
	print(f"Classification report for classifier {model}:\n{report}\n")
	disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
	disp.figure_.suptitle("Confusion Matrix")
	plt.show()
	return predicted



