# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

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
