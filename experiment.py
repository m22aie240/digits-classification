# Standard scientific Python imports
import matplotlib.pyplot as plt
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval
from sklearn import svm, metrics


# 1. Get the dataset

X, y = read_digits()

#2.Splitting
#X_train, X_test, y_train, y_test = split_data(X,y,test_size=0.3)
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=0.3, dev_size=0.1)

#3.Preprocessing
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)
X_test = preprocess_data(X_test)


#4. Model Training
model = train_model(X_train,y_train, {'gamma':0.001}, model_type="svm")

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)
# Learn the digits on the train subset
#clf.fit(X_train, y_train)


#5. Model Prediction
# Predict the value of the digit on the Dev subset
dev_predictions = predict_and_eval(model, X_dev, y_dev)


#6.Model evaluation on Test Set
test_predictions = predict_and_eval(model, X_test, y_test)

