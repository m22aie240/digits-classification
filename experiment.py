# Standard scientific Python imports
import matplotlib.pyplot as plt
from utils import preprocess_data, split_data, train_model, read_digits
from sklearn import svm, metrics


# 1. Get the dataset

X, y = read_digits()

#2.Splitting
X_train, X_test, y_train, y_test = split_data(X,y,test_size=0.3)


#3.Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)


#4. Model Training
model = train_model(X_train,y_train, {'gamma':0.001}, model_type="svm")

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
# Learn the digits on the train subset
clf.fit(X_train, y_train)

#5. Model Prediction
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

#6. Qualitative sanity check for predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

#6.Model evaluation
print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)

