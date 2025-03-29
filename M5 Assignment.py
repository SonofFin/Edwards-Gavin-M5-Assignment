# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split

# Load dataset + shuffle it to reduce bias
from ucimlrepo import fetch_ucirepo

# fetch the dataset
student_performance = fetch_ucirepo(id=320)

X = student_performance.data.features
y = student_performance.data.targets

# print the metadata
print(student_performance.metadata)

# print the variable info
print(student_performance.variables)

# data is shuffled
X, y = shuffle(X, y, random_state=7)

# Data is split into an 80/20 format. This is training and testing. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Using a linear kernel, train and create the sv regressor.
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)

# Regressor is run on test data.
y_test_pred = sv_regressor.predict(X_test)

# Print MSE and EVS.
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("Mean Squared Error (MSE):", mse)
print("Explained Variance Score (EVS):", evs)

# Set threshold to 12 and binarize the data
threshold = 12.0
y_test_pred_label = np.where(y_test_pred >= threshold, 1, 0)
y_test_label      = np.where(y_test >= threshold,      1, 0)

# Make confusion matrix.
confusion_mat = confusion_matrix(y_test_label, y_test_pred_label)
print("\nConfusion Matrix:\n", confusion_mat)

# Visualize the confusion matrix.
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Show the results of the classification report.
report = classification_report(y_test_label, y_test_pred_label)
print("\nClassification Report:\n", report)
