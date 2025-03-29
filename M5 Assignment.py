# Import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split

# Load the dataset and shuffle the data so that you don't bias your analysis.
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets

# metadata
print(student_performance.metadata)

# variable information
print(student_performance.variables)

# shuffle the data to avoid bias
X, y = shuffle(X, y, random_state=7)

# Split the dataset into training and testing in an 80/20 format:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Create and train the Support Vector Regressor using a linear kernel.
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)

# Run the regressor on the testing data and predict the output (predicted labels).
y_test_pred = sv_regressor.predict(X_test)

# Evaluate the performance of the regressor and print the initial metrics.
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("Mean Squared Error (MSE):", mse)
print("Explained Variance Score (EVS):", evs)

# Binarize the predicted values & the actual values using threshold of 12.0.
threshold = 12.0
y_test_pred_label = np.where(y_test_pred >= threshold, 1, 0)
y_test_label      = np.where(y_test >= threshold,      1, 0)

# Create the confusion matrix using the predicted labes and the actual labels.
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

# Print the classification report based on the confusion matrix.
report = classification_report(y_test_label, y_test_pred_label)
print("\nClassification Report:\n", report)
