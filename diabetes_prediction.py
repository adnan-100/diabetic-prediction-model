# -*- coding: utf-8 -*-
"""Diabetes Prediction with Deep Learning Added"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print(os.path.exists('path/to/my_model.keras'))  # Check if file exists
import os
print(os.getcwd())  # This will print the current working directory




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers 



# loading the diabetes datasets to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define traditional ML models
models = {
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'XGBoost': XGBClassifier()
}
deep_model = tf.keras.models.load_model('deep_model.h5')


accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# Deep Learning Model
# Build the model
deep_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
deep_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# Evaluate the model
deeplearning_loss, deeplearning_accuracy = deep_model.evaluate(X_test, y_test, verbose=0)
accuracies['Deep Learning (Neural Network)'] = deeplearning_accuracy

# Find the best model
best_model = max(accuracies, key=accuracies.get)

# Print accuracy of each model and the best model
print("Model Accuracies:")
for model, acc in accuracies.items():
    print(f"{model}: {acc:.4f}")

print(f"\nBest Model: {best_model} with accuracy {accuracies[best_model]:.4f}")

# Save AdaBoost model
AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, y_train)
pickle.dump(AdaBoost, open('model.pkl', 'wb'))

# Make prediction for a sample input using AdaBoost
input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = AdaBoost.predict(std_data)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

# Create and train ensemble model
ensemble_estimators = [(name, model) for name, model in models.items()]
ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {ensemble_accuracy:.4f}')

# Save Deep Learning model separately
model.save("C:/Users/HP/OneDrive/Desktop/diabetic prediction/my_model.keras", save_format="keras_v3")

