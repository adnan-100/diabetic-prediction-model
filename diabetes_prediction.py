# -*- coding: utf-8 -*-
"""Diabetes Prediction with Deep Learning Added"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras import layers 

import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


diabetes_dataset = pd.read_csv('diabetes_expand.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Class distribution before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train))

models = {
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'XGBoost': XGBClassifier()
}
deep_model = tf.keras.models.load_model('deep_model.h5')


metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

# Deep Learning Model
deep_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

deep_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

deeplearning_pred = (deep_model.predict(X_test) > 0.5).astype(int)
metrics['Deep Learning (Neural Network)'] = {
    'accuracy': accuracy_score(y_test, deeplearning_pred),
    'precision': precision_score(y_test, deeplearning_pred),
    'recall': recall_score(y_test, deeplearning_pred),
    'f1_score': f1_score(y_test, deeplearning_pred)
}

print("\nModel Performance Metrics:")
for model, scores in metrics.items():
    print(f"{model} -> Accuracy: {scores['accuracy']:.4f}, Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F1 Score: {scores['f1_score']:.4f}")

best_model = max(metrics, key=lambda m: metrics[m]['accuracy'])
print(f"\nBest Model: {best_model} with Accuracy {metrics[best_model]['accuracy']:.4f}")

AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, y_train)
pickle.dump(AdaBoost, open('model.pkl', 'wb'))


input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31 ,2 ,2 ,85 , 0.032258065)
input_df = pd.DataFrame([input_data], columns=X.columns)  
std_data = scaler.transform(input_df)
prediction = AdaBoost.predict(std_data)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')


ensemble_estimators = [(name, model) for name, model in models.items()]
ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f'\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}')

deep_model.save("C:/Users/HP/OneDrive/Desktop/diabetic prediction/my_model.keras", save_format="keras_v3")


