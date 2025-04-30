
# loading the diabetes datasets to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# printing the first 5 rows of the dataset
diabetes_dataset.head()

#number of rows and columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

# Splitting data into features and target variable
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X.shape, X_train.shape, X_test.shape)

# Define models
models = {
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Voting Classifier': VotingClassifier(estimators=[('svm', SVC()), ('rf', RandomForestClassifier()), ('lr', LogisticRegression())]),
    'XGBoost': XGBClassifier()
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    
     # Deep Learning Model
# Build the model
deep_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
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