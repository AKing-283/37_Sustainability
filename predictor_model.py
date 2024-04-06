import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
​
# Read data from the Excel file
data = pd.read_excel("/kaggle/input/content/all_wind_power_data.xlsx")
​
# Drop the datetime column
data = data.drop(columns=['DateTime'])
​
# Assume the last column contains the target labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target labels
​
# Encode categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Initialize and train the model
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
​
# Get the best model
best_rf_model = grid_search.best_estimator_
​
# Make predictions
y_pred = best_rf_model.predict(X_test)
​
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)
​
# Print classification report for more detailed evaluation
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
​
