import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data from the Excel file
data = pd.read_excel("/content/wind_power_gen_3months_validation_data.xlsx")

# Drop the DateTime column if not needed
data = data.drop(columns=['DateTime'])

# Convert DataFrame to numpy array
data_array = data.values

# Split the data into features and target
X = data_array[:, :-3]  # Features: energy produced data
y = data_array[:, -3:]  # Targets: grid stability, unit consumption, and price per unit

# Apply scaling to the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(1, input_shape=(X_train.shape[0], X_train.shape[1])),
    tf.keras.layers.Dense(3)  # Output layer for three targets
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Print the test loss
print("Test Loss:", test_loss)

# Calculate R-squared for each target
for i in range(3):
    train_r2 = r2_score(y_train[:, i], model.predict(X_train)[:, i])
    test_r2 = r2_score(y_test[:, i], predictions[:, i])
    print(f"Target {i+1} Train R-squared:", train_r2)
    print(f"Target {i+1} Test R-squared:", test_r2)
