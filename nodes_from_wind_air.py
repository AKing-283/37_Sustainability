import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data from the Excel file
data = pd.read_excel("/content/all_wind_power_data.xlsx")

# Drop the DateTime column if not needed
data = data.drop(columns=['DateTime'])

# Define and fit the StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into features and target
X = data_scaled[:, :-3]  # Features: energy produced data
y = data_scaled[:, -3:]  # Targets: grid stability, unit consumption, and price per unit

# Reshape the input data for LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
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

# Calculate accuracy for each target (you may use different metrics for each target)
# Assuming accuracy means within a certain threshold for regression values
threshold = 0.1  # Define a threshold for considering predictions as accurate
accuracies = []
for i in range(3):
    accurate_predictions = np.abs(predictions[:, i] - y_test[:, i]) <= threshold
    accuracy = np.mean(accurate_predictions)
    accuracies.append(accuracy)
    print("Accuracy for Target {}: {:.2f}%".format(i+1, accuracy * 100))

# Calculate the total power generated (p)
total_power_generated = data['Total Power Generated'].sum()

# Define the percentages for each node
percentage_node1 = 0.20
percentage_node2 = 0.45
percentage_node3 = 0.35

# Calculate the power allocated to each node
power_node1 = total_power_generated * percentage_node1
power_node2 = total_power_generated * percentage_node2
power_node3 = total_power_generated * percentage_node3

print("Power for Node 1:", power_node1)
print("Power for Node 2:", power_node2)
print("Power for Node 3:", power_node3)

total_power = power_node1 + power_node2 + power_node3
print("The Total power is ",total_power)
