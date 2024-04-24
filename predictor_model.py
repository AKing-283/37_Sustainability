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


X = data_scaled[:, :-3]  
y = data_scaled[:, -3:]  

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

# Make predictions
predictions = model.predict(X_test)

# Create a DataFrame with predictions and original features
predictions_df = pd.DataFrame(np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions], axis=1),
                              columns=data.columns[:-3].tolist() + ['predicted_c1', 'predicted_c2', 'predicted_c3'])

# Export the DataFrame to a CSV file
predictions_df.to_csv("predictions.csv", index=False)

