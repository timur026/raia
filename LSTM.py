import tensorflow as tf

# Specify the GPU device settings
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.set_logical_device_configuration(
#         physical_devices[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Set memory limit as desired


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import winsound
import time
import matplotlib.pyplot as plt

# Specify the file path of the CSV file
csv_file_path = 'transactions_rev4.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Separate the input features (procedure_area, has_parking, and instance_date) and the target variable (actual_worth)
X_numerical = data[['procedure_area', 'has_parking']].values

num_categories = [8, 4, 6, 15,
                  2, 15, 2,
                  29, 151, 62, 3,
                  139, 1362, 2631]
X_categorical = data[['property_sub_type_en', 'property_usage_en', 'nearest_mall_en', 'rooms_en',
                      'reg_type_en', 'nearest_landmark_en', 'trans_group_en',
                      'procedure_name_en', 'area_name_en', 'nearest_metro_en', 'property_type_en',
                      'master_project_en', 'project_name_en', 'building_name_en']].values
y = data['actual_worth'].values

# Normalize the numerical input features (procedure_area and has_parking)
scaler = StandardScaler()
X_numerical_normalized = scaler.fit_transform(X_numerical)

# Convert instance_date to numerical representation
date_strings = data['instance_date'].values
timestamps = [datetime.strptime(date_string, "%Y-%m-%d").timestamp() for date_string in date_strings]

timestamps = np.array(timestamps).reshape(-1, 1)

# Perform one-hot encoding on the categorical inputs
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Concatenate the normalized numerical input features, timestamps, and the encoded categorical inputs
X = np.concatenate((X_numerical_normalized, timestamps, X_categorical_encoded), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Determine the size of hidden layers based on the total number of categories
hidden_layer_sizes = [50, 50, 50, 100, 100, 100]
print("hidden layer sizes:", hidden_layer_sizes)

# Create the Recurrent Neural Network (RNN) model with LSTM layer
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(hidden_layer_sizes[0], activation='relu', input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(hidden_layer_sizes[1], activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Start the timer
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=1024, verbose=1, validation_data=(X_test, y_test))

# End the timer and calculate the total time
total_time = time.time() - start_time

# Convert total time to HH:MM:SS format
total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))

# Evaluate the model on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("R^2 Score: {:.2f}%".format(r2 * 100))

# Print the total time
print("Total Time: {}".format(total_time_str))

# Play a sound signal
winsound.Beep(1000, 500)  # Plays a beep sound at 1000 Hz for 500 milliseconds

# Access the validation loss from the history object
val_loss = history.history['val_loss']

# Plot the validation loss
epochs = range(1, len(val_loss) + 1)
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()
