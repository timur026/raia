import tensorflow as tf

# Specify the GPU device settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Set memory limit as desired

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import winsound
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Specify the file path of the CSV file
# csv_file_path = r"C:\Users\Timur\Desktop\Dubai_rev2\dubai_all_residential_only_sell_2022_2023.csv"
csv_file_path = r"C:\Users\Timur\Desktop\Dubai_rev2\Transactions.csv"

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path)

print("Records in dataset:", len(data))

# Separate the input features (procedure_area and categorical inputs) and the target variable (actual_worth)
X_numerical = data['procedure_area'].values

# Reshape 'X_numerical' to a 2D array to match the shape of 'timestamps'
X_numerical = X_numerical.reshape(-1, 1)

X_categorical = data[['property_sub_type_en', 'nearest_mall_en', 'rooms_en',
                      'reg_type_en', 'nearest_landmark_en', 'trans_group_en',
                      'procedure_name_en', 'area_name_en', 'nearest_metro_en',
                      'master_project_en', 'project_name_en', 'building_name_en',
                      'has_parking']].values
y = data['actual_worth'].values


# Convert instance_date to numerical representation
date_strings = data['instance_date'].values
timestamps = [datetime.strptime(date_string, "%Y-%m-%d").timestamp() for date_string in date_strings]
# timestamps = [datetime.strptime(date_string, "%d-%m-%Y").timestamp() for date_string in date_strings]

timestamps = np.array(timestamps).reshape(-1, 1)


# Normalize the numerical input features (procedure_area and instance_date)
scaler = StandardScaler()
X_numerical_normalized = scaler.fit_transform(np.concatenate((X_numerical, timestamps), axis=1))


# Perform one-hot encoding on the categorical inputs
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Concatenate the normalized numerical input feature and the encoded categorical inputs
X = np.concatenate((X_numerical_normalized, X_categorical_encoded), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Determine the size of hidden layers based on the total number of categories
hidden_layer_sizes = [50, 50, 100, 100, 100, 100]
# hidden_layer_sizes = [800, 400, 200, 100, 50, 25]

print("hidden layer sizes:", hidden_layer_sizes)

# Create the Feedforward Neural Network (FFNN) model
model = tf.keras.Sequential([
    # tf.keras.layers.SimpleRNN(hidden_layer_sizes[0], activation='relu', input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Apply dropout with a rate of 0.1 (10% of the neurons will be dropped)
    # tf.keras.layers.Dense(hidden_layer_sizes[1], activation='relu'),
    # tf.keras.layers.Dropout(0.2),  # Apply dropout with a rate of 0.1
    # tf.keras.layers.Dense(hidden_layer_sizes[2], activation='relu'),
    # tf.keras.layers.Dropout(0.2),  # Apply dropout with a rate of 0.1
    # tf.keras.layers.Dense(hidden_layer_sizes[3], activation='relu'),
    # tf.keras.layers.Dropout(0.1),  # Apply dropout with a rate of 0.1
    # tf.keras.layers.Dense(hidden_layer_sizes[4], activation='relu'),
    # tf.keras.layers.Dropout(0.1),  # Apply dropout with a rate of 0.1
    # tf.keras.layers.Dense(hidden_layer_sizes[5], activation='relu'),
    # tf.keras.layers.Dropout(0.1),  # Apply dropout with a rate of 0.1
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Start the timer
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

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

# Save the trained model
# model.save("model_ffnn.keras")
# print("Trained model saved.")

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
