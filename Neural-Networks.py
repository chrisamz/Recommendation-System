# neural_networks.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Define file paths
train_data_path = 'data/processed/train_data.csv'
test_data_path = 'data/processed/test_data.csv'
neural_network_model_path = 'models/neural_network_model.h5'

# Create directories if they don't exist
os.makedirs(os.path.dirname(neural_network_model_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Encode user_ids and product_ids
print("Encoding user_ids and product_ids...")
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

train_data['user_id'] = user_encoder.fit_transform(train_data['user_id'])
train_data['product_id'] = product_encoder.fit_transform(train_data['product_id'])

test_data['user_id'] = user_encoder.transform(test_data['user_id'])
test_data['product_id'] = product_encoder.transform(test_data['product_id'])

# Prepare data for the neural network
num_users = train_data['user_id'].nunique()
num_products = train_data['product_id'].nunique()

X_train = [train_data['user_id'].values, train_data['product_id'].values]
y_train = train_data['interaction_value'].values

X_test = [test_data['user_id'].values, test_data['product_id'].values]
y_test = test_data['interaction_value'].values

# Build the neural network model
print("Building the neural network model...")

# Input layers
user_input = Input(shape=(1,), name='user_input')
product_input = Input(shape=(1,), name='product_input')

# Embedding layers
user_embedding = Embedding(input_dim=num_users, output_dim=50, name='user_embedding')(user_input)
product_embedding = Embedding(input_dim=num_products, output_dim=50, name='product_embedding')(product_input)

# Flatten embedding layers
user_vec = Flatten()(user_embedding)
product_vec = Flatten()(product_embedding)

# Concatenate user and product vectors
concat = Concatenate()([user_vec, product_vec])

# Add dense layers
dense = Dense(128, activation='relu')(concat)
dense = Dropout(0.5)(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dropout(0.5)(dense)
output = Dense(1)(dense)

# Compile the model
model = Model([user_input, product_input], output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
print("Training the model...")
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse_value}")

# Save the model
print("Saving the model...")
model.save(neural_network_model_path)

print("Neural network model training and evaluation completed!")
