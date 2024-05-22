# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Define file paths
user_data_path = 'data/raw/users.csv'
product_data_path = 'data/raw/products.csv'
interaction_data_path = 'data/raw/interactions.csv'

processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_train_data_path), exist_ok=True)

# Load Data
print("Loading data...")
user_data = pd.read_csv(user_data_path)
product_data = pd.read_csv(product_data_path)
interaction_data = pd.read_csv(interaction_data_path)

# Display the first few rows of each dataset
print("User Data:")
print(user_data.head())

print("Product Data:")
print(product_data.head())

print("Interaction Data:")
print(interaction_data.head())

# Data Cleaning
# Handling missing values
print("Handling missing values...")
user_data.fillna('', inplace=True)
product_data.fillna('', inplace=True)
interaction_data.fillna(0, inplace=True)

# Convert date columns to datetime
print("Converting date columns to datetime...")
interaction_data['date'] = pd.to_datetime(interaction_data['date'])

# Display the data types after conversion
print("Data types after conversion:")
print(interaction_data.dtypes)

# Feature Engineering
# Create new features based on existing data
print("Creating new features...")
user_data['age'] = 2023 - user_data['year_of_birth']
user_data.drop('year_of_birth', axis=1, inplace=True)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in ['gender', 'location']:
    label_encoders[column] = LabelEncoder()
    user_data[column] = label_encoders[column].fit_transform(user_data[column])

# Merge data for collaborative filtering
print("Merging data for collaborative filtering...")
merged_data = interaction_data.merge(user_data, on='user_id', how='left')
merged_data = merged_data.merge(product_data, on='product_id', how='left')

# Display the merged data
print("Merged Data:")
print(merged_data.head())

# Normalize numerical features
print("Normalizing numerical features...")
scaler = MinMaxScaler()
merged_data[['age', 'price']] = scaler.fit_transform(merged_data[['age', 'price']])

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Save processed data
print("Saving processed data...")
train_data.to_csv(processed_train_data_path, index=False)
test_data.to_csv(processed_test_data_path, index=False)

print("Data preprocessing completed!")
