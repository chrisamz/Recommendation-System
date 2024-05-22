# matrix_factorization.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
import os

# Define file paths
train_data_path = 'data/processed/train_data.csv'
test_data_path = 'data/processed/test_data.csv'
matrix_factorization_model_path = 'models/matrix_factorization_model.npz'

# Create directories if they don't exist
os.makedirs(os.path.dirname(matrix_factorization_model_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Create user-item interaction matrix
print("Creating user-item interaction matrix...")
user_item_matrix = train_data.pivot(index='user_id', columns='product_id', values='interaction_value').fillna(0)

# Normalize user-item interaction matrix by subtracting mean user ratings
print("Normalizing user-item interaction matrix...")
user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
user_item_matrix_normalized = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

# Perform Singular Value Decomposition
print("Performing Singular Value Decomposition (SVD)...")
U, sigma, Vt = svds(user_item_matrix_normalized, k=50)  # k is the number of latent factors
sigma = np.diag(sigma)

# Reconstruct the user-item interaction matrix
print("Reconstructing user-item interaction matrix...")
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Evaluate the matrix factorization model using RMSE
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, ground_truth))

print("Evaluating model...")
rmse_value = rmse(predicted_ratings, user_item_matrix.values)
print(f"RMSE: {rmse_value}")

# Save the matrix factorization model components
print("Saving matrix factorization model components...")
np.savez(matrix_factorization_model_path, U=U, sigma=sigma, Vt=Vt, user_ratings_mean=user_ratings_mean)

print("Matrix factorization model training and evaluation completed!")
