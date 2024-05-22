# collaborative_filtering.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load processed training data
train_data_path = 'data/processed/train_data.csv'
test_data_path = 'data/processed/test_data.csv'

print("Loading processed data...")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Create user-item interaction matrix
print("Creating user-item interaction matrix...")
user_item_matrix = train_data.pivot(index='user_id', columns='product_id', values='interaction_value').fillna(0)

# Compute cosine similarity between users
print("Computing cosine similarity between users...")
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to make predictions based on user similarity
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings.T - mean_user_rating).T
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

# Predict user ratings
print("Predicting user ratings...")
user_predictions = predict(user_item_matrix.values, user_similarity_df.values, type='user')

# Convert predictions to a DataFrame
user_pred_df = pd.DataFrame(user_predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Evaluate the collaborative filtering model using RMSE
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, ground_truth))

print("Evaluating model...")
rmse_value = rmse(user_predictions, user_item_matrix.values)
print(f"RMSE: {rmse_value}")

# Save the user similarity matrix and predictions
print("Saving model...")
with open('models/user_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(user_similarity_df, f)

with open('models/user_predictions.pkl', 'wb') as f:
    pickle.dump(user_pred_df, f)

print("Collaborative filtering model training and evaluation completed!")
