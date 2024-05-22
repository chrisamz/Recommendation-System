# content_based_filtering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Define file paths
product_data_path = 'data/raw/products.csv'
interaction_data_path = 'data/raw/interactions.csv'

cosine_sim_matrix_path = 'models/cosine_similarity_matrix.npy'

# Create directories if they don't exist
os.makedirs(os.path.dirname(cosine_sim_matrix_path), exist_ok=True)

# Load Data
print("Loading data...")
product_data = pd.read_csv(product_data_path)
interaction_data = pd.read_csv(interaction_data_path)

# Display the first few rows of each dataset
print("Product Data:")
print(product_data.head())

print("Interaction Data:")
print(interaction_data.head())

# Data Cleaning
# Fill missing values
print("Handling missing values...")
product_data.fillna('', inplace=True)

# Feature Engineering
# Combine relevant features into a single string
print("Creating combined feature for TF-IDF...")
product_data['combined_features'] = product_data['title'] + ' ' + product_data['category'] + ' ' + product_data['description']

# Vectorize text data using TF-IDF
print("Vectorizing text data...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_data['combined_features'])

# Compute cosine similarity matrix
print("Computing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get product recommendations
def get_recommendations(product_id, cosine_sim=cosine_sim):
    idx = product_data[product_data['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar products
    product_indices = [i[0] for i in sim_scores]
    return product_data.iloc[product_indices]

# Test the recommendation system with a sample product
sample_product_id = product_data['product_id'].iloc[0]  # Replace with a product ID from your dataset
print("Product ID:", sample_product_id)
recommended_products = get_recommendations(sample_product_id)
print("Recommended Products:")
print(recommended_products)

# Save the cosine similarity matrix
print("Saving cosine similarity matrix...")
np.save(cosine_sim_matrix_path, cosine_sim)

print("Content-based filtering model completed!")
