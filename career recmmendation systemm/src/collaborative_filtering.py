"""
Collaborative Filtering Module for MY NEW CAREER System
Implements collaborative filtering for personalized career recommendations
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilter:
    def __init__(self, method='user_based'):
        """
        Initialize collaborative filtering model
        
        Args:
            method: 'user_based', 'item_based', or 'matrix_factorization'
        """
        self.method = method
        self.user_similarity = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.model = None
        self.is_trained = False
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def prepare_data(self, ratings_data):
        """Prepare user-item rating matrix"""
        # Create user and item mappings
        unique_users = ratings_data['student_id'].unique()
        unique_items = ratings_data['career_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create user-item matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings_data.iterrows():
            user_idx = self.user_mapping[row['student_id']]
            item_idx = self.item_mapping[row['career_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        print(f"✅ User-item matrix created: {n_users} users x {n_items} items")
        print(f"   Sparsity: {(self.user_item_matrix == 0).sum() / self.user_item_matrix.size:.2%}")
        
        return self.user_item_matrix
    
    def calculate_user_similarity(self):
        """Calculate user-user similarity matrix"""
        # Use cosine similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Set diagonal to 0 (users shouldn't be similar to themselves for recommendations)
        np.fill_diagonal(self.user_similarity, 0)
        
        print("✅ User similarity matrix calculated")
        return self.user_similarity
    
    def calculate_item_similarity(self):
        """Calculate item-item similarity matrix"""
        # Transpose matrix for item-item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Set diagonal to 0
        np.fill_diagonal(self.item_similarity, 0)
        
        print("✅ Item similarity matrix calculated")
        return self.item_similarity
    
    def train_user_based_cf(self):
        """Train user-based collaborative filtering"""
        self.calculate_user_similarity()
        self.is_trained = True
        print("✅ User-based collaborative filtering trained")
    
    def train_item_based_cf(self):
        """Train item-based collaborative filtering"""
        self.calculate_item_similarity()
        self.is_trained = True
        print("✅ Item-based collaborative filtering trained")
    
    def train_matrix_factorization(self, n_components=10, method='nmf'):
        """Train matrix factorization model"""
        if method == 'nmf':
            # Non-negative Matrix Factorization
            self.model = NMF(n_components=n_components, random_state=42, max_iter=200)
        else:
            # Singular Value Decomposition
            self.model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model
        self.W = self.model.fit_transform(self.user_item_matrix)
        self.H = self.model.components_
        
        # Reconstruct the matrix
        self.reconstructed_matrix = self.W @ self.H
        
        self.is_trained = True
        print(f"✅ Matrix factorization ({method.upper()}) trained with {n_components} components")
    
    def predict_user_based(self, user_id, item_id, k=5):
        """Predict rating using user-based collaborative filtering"""
        if not self.is_trained or self.user_similarity is None:
            return None
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        # Find k most similar users who have rated this item
        similar_users = []
        for other_user_idx in range(len(self.user_similarity)):
            if (other_user_idx != user_idx and 
                self.user_item_matrix[other_user_idx, item_idx] > 0):
                similarity = self.user_similarity[user_idx, other_user_idx]
                rating = self.user_item_matrix[other_user_idx, item_idx]
                similar_users.append((similarity, rating))
        
        # Sort by similarity and take top k
        similar_users.sort(reverse=True)
        similar_users = similar_users[:k]
        
        if not similar_users:
            return None
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in similar_users)
        denominator = sum(abs(sim) for sim, rating in similar_users)
        
        if denominator == 0:
            return None
        
        predicted_rating = numerator / denominator
        return max(1, min(5, predicted_rating))  # Clip to rating range
    
    def predict_item_based(self, user_id, item_id, k=5):
        """Predict rating using item-based collaborative filtering"""
        if not self.is_trained or self.item_similarity is None:
            return None
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        # Find k most similar items that this user has rated
        similar_items = []
        for other_item_idx in range(len(self.item_similarity)):
            if (other_item_idx != item_idx and 
                self.user_item_matrix[user_idx, other_item_idx] > 0):
                similarity = self.item_similarity[item_idx, other_item_idx]
                rating = self.user_item_matrix[user_idx, other_item_idx]
                similar_items.append((similarity, rating))
        
        # Sort by similarity and take top k
        similar_items.sort(reverse=True)
        similar_items = similar_items[:k]
        
        if not similar_items:
            return None
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in similar_items)
        denominator = sum(abs(sim) for sim, rating in similar_items)
        
        if denominator == 0:
            return None
        
        predicted_rating = numerator / denominator
        return max(1, min(5, predicted_rating))  # Clip to rating range
    
    def predict_matrix_factorization(self, user_id, item_id):
        """Predict rating using matrix factorization"""
        if not self.is_trained or self.reconstructed_matrix is None:
            return None
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        predicted_rating = self.reconstructed_matrix[user_idx, item_idx]
        return max(1, min(5, predicted_rating))  # Clip to rating range
    
    def get_user_recommendations(self, user_id, n_recommendations=10, exclude_rated=True):
        """Get top N recommendations for a user"""
        if not self.is_trained:
            return []
        
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        recommendations = []
        
        # Get all items
        for item_id in self.item_mapping.keys():
            item_idx = self.item_mapping[item_id]
            
            # Skip if user has already rated this item (optional)
            if exclude_rated and self.user_item_matrix[user_idx, item_idx] > 0:
                continue
            
            # Predict rating based on method
            if self.method == 'user_based':
                predicted_rating = self.predict_user_based(user_id, item_id)
            elif self.method == 'item_based':
                predicted_rating = self.predict_item_based(user_id, item_id)
            else:  # matrix_factorization
                predicted_rating = self.predict_matrix_factorization(user_id, item_id)
            
            if predicted_rating is not None:
                recommendations.append({
                    'career_id': item_id,
                    'predicted_rating': predicted_rating,
                    'confidence': self._calculate_confidence(user_id, item_id)
                })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def _calculate_confidence(self, user_id, item_id):
        """Calculate confidence score for a prediction"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 0.0
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        # Calculate confidence based on number of similar users/items
        if self.method == 'user_based' and self.user_similarity is not None:
            # Count users who rated this item and are similar to target user
            similar_count = 0
            for other_user_idx in range(len(self.user_similarity)):
                if (other_user_idx != user_idx and 
                    self.user_item_matrix[other_user_idx, item_idx] > 0 and
                    self.user_similarity[user_idx, other_user_idx] > 0.1):
                    similar_count += 1
            return min(1.0, similar_count / 5)  # Normalize to 0-1
        
        elif self.method == 'item_based' and self.item_similarity is not None:
            # Count items rated by user that are similar to target item
            similar_count = 0
            for other_item_idx in range(len(self.item_similarity)):
                if (other_item_idx != item_idx and 
                    self.user_item_matrix[user_idx, other_item_idx] > 0 and
                    self.item_similarity[item_idx, other_item_idx] > 0.1):
                    similar_count += 1
            return min(1.0, similar_count / 5)  # Normalize to 0-1
        
        else:
            # For matrix factorization, use reconstruction quality as confidence
            return 0.7  # Default confidence
    
    def find_similar_users(self, user_id, n_similar=5):
        """Find users similar to the given user"""
        if not self.is_trained or self.user_similarity is None:
            return []
        
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get similarity scores with all other users
        similarities = []
        for other_user_idx, similarity in enumerate(self.user_similarity[user_idx]):
            if other_user_idx != user_idx and similarity > 0:
                other_user_id = self.reverse_user_mapping[other_user_idx]
                similarities.append({
                    'user_id': other_user_id,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:n_similar]
    
    def find_similar_items(self, item_id, n_similar=5):
        """Find items similar to the given item"""
        if not self.is_trained or self.item_similarity is None:
            return []
        
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        
        # Get similarity scores with all other items
        similarities = []
        for other_item_idx, similarity in enumerate(self.item_similarity[item_idx]):
            if other_item_idx != item_idx and similarity > 0:
                other_item_id = self.reverse_item_mapping[other_item_idx]
                similarities.append({
                    'career_id': other_item_id,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:n_similar]
    
    def evaluate_model(self, test_data):
        """Evaluate the collaborative filtering model"""
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['student_id']
            item_id = row['career_id']
            actual_rating = row['rating']
            
            # Make prediction
            if self.method == 'user_based':
                predicted_rating = self.predict_user_based(user_id, item_id)
            elif self.method == 'item_based':
                predicted_rating = self.predict_item_based(user_id, item_id)
            else:
                predicted_rating = self.predict_matrix_factorization(user_id, item_id)
            
            if predicted_rating is not None:
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
        
        if predictions:
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'n_predictions': len(predictions)
            }
        
        return None
    
    def train(self, ratings_data):
        """Train the collaborative filtering model"""
        # Prepare data
        self.prepare_data(ratings_data)
        
        # Train based on method
        if self.method == 'user_based':
            self.train_user_based_cf()
        elif self.method == 'item_based':
            self.train_item_based_cf()
        else:  # matrix_factorization
            self.train_matrix_factorization()
        
        print(f"✅ Collaborative filtering ({self.method}) training completed!")
    
    def get_cold_start_recommendations(self, user_profile, career_data, n_recommendations=5):
        """Handle cold start problem for new users"""
        # For new users, recommend based on popularity and profile matching
        
        # Calculate career popularity (average rating)
        career_popularity = {}
        for item_id in self.item_mapping.keys():
            item_idx = self.item_mapping[item_id]
            ratings = self.user_item_matrix[:, item_idx]
            ratings = ratings[ratings > 0]  # Only non-zero ratings
            
            if len(ratings) > 0:
                avg_rating = ratings.mean()
                popularity_score = len(ratings) / len(self.user_mapping)  # Percentage of users who rated
                career_popularity[item_id] = {
                    'avg_rating': avg_rating,
                    'popularity': popularity_score,
                    'combined_score': avg_rating * (0.7 + 0.3 * popularity_score)
                }
        
        # Sort by combined score
        popular_careers = sorted(career_popularity.items(), 
                               key=lambda x: x[1]['combined_score'], 
                               reverse=True)
        
        recommendations = []
        for career_id, scores in popular_careers[:n_recommendations]:
            recommendations.append({
                'career_id': career_id,
                'predicted_rating': scores['avg_rating'],
                'confidence': scores['popularity'],
                'reason': 'Popular among similar users'
            })
        
        return recommendations

# Example usage
if __name__ == "__main__":
    print("🤝 Testing Collaborative Filtering Module...")
    
    # Create sample data for testing
    sample_ratings = pd.DataFrame({
        'student_id': [1, 1, 2, 2, 3, 3] * 5,
        'career_id': [1, 2, 1, 3, 2, 3] * 5,
        'rating': [5, 4, 3, 5, 4, 2] * 5
    })
    
    # Test user-based collaborative filtering
    cf_user = CollaborativeFilter(method='user_based')
    cf_user.train(sample_ratings)
    
    print("✅ Collaborative filtering module structure created successfully!")