"""
Main Recommendation Engine for MY NEW CAREER System
Combines NLP, Decision Tree, and Collaborative Filtering for comprehensive career recommendations
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from data_preprocessing import DataPreprocessor
    from nlp_processor import NLPProcessor
    from decision_tree_model import CareerDecisionTree
    from collaborative_filtering import CollaborativeFilter
except ImportError:
    # Try absolute imports if relative imports fail
    from src.data_preprocessing import DataPreprocessor
    from src.nlp_processor import NLPProcessor
    from src.decision_tree_model import CareerDecisionTree
    from src.collaborative_filtering import CollaborativeFilter

class CareerRecommendationEngine:
    def __init__(self, data_dir="data"):
        """Initialize the main recommendation engine"""
        self.data_dir = data_dir
        
        # Initialize components
        self.preprocessor = DataPreprocessor(data_dir)
        self.nlp_processor = NLPProcessor()
        self.decision_tree = CareerDecisionTree()
        self.collaborative_filter_user = CollaborativeFilter(method='user_based')
        self.collaborative_filter_item = CollaborativeFilter(method='item_based')
        self.collaborative_filter_mf = CollaborativeFilter(method='matrix_factorization')
        
        # Data storage
        self.student_data = None
        self.career_data = None
        self.ratings_data = None
        
        # Training status
        self.is_initialized = False
        self.is_trained = False
        
    def initialize_system(self):
        """Initialize and load all data"""
        print("🚀 Initializing MY NEW CAREER Recommendation System...")
        
        try:
            # Run data preprocessing
            success = self.preprocessor.run_preprocessing()
            if not success:
                return False
            
            # Store processed data
            self.student_data = self.preprocessor.processed_students
            self.career_data = self.preprocessor.processed_careers
            self.ratings_data = self.preprocessor.user_ratings
            
            # Initialize NLP components
            self._initialize_nlp()
            
            self.is_initialized = True
            print("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    def _initialize_nlp(self):
        """Initialize NLP components with career data"""
        try:
            # Prepare career descriptions for NLP processing
            career_descriptions = []
            for _, career in self.career_data.iterrows():
                description = f"{career['career_title']} {career['industry']} {career['required_skills']} {career['job_description']}"
                career_descriptions.append(description)
            
            # Create TF-IDF vectors
            self.nlp_processor.create_tfidf_vectors(career_descriptions)
            
            print("✅ NLP components initialized")
            
        except Exception as e:
            print(f"⚠️  NLP initialization warning: {str(e)}")
    
    def train_models(self):
        """Train all machine learning models"""
        if not self.is_initialized:
            print("❌ System not initialized. Please run initialize_system() first.")
            return False
        
        print("🔧 Training recommendation models...")
        
        try:
            # Train Decision Tree
            print("  Training Decision Tree model...")
            X, y, merged_data = self.decision_tree.prepare_features(
                self.student_data, self.career_data, self.ratings_data
            )
            self.decision_tree.train_model(X, y)
            
            # Train ensemble model
            self.decision_tree.train_ensemble_model(X, y)
            
            # Train Collaborative Filtering models
            print("  Training Collaborative Filtering models...")
            self.collaborative_filter_user.train(self.ratings_data)
            self.collaborative_filter_item.train(self.ratings_data)
            self.collaborative_filter_mf.train(self.ratings_data)
            
            self.is_trained = True
            print("✅ All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False
    
    def get_comprehensive_recommendations(self, user_profile: Dict, 
                                        n_recommendations: int = 10,
                                        include_explanations: bool = True) -> List[Dict]:
        """
        Get comprehensive career recommendations using all algorithms
        
        Args:
            user_profile: Dictionary containing user information
            n_recommendations: Number of recommendations to return
            include_explanations: Whether to include explanation for each recommendation
        
        Returns:
            List of career recommendations with scores and explanations
        """
        if not self.is_trained:
            print("❌ Models not trained. Please run train_models() first.")
            return []
        
        try:
            recommendations = []
            
            # Get recommendations from each algorithm
            nlp_recs = self._get_nlp_recommendations(user_profile, n_recommendations * 2)
            dt_recs = self._get_decision_tree_recommendations(user_profile, n_recommendations * 2)
            cf_recs = self._get_collaborative_filtering_recommendations(user_profile, n_recommendations * 2)
            
            # Combine and score recommendations
            career_scores = self._combine_recommendations(nlp_recs, dt_recs, cf_recs)
            
            # Sort by combined score
            sorted_careers = sorted(career_scores.items(), 
                                  key=lambda x: x[1]['combined_score'], 
                                  reverse=True)
            
            # Prepare final recommendations
            for career_id, scores in sorted_careers[:n_recommendations]:
                career_info = self.career_data[self.career_data['career_id'] == career_id].iloc[0]
                
                recommendation = {
                    'career_id': career_id,
                    'career_title': career_info['career_title'],
                    'industry': career_info['industry'],
                    'combined_score': scores['combined_score'],
                    'nlp_score': scores.get('nlp_score', 0),
                    'decision_tree_score': scores.get('dt_score', 0),
                    'collaborative_filtering_score': scores.get('cf_score', 0),
                    'salary_range': career_info['salary_range'],
                    'growth_outlook': career_info['growth_outlook'],
                    'required_skills': career_info['required_skills'],
                    'job_description': career_info['job_description']
                }
                
                if include_explanations:
                    recommendation['explanation'] = self._generate_explanation(
                        user_profile, career_info, scores
                    )
                    recommendation['skill_match'] = self._analyze_skill_match(
                        user_profile, career_info
                    )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {str(e)}")
            return []
    
    def _get_nlp_recommendations(self, user_profile: Dict, n_recs: int) -> Dict:
        """Get recommendations using NLP similarity"""
        try:
            # Create user description
            user_description = f"{user_profile.get('academic_background', '')} {user_profile.get('skills', '')} {user_profile.get('interests', '')} {user_profile.get('personality_traits', '')}"
            
            # Find similar careers
            similar_careers = self.nlp_processor.find_similar_careers(user_description, n_recs)
            
            # Convert to dictionary with career_id as key
            nlp_recs = {}
            for rec in similar_careers:
                career_idx = rec['career_index']
                if career_idx < len(self.career_data):
                    career_id = self.career_data.iloc[career_idx]['career_id']
                    nlp_recs[career_id] = {
                        'score': rec['similarity_score'],
                        'rank': rec['rank']
                    }
            
            return nlp_recs
            
        except Exception as e:
            print(f"⚠️  NLP recommendations error: {str(e)}")
            return {}
    
    def _get_decision_tree_recommendations(self, user_profile: Dict, n_recs: int) -> Dict:
        """Get recommendations using Decision Tree"""
        try:
            dt_recs = {}
            
            # Test each career for the user
            for _, career in self.career_data.iterrows():
                # Prepare combined profile
                combined_profile = {
                    **user_profile,
                    'industry': career['industry'],
                    'education_requirements': career['education_requirements'],
                    'salary_avg': career.get('avg_salary', 70000),
                    'growth_score': self.decision_tree._encode_growth(career['growth_outlook'])
                }
                
                # Get prediction
                prediction = self.decision_tree.predict_career_fit(combined_profile)
                
                if prediction:
                    dt_recs[career['career_id']] = {
                        'score': prediction['fit_probability'],
                        'ensemble_score': prediction.get('ensemble_probability', 0)
                    }
            
            return dt_recs
            
        except Exception as e:
            print(f"⚠️  Decision Tree recommendations error: {str(e)}")
            return {}
    
    def _get_collaborative_filtering_recommendations(self, user_profile: Dict, n_recs: int) -> Dict:
        """Get recommendations using Collaborative Filtering"""
        try:
            cf_recs = {}
            
            # Check if user exists in training data
            user_id = user_profile.get('student_id')
            
            if user_id and user_id in self.collaborative_filter_user.user_mapping:
                # Existing user - use collaborative filtering
                user_recommendations = self.collaborative_filter_user.get_user_recommendations(
                    user_id, n_recs
                )
                
                for rec in user_recommendations:
                    cf_recs[rec['career_id']] = {
                        'score': rec['predicted_rating'] / 5.0,  # Normalize to 0-1
                        'confidence': rec['confidence']
                    }
            else:
                # New user - use cold start recommendations
                cold_start_recs = self.collaborative_filter_user.get_cold_start_recommendations(
                    user_profile, self.career_data, n_recs
                )
                
                for rec in cold_start_recs:
                    cf_recs[rec['career_id']] = {
                        'score': rec['predicted_rating'] / 5.0,  # Normalize to 0-1
                        'confidence': rec['confidence']
                    }
            
            return cf_recs
            
        except Exception as e:
            print(f"⚠️  Collaborative Filtering recommendations error: {str(e)}")
            return {}
    
    def _combine_recommendations(self, nlp_recs: Dict, dt_recs: Dict, cf_recs: Dict) -> Dict:
        """Combine recommendations from all algorithms"""
        # Weights for different algorithms
        weights = {
            'nlp': 0.3,
            'dt': 0.4,
            'cf': 0.3
        }
        
        # Get all unique career IDs
        all_career_ids = set()
        all_career_ids.update(nlp_recs.keys())
        all_career_ids.update(dt_recs.keys())
        all_career_ids.update(cf_recs.keys())
        
        combined_scores = {}
        
        for career_id in all_career_ids:
            # Get scores from each algorithm (default to 0 if not present)
            nlp_score = nlp_recs.get(career_id, {}).get('score', 0)
            dt_score = dt_recs.get(career_id, {}).get('score', 0)
            cf_score = cf_recs.get(career_id, {}).get('score', 0)
            
            # Calculate weighted combination
            combined_score = (
                weights['nlp'] * nlp_score +
                weights['dt'] * dt_score +
                weights['cf'] * cf_score
            )
            
            # Boost score if recommended by multiple algorithms
            num_algorithms = sum([
                1 if nlp_score > 0 else 0,
                1 if dt_score > 0 else 0,
                1 if cf_score > 0 else 0
            ])
            
            if num_algorithms > 1:
                combined_score *= (1 + 0.1 * (num_algorithms - 1))  # 10% boost per additional algorithm
            
            combined_scores[career_id] = {
                'combined_score': combined_score,
                'nlp_score': nlp_score,
                'dt_score': dt_score,
                'cf_score': cf_score,
                'num_algorithms': num_algorithms
            }
        
        return combined_scores
    
    def _generate_explanation(self, user_profile: Dict, career_info: pd.Series, scores: Dict) -> str:
        """Generate explanation for why this career was recommended"""
        explanations = []
        
        # NLP-based explanation
        if scores.get('nlp_score', 0) > 0.5:
            explanations.append(f"Your background in {user_profile.get('academic_background', 'your field')} aligns well with this career")
        
        # Decision tree explanation
        if scores.get('dt_score', 0) > 0.6:
            explanations.append("Based on similar successful career transitions")
        
        # Collaborative filtering explanation
        if scores.get('cf_score', 0) > 0.6:
            explanations.append("Highly rated by users with similar profiles")
        
        # Skills match explanation
        user_skills = user_profile.get('skills', '').lower().split('|')
        required_skills = career_info['required_skills'].lower().split('|')
        matching_skills = set(user_skills) & set(required_skills)
        
        if matching_skills:
            explanations.append(f"You have {len(matching_skills)} matching skills including {list(matching_skills)[0]}")
        
        # Growth and salary explanation
        if career_info['growth_outlook'].lower() in ['high', 'very high']:
            explanations.append("Excellent growth prospects in this field")
        
        return " • ".join(explanations) if explanations else "Good overall fit based on multiple factors"
    
    def _analyze_skill_match(self, user_profile: Dict, career_info: pd.Series) -> Dict:
        """Analyze skill match between user and career"""
        user_skills_str = user_profile.get('skills', '')
        required_skills_str = career_info['required_skills']
        
        return self.nlp_processor.analyze_skill_gaps(user_skills_str, required_skills_str)
    
    def get_career_details(self, career_id: int) -> Optional[Dict]:
        """Get detailed information about a specific career"""
        try:
            career = self.career_data[self.career_data['career_id'] == career_id]
            if career.empty:
                return None
            
            career_info = career.iloc[0]
            
            # Get additional insights
            similar_careers = self.collaborative_filter_item.find_similar_items(career_id, 5)
            
            details = {
                'career_id': career_id,
                'career_title': career_info['career_title'],
                'industry': career_info['industry'],
                'required_skills': career_info['required_skills'],
                'job_description': career_info['job_description'],
                'salary_range': career_info['salary_range'],
                'growth_outlook': career_info['growth_outlook'],
                'education_requirements': career_info['education_requirements'],
                'work_environment': career_info['work_environment'],
                'similar_careers': similar_careers,
                'average_rating': self._get_average_rating(career_id)
            }
            
            return details
            
        except Exception as e:
            print(f"❌ Error getting career details: {str(e)}")
            return None
    
    def _get_average_rating(self, career_id: int) -> float:
        """Get average rating for a career"""
        career_ratings = self.ratings_data[self.ratings_data['career_id'] == career_id]
        if not career_ratings.empty:
            return career_ratings['rating'].mean()
        return 0.0
    
    def save_models(self, models_dir="models"):
        """Save all trained models"""
        if not self.is_trained:
            print("❌ No trained models to save!")
            return False
        
        try:
            os.makedirs(models_dir, exist_ok=True)
            
            # Save decision tree
            dt_path = os.path.join(models_dir, "decision_tree_model.joblib")
            self.decision_tree.save_model(dt_path)
            
            print(f"✅ Models saved to {models_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        if not self.is_initialized:
            return {}
        
        stats = {
            'total_students': len(self.student_data),
            'total_careers': len(self.career_data),
            'total_ratings': len(self.ratings_data),
            'average_rating': self.ratings_data['rating'].mean(),
            'unique_skills': len(self.preprocessor.all_skills),
            'unique_industries': self.career_data['industry'].nunique(),
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            stats['decision_tree_accuracy'] = getattr(self.decision_tree, 'training_results', {}).get('accuracy', 0)
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("🎯 Testing MY NEW CAREER Recommendation Engine...")
    
    # Initialize the system
    engine = CareerRecommendationEngine()
    
    # Test initialization
    if engine.initialize_system():
        print("✅ System initialized successfully!")
        
        # Test training
        if engine.train_models():
            print("✅ Models trained successfully!")
            
            # Test recommendation
            sample_user = {
                'student_id': 999,  # New user
                'academic_background': 'Computer Science',
                'skills': 'Python|Machine Learning|Data Analysis',
                'interests': 'Technology|Problem Solving',
                'gpa': 3.8,
                'preferred_work_environment': 'Tech Company',
                'personality_traits': 'Analytical|Creative'
            }
            
            recommendations = engine.get_comprehensive_recommendations(sample_user, 5)
            
            if recommendations:
                print(f"✅ Generated {len(recommendations)} recommendations!")
                print("Top recommendation:", recommendations[0]['career_title'])
            
            # Print system stats
            stats = engine.get_system_stats()
            print("\n📊 System Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
    print("✅ MY NEW CAREER Recommendation Engine ready for deployment!")