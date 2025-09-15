"""
Decision Tree Model for MY NEW CAREER System
Implements decision tree classifier for career prediction based on student profiles
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class CareerDecisionTree:
    def __init__(self):
        """Initialize the Decision Tree model"""
        self.model = None
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.feature_importance = None
        self.is_trained = False
        
    def prepare_features(self, student_data, career_data, ratings_data):
        """Prepare features for decision tree training"""
        # Merge all data
        merged_data = ratings_data.merge(
            student_data[['student_id', 'academic_background', 'gpa', 
                         'preferred_work_environment', 'personality_traits']],
            on='student_id'
        ).merge(
            career_data[['career_id', 'career_title', 'industry', 
                        'growth_outlook', 'salary_range', 'education_requirements']],
            on='career_id'
        )
        
        # Create additional features
        merged_data['salary_avg'] = merged_data['salary_range'].apply(self._extract_avg_salary)
        merged_data['growth_score'] = merged_data['growth_outlook'].apply(self._encode_growth)
        merged_data['gpa_category'] = pd.cut(merged_data['gpa'], 
                                           bins=[0, 3.0, 3.5, 4.0], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Encode categorical features
        categorical_features = ['academic_background', 'preferred_work_environment',
                               'industry', 'education_requirements', 'gpa_category']
        
        for feature in categorical_features:
            if feature in merged_data.columns:
                le = LabelEncoder()
                merged_data[f'{feature}_encoded'] = le.fit_transform(merged_data[feature].astype(str))
                self.feature_encoders[feature] = le
        
        # Select features for training
        feature_columns = [f'{f}_encoded' for f in categorical_features if f in merged_data.columns]
        feature_columns.extend(['gpa', 'salary_avg', 'growth_score'])
        
        X = merged_data[feature_columns]
        
        # Create target variable (binary: high rating vs low rating)
        y = (merged_data['rating'] >= 4).astype(int)  # 1 for rating >= 4, 0 otherwise
        
        self.feature_names = feature_columns
        
        return X, y, merged_data
    
    def _extract_avg_salary(self, salary_range):
        """Extract average salary from range string"""
        try:
            if pd.isna(salary_range):
                return 70000  # Default salary
            
            # Extract numbers from salary range
            import re
            numbers = re.findall(r'\d+', str(salary_range))
            if len(numbers) >= 2:
                return (int(numbers[0]) + int(numbers[1])) / 2
            elif len(numbers) == 1:
                return int(numbers[0])
            return 70000
        except:
            return 70000
    
    def _encode_growth(self, growth_outlook):
        """Encode growth outlook to numeric score"""
        growth_mapping = {
            'very high': 5,
            'high': 4,
            'medium': 3,
            'low': 2,
            'very low': 1
        }
        
        if pd.isna(growth_outlook):
            return 3
        
        return growth_mapping.get(str(growth_outlook).lower(), 3)
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the decision tree model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train basic decision tree
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Store training results
        self.training_results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"✅ Decision Tree model trained successfully!")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return accuracy
    
    def optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using GridSearchCV"""
        param_grid = {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        print(f"✅ Hyperparameter optimization completed!")
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_
    
    def train_ensemble_model(self, X, y):
        """Train a Random Forest ensemble model for better performance"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.ensemble_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.ensemble_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store ensemble feature importance
        self.ensemble_feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ensemble_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Random Forest ensemble model trained!")
        print(f"   Ensemble accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def predict_career_fit(self, student_profile):
        """Predict if a student would be a good fit for careers"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        # Prepare student features
        student_features = self._prepare_student_features(student_profile)
        
        if student_features is None:
            return None
        
        # Make prediction using decision tree
        probability = self.model.predict_proba([student_features])[0]
        prediction = self.model.predict([student_features])[0]
        
        # If ensemble model exists, get ensemble prediction
        ensemble_prediction = None
        ensemble_probability = None
        if hasattr(self, 'ensemble_model'):
            ensemble_probability = self.ensemble_model.predict_proba([student_features])[0]
            ensemble_prediction = self.ensemble_model.predict([student_features])[0]
        
        return {
            'fit_prediction': prediction,
            'fit_probability': probability[1],  # Probability of good fit
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability': ensemble_probability[1] if ensemble_probability is not None else None
        }
    
    def _prepare_student_features(self, student_profile):
        """Prepare student profile features for prediction"""
        try:
            features = []
            
            # Encode categorical features
            categorical_features = ['academic_background', 'preferred_work_environment',
                                   'industry', 'education_requirements', 'gpa_category']
            
            for feature in categorical_features:
                if f'{feature}_encoded' in self.feature_names:
                    if feature in student_profile:
                        value = str(student_profile[feature])
                        if feature in self.feature_encoders:
                            try:
                                encoded_value = self.feature_encoders[feature].transform([value])[0]
                            except:
                                # Handle unseen categories
                                encoded_value = 0
                        else:
                            encoded_value = 0
                        features.append(encoded_value)
                    else:
                        features.append(0)  # Default value
            
            # Add numeric features
            if 'gpa' in self.feature_names:
                features.append(student_profile.get('gpa', 3.0))
            
            if 'salary_avg' in self.feature_names:
                features.append(student_profile.get('salary_avg', 70000))
            
            if 'growth_score' in self.feature_names:
                features.append(student_profile.get('growth_score', 3))
            
            return features
            
        except Exception as e:
            print(f"❌ Error preparing student features: {str(e)}")
            return None
    
    def get_decision_rules(self, max_depth=5):
        """Extract human-readable decision rules from the tree"""
        if not self.is_trained:
            return "Model not trained yet!"
        
        # Get tree rules in text format
        tree_rules = export_text(self.model, feature_names=self.feature_names, max_depth=max_depth)
        
        return tree_rules
    
    def analyze_feature_importance(self):
        """Analyze and return feature importance"""
        if not self.is_trained:
            return None
        
        return self.feature_importance
    
    def visualize_tree(self, max_depth=3, figsize=(15, 10)):
        """Visualize the decision tree"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 class_names=['Poor Fit', 'Good Fit'],
                 filled=True,
                 rounded=True,
                 max_depth=max_depth)
        plt.title("Career Recommendation Decision Tree")
        plt.tight_layout()
        return plt
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            print("❌ No trained model to save!")
            return False
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and related data
            model_data = {
                'decision_tree': self.model,
                'ensemble_model': getattr(self, 'ensemble_model', None),
                'feature_names': self.feature_names,
                'feature_encoders': self.feature_encoders,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, model_path)
            print(f"✅ Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['decision_tree']
            self.ensemble_model = model_data.get('ensemble_model')
            self.feature_names = model_data['feature_names']
            self.feature_encoders = model_data['feature_encoders']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation to evaluate model performance"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'min_accuracy': cv_scores.min(),
            'max_accuracy': cv_scores.max()
        }
        
        print(f"✅ Cross-validation completed ({cv}-fold)")
        print(f"   Mean accuracy: {results['mean_accuracy']:.3f} (+/- {results['std_accuracy']*2:.3f})")
        
        return results

# Example usage
if __name__ == "__main__":
    print("🌳 Testing Decision Tree Model...")
    
    # This would typically be called with real data
    dt_model = CareerDecisionTree()
    
    # Create some sample data for testing
    sample_student = {
        'academic_background': 'Computer Science',
        'gpa': 3.8,
        'preferred_work_environment': 'Tech Company',
        'industry': 'Technology',
        'education_requirements': "Bachelor's in Computer Science",
        'gpa_category': 'High',
        'salary_avg': 100000,
        'growth_score': 4
    }
    
    print("✅ Decision Tree module structure created successfully!")