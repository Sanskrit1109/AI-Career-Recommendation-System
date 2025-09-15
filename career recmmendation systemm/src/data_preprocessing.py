"""
Data Preprocessing Module for MY NEW CAREER System
Handles data loading, cleaning, and preparation for ML models
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        """Initialize the data preprocessor"""
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all datasets"""
        try:
            # Load student profiles
            self.student_profiles = pd.read_csv(
                os.path.join(self.data_dir, "student_profiles.csv")
            )
            
            # Load career database
            self.career_database = pd.read_csv(
                os.path.join(self.data_dir, "career_database.csv")
            )
            
            # Load user ratings
            self.user_ratings = pd.read_csv(
                os.path.join(self.data_dir, "user_ratings.csv")
            )
            
            print("✅ All datasets loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def clean_text_data(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters except pipes (used as separators)
        text = re.sub(r'[^\w\s\|]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_skills(self, skills_text):
        """Process skills column to create skill lists"""
        if pd.isna(skills_text):
            return []
        
        # Split by pipe and clean each skill
        skills = [skill.strip().lower() for skill in str(skills_text).split('|')]
        return [skill for skill in skills if skill]  # Remove empty strings
    
    def preprocess_student_data(self):
        """Preprocess student profile data"""
        df = self.student_profiles.copy()
        
        # Clean text columns
        text_columns = ['academic_background', 'skills', 'interests', 
                       'preferred_work_environment', 'career_experience', 
                       'personality_traits']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text_data)
        
        # Process skills into lists
        df['skills_list'] = df['skills'].apply(self.process_skills)
        df['interests_list'] = df['interests'].apply(self.process_skills)
        df['personality_list'] = df['personality_traits'].apply(self.process_skills)
        
        # Encode categorical variables
        categorical_columns = ['academic_background', 'preferred_work_environment']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Handle GPA - ensure it's numeric
        if 'gpa' in df.columns:
            df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
            df.loc[:, 'gpa'] = df['gpa'].fillna(df['gpa'].mean())
        
        self.processed_students = df
        print("✅ Student data preprocessed successfully!")
        return df
    
    def preprocess_career_data(self):
        """Preprocess career database"""
        df = self.career_database.copy()
        
        # Clean text columns
        text_columns = ['career_title', 'required_skills', 'industry', 
                       'education_requirements', 'work_environment', 'job_description']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text_data)
        
        # Process required skills into lists
        df['required_skills_list'] = df['required_skills'].apply(self.process_skills)
        
        # Process salary range
        if 'salary_range' in df.columns:
            df['min_salary'] = df['salary_range'].apply(self.extract_min_salary)
            df['max_salary'] = df['salary_range'].apply(self.extract_max_salary)
            df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
        
        # Encode categorical variables
        categorical_columns = ['industry', 'growth_outlook', 'work_environment']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[f'career_{col}'] = le
        
        self.processed_careers = df
        print("✅ Career data preprocessed successfully!")
        return df
    
    def extract_min_salary(self, salary_range):
        """Extract minimum salary from range string"""
        try:
            if pd.isna(salary_range):
                return 0
            
            # Extract numbers from salary range (e.g., "80000-150000")
            numbers = re.findall(r'\d+', str(salary_range))
            if numbers:
                return int(numbers[0])
            return 0
        except:
            return 0
    
    def extract_max_salary(self, salary_range):
        """Extract maximum salary from range string"""
        try:
            if pd.isna(salary_range):
                return 0
            
            # Extract numbers from salary range
            numbers = re.findall(r'\d+', str(salary_range))
            if len(numbers) >= 2:
                return int(numbers[1])
            elif len(numbers) == 1:
                return int(numbers[0])
            return 0
        except:
            return 0
    
    def create_skill_matrix(self):
        """Create a skills matrix for similarity calculations"""
        # Get all unique skills from both students and careers
        all_skills = set()
        
        # Add student skills
        for skills_list in self.processed_students['skills_list']:
            all_skills.update(skills_list)
        
        # Add career required skills
        for skills_list in self.processed_careers['required_skills_list']:
            all_skills.update(skills_list)
        
        self.all_skills = sorted(list(all_skills))
        
        # Create student skill matrix
        student_skill_matrix = []
        for _, row in self.processed_students.iterrows():
            skill_vector = [1 if skill in row['skills_list'] else 0 for skill in self.all_skills]
            student_skill_matrix.append(skill_vector)
        
        self.student_skill_matrix = np.array(student_skill_matrix)
        
        # Create career skill matrix
        career_skill_matrix = []
        for _, row in self.processed_careers.iterrows():
            skill_vector = [1 if skill in row['required_skills_list'] else 0 for skill in self.all_skills]
            career_skill_matrix.append(skill_vector)
        
        self.career_skill_matrix = np.array(career_skill_matrix)
        
        print(f"✅ Skill matrix created with {len(self.all_skills)} unique skills!")
        return self.student_skill_matrix, self.career_skill_matrix
    
    def prepare_training_data(self):
        """Prepare data for machine learning models"""
        # Merge user ratings with student and career data
        training_data = self.user_ratings.merge(
            self.processed_students[['student_id', 'academic_background_encoded', 
                                   'preferred_work_environment_encoded', 'gpa']],
            on='student_id'
        ).merge(
            self.processed_careers[['career_id', 'industry_encoded', 
                                  'growth_outlook_encoded', 'avg_salary']],
            on='career_id'
        )
        
        # Features for decision tree
        feature_columns = ['academic_background_encoded', 'preferred_work_environment_encoded', 
                          'gpa', 'industry_encoded', 'growth_outlook_encoded', 'avg_salary']
        
        X = training_data[feature_columns]
        y = training_data['rating']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.training_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns
        }
        
        print("✅ Training data prepared successfully!")
        return self.training_data
    
    def get_user_skill_vector(self, user_skills):
        """Convert user skills to vector format"""
        if isinstance(user_skills, str):
            user_skills = self.process_skills(user_skills)
        
        skill_vector = [1 if skill.lower() in [s.lower() for s in user_skills] else 0 
                       for skill in self.all_skills]
        return np.array(skill_vector)
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 Starting data preprocessing pipeline...")
        
        # Load data
        if not self.load_data():
            return False
        
        # Preprocess data
        self.preprocess_student_data()
        self.preprocess_career_data()
        
        # Create skill matrices
        self.create_skill_matrix()
        
        # Prepare training data
        self.prepare_training_data()
        
        print("🎉 Data preprocessing completed successfully!")
        return True
    
    def get_summary(self):
        """Get a summary of the preprocessed data"""
        summary = {
            'total_students': len(self.processed_students),
            'total_careers': len(self.processed_careers),
            'total_ratings': len(self.user_ratings),
            'unique_skills': len(self.all_skills),
            'academic_backgrounds': self.processed_students['academic_background'].nunique(),
            'industries': self.processed_careers['industry'].nunique()
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    success = preprocessor.run_preprocessing()
    
    if success:
        # Display summary
        summary = preprocessor.get_summary()
        print("\n📊 Data Summary:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")