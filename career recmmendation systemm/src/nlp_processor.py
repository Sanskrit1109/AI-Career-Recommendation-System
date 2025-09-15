"""
Natural Language Processing Module for MY NEW CAREER System
Handles text processing, similarity analysis, and NLP-based recommendations
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

class NLPProcessor:
    def __init__(self):
        """Initialize the NLP processor"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.career_vectors = None
        self.student_vectors = None
        
        # Load spaCy model if available
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            print("⚠️  spaCy model not found. Using basic NLP processing.")
            self.nlp = None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_keywords(self, text, top_k=10):
        """Extract key terms from text"""
        if pd.isna(text) or text == "":
            return []
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if self.nlp:
            # Use spaCy for advanced keyword extraction
            doc = self.nlp(processed_text)
            
            # Extract nouns and adjectives as keywords
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ'] and 
                    not token.is_stop and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_)
            
            # Count frequency and return top keywords
            keyword_freq = Counter(keywords)
            return [word for word, _ in keyword_freq.most_common(top_k)]
        else:
            # Basic keyword extraction using frequency
            words = processed_text.split()
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_k)]
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        if pd.isna(text1) or pd.isna(text2) or text1 == "" or text2 == "":
            return 0.0
        
        # Preprocess both texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        if processed_text1 == "" or processed_text2 == "":
            return 0.0
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if pd.isna(text) or text == "":
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(str(text))
            return sentiment_scores
        except:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def create_tfidf_vectors(self, career_descriptions, student_descriptions=None):
        """Create TF-IDF vectors for career and student descriptions"""
        # Preprocess all descriptions
        processed_careers = [self.preprocess_text(desc) for desc in career_descriptions]
        
        # Filter out empty descriptions
        processed_careers = [desc for desc in processed_careers if desc.strip()]
        
        if not processed_careers:
            print("⚠️  No valid career descriptions found")
            return None, None
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit on career descriptions
        self.career_vectors = self.tfidf_vectorizer.fit_transform(processed_careers)
        
        # Transform student descriptions if provided
        if student_descriptions is not None:
            processed_students = [self.preprocess_text(desc) for desc in student_descriptions]
            processed_students = [desc if desc.strip() else "general skills experience" 
                                for desc in processed_students]
            self.student_vectors = self.tfidf_vectorizer.transform(processed_students)
        
        print(f"✅ TF-IDF vectors created: {self.career_vectors.shape[0]} careers, "
              f"{len(self.tfidf_vectorizer.get_feature_names_out())} features")
        
        return self.career_vectors, self.student_vectors
    
    def find_similar_careers(self, user_description, top_k=5):
        """Find careers similar to user description using NLP"""
        if self.tfidf_vectorizer is None or self.career_vectors is None:
            print("⚠️  TF-IDF vectors not initialized")
            return []
        
        # Preprocess user description
        processed_desc = self.preprocess_text(user_description)
        if not processed_desc.strip():
            processed_desc = "general skills experience"
        
        # Transform user description
        user_vector = self.tfidf_vectorizer.transform([processed_desc])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.career_vectors)[0]
        
        # Get top similar careers
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append({
                    'career_index': idx,
                    'similarity_score': similarities[idx],
                    'rank': len(results) + 1
                })
        
        return results
    
    def analyze_skill_gaps(self, user_skills, required_skills):
        """Analyze gaps between user skills and required skills"""
        if isinstance(user_skills, str):
            user_skills = [skill.strip().lower() for skill in user_skills.split('|')]
        if isinstance(required_skills, str):
            required_skills = [skill.strip().lower() for skill in required_skills.split('|')]
        
        user_skills_set = set(user_skills)
        required_skills_set = set(required_skills)
        
        # Find gaps and matches
        missing_skills = required_skills_set - user_skills_set
        matching_skills = required_skills_set & user_skills_set
        
        # Calculate match percentage
        if required_skills_set:
            match_percentage = len(matching_skills) / len(required_skills_set)
        else:
            match_percentage = 0.0
        
        return {
            'missing_skills': list(missing_skills),
            'matching_skills': list(matching_skills),
            'match_percentage': match_percentage,
            'skill_score': match_percentage
        }
    
    def generate_career_summary(self, career_info):
        """Generate a summary of career information using NLP"""
        summary_parts = []
        
        # Add career title
        if 'career_title' in career_info:
            summary_parts.append(f"Career: {career_info['career_title']}")
        
        # Add industry
        if 'industry' in career_info:
            summary_parts.append(f"Industry: {career_info['industry']}")
        
        # Add key skills
        if 'required_skills' in career_info:
            skills = career_info['required_skills'].split('|')[:3]  # Top 3 skills
            summary_parts.append(f"Key Skills: {', '.join(skills)}")
        
        # Add salary info
        if 'salary_range' in career_info:
            summary_parts.append(f"Salary Range: {career_info['salary_range']}")
        
        # Add growth outlook
        if 'growth_outlook' in career_info:
            summary_parts.append(f"Growth Outlook: {career_info['growth_outlook']}")
        
        return " | ".join(summary_parts)
    
    def extract_interests_from_text(self, text):
        """Extract potential interests from free text"""
        if pd.isna(text) or text == "":
            return []
        
        # Define interest-related keywords
        interest_keywords = [
            'technology', 'science', 'research', 'creativity', 'design', 'art',
            'business', 'finance', 'healthcare', 'education', 'environment',
            'social', 'engineering', 'programming', 'data', 'analysis',
            'management', 'leadership', 'communication', 'writing', 'music',
            'sports', 'travel', 'learning', 'innovation', 'problem solving'
        ]
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Find matching interests
        found_interests = []
        for keyword in interest_keywords:
            if keyword in processed_text.lower():
                found_interests.append(keyword)
        
        return found_interests
    
    def calculate_semantic_similarity(self, student_profile, career_profile):
        """Calculate semantic similarity between student and career profiles"""
        # Combine relevant text fields
        student_text = " ".join([
            str(student_profile.get('academic_background', '')),
            str(student_profile.get('skills', '')),
            str(student_profile.get('interests', '')),
            str(student_profile.get('personality_traits', ''))
        ])
        
        career_text = " ".join([
            str(career_profile.get('career_title', '')),
            str(career_profile.get('required_skills', '')),
            str(career_profile.get('job_description', '')),
            str(career_profile.get('industry', ''))
        ])
        
        # Calculate text similarity
        similarity = self.calculate_text_similarity(student_text, career_text)
        
        return similarity
    
    def process_feedback_sentiment(self, feedback_text):
        """Process user feedback to understand preferences"""
        if pd.isna(feedback_text) or feedback_text == "":
            return {'sentiment': 'neutral', 'score': 0.0, 'keywords': []}
        
        # Analyze sentiment
        sentiment_scores = self.analyze_sentiment(feedback_text)
        
        # Determine overall sentiment
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Extract keywords from feedback
        keywords = self.extract_keywords(feedback_text, top_k=5)
        
        return {
            'sentiment': sentiment,
            'score': sentiment_scores['compound'],
            'keywords': keywords,
            'detailed_scores': sentiment_scores
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize NLP processor
    nlp_processor = NLPProcessor()
    
    # Test text preprocessing
    sample_text = "I love working with Python programming and machine learning algorithms!"
    processed = nlp_processor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    
    # Test keyword extraction
    keywords = nlp_processor.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
    
    # Test sentiment analysis
    sentiment = nlp_processor.analyze_sentiment("I really enjoyed this career recommendation!")
    print(f"Sentiment: {sentiment}")
    
    # Test similarity calculation
    text1 = "software engineering programming"
    text2 = "computer science coding development"
    similarity = nlp_processor.calculate_text_similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity:.3f}")
    
    print("✅ NLP module tested successfully!")