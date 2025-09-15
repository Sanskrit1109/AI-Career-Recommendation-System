"""
MY NEW CAREER - AI-Powered Career Recommendation System
Streamlit Web Application Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
try:
    from src.recommendation_engine import CareerRecommendationEngine
    from src.data_preprocessing import DataPreprocessor
    from src.nlp_processor import NLPProcessor
except ImportError:
    st.error("Could not import recommendation modules. Please ensure all files are in the correct location.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MY NEW CAREER - AI Career Recommendations",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 2em;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1.5em;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1em 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1em;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5em 0;
    }
    .skill-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.3em 0.6em;
        border-radius: 15px;
        margin: 0.2em;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def initialize_system():
    """Initialize the recommendation system"""
    if st.session_state.recommendation_engine is None:
        with st.spinner("🚀 Initializing MY NEW CAREER System..."):
            try:
                engine = CareerRecommendationEngine()
                success = engine.initialize_system()
                
                if success:
                    st.session_state.recommendation_engine = engine
                    st.session_state.system_initialized = True
                    return True
                else:
                    st.error("Failed to initialize the system. Please check the data files.")
                    return False
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
                return False
    return True

def train_models():
    """Train the ML models"""
    if not st.session_state.models_trained:
        with st.spinner("🔧 Training AI models... This may take a moment."):
            try:
                success = st.session_state.recommendation_engine.train_models()
                if success:
                    st.session_state.models_trained = True
                    return True
                else:
                    st.error("Failed to train models.")
                    return False
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                return False
    return True

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 MY NEW CAREER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Career Recommendation System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "📝 Get Career Recommendations", 
        "📊 Career Explorer",
        "🔍 Career Details",
        "📈 System Analytics",
        "ℹ️ About"
    ])
    
    # Initialize system if not already done
    if not st.session_state.system_initialized:
        if not initialize_system():
            st.stop()
    
    # Train models if not already done
    if not st.session_state.models_trained:
        if not train_models():
            st.stop()
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Get Career Recommendations":
        show_recommendation_page()
    elif page == "📊 Career Explorer":
        show_career_explorer()
    elif page == "🔍 Career Details":
        show_career_details()
    elif page == "📈 System Analytics":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display the home page"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to MY NEW CAREER! 🎉
        
        Your intelligent career guidance system powered by advanced AI algorithms including:
        
        - 🧠 **Natural Language Processing** for understanding your profile
        - 🌳 **Decision Trees** for intelligent career matching
        - 🤝 **Collaborative Filtering** for personalized recommendations
        - 📊 **Data Analytics** for comprehensive insights
        """)
    
    # Quick stats
    if st.session_state.recommendation_engine:
        stats = st.session_state.recommendation_engine.get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Careers", stats.get('total_careers', 0))
        with col2:
            st.metric("Student Profiles", stats.get('total_students', 0))
        with col3:
            st.metric("Unique Skills", stats.get('unique_skills', 0))
        with col4:
            st.metric("Industries", stats.get('unique_industries', 0))
    
    # Features section
    st.markdown("---")
    st.markdown("## 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Personalized Recommendations
        - Tailored career suggestions based on your unique profile
        - Multiple AI algorithms working together
        - Detailed explanations for each recommendation
        """)
        
        st.markdown("""
        ### 📊 Comprehensive Analysis
        - Skill gap analysis
        - Career growth insights
        - Salary expectations
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Career Exploration
        - Browse careers by industry
        - Compare different career paths
        - Discover similar careers
        """)
        
        st.markdown("""
        ### 📈 Data-Driven Insights
        - Success rate predictions
        - Industry trends
        - Educational requirements
        """)
    
    # Call to action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Get My Career Recommendations", type="primary", use_container_width=True):
            st.info("👈 Please use the sidebar navigation to go to 'Get Career Recommendations' page!")
            st.balloons()

def show_recommendation_page():
    """Display the career recommendation page"""
    
    st.markdown("## 📝 Get Your Personalized Career Recommendations")
    st.markdown("Fill out the form below to receive AI-powered career suggestions tailored to your profile.")
    
    # User input form
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎓 Academic Background")
            academic_background = st.selectbox(
                "Your Academic Field",
                ["Computer Science", "Business Administration", "Engineering", "Psychology", 
                 "Marketing", "Data Science", "Biology", "English Literature", "Finance",
                 "Art", "Music", "Education", "Medicine", "Law", "Other"]
            )
            
            gpa = st.slider("GPA (if applicable)", 2.0, 4.0, 3.5, 0.1)
            
            st.markdown("### 💼 Work Preferences")
            work_environment = st.selectbox(
                "Preferred Work Environment",
                ["Tech Company", "Corporate Office", "Healthcare Setting", "Educational Institution",
                 "Government Agency", "Non-profit", "Startup", "Remote Work", "Freelance"]
            )
            
            career_experience = st.text_area(
                "Previous Experience (internships, jobs, projects)",
                placeholder="Describe any relevant work experience, internships, or projects..."
            )
        
        with col2:
            st.markdown("### 🛠️ Skills")
            skills_input = st.text_area(
                "Your Skills (separate with | )",
                placeholder="e.g., Python|Data Analysis|Communication|Project Management",
                help="List your skills separated by the | symbol"
            )
            
            st.markdown("### 🎯 Interests")
            interests_input = st.text_area(
                "Your Interests (separate with | )",
                placeholder="e.g., Technology|Problem Solving|Creativity|Healthcare",
                help="List your interests separated by the | symbol"
            )
            
            st.markdown("### 🧠 Personality Traits")
            personality_input = st.text_area(
                "Your Personality Traits (separate with | )",
                placeholder="e.g., Analytical|Creative|Detail-oriented|Leadership",
                help="Describe your personality traits separated by the | symbol"
            )
        
        # Submit button
        submitted = st.form_submit_button("🎯 Get My Career Recommendations", type="primary")
        
        if submitted:
            # Validate inputs
            if not skills_input.strip():
                st.error("Please enter at least some skills.")
                return
            
            # Create user profile
            user_profile = {
                'student_id': 9999,  # New user ID
                'academic_background': academic_background,
                'skills': skills_input,
                'interests': interests_input or "General interests",
                'gpa': gpa,
                'preferred_work_environment': work_environment,
                'career_experience': career_experience or "Limited experience",
                'personality_traits': personality_input or "Well-rounded"
            }
            
            # Get recommendations
            with st.spinner("🤖 AI is analyzing your profile and generating recommendations..."):
                try:
                    recommendations = st.session_state.recommendation_engine.get_comprehensive_recommendations(
                        user_profile, n_recommendations=8, include_explanations=True
                    )
                    
                    if recommendations:
                        st.success(f"✅ Generated {len(recommendations)} personalized career recommendations!")
                        display_recommendations(recommendations, user_profile)
                    else:
                        st.warning("No recommendations could be generated. Please try adjusting your profile.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")

def display_recommendations(recommendations, user_profile):
    """Display the career recommendations"""
    
    st.markdown("---")
    st.markdown("## 🎯 Your Personalized Career Recommendations")
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"#{i+1} {rec['career_title']} - {rec['industry']}", expanded=(i < 3)):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Career details
                st.markdown(f"**Industry:** {rec['industry']}")
                st.markdown(f"**Salary Range:** {rec['salary_range']}")
                st.markdown(f"**Growth Outlook:** {rec['growth_outlook']}")
                
                # Job description
                st.markdown("**Description:**")
                st.write(rec['job_description'])
                
                # Explanation
                st.markdown("**Why this career fits you:**")
                st.info(rec['explanation'])
                
                # Required skills
                st.markdown("**Required Skills:**")
                skills = rec['required_skills'].split('|')
                skill_html = " ".join([f'<span class="skill-tag">{skill.strip()}</span>' for skill in skills])
                st.markdown(skill_html, unsafe_allow_html=True)
            
            with col2:
                # Scores visualization
                fig = go.Figure(go.Bar(
                    x=['Combined', 'NLP', 'Decision Tree', 'Collaborative'],
                    y=[rec['combined_score'], rec['nlp_score'], 
                       rec['decision_tree_score'], rec['collaborative_filtering_score']],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                ))
                fig.update_layout(
                    title="Recommendation Scores",
                    xaxis_title="Algorithm",
                    yaxis_title="Score",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"scores_chart_{i}")
                
                # Skill match analysis
                if 'skill_match' in rec:
                    skill_match = rec['skill_match']
                    match_percentage = skill_match['match_percentage'] * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = match_percentage,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Skill Match"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True, key=f"skill_gauge_{i}")
                    
                    if skill_match['missing_skills']:
                        st.markdown("**Skills to develop:**")
                        for skill in skill_match['missing_skills'][:3]:
                            st.markdown(f"• {skill}")

def show_career_explorer():
    """Display the career explorer page"""
    
    st.markdown("## 📊 Career Explorer")
    st.markdown("Browse and explore different careers by industry, skills, and other criteria.")
    
    if st.session_state.recommendation_engine:
        career_data = st.session_state.recommendation_engine.career_data
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            industries = ['All'] + sorted(career_data['industry'].unique().tolist())
            selected_industry = st.selectbox("Filter by Industry", industries)
        
        with col2:
            growth_outlooks = ['All'] + sorted(career_data['growth_outlook'].unique().tolist())
            selected_growth = st.selectbox("Filter by Growth Outlook", growth_outlooks)
        
        with col3:
            salary_range = st.slider("Minimum Average Salary (K)", 0, 200, 0, 10)
        
        # Apply filters
        filtered_data = career_data.copy()
        
        if selected_industry != 'All':
            filtered_data = filtered_data[filtered_data['industry'] == selected_industry]
        
        if selected_growth != 'All':
            filtered_data = filtered_data[filtered_data['growth_outlook'] == selected_growth]
        
        if salary_range > 0:
            filtered_data = filtered_data[filtered_data['avg_salary'] >= salary_range * 1000]
        
        # Display results
        st.markdown(f"**Found {len(filtered_data)} careers matching your criteria:**")
        
        # Career cards
        for _, career in filtered_data.iterrows():
            with st.expander(f"{career['career_title']} - {career['industry']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Industry:** {career['industry']}")
                    st.markdown(f"**Required Skills:** {career['required_skills']}")
                    st.markdown(f"**Education:** {career['education_requirements']}")
                    st.markdown(f"**Description:** {career['job_description']}")
                
                with col2:
                    st.metric("Salary Range", career['salary_range'])
                    st.metric("Growth Outlook", career['growth_outlook'])
                    
                    if st.button(f"View Details", key=f"details_{career['career_id']}"):
                        st.session_state.selected_career_id = career['career_id']
        
        # Industry distribution chart
        st.markdown("---")
        st.markdown("### 📈 Career Distribution by Industry")
        
        industry_counts = career_data['industry'].value_counts()
        fig = px.pie(values=industry_counts.values, names=industry_counts.index, 
                     title="Careers by Industry")
        st.plotly_chart(fig, use_container_width=True, key="career_explorer_pie")

def show_career_details():
    """Display detailed information about a specific career"""
    
    st.markdown("## 🔍 Career Details")
    
    if st.session_state.recommendation_engine:
        career_data = st.session_state.recommendation_engine.career_data
        
        # Career selection
        career_titles = career_data.apply(lambda x: f"{x['career_title']} ({x['industry']})", axis=1).tolist()
        career_ids = career_data['career_id'].tolist()
        
        selected_idx = st.selectbox("Select a career to explore:", range(len(career_titles)), 
                                   format_func=lambda x: career_titles[x])
        
        selected_career_id = career_ids[selected_idx]
        
        # Get career details
        career_details = st.session_state.recommendation_engine.get_career_details(selected_career_id)
        
        if career_details:
            # Main career information
            st.markdown(f"# {career_details['career_title']}")
            st.markdown(f"**Industry:** {career_details['industry']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Salary Range", career_details['salary_range'])
            with col2:
                st.metric("Growth Outlook", career_details['growth_outlook'])
            with col3:
                st.metric("Average Rating", f"{career_details['average_rating']:.1f}/5.0")
            
            # Detailed information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Job Description")
                st.write(career_details['job_description'])
                
                st.markdown("### 🎓 Education Requirements")
                st.write(career_details['education_requirements'])
                
                st.markdown("### 🏢 Work Environment")
                st.write(career_details['work_environment'])
            
            with col2:
                st.markdown("### 🛠️ Required Skills")
                skills = career_details['required_skills'].split('|')
                for skill in skills:
                    st.markdown(f"• {skill.strip()}")
                
                st.markdown("### 🔗 Similar Careers")
                if career_details['similar_careers']:
                    for similar in career_details['similar_careers'][:5]:
                        similar_career = career_data[career_data['career_id'] == similar['career_id']]
                        if not similar_career.empty:
                            st.markdown(f"• {similar_career.iloc[0]['career_title']} (Similarity: {similar['similarity']:.2f})")
                else:
                    st.write("No similar careers found in the database.")
            
            # Skills word cloud
            st.markdown("### ☁️ Skills Word Cloud")
            skills_text = career_details['required_skills'].replace('|', ' ')
            
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(skills_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except:
                st.write("Could not generate word cloud for skills.")

def show_analytics_page():
    """Display system analytics and insights"""
    
    st.markdown("## 📈 System Analytics")
    st.markdown("Insights and statistics about the career recommendation system.")
    
    if st.session_state.recommendation_engine:
        engine = st.session_state.recommendation_engine
        stats = engine.get_system_stats()
        
        # System overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Careers", stats.get('total_careers', 0))
        with col2:
            st.metric("Student Profiles", stats.get('total_students', 0))
        with col3:
            st.metric("Total Ratings", stats.get('total_ratings', 0))
        with col4:
            st.metric("Average Rating", f"{stats.get('average_rating', 0):.2f}")
        
        # Model performance
        if stats.get('is_trained'):
            st.markdown("### 🎯 Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                accuracy = stats.get('decision_tree_accuracy', 0)
                st.metric("Decision Tree Accuracy", f"{accuracy:.1%}")
            
            # Feature importance
            if hasattr(engine.decision_tree, 'feature_importance'):
                st.markdown("### 📊 Feature Importance (Decision Tree)")
                importance_df = engine.decision_tree.feature_importance.head(10)
                
                fig = px.bar(importance_df, x='importance', y='feature', 
                           orientation='h', title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True, key="feature_importance")
        
        # Data insights
        career_data = engine.career_data
        student_data = engine.student_data
        ratings_data = engine.ratings_data
        
        st.markdown("### 📊 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Industry distribution
            industry_counts = career_data['industry'].value_counts()
            fig = px.bar(x=industry_counts.index, y=industry_counts.values,
                        title="Careers by Industry")
            fig.update_layout(xaxis_title="Industry", yaxis_title="Number of Careers")
            st.plotly_chart(fig, use_container_width=True, key="industry_distribution")
        
        with col2:
            # Growth outlook distribution
            growth_counts = career_data['growth_outlook'].value_counts()
            fig = px.pie(values=growth_counts.values, names=growth_counts.index,
                        title="Growth Outlook Distribution")
            st.plotly_chart(fig, use_container_width=True, key="growth_outlook_pie")
        
        # Rating distribution
        st.markdown("### 📈 Rating Distribution")
        fig = px.histogram(ratings_data, x='rating', nbins=5, 
                          title="Distribution of Career Ratings")
        fig.update_layout(xaxis_title="Rating", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True, key="rating_distribution")
        
        # Academic background distribution
        st.markdown("### 🎓 Academic Background Distribution")
        academic_counts = student_data['academic_background'].value_counts()
        fig = px.bar(x=academic_counts.values, y=academic_counts.index,
                    orientation='h', title="Students by Academic Background")
        fig.update_layout(xaxis_title="Number of Students", yaxis_title="Academic Background")
        st.plotly_chart(fig, use_container_width=True, key="academic_background_bar")

def show_about_page():
    """Display the about page"""
    
    st.markdown("## ℹ️ About MY NEW CAREER")
    
    st.markdown("""
    ### 🎯 Mission
    MY NEW CAREER is an AI-powered career recommendation system designed to help students and professionals 
    discover their ideal career paths based on their academic background, skills, interests, and personality traits.
    
    ### 🧠 Technology Stack
    Our system uses cutting-edge AI and machine learning technologies:
    
    - **Natural Language Processing (NLP)**: For understanding and processing text data from user profiles and career descriptions
    - **Decision Trees**: For intelligent decision making based on user characteristics and career requirements
    - **Collaborative Filtering**: For personalized recommendations based on similar user preferences
    - **Data Preprocessing**: Advanced data cleaning and feature engineering with Python
    - **Streamlit**: Modern web interface for easy interaction
    
    ### 🔬 How It Works
    
    1. **Data Collection**: The system analyzes your academic background, skills, interests, and preferences
    2. **Multi-Algorithm Analysis**: Three different AI algorithms work together:
       - NLP analyzes text similarity between your profile and career descriptions
       - Decision Trees predict career fit based on structured data
       - Collaborative Filtering finds careers liked by similar users
    3. **Intelligent Combination**: Results from all algorithms are combined using weighted scoring
    4. **Personalized Recommendations**: You receive ranked career suggestions with detailed explanations
    
    ### 📊 Features
    
    - **Comprehensive Profiling**: Detailed user profile analysis
    - **Multi-Algorithm Recommendations**: Three AI approaches for better accuracy
    - **Skill Gap Analysis**: Identification of skills needed for target careers
    - **Career Exploration**: Browse careers by industry, growth outlook, and salary
    - **Detailed Career Information**: In-depth information about each career path
    - **Analytics Dashboard**: System insights and performance metrics
    
    ### 🎓 Educational Value
    
    This system showcases practical applications of:
    - Machine Learning in career guidance
    - Natural Language Processing for text analysis
    - Collaborative filtering techniques
    - Data preprocessing and feature engineering
    - Interactive web application development
    
    ### 🚀 Future Enhancements
    
    - Integration with job market APIs for real-time data
    - Advanced deep learning models
    - Mobile application development
    - LinkedIn profile integration
    - Career path visualization
    - Mentorship matching
    
    ### 👥 Target Audience
    
    - **Students**: Exploring career options after graduation
    - **Career Changers**: Professionals seeking new career paths
    - **Career Counselors**: Tools for advising clients
    - **Educational Institutions**: Career guidance departments
    
    ### 📈 Impact
    
    MY NEW CAREER aims to:
    - Reduce career decision uncertainty
    - Improve job satisfaction through better matching
    - Provide data-driven career insights
    - Support educational and career planning
    
    ---
    
    ### 🛠️ Technical Implementation
    
    **Data Processing Pipeline:**
    - CSV data loading and validation
    - Text preprocessing and cleaning
    - Feature encoding and normalization
    - Skill matrix creation
    
    **Machine Learning Models:**
    - Decision Tree Classifier with hyperparameter optimization
    - Random Forest ensemble for improved accuracy
    - User-based and item-based collaborative filtering
    - Matrix factorization using NMF and SVD
    
    **NLP Processing:**
    - TF-IDF vectorization for text similarity
    - Sentiment analysis for feedback processing
    - Keyword extraction and importance ranking
    - Semantic similarity calculations
    
    **Web Interface:**
    - Interactive Streamlit application
    - Responsive design with custom CSS
    - Real-time recommendation generation
    - Data visualization with Plotly and Matplotlib
    
    ### 📞 Contact & Support
    
    For questions, suggestions, or support:
    - Use the feedback form in the application
    - Check the documentation for technical details
    - Review the source code for implementation insights
    
    ---
    
    *MY NEW CAREER - Empowering career decisions through artificial intelligence* 🎯
    """)

if __name__ == "__main__":
    main()