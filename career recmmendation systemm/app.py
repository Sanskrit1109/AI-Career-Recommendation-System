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
import hashlib
import json
from datetime import datetime

# OpenAI imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️  OpenAI package not installed. Career assistant will not be available.")

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

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Make sidebar thinner */
    .css-1d391kg {
        width: 180px !important;
    }
    .css-1oe5cao {
        max-width: 180px !important;
        min-width: 180px !important;
    }
    section[data-testid="stSidebar"] {
        width: 180px !important;
        max-width: 180px !important;
        min-width: 180px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 180px !important;
        max-width: 180px !important;
        min-width: 180px !important;
    }
    
    /* Enhanced header styling */
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.2em;
        font-family: 'Arial', sans-serif;
    }
    .sub-header {
        font-size: 1.1em;
        color: #666;
        margin-bottom: 1em;
        font-style: italic;
    }
    
    /* Modern card styling */
    .modern-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 2em;
        border-radius: 20px;
        margin: 1.5em 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    .modern-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .gradient-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2em;
        border-radius: 20px;
        color: white;
        margin: 1.5em 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    .gradient-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5em;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 0.8em 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .skill-tag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5em 1em;
        border-radius: 25px;
        margin: 0.3em;
        font-size: 0.9em;
        font-weight: 500;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    .skill-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced navigation */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75em 1.5em;
        transition: all 0.3s ease;
        border: 2px solid #667eea;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        min-width: 80px;
        min-height: 40px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: #764ba2;
    }
    
    /* Circular Sign-In Button Styling */
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="primary"] {
        width: 150px !important;
        height: 150px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: 4px solid white !important;
        color: white !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.2 !important;
        margin: 1em auto !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.4s ease !important;
    }
    
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="primary"]:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7) !important;
        border-color: #f0f0f0 !important;
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
    }
    
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="primary"]:active {
        transform: scale(0.95) !important;
        transition: all 0.1s ease !important;
    }
    
    /* Add glow effect */
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="primary"]::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: rgba(255,255,255,0.1) !important;
        border-radius: 50% !important;
        opacity: 0 !important;
        transition: opacity 0.3s ease !important;
    }
    
    div[data-testid="column"]:nth-child(2) .stButton > button[kind="primary"]:hover::before {
        opacity: 1 !important;
    }
    
    /* Square box styling for sign-in button */
    .signin-button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.8em 1.2em !important;
        border: 2px solid #667eea !important;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        min-width: 100px !important;
        min-height: 45px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Modern animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Streamlit specific overrides */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .modern-card {
            margin: 1em 0;
            padding: 1.5em;
        }
        .gradient-card {
            margin: 1em 0;
            padding: 1.5em;
        }
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
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = {}
if 'current_user_email' not in st.session_state:
    st.session_state.current_user_email = None

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users_database():
    """Load users database from file"""
    try:
        if os.path.exists('data/users_database.json'):
            with open('data/users_database.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading users database: {str(e)}")
        return {}

def save_users_database(users_db):
    """Save users database to file"""
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/users_database.json', 'w') as f:
            json.dump(users_db, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving users database: {str(e)}")
        return False

def register_user(username, email, password):
    """Register a new user"""
    users_db = load_users_database()
    
    # Check if email already exists
    for user_data in users_db.values():
        if user_data['email'] == email:
            return False, "Email already registered"
    
    # Check if username already exists
    if username in users_db:
        return False, "Username already exists"
    
    # Register new user
    users_db[username] = {
        'email': email,
        'password': hash_password(password),
        'registration_date': datetime.now().isoformat(),
        'profile_completed': False
    }
    
    if save_users_database(users_db):
        return True, "Registration successful"
    else:
        return False, "Error saving user data"

def authenticate_user_new(username, email, password):
    """Authenticate user with username, email, and password"""
    users_db = load_users_database()
    
    if username in users_db:
        user_data = users_db[username]
        if (user_data['email'] == email and 
            user_data['password'] == hash_password(password)):
            return True, user_data
    
    return False, None

def get_user_profile(username):
    """Get user profile from student_profiles.csv or create default"""
    try:
        student_data = pd.read_csv('data/student_profiles.csv')
        # Try to find user by academic background matching username
        user_match = student_data[
            student_data['academic_background'].str.lower().str.contains(username.lower(), na=False)
        ]
        
        if not user_match.empty:
            return user_match.iloc[0].to_dict()
        else:
            # Create default profile
            return {
                'student_id': 999,
                'academic_background': username.title(),
                'skills': 'To be updated',
                'interests': 'To be updated',
                'gpa': 0.0,
                'preferred_work_environment': 'To be updated',
                'career_experience': 'To be updated',
                'personality_traits': 'To be updated'
            }
    except Exception as e:
        st.error(f"Error loading user profile: {str(e)}")
        return None
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

def load_student_data():
    """Load student data (kept for compatibility)"""
    try:
        student_data = pd.read_csv('data/student_profiles.csv')
        return student_data
    except Exception as e:
        st.error(f"Error loading student data: {str(e)}")
        return None

def show_login_page():
    """Display the login page with username, email, and password authentication"""
    
    # Login page styling with circular frame
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3em 2em;
        border-radius: 20px;
        margin: 2em 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    ">
        <div style="
            width: 80px;
            height: 80px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5em;
            margin: 0 auto 1em auto;
        ">
        🎓
        </div>
        <h1 style="margin: 0 0 0.5em 0; font-size: 2.5em;">Welcome to MY NEW CAREER</h1>
        <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">AI-Powered Career Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered login/registration form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Tab selection for Login/Register
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Sign In to Your Account")
            
            with st.form("login_form"):
                # Username input
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    help="Your unique username"
                )
                
                # Email input
                email = st.text_input(
                    "Email",
                    placeholder="Enter your email address",
                    help="Your registered email address"
                )
                
                # Password input
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Your account password"
                )
                
                # Custom CSS for circular sign-in button
                st.markdown("""
                <style>
                .login-button-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 2em 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Centered circular sign-in button
                st.markdown('<div class="login-button-container">', unsafe_allow_html=True)
                
                signin_col1, signin_col2, signin_col3 = st.columns([1, 1, 1])
                
                with signin_col2:
                    login_clicked = st.form_submit_button(
                        "🔐\nSignIn",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Process login
                if login_clicked:
                    if not username.strip() or not email.strip() or not password.strip():
                        st.error("Please fill in all fields.")
                    else:
                        with st.spinner("🔍 Authenticating..."):
                            success, user_data = authenticate_user_new(username, email, password)
                            
                            if success:
                                st.session_state.is_authenticated = True
                                st.session_state.current_user = username
                                st.session_state.current_user_email = email
                                st.session_state.user_data = get_user_profile(username)
                                st.success(f"✅ Welcome back, {username}!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials. Please check your username, email, and password.")
        
        with tab2:
            st.markdown("### Create New Account")
            
            with st.form("register_form"):
                # Registration fields
                new_username = st.text_input(
                    "Choose Username",
                    placeholder="Enter a unique username",
                    help="This will be your login username"
                )
                
                new_email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email address",
                    help="We'll use this for account verification"
                )
                
                new_password = st.text_input(
                    "Create Password",
                    type="password",
                    placeholder="Create a secure password",
                    help="Choose a strong password for your account"
                )
                
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password",
                    help="Confirm your password"
                )
                
                # Registration button
                register_clicked = st.form_submit_button(
                    "📝 Create Account",
                    type="primary",
                    use_container_width=True
                )
                
                # Process registration
                if register_clicked:
                    if not new_username.strip() or not new_email.strip() or not new_password.strip():
                        st.error("Please fill in all fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif '@' not in new_email or '.' not in new_email:
                        st.error("Please enter a valid email address.")
                    else:
                        with st.spinner("📝 Creating account..."):
                            success, message = register_user(new_username, new_email, new_password)
                            
                            if success:
                                st.success(f"✅ {message}! You can now login with your credentials.")
                                st.balloons()
                            else:
                                st.error(f"❌ {message}")


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
    
    # Simple Header without nested columns
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1em 0 2em 0;
        padding: 1em 0;
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
    ">
        <div style="display: flex; align-items: center;">
            <div style="
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #1f77b4, #ff7f0e);
                border-radius: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5em;
                color: white;
                margin-right: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            ">
            🎯
            </div>
            <div>
                <h1 style="
                    font-size: 2.2em;
                    color: #1f77b4;
                    font-weight: bold;
                    margin: 0;
                    font-family: 'Arial', sans-serif;
                ">MY NEW CAREER</h1>
                <p style="
                    font-size: 1.0em;
                    color: #666;
                    margin: 0;
                    font-style: italic;
                ">AI-Powered Career Recommendation System</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # User menu - simple approach without columns
    if st.session_state.is_authenticated:
        username = st.session_state.current_user
        email = st.session_state.current_user_email
        
        st.markdown(f"""
        <div style="text-align: right; margin: -3em 1em 1em 0;">
            <span style="color: #666; font-size: 0.9em;">👋 Welcome, <strong>{username}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout button in sidebar instead
        with st.sidebar:
            if st.button("🚪 Logout", key="logout_btn", type="primary"):
                st.session_state.is_authenticated = False
                st.session_state.current_user = None
                st.session_state.current_user_email = None
                st.session_state.user_data = None
                st.success("✅ Logged out successfully!")
                st.rerun()
    
    # Check authentication status
    if not st.session_state.is_authenticated:
        show_login_page()
        return
    
    # Sidebar (only show when authenticated)
    # Sidebar logo
    st.sidebar.markdown("""
    <div style="
        display: flex;
        align-items: center;
        margin-bottom: 1em;
    ">
        <div style="
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #1f77b4, #ff7f0e);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2em;
            margin-right: 10px;
        ">
        🎯
        </div>
        <div style="
            font-size: 1.2em;
            font-weight: bold;
            color: #1f77b4;
        ">MY NEW CAREER</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    
    # Show user info in sidebar
    if st.session_state.user_data:
        user_data = st.session_state.user_data
        st.sidebar.markdown(f"**👤 Student ID:** {user_data.get('student_id', 'N/A')}")
        st.sidebar.markdown(f"**🎓 Field:** {user_data.get('academic_background', 'N/A')}")
        st.sidebar.markdown(f"**📏 GPA:** {user_data.get('gpa', 'N/A')}")
        st.sidebar.markdown("---")
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
    """Display the home page with modern design"""
    
    # Modern Hero Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 4rem 2rem;
        border-radius: 25px;
        margin: 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white" opacity="0.1"/><circle cx="80" cy="40" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="80" r="1.5" fill="white" opacity="0.1"/><circle cx="90" cy="80" r="1" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="1" fill="white" opacity="0.1"/></svg>');
            animation: float 20s linear infinite;
        "></div>
        <div style="position: relative; z-index: 2;">
            <div style="
                font-size: 4rem;
                margin-bottom: 1rem;
                text-shadow: 0 4px 8px rgba(0,0,0,0.3);
            ">🚀</div>
            <h1 style="
                font-size: 3.5rem;
                margin: 0;
                font-weight: 800;
                text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                letter-spacing: -1px;
            ">MY NEW CAREER</h1>
            <p style="
                font-size: 1.4rem;
                margin: 1rem 0 0 0;
                opacity: 0.95;
                font-weight: 300;
            ">Discover Your Future with AI-Powered Career Intelligence</p>
        </div>
    </div>
    
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0% { transform: translateX(-100%) translateY(-100%) rotate(0deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Introduction Section with Cards - Fixed Layout
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    ">
        <div style="text-align: center; margin-bottom: 3rem;">
            <h2 style="
                font-size: 2.5rem;
                color: #2d3748;
                margin: 0;
                font-weight: 700;
            ">🧠 Powered by Advanced AI</h2>
            <p style="
                font-size: 1.2rem;
                color: #4a5568;
                margin: 1rem 0 0 0;
            ">Our intelligent system combines multiple AI technologies to find your perfect career match</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Technologies Cards using Streamlit columns
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🧠</div>
            <h4 style="color: #2d3748; margin: 0 0 0.5rem 0; font-size: 1.3rem;">Natural Language Processing</h4>
            <p style="color: #666; font-size: 1rem; margin: 0; line-height: 1.5;">Understanding your profile with advanced text analysis and semantic matching</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🌳</div>
            <h4 style="color: #2d3748; margin: 0 0 0.5rem 0; font-size: 1.3rem;">Decision Trees</h4>
            <p style="color: #666; font-size: 1rem; margin: 0; line-height: 1.5;">Intelligent career matching with machine learning algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🤝</div>
            <h4 style="color: #2d3748; margin: 0 0 0.5rem 0; font-size: 1.3rem;">Collaborative Filtering</h4>
            <p style="color: #666; font-size: 1rem; margin: 0; line-height: 1.5;">Personalized recommendations based on user patterns and preferences</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: transform 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #2d3748; margin: 0 0 0.5rem 0; font-size: 1.3rem;">Data Analytics</h4>
            <p style="color: #666; font-size: 1rem; margin: 0; line-height: 1.5;">Comprehensive insights from career data analysis and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats with Modern Cards
    if st.session_state.recommendation_engine:
        stats = st.session_state.recommendation_engine.get_system_stats()
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 3rem 0;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        ">
            <h2 style="
                text-align: center;
                font-size: 2.5rem;
                margin: 0 0 3rem 0;
                font-weight: 700;
            ">📈 System Statistics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ff6b6b, #ee5a52);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 25px rgba(255, 107, 107, 0.3);
                margin-top: -2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💼</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{stats.get('total_careers', 0)}</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Total Careers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4ecdc4, #44a08d);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
                margin-top: -2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">👥</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{stats.get('total_students', 0)}</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Student Profiles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea, #764ba2);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
                margin-top: -2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🛠️</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{stats.get('unique_skills', 0)}</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Unique Skills</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f093fb, #f5576c);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 25px rgba(240, 147, 251, 0.3);
                margin-top: -2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🏢</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{stats.get('unique_industries', 0)}</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Industries</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Grid Section
    st.markdown("""
    <div style="
        margin: 4rem 0;
    ">
        <h2 style="
            text-align: center;
            font-size: 2.5rem;
            color: #2d3748;
            margin: 0 0 3rem 0;
            font-weight: 700;
        ">🌟 Powerful Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.2);
        ">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; text-align: center;">🎯</div>
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">Personalized Recommendations</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Tailored career suggestions based on your unique profile</li>
                <li>Multiple AI algorithms working together</li>
                <li>Detailed explanations for each recommendation</li>
            </ul>
        </div>
        
        <div style="
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            padding: 2.5rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 15px 30px rgba(78, 205, 196, 0.2);
        ">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; text-align: center;">📊</div>
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">Comprehensive Analysis</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Skill gap analysis</li>
                <li>Career growth insights</li>
                <li>Salary expectations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb, #f5576c);
            padding: 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(240, 147, 251, 0.2);
        ">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; text-align: center;">🔍</div>
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">Career Exploration</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Browse careers by industry</li>
                <li>Compare different career paths</li>
                <li>Discover similar careers</li>
            </ul>
        </div>
        
        <div style="
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            padding: 2.5rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 15px 30px rgba(255, 107, 107, 0.2);
        ">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; text-align: center;">📈</div>
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">Data-Driven Insights</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Success rate predictions</li>
                <li>Industry trends</li>
                <li>Educational requirements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 25px;
        margin: 4rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    ">
        <h2 style="
            font-size: 2.5rem;
            margin: 0 0 1rem 0;
            font-weight: 700;
        ">Ready to Discover Your Dream Career?</h2>
        <p style="
            font-size: 1.3rem;
            margin: 0 0 2rem 0;
            opacity: 0.9;
        ">Get personalized AI-powered recommendations tailored to your unique profile</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Get My Career Recommendations", type="primary", use_container_width=True):
            st.info("👈 Please use the sidebar navigation to go to 'Get Career Recommendations' page!")
            st.balloons()
    
    # Career guidance section removed - focus on core recommendation features

# GUIDE assistant function removed to simplify the application
# Focus on core career recommendation features

def show_career_assistant():
    """Display the OpenAI career assistant on the recommendations page"""
    
    st.markdown("""---""")
    st.markdown("### 🤖 Career Assistant")
    st.markdown("Get personalized career advice and answers to your questions about career development.")
    
    if not OPENAI_AVAILABLE:
        st.error("😞 OpenAI is not available. Please install the openai package to use the Career Assistant.")
        st.code("pip install openai>=1.0.0", language="bash")
        return
    
    # API Key input in an expander
    with st.expander("🔑 OpenAI API Key Setup", expanded=False):
        api_key = st.session_state.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        
        if not api_key:
            st.info("""
            **To use the Career Assistant, you need an OpenAI API key:**
            1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign in or create an account
            3. Click "Create new secret key"
            4. Copy the key (starts with 'sk-')
            5. Paste it below
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                api_key_input = st.text_input(
                    "Enter your OpenAI API Key",
                    type="password",
                    placeholder="sk-proj-... or sk-...",
                    help="Your API key should start with 'sk-' and be around 50+ characters long"
                )
            
            with col2:
                if st.button("Validate Key", type="primary"):
                    if api_key_input:
                        # Basic validation
                        if not api_key_input.startswith('sk-'):
                            st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
                        elif len(api_key_input) < 20:
                            st.error("❌ API key seems too short. Please check you copied the complete key.")
                        else:
                            # Test the API key
                            with st.spinner("🔍 Validating API key..."):
                                try:
                                    test_client = OpenAI(api_key=api_key_input)
                                    # Test with a minimal request
                                    test_response = test_client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "user", "content": "Hello"}
                                        ],
                                        max_tokens=5
                                    )
                                    st.session_state.openai_api_key = api_key_input
                                    st.success("✅ API key validated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    if "401" in error_msg or "invalid_api_key" in error_msg:
                                        st.error("❌ Invalid API key. Please check your key and try again.")
                                    elif "quota" in error_msg.lower():
                                        st.error("❌ API quota exceeded. Please check your OpenAI billing.")
                                    elif "rate_limit" in error_msg.lower():
                                        st.error("❌ Rate limit exceeded. Please try again in a moment.")
                                    else:
                                        st.error(f"❌ Error validating API key: {error_msg}")
                    else:
                        st.error("Please enter an API key")
        else:
            st.success("✅ OpenAI API Key configured and validated")
            if st.button("🗑️ Clear API Key"):
                if 'openai_api_key' in st.session_state:
                    del st.session_state.openai_api_key
                if 'career_chat_history' in st.session_state:
                    del st.session_state.career_chat_history
                st.success("🧹 API key and chat history cleared")
                st.rerun()
    
    # Only show chat if API key is configured
    api_key = st.session_state.get('openai_api_key')
    if not api_key:
        st.info("👆 Please configure your OpenAI API key above to use the Career Assistant.")
        return
    
    # Initialize chat history for career assistant
    if 'career_chat_history' not in st.session_state:
        st.session_state.career_chat_history = [
            {
                "role": "system",
                "content": """
You are a Career Assistant for the MY NEW CAREER recommendation system. You help users with:

1. Career planning and goal setting
2. Understanding career recommendations they've received
3. Skill development advice
4. Industry insights and trends
5. Interview preparation
6. Resume and portfolio guidance
7. Career transition strategies

You are supportive, knowledgeable, and provide actionable advice. Keep responses concise but helpful.
Encourage users to use the main recommendation system for personalized career matching.
"""
            },
            {
                "role": "assistant",
                "content": "Hello! I'm your Career Assistant. I can help you understand your career recommendations, plan your next steps, or answer any career-related questions. What would you like to discuss?"
            }
        ]
    
    # Display chat in a container
    chat_container = st.container()
    
    with chat_container:
        # Show recent chat messages (last 6 to keep it manageable)
        recent_messages = st.session_state.career_chat_history[1:]  # Skip system message
        if len(recent_messages) > 6:
            recent_messages = recent_messages[-6:]
        
        for message in recent_messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(f"**Career Assistant:** {message['content']}")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.write(message['content'])
    
    # Chat input
    user_question = st.chat_input(
        "Ask about career planning, skills, or any career-related questions...",
        key="career_assistant_input"
    )
    
    if user_question:
        # Add user message to chat history
        st.session_state.career_chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(user_question)
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Career Assistant is thinking..."):
                try:
                    client = OpenAI(api_key=api_key)
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.career_chat_history,
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    ai_response = response.choices[0].message.content or "Sorry, I couldn't generate a response."
                    
                    # Add AI response to chat history
                    st.session_state.career_chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    st.write(f"**Career Assistant:** {ai_response}")
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    if "401" in error_msg or "invalid_api_key" in error_msg:
                        st.error("❌ Invalid API Key. Please update your key in the setup section above.")
                        if 'openai_api_key' in st.session_state:
                            del st.session_state.openai_api_key
                    elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                        st.error("❌ API quota exceeded. Please check your OpenAI billing.")
                    elif "rate_limit" in error_msg.lower():
                        st.error("❌ Rate limited. Please wait a moment and try again.")
                    else:
                        st.error(f"❌ Error: {error_msg}")
    
    # Clear chat button
    if len(st.session_state.career_chat_history) > 2:  # More than just system + initial message
        if st.button("🗑️ Clear Chat History", key="clear_career_chat"):
            st.session_state.career_chat_history = [
                {
                    "role": "system",
                    "content": """
You are a Career Assistant for the MY NEW CAREER recommendation system. You help users with:

1. Career planning and goal setting
2. Understanding career recommendations they've received
3. Skill development advice
4. Industry insights and trends
5. Interview preparation
6. Resume and portfolio guidance
7. Career transition strategies

You are supportive, knowledgeable, and provide actionable advice. Keep responses concise but helpful.
Encourage users to use the main recommendation system for personalized career matching.
"""
                },
                {
                    "role": "assistant",
                    "content": "Hello! I'm your Career Assistant. I can help you understand your career recommendations, plan your next steps, or answer any career-related questions. What would you like to discuss?"
                }
            ]
            st.success("🧹 Chat history cleared!")
            st.rerun()

def show_recommendation_page():
    """Display the career recommendation page"""
    
    st.markdown("## 📝 Get Your Personalized Career Recommendations")
    
    # Show user info if logged in
    if st.session_state.user_data:
        user_data = st.session_state.user_data
        st.info(f"👤 Logged in as: **{user_data.get('academic_background', 'Student')}** (ID: {user_data.get('student_id', 'N/A')})")
        st.markdown("*Your profile data has been pre-filled. You can modify any fields below.*")
    else:
        st.markdown("Fill out the form below to receive AI-powered career suggestions tailored to your profile.")
    
    # Get default values from user data if available
    user_data = st.session_state.user_data or {}
    default_academic = user_data.get('academic_background', 'Computer Science')
    default_gpa = user_data.get('gpa', 3.5)
    default_skills = user_data.get('skills', '')
    default_interests = user_data.get('interests', '')
    default_traits = user_data.get('personality_traits', '')
    default_environment = user_data.get('preferred_work_environment', 'Tech Company')
    
    # User input form
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎓 Academic Background")
            
            # Find index for selectbox
            academic_options = ["Computer Science", "Business Administration", "Engineering", "Psychology", 
                               "Marketing", "Data Science", "Biology", "English Literature", "Finance",
                               "Art", "Music", "Education", "Medicine", "Law", "Other"]
            
            try:
                default_index = academic_options.index(default_academic) if default_academic in academic_options else 0
            except:
                default_index = 0
            
            academic_background = st.selectbox(
                "Your Academic Field",
                academic_options,
                index=default_index
            )
            
            gpa = st.slider("GPA (if applicable)", 2.0, 4.0, float(default_gpa), 0.1)
            
            st.markdown("### 💼 Work Preferences")
            
            environment_options = ["Tech Company", "Corporate Office", "Healthcare Setting", "Educational Institution",
                                  "Government Agency", "Non-profit", "Startup", "Remote Work", "Freelance"]
            
            try:
                env_index = environment_options.index(default_environment) if default_environment in environment_options else 0
            except:
                env_index = 0
                
            work_environment = st.selectbox(
                "Preferred Work Environment",
                environment_options,
                index=env_index
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
    
    # Add career assistant at the bottom of the recommendations page
    show_career_assistant()

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
    
    # Add career assistant after recommendations
    show_career_assistant()

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
    
    # Analytics header with icon
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        margin-bottom: 2em;
    ">
        <div style="
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.8em;
            margin-right: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
        📈
        </div>
        <div>
            <h2 style="margin: 0; color: #1f77b4;">System Analytics</h2>
            <p style="margin: 0; color: #666;">Insights and statistics about the career recommendation system.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
            chart_fig = px.bar(x=industry_counts.index, y=industry_counts.values,
                        title="Careers by Industry")
            chart_fig.update_layout(xaxis_title="Industry", yaxis_title="Number of Careers")
            st.plotly_chart(chart_fig, use_container_width=True, key="industry_distribution")
        
        with col2:
            # Growth outlook distribution
            growth_counts = career_data['growth_outlook'].value_counts()
            pie_fig = px.pie(values=growth_counts.values, names=growth_counts.index,
                        title="Growth Outlook Distribution")
            st.plotly_chart(pie_fig, use_container_width=True, key="growth_outlook_pie")
        
        # Rating distribution
        st.markdown("### 📈 Rating Distribution")
        hist_fig = px.histogram(ratings_data, x='rating', nbins=5, 
                          title="Distribution of Career Ratings")
        hist_fig.update_layout(xaxis_title="Rating", yaxis_title="Frequency")
        st.plotly_chart(hist_fig, use_container_width=True, key="rating_distribution")
        
        # Academic background distribution
        st.markdown("### 🎓 Academic Background Distribution")
        academic_counts = student_data['academic_background'].value_counts()
        bar_fig = px.bar(x=academic_counts.values, y=academic_counts.index,
                    orientation='h', title="Students by Academic Background")
        bar_fig.update_layout(xaxis_title="Number of Students", yaxis_title="Academic Background")
        st.plotly_chart(bar_fig, use_container_width=True, key="academic_background_bar")

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