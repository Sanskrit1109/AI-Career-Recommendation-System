"""
Test script for MY NEW CAREER system
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_preprocessing():
    """Test data preprocessing module"""
    print("🧪 Testing Data Preprocessing...")
    try:
        from src.data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(data_dir="data")
        success = preprocessor.run_preprocessing()
        
        if success:
            summary = preprocessor.get_summary()
            print("✅ Data preprocessing successful!")
            print(f"   - Students: {summary['total_students']}")
            print(f"   - Careers: {summary['total_careers']}")
            print(f"   - Ratings: {summary['total_ratings']}")
            print(f"   - Skills: {summary['unique_skills']}")
            return True
        else:
            print("❌ Data preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Data preprocessing error: {str(e)}")
        return False

def test_nlp_processor():
    """Test NLP processing module"""
    print("\n🧪 Testing NLP Processor...")
    try:
        from src.nlp_processor import NLPProcessor
        
        nlp = NLPProcessor()
        
        # Test text preprocessing
        sample_text = "I love working with Python programming and machine learning algorithms!"
        processed = nlp.preprocess_text(sample_text)
        print(f"✅ Text preprocessing: '{sample_text[:30]}...' → '{processed[:30]}...'")
        
        # Test keyword extraction
        keywords = nlp.extract_keywords(sample_text, top_k=3)
        print(f"✅ Keyword extraction: {keywords}")
        
        # Test sentiment analysis
        sentiment = nlp.analyze_sentiment("I really enjoyed this career recommendation!")
        print(f"✅ Sentiment analysis: {sentiment['compound']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ NLP processor error: {str(e)}")
        return False

def test_recommendation_engine():
    """Test the main recommendation engine"""
    print("\n🧪 Testing Recommendation Engine...")
    try:
        from src.recommendation_engine import CareerRecommendationEngine
        
        # Initialize engine
        engine = CareerRecommendationEngine(data_dir="data")
        
        # Initialize system
        print("   Initializing system...")
        if not engine.initialize_system():
            print("❌ System initialization failed")
            return False
        
        # Train models
        print("   Training models...")
        if not engine.train_models():
            print("❌ Model training failed")
            return False
        
        # Test recommendations
        print("   Testing recommendations...")
        sample_user = {
            'student_id': 999,
            'academic_background': 'Computer Science',
            'skills': 'Python|Machine Learning|Data Analysis',
            'interests': 'Technology|Problem Solving',
            'gpa': 3.8,
            'preferred_work_environment': 'Tech Company',
            'personality_traits': 'Analytical|Creative'
        }
        
        recommendations = engine.get_comprehensive_recommendations(sample_user, 3)
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} recommendations!")
            print(f"   Top recommendation: {recommendations[0]['career_title']}")
            print(f"   Combined score: {recommendations[0]['combined_score']:.3f}")
            return True
        else:
            print("❌ No recommendations generated")
            return False
            
    except Exception as e:
        print(f"❌ Recommendation engine error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_system_stats():
    """Test system statistics"""
    print("\n🧪 Testing System Statistics...")
    try:
        from src.recommendation_engine import CareerRecommendationEngine
        
        engine = CareerRecommendationEngine(data_dir="data")
        engine.initialize_system()
        engine.train_models()
        
        stats = engine.get_system_stats()
        print("✅ System statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ System statistics error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing MY NEW CAREER System\n")
    
    tests = [
        test_data_preprocessing,
        test_nlp_processor,
        test_recommendation_engine,
        test_system_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {str(e)}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()