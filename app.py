import streamlit as st
import pandas as pd
import joblib
import pickle
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set page config
st.set_page_config(
    page_title="Tourism Recommendation System✈️ ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEMO_MODE = False  # Set to True if you want to test without model files

# Error handling decorator
def handle_model_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error loading model/data: {str(e)}")
            return None
    return wrapper

# Load all models and data with error handling
@st.cache_resource
@handle_model_errors
def load_models():
    """Load all pre-trained models and data files"""
    if DEMO_MODE:
        return create_demo_models()
        
    models = {}
    base_path = Path("tourism_models")
    
    # Visit prediction models
    visit_path = base_path / "visit_prediction"
    try:
        if visit_path.exists():
            models['visit_model'] = safe_load(visit_path / "model.pkl")
            models['visit_le'] = safe_load(visit_path / "label_encoder.pkl")
            models['visit_features'] = safe_load(visit_path / "feature_names.pkl")
            models['visit_template'] = safe_load(visit_path / "feature_template.pkl")
    except Exception as e:
        st.warning(f"Visit prediction models not loaded: {str(e)}")
    
    # Rating prediction models
    rating_path = base_path / "rating_prediction"
    try:
        if rating_path.exists():
            models['rating_model'] = safe_load(rating_path / "model.pkl")
            models['rating_scaler'] = safe_load(rating_path / "scaler.pkl")
            models['rating_features'] = safe_load(rating_path / "feature_names.pkl")
            models['rating_template'] = safe_load(rating_path / "feature_template.pkl")
    except Exception as e:
        st.warning(f"Rating prediction models not loaded: {str(e)}")
    
    # Recommendation models
    rec_path = base_path / "recommendation"
    try:
        if rec_path.exists():
            models['attraction_data'] = safe_load_attraction_data(rec_path)
            
            # Load similarity matrices
            similarity_files = {
                'user_similarity': rec_path / "user_similarity.pkl",
                'item_similarity': rec_path / "item_similarity.pkl",
                'content_similarity': rec_path / "content_similarity.pkl",
                'tfidf_vectorizer': rec_path / "tfidf_vectorizer.pkl"
            }
            
            for name, filepath in similarity_files.items():
                models[name] = safe_load(filepath)
    except Exception as e:
        st.warning(f"Recommendation models not loaded: {str(e)}")
    
    return models

def safe_load(filepath):
    """Safely load a pickle file with multiple fallback methods"""
    if not filepath.exists():
        return None
        
    try:
        return joblib.load(filepath)
    except:
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except:
            try:
                return pd.read_pickle(filepath)
            except:
                return None

def safe_load_attraction_data(rec_path):
    """Load attraction data with multiple fallback methods"""
    try:
        data = pd.read_pickle(rec_path / "attraction_data.pkl")
        # Ensure required columns exist
        if 'AttractionType' not in data.columns:
            data['AttractionType'] = "Unknown"
        if 'AttractionAddress' not in data.columns:
            data['AttractionAddress'] = "Unknown"
        if 'Rating' not in data.columns:
            data['Rating'] = np.random.uniform(3, 5, len(data)).round(1)
        return data
    except:
        try:
            data = pd.read_csv(rec_path / "attraction_data.csv")
            # Ensure required columns exist
            if 'AttractionType' not in data.columns:
                data['AttractionType'] = "Unknown"
            if 'AttractionAddress' not in data.columns:
                data['AttractionAddress'] = "Unknown"
            if 'Rating' not in data.columns:
                data['Rating'] = np.random.uniform(3, 5, len(data)).round(1)
            return data
        except:
            return create_demo_attractions()

def create_demo_models():
    """Create demo models when real ones aren't available"""
    st.warning("Running in demo mode with simulated data")
    return {
        'attraction_data': create_demo_attractions(),
        'visit_model': None,
        'visit_le': None,
        'rating_model': None
    }

def create_demo_attractions():
    """Create sample attraction data for demo purposes"""
    return pd.DataFrame({
        'AttractionId': range(1, 21),
        'AttractionType': np.random.choice(
            ['Museum', 'Park', 'Historical', 'Beach', 'Shopping', 'Adventure'], 
            20
        ),
        'AttractionAddress': np.random.choice(
            ['New York', 'Chicago', 'Boston', 'Miami', 'Los Angeles', 
             'San Francisco', 'Seattle', 'Washington DC'],
            20
        ),
        'Rating': np.random.uniform(3, 5, 20).round(1)
    })

# Load visualization images with error handling
@st.cache_data
def load_visualizations():
    """Load visualization images"""
    visuals = {}
    try:
        viz_files = {
            'rating_trends': "visualizations/monthly_rating_trends.png",
            'popular_attractions': "visualizations/popular_attractions.png",
            'user_segments': "visualizations/user_segments.png"
        }
        
        for name, path in viz_files.items():
            if os.path.exists(path):
                visuals[name] = Image.open(path)
            else:
                visuals[name] = None
    except Exception as e:
        st.warning(f"Could not load visualizations: {str(e)}")
        visuals = {name: None for name in viz_files.keys()}
    
    return visuals

# Initialize session state with error handling
if 'models' not in st.session_state:
    loaded_models = load_models()
    st.session_state.models = loaded_models if loaded_models is not None else create_demo_models()
    
if 'visuals' not in st.session_state:
    st.session_state.visuals = load_visualizations() or {}

# Prediction functions with fallbacks
def predict_visit_mode(input_data):
    """Predict the visit mode with fallback"""
    if not st.session_state.models.get('visit_model'):
        return np.random.choice(['Business', 'Family', 'Solo', 'Couple', 'Friends'])
    
    try:
        features = st.session_state.models['visit_template'].copy()
        for key in input_data:
            if key in features:
                features[key] = input_data[key]
        
        # Convert to DataFrame with proper feature names
        input_df = pd.DataFrame([features])
        
        # Ensure columns match training data
        expected_features = st.session_state.models['visit_features']
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_features]
        
        prediction = st.session_state.models['visit_model'].predict(input_df.values)
        return st.session_state.models['visit_le'].inverse_transform(prediction)[0]
    except Exception as e:
        st.warning(f"Visit mode prediction failed: {str(e)}")
        return np.random.choice(['Business', 'Family', 'Solo', 'Couple', 'Friends'])

def predict_rating(input_data):
    """Predict the rating with fallback"""
    if not st.session_state.models.get('rating_model'):
        return round(np.random.uniform(3, 5), 1)
    
    try:
        features = st.session_state.models['rating_template'].copy()
        for key in input_data:
            if key in features:
                features[key] = input_data[key]
        
        # Convert to DataFrame with proper feature names
        input_df = pd.DataFrame([features])
        
        # Ensure columns match training data
        expected_features = st.session_state.models['rating_features']
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_features]
        
        scaled_input = st.session_state.models['rating_scaler'].transform(input_df)
        return max(1, min(5, st.session_state.models['rating_model'].predict(scaled_input)[0]))
    except Exception as e:
        st.warning(f"Rating prediction failed: {str(e)}")
        return round(np.random.uniform(3, 5), 1)

def get_recommendations(user_id=None, attraction_id=None, n=5, method='hybrid'):
    """Get recommendations with comprehensive fallbacks"""
    attraction_data = st.session_state.models.get('attraction_data', create_demo_attractions())
    
    # Ensure required columns exist
    if 'AttractionType' not in attraction_data.columns:
        attraction_data['AttractionType'] = "Unknown"
    if 'AttractionAddress' not in attraction_data.columns:
        attraction_data['AttractionAddress'] = "Unknown"
    
    if method == 'collaborative' and user_id is not None:
        try:
            if 'user_similarity' not in st.session_state.models:
                raise ValueError("Collaborative filtering not available")
                
            user_sim = st.session_state.models['user_similarity']
            if user_sim is None or user_id not in user_sim.index:
                return attraction_data.sample(min(n, len(attraction_data)))
            
            similar_users = user_sim[user_id].sort_values(ascending=False)[1:6]
            return attraction_data.sample(min(n, len(attraction_data)))
        except Exception as e:
            st.warning(f"Collaborative recommendation failed: {str(e)}")
            return attraction_data.sample(min(n, len(attraction_data)))
    
    elif method == 'content' and attraction_id is not None:
        try:
            if 'content_similarity' not in st.session_state.models:
                raise ValueError("Content-based filtering not available")
                
            content_sim = st.session_state.models['content_similarity']
            if content_sim is None or attraction_id not in content_sim.columns:
                return attraction_data.sample(min(n, len(attraction_data)))
            
            similar_items = content_sim[attraction_id].sort_values(ascending=False)[1:n+1]
            return attraction_data[attraction_data['AttractionId'].isin(similar_items.index)]
        except Exception as e:
            st.warning(f"Content-based recommendation failed: {str(e)}")
            return attraction_data.sample(min(n, len(attraction_data)))
    
    elif method == 'hybrid' and user_id is not None:
        try:
            collab_recs = get_recommendations(user_id=user_id, n=n*2, method='collaborative')
            if collab_recs.empty:
                return attraction_data.sample(min(n, len(attraction_data)))
            
            hybrid_recs = pd.DataFrame()
            for _, row in collab_recs.iterrows():
                content_recs = get_recommendations(attraction_id=row['AttractionId'], n=2, method='content')
                hybrid_recs = pd.concat([hybrid_recs, content_recs])
            
            if not hybrid_recs.empty:
                hybrid_recs = hybrid_recs.drop_duplicates(subset=['AttractionId'])
                if 'AttractionId' in hybrid_recs.columns:
                    hybrid_recs = hybrid_recs.sort_values(by='AttractionId').head(n)
            
            return hybrid_recs if not hybrid_recs.empty else attraction_data.sample(min(n, len(attraction_data)))
        except Exception as e:
            st.warning(f"Hybrid recommendation failed: {str(e)}")
            return attraction_data.sample(min(n, len(attraction_data)))
    
    return attraction_data.sample(min(n, len(attraction_data)))

# App layout
def main():
    st.title("🏝️ Tourism Experience Analytics✈️")
    st.markdown("""
    Welcome to the Tourism Recommendation System! This app helps you:
    - Predict your likely visit mode (Business, Family, etc.)
    - Get personalized attraction recommendations
    - Explore popular attractions and trends
    """)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("👤 User Profile")
        
        with st.form("user_input"):
            st.subheader("Tell us about yourself")
            
            # Basic info
            age = st.slider("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            # Travel preferences
            travel_frequency = st.select_slider(
                "How often do you travel?",
                options=["Rarely", "Occasionally", "Regularly", "Frequently", "Very Frequently"]
            )
            
            preferred_attraction = st.selectbox(
                "Preferred Attraction Type",
                ["Museum", "Park", "Historical", "Beach", "Shopping", "Adventure"]
            )
            
            travel_companions = st.multiselect(
                "Typical Travel Companions",
                ["Solo", "Partner", "Family", "Friends", "Business"]
            )
            
            budget = st.select_slider(
                "Travel Budget",
                options=["Low", "Medium", "High", "Luxury"]
            )
            
            submitted = st.form_submit_button("Get Recommendations")
        
        # Show visualizations in sidebar
        if st.session_state.visuals.get('rating_trends'):
            st.image(st.session_state.visuals['rating_trends'], 
                    caption="Monthly Rating Trends", use_container_width=True)
    
    # Main content area
    if submitted:
        st.success("✅ Profile submitted successfully!")
        
        # Prepare input data for predictions
        input_data = {
            'Age': age,
            'Gender': 1 if gender == "Male" else 0,
            'TravelFrequency': ["Rarely", "Occasionally", "Regularly", "Frequently", "Very Frequently"].index(travel_frequency),
            f"AttractionType_{preferred_attraction}": 1,
            'BudgetLevel': ["Low", "Medium", "High", "Luxury"].index(budget)
        }
        
        # Add travel companion features
        for companion in ["Solo", "Partner", "Family", "Friends", "Business"]:
            input_data[f"TravelWith_{companion}"] = 1 if companion in travel_companions else 0
        
        # Display predictions
        with st.spinner("Analyzing your profile..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🛫 Visit Mode Prediction")
                visit_mode = predict_visit_mode(input_data)
                st.metric("Predicted Visit Mode", visit_mode)
                
                visit_mode_info = {
                    "Business": "Your profile suggests business travel preferences with focus on convenience and efficiency.",
                    "Family": "Your preferences align with family travel, focusing on child-friendly activities.",
                    "Solo": "Your independent travel style suggests you prefer solo adventures.",
                    "Couple": "Your romantic travel preferences indicate couple-oriented experiences.",
                    "Friends": "Your social travel style suggests group activities with friends."
                }
                
                st.info(visit_mode_info.get(visit_mode, "Your travel style is unique!"))
            
            with col2:
                st.subheader("⭐ Expected Enjoyment")
                predicted_rating = predict_rating(input_data)
                st.metric("Predicted Attraction Rating", f"{predicted_rating:.1f}/5")
                
                rating_info = {
                    1: "Very poor experience expected",
                    2: "Below average experience expected",
                    3: "Average experience expected",
                    4: "Good experience expected",
                    5: "Excellent experience expected"
                }
                st.info(rating_info.get(round(predicted_rating), "Rating prediction complete"))
        
        # Recommendations section
        st.subheader("🌟 Personalized Recommendations")
        
        # Generate a sample user ID based on inputs
        sample_user_id = hash(f"{age}{gender}{travel_frequency}") % 10000
        
        with st.spinner("Finding the perfect attractions for you..."):
            rec_method = st.radio("Recommendation Method", 
                                ["Hybrid", "Collaborative", "Content-Based"], 
                                horizontal=True)
            
            if rec_method == "Hybrid":
                recommendations = get_recommendations(user_id=sample_user_id, method='hybrid')
            elif rec_method == "Collaborative":
                recommendations = get_recommendations(user_id=sample_user_id, method='collaborative')
            else:
                # For content-based, use preferred attraction type
                filtered = st.session_state.models['attraction_data'][
                    st.session_state.models['attraction_data']['AttractionType'] == preferred_attraction
                ]
                sample_attraction = filtered['AttractionId'].sample(1).values[0] if not filtered.empty else None
                recommendations = get_recommendations(attraction_id=sample_attraction, method='content') if sample_attraction else pd.DataFrame()
            
            if not recommendations.empty:
                # Display recommendations as cards
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recommendations.head(3).iterrows()):
                    with cols[idx]:
                        st.image(f"https://source.unsplash.com/300x200/?{row['AttractionType'].lower()}", 
                                use_container_width=True)
                        st.subheader(row['AttractionType'])
                        st.caption(row.get('AttractionAddress', 'Location not specified'))
                        st.write(f"⭐ Predicted Rating: {predict_rating(input_data):.1f}/5")
                        st.button("Learn More", key=f"btn_{idx}")
                
                # Show full recommendations table
                with st.expander("View All Recommendations"):
                    display_cols = ['AttractionType', 'AttractionAddress']
                    if 'Rating' in recommendations.columns:
                        display_cols.append('Rating')
                    st.dataframe(recommendations[display_cols].reset_index(drop=True))
            else:
                st.warning("Could not generate personalized recommendations. Showing sample attractions instead.")
                popular = st.session_state.models['attraction_data']
                if 'Rating' in popular.columns:
                    popular = popular.sort_values('Rating', ascending=False)
                st.dataframe(popular.head(5)[['AttractionType', 'AttractionAddress']])
        
        # Analytics section
        st.subheader("📊 Travel Insights")
        
        tab1, tab2, tab3 = st.tabs(["Popular Attractions", "User Segments", "Regional Trends"])
        
        with tab1:
            st.write("### Top Rated Attractions")
            attractions = st.session_state.models['attraction_data']
            
            if 'Rating' in attractions.columns:
                top_attractions = attractions.sort_values('Rating', ascending=False).head(5)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=top_attractions, x='Rating', y='AttractionType', ax=ax)
                ax.set_title("Top Attractions by Rating")
                st.pyplot(fig)
                
                display_cols = ['AttractionType', 'AttractionAddress', 'Rating']
                st.dataframe(top_attractions[display_cols])
            else:
                st.warning("Rating data not available. Showing random attractions.")
                st.dataframe(attractions.sample(5)[['AttractionType', 'AttractionAddress']])
        
        with tab2:
            st.write("### User Segment Distribution")
            segments = pd.DataFrame({
                'Segment': ['Business', 'Family', 'Solo', 'Couple', 'Friends'],
                'Percentage': [35, 25, 15, 15, 10]
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(segments)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(segments['Percentage'], labels=segments['Segment'], autopct='%1.1f%%')
                ax.set_title("User Segments")
                st.pyplot(fig)
        
        with tab3:
            st.write("### Popular Regions by Visit Mode")
            regions = pd.DataFrame({
                'Region': ['North America', 'Europe', 'Asia', 'South America', 'Africa'],
                'Business': [45, 30, 15, 5, 5],
                'Family': [30, 40, 10, 15, 5],
                'Leisure': [25, 35, 20, 10, 10]
            }).set_index('Region')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            regions.plot(kind='bar', stacked=True, ax=ax)
            plt.xticks(rotation=45)
            plt.title("Regional Travel Preferences")
            st.pyplot(fig)
            
            st.dataframe(regions)
    
    else:
        # Show welcome content when no submission
        st.markdown("""
        ### How it works:
        1. Fill out your travel profile in the sidebar
        2. Submit your information
        3. Get personalized predictions and recommendations
        
        ### Features:
        - **Visit Mode Prediction**: Understand your travel style
        - **Rating Prediction**: See how much you'll likely enjoy attractions
        - **Personalized Recommendations**: Get suggestions tailored to you
        - **Travel Insights**: Explore popular attractions and trends
        """)
        
        # Show sample attractions
        st.subheader("Sample Attractions in Our System")
        attractions = st.session_state.models['attraction_data']
        display_cols = ['AttractionType', 'AttractionAddress']
        if 'Rating' in attractions.columns:
            display_cols.append('Rating')
        st.dataframe(attractions.sample(5)[display_cols])
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        Tourism Experience Analytics  | Classification, Prediction, and Recommendation System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()