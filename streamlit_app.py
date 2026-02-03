import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Diet Recommendation System", layout="wide")

# Load models
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model_diet.pkl')
    scaler = joblib.load('scaler_diet.pkl')
    return kmeans_model, scaler

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('recipes_clusters_diet.csv')
    return df

try:
    kmeans_model, scaler = load_models()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure 'kmeans_model_diet.pkl', 'scaler_diet.pkl', and 'recipes_clusters_diet.csv' are in the directory.")
    st.stop()

# Cluster generation logic (if missing)
if 'Cluster' not in df.columns:
    with st.spinner("Generating 'Cluster' column (one-time setup)..."):
        nutrient_columns = [
            'Calories', 'FatContent', 'CholesterolContent', 'SodiumContent', 
            'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
        ]
        
        df_clustering = df.copy()
        
        for col in nutrient_columns:
            if col in df.columns:
                # Binning
                df_clustering[col] = pd.qcut(df[col], q=4, labels=[1, 2, 3, 4], duplicates='drop')
                df_clustering[col] = df_clustering[col].astype(int)
        
        X = df_clustering[nutrient_columns]
        X_scaled = scaler.transform(X)
        df['Cluster'] = kmeans_model.predict(X_scaled)
        # st.success("Clusters generated successfully!")

def recommend_recipes(nutrient_ranges):
    nutrient_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    
    user_nutrients = {
        'Calories': nutrient_mapping.get(nutrient_ranges['Calories']),
        'FatContent': nutrient_mapping.get(nutrient_ranges['FatContent']),
        'CholesterolContent': nutrient_mapping.get(nutrient_ranges['CholesterolContent']),
        'SodiumContent': nutrient_mapping.get(nutrient_ranges['SodiumContent']),
        'CarbohydrateContent': nutrient_mapping.get(nutrient_ranges['CarbohydrateContent']),
        'FiberContent': nutrient_mapping.get(nutrient_ranges['FiberContent']),
        'SugarContent': nutrient_mapping.get(nutrient_ranges['SugarContent']),
        'ProteinContent': nutrient_mapping.get(nutrient_ranges['ProteinContent'])
    }
    
    user_nutrients_df = pd.DataFrame([user_nutrients])
    user_nutrients_scaled = scaler.transform(user_nutrients_df)
    cluster_label = kmeans_model.predict(user_nutrients_scaled)[0]
    
    # Recommend
    recommended_recipes = df[df['Cluster'] == cluster_label].sample(n=3)
    return recommended_recipes[['Name', 'Cluster', 'Description', 'Calories', 'FatContent', 'ProteinContent']]

# UI
st.title("🥗 Diet Recommendation System")
st.markdown("Select your nutrient preferences to get customized recipe recommendations.")

with st.form("nutrient_form"):
    col1, col2 = st.columns(2)
    
    options = ['Low', 'Medium', 'High', 'Very High']
    
    with col1:
        calories = st.selectbox('Calories', options)
        fat = st.selectbox('Fat Content', options)
        cholesterol = st.selectbox('Cholesterol Content', options)
        sodium = st.selectbox('Sodium Content', options)
        
    with col2:
        carbohydrate = st.selectbox('Carbohydrate Content', options)
        fiber = st.selectbox('Fiber Content', options)
        sugar = st.selectbox('Sugar Content', options)
        protein = st.selectbox('Protein Content', options)
        
    submit_button = st.form_submit_button(label='Get Recommendations')

if submit_button:
    nutrient_ranges = {
        'Calories': calories,
        'FatContent': fat,
        'CholesterolContent': cholesterol,
        'SodiumContent': sodium,
        'CarbohydrateContent': carbohydrate,
        'FiberContent': fiber,
        'SugarContent': sugar,
        'ProteinContent': protein,
    }
    
    recommendations = recommend_recipes(nutrient_ranges)
    
    st.subheader(f"Recommended Recipes (Cluster {recommendations['Cluster'].iloc[0]})")
    st.dataframe(recommendations, hide_index=True)
    
    for idx, row in recommendations.iterrows():
        with st.expander(f"📄 {row['Name']}"):
            # Clean description
            description = str(row['Description']).replace('Make and share this', 'Enjoy this').replace('from Food.com.', '').replace('Food.com', '')
            st.write(f"**Description:** {description}")
            st.write(f"**Nutrients:** {row['Calories']} cal, {row['ProteinContent']}g Protein, {row['FatContent']}g Fat")
