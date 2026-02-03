# 🥗 Diet Recommendation System

A standardized diet recommendation system built with **Streamlit** and **Machine Learning** (K-Means Clustering) to suggest recipes based on your nutritional requirements.

## 📊 Dataset

The dataset used in this project is sourced from the **Food.com Recipes and Reviews** dataset on Kaggle. It originally consists of 522,715 rows and 28 columns containing detailed recipe information.

-   **Source**: [Kaggle - Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv)
-   **Description**: Contains recipes, nutritional information, and other metadata.

## 🚀 Features

-   **Personalized Recommendations**: Input your desired nutrient levels (Calories, Fat, Protein, etc.) as "Low", "Medium", "High", or "Very High".
-   **Machine Learning**: Uses K-Means clustering to group similar recipes and `StandardScaler` for data normalization.
-   **Interactive UI**: Built with Streamlit for a clean, responsive web interface.
-   **Smart Descriptions**: Automatically cleans and displays recipe descriptions.

## 🛠️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AdarshTomy1/Diet-Recommendation-System.git
    cd Diet-Recommendation-System
    ```

2.  **Install Dependencies**
    Make sure you have Python installed, then run:
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```

3.  **Run the Application**
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **View locally**
    Open your browser to `http://localhost:8501`.

## 📂 Project Structure

-   `streamlit_app.py`: The main application file.
-   `kmeans_model_diet.pkl`: Pre-trained K-Means model.
-   `scaler_diet.pkl`: Pre-trained scaler for nutrient normalization.
-   `recipes_clusters_diet.csv`: The processed dataset used by the app.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
