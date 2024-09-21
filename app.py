import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Title and description
st.title("Wine Quality Prediction")
st.write("This app predicts the quality of red wine based on various chemical properties.")

# Sidebar for uploading data
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded dataset
    data = pd.read_csv(uploaded_file, delimiter=";")
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Feature selection and target variable
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Data preprocessing
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the model to file
    joblib.dump(model, "wine_quality_model.pkl")
    st.write("Model trained and saved successfully!")

    # Make predictions
    y_pred = model.predict(X_test)

    # Display evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    # Feature importance visualization
    st.write("Feature Importance:")
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importance_df.set_index('Feature'))

else:
    st.write("Please upload a CSV file to continue.")

