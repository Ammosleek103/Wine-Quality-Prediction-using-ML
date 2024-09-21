# Wine-Quality-Prediction-using-ML

---

## Wine Quality Check - End-to-End Machine Learning Project and Streamlit App

### Overview

This project uses **machine learning** to predict the quality of red wine based on several physicochemical features such as pH, alcohol content, citric acid, etc. The project includes data cleaning, exploratory data analysis, feature engineering, model building, and deployment using **Streamlit**.

### Features

- **End-to-end machine learning pipeline** for red wine quality prediction.
- **Multiple regression algorithms** are tested, with **Extra Trees Regressor** selected as the final model.
- **Hyperparameter tuning** using **RandomizedSearchCV** to improve model accuracy.
- A **Streamlit web app** for users to predict wine quality based on input features.
- The project is implemented in **Python**, and the model is saved and loaded for inference in the web app.

---

### Dataset

The dataset used for this project is the **Wine Quality Dataset**, sourced from **Kaggle**. It contains 1599 samples of red wine with 11 physicochemical features, and the target variable is wine quality (a score between 0 and 10).

- **Source**: [Wine Quality Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

#### Data Features:

- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **density**
- **pH**
- **sulphates**
- **alcohol**
- **quality** (target)

---

### Project Structure

```bash
.
├── app.py                   # Streamlit app code
├── model_training.ipynb      # Jupyter notebook for model training
├── extra_trees_model.pkl     # Saved model file for prediction
├── winequality-red.csv       # Wine quality dataset (from Kaggle)
├── requirements.txt          # Required libraries for the project
└── README.md                 # Project readme file
```

---

### Steps for Model Building

1. **Look at the Big Picture**  
   Understand the problem, target prediction, and explore the dataset.

2. **Get the Data**  
   Load the dataset from **Kaggle** and check for any missing or duplicate values. Clean the data if necessary.

3. **Discover and Visualize the Data**  
   Perform exploratory data analysis using correlation matrix and feature visualizations to understand relationships between features and the target.

4. **Prepare the Data for Machine Learning Algorithms**  
   Perform data preprocessing such as scaling, normalization, and handling skewed features.

5. **Select a Model and Train It**  
   Train multiple models like **Random Forest**, **Gradient Boosting**, and **Extra Trees**. Use cross-validation to find the best-performing model.

6. **Fine-tune the Model**  
   Use **RandomizedSearchCV** to tune hyperparameters and further improve the model's accuracy.

7. **Test the Model on the Test Dataset**  
   Evaluate the final model on the test data and save the trained model.

8. **Deploy the Model Using Streamlit**  
   Create a **Streamlit** app for users to input wine features and get quality predictions.

---

### Installation

To run the project locally, follow the steps below:

1. **Clone the repository**:

```bash
git clone https://github.com/Ammosleek103/Wine-Quality-Prediction-using-ML.git
cd Wine-Quality-Prediction-using-ML
```

2. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook to train the model**:

If you want to retrain the model or modify the training process, open the `model_training.ipynb` notebook in Jupyter and follow the steps to build the model.

4. **Run the Streamlit app**:

Once the model is trained and saved, you can run the **Streamlit** app to make predictions.

```bash
streamlit run app.py
```

---

### Usage

After running the Streamlit app, you will be able to input values for the wine features like acidity, alcohol content, etc., and the app will output the predicted quality of the wine.

### Example Input for the App

- **Fixed Acidity**: 7.4
- **Volatile Acidity**: 0.7
- **Citric Acid**: 0
- **Residual Sugar**: 1.9
- **Chlorides**: 0.076
- **Free Sulfur Dioxide**: 11
- **Total Sulfur Dioxide**: 34
- **Density**: 0.9978
- **pH**: 3.51
- **Sulphates**: 0.56
- **Alcohol**: 9.4

Click the **Predict** button, and the app will display the predicted wine quality.

---

### Model Performance

The final model selected is the **Extra Trees Regressor**, which was tuned using **RandomizedSearchCV** to achieve the best performance. Here are the key evaluation metrics:

- **R-squared (R²)**: 0.55 (This may vary depending on your random splits and parameter tuning)
- **Mean Squared Error (MSE)**: 0.37

---

### Future Improvements

- Add more data visualization techniques in the Streamlit app.
- Implement feature engineering for better model accuracy.
- Try different machine learning algorithms and tuning techniques for further accuracy improvement.
- Deploy the app to **Heroku** or any other cloud platform for public access.

---

### License

This project is licensed under the **MIT License**.

---

### Author

**S. Charith**

If you have any questions or suggestions, feel free to contact me!

---

### Acknowledgments

Thanks to **Kaggle** for providing the dataset.

---

### Requirements

- Python 3.12.6
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

Ensure all dependencies are listed in the `requirements.txt`.

```bash
# Example requirements.txt
pandas
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---
