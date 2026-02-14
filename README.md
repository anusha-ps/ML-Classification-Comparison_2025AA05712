# ML-Classification-Comparison
## <b>Problem Statement</b>

The objective of this project is to build and compare multiple supervised machine learning classification models for predicting whether a breast tumor is Benign (B) or Malignant (M) using diagnostic features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

The project involves:
<li>Training 6 different classification models</li>
<li>Evaluating them using multiple performance metrics</li>
<li>Building an interactive Streamlit web application</li>
<li>Deploying the application on Streamlit Community Cloud</li>

</n></n>

## <b>Dataset Description</b>

Dataset Name: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository

The dataset contains features computed from a digitized image of a breast mass. These features describe characteristics of the cell nuclei present in the image.

Dataset Characteristics:
<li>Total Instances: 569</li>
<li>Total Features: 30 numerical features</li>
<li>Target Variable: diagnosis
    <li>M = Malignant (mapped to 1)</li>
    <li>B = Benign (mapped to 0)</li></li>
    
</n></n>

Train-Test Split:
<li> 80% Training Data </li>
<li> 20% Test Data </li>

Stratified sampling used to preserve class distribution with Random State = 42


## <b> Models Used </b>


The following six classification models were implemented and evaluated on the same dataset:
<li>Logistic Regression</li>
<li>Decision Tree Classifier</li>
<li>K-Nearest Neighbors (kNN)</li>
<li>Naive Bayes (Gaussian)</li>
<li>Random Forest (Ensemble)</li>
<li>XGBoost (Ensemble)</li>



All models were evaluated using:

<li>Accuracy</li>
<li>AUC Score</li>
<li>Precision</li>
<li>Recall</li>
<li>F1 Score</li>
<li>Matthews Correlation Coefficient (MCC)</li>

| ML Model Name            | Accuracy   | AUC    | Precision  | Recall | F1         | MCC        |
| ------------------------ | ---------- | ------ | ---------- | ------ | ---------- | ---------- |
| Logistic Regression      | 0.9649     | 0.9960 | 0.9750     | 0.9286 | 0.9512     | 0.9245     |
| Decision Tree            | 0.9211     | 0.9127 | 0.9024     | 0.8810 | 0.8916     | 0.8297     |
| kNN                      | 0.9561     | 0.9828 | 0.9744     | 0.9048 | 0.9383     | 0.9058     |
| Naive Bayes              | 0.9211     | 0.9894 | 0.9231     | 0.8571 | 0.8889     | 0.8292     |
| Random Forest (Ensemble) | 0.9649     | 0.9937 | 0.9750     | 0.9286 | 0.9512     | 0.9245     |
| XGBoost (Ensemble)       | **0.9737** | 0.9950 | **1.0000** | 0.9286 | **0.9630** | **0.9442** |

## <b> Observations on Model </b> 

| ML Model Name            | Observation about model performance                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performs extremely well with high AUC (0.996). Shows strong generalization and balanced precision-recall performance. Suitable for linearly separable medical datasets. |
| Decision Tree            | Lower performance compared to ensemble models. Prone to overfitting and less stable than Random Forest and XGBoost.                                                     |
| kNN                      | Strong performance with high precision. Sensitive to feature scaling. Performs well but slightly lower recall compared to Logistic Regression.                          |
| Naive Bayes              | High AUC but lower recall and F1 score. Assumption of feature independence may reduce effectiveness.                                                                    |
| Random Forest (Ensemble) | Excellent performance with stable metrics. Reduces overfitting seen in Decision Tree. Performs nearly equal to Logistic Regression.                                     |
| XGBoost (Ensemble)       | Best performing model overall. Highest accuracy (97.37%), perfect precision (1.0), and highest MCC. Most robust and powerful model among all tested classifiers.        |

## Project Structure

<pre>
breast-cancer-ml-streamlit/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Required Python libraries
├── ML_Assignment2.ipynb
│
├── data/
│   ├── sample_test_data.csv        # Sample test dataset for experimentation
│   ├── breast_cancer_wisconsin.csv # Dataset used
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
└── results/
    └── model_reports.json</pre>

## Streamlit Application Features

The deployed Streamlit app includes:

📂 Download Sample Test Dataset

📤 Upload Test Dataset (CSV)

🤖 Model Selection Dropdown

⚡ Predict Button

📈 Evaluation Metrics Display

📊 Confusion Matrix

🧾 Classification Report

The app allows users to dynamically evaluate trained models on uploaded test data.

## How to Run the Project Locally
Step 1: Clone Repository
    <li> git clone <your-github-link> </li>
    <li> cd reast-cancer-ml-streamlit </li>

Step 2: Install Dependencies
    <li> pipinstall -r requirements.txt </li>

Step 3: Run Streamlit App
    <li> streamlit run app.py</li>

## Deployment

The application is deployed using Streamlit Community Cloud.

Steps:
  
  Push repository to GitHub
  
  Go to https://streamlit.io/cloud
  
  Select repository
  
  Choose app.py
  
  Deploy

## Conclusion

This project demonstrates:

<li>End-to-end ML pipeline development</li>

<li>Comparison of classical and ensemble methods</li>

<li>Importance of evaluation metrics in medical diagnosis</li>

<li>Real-world deployment using Streamlit</li>

<li>Among all models, XGBoost achieved the best overall performance, making it the most suitable model for breast cancer diagnosis in this dataset. </li>

## Libraries Used

Python

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib

Seaborn

Streamlit

Joblib
