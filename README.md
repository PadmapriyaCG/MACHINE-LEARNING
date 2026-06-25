# KNN Classification Model — Heart Disease Dataset

This project implements a K-Nearest Neighbors (KNN) classification model to predict the presence of heart disease based on various patient attributes. The model is trained and evaluated using the `heart.csv` dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Notebook Steps](#notebook-steps)
- [How to Use](#how-to-use)
- [Output Files](#output-files)

## Project Overview
This Jupyter Notebook (compatible with Google Colab) guides you through the process of building, training, evaluating, and saving a KNN classification model. The goal is to classify whether a patient has heart disease (target=1) or not (target=0) using various physiological and clinical features.

## Dataset
The project uses the `heart.csv` dataset, which contains 14 features related to heart disease diagnosis. Key features include:
- `age`: Age of the patient (years)
- `sex`: Sex (1 = Male, 0 = Female)
- `cp`: Chest pain type (0–3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1=True, 0=False)
- `restecg`: Resting ECG results (0–2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1=Yes, 0=No)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment (0–2)
- `ca`: Number of major vessels (0–4)
- `thal`: Thalassemia type (0=Normal, 1=Fixed defect, 2=Reversible defect)
- `target`: Heart Disease (1=Yes, 0=No)

## Notebook Steps
The notebook is structured into the following steps:
1.  **Install & Import Libraries**: Loads necessary Python libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn.
2.  **Load Dataset**: Prompts the user to upload the `heart.csv` file (e.g., in Google Colab).
3.  **Understand the Data**: Performs initial data exploration, including displaying column headers, data types, statistical summaries, missing values, and target class distribution.
4.  **Visualizations**: Generates visualizations such as target class distribution, age distribution, and a feature correlation heatmap.
5.  **Preprocessing**: Separates features (X) and target (y), performs an 80/20 train-test split, and scales features using `StandardScaler`.
6.  **Find Best K**: Uses cross-validation to determine the optimal `k` value for the KNN model.
7.  **Train Final KNN Model**: Trains the KNN classifier using the best `k` found.
8.  **Evaluate the Model**: Assesses model performance using accuracy score, classification report, and a confusion matrix.
9.  **Save the Model**: Saves the trained KNN model and the `StandardScaler` using `joblib`.
10. **Predict on New Data**: Demonstrates how to use the saved model to make predictions on new, unseen patient data.

## How to Use
1.  **Open the Notebook**: Open this `.ipynb` file in Google Colab or any Jupyter environment.
2.  **Run Cells Sequentially**: Execute each code cell in order.
3.  **Upload `heart.csv`**: When prompted in 'STEP 2', upload your `heart.csv` file.
4.  **Review Outputs**: Observe the data insights, plots, model performance, and sample predictions.

## Output Files
Upon execution, the notebook generates the following files:
-   `eda_plots.png`: Image file containing EDA visualizations.
-   `k_vs_accuracy.png`: Plot showing cross-validation accuracy for different K values.
-   `confusion_matrix.png`: Image file of the confusion matrix for model evaluation.
-   `knn_heart_model.pkl`: The saved trained KNN classification model.
-   `scaler.pkl`: The saved `StandardScaler` object used for feature scaling.
