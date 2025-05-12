# Heart_disease-Predictor

This project predicts the presence of heart disease in a patient using various machine learning models based on clinical features. The dataset used is a standard heart disease dataset containing several medical attributes and a target variable.

## Files

* `Heart_disease Predictor.ipynb`: Jupyter notebook implementing the entire ML pipeline.
* `heart.csv`: Dataset used for training and evaluating models.

## Dataset Overview

The dataset includes the following features:

* `age`: Age of the patient
* `sex`: Gender (1 = male; 0 = female)
* `cp`: Chest pain type (0–3)
* `trestbps`: Resting blood pressure (in mm Hg)
* `chol`: Serum cholesterol in mg/dl
* `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
* `restecg`: Resting electrocardiographic results (0–2)
* `thalach`: Maximum heart rate achieved
* `exang`: Exercise-induced angina (1 = yes; 0 = no)
* `oldpeak`: ST depression induced by exercise relative to rest
* `slope`: Slope of the peak exercise ST segment
* `ca`: Number of major vessels (0–4) colored by fluoroscopy
* `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
* `target`: Presence of heart disease (1 = yes; 0 = no)

## Models Trained

The notebook explores multiple classification models:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest (optimized via random state tuning)
* XGBoost
* Neural Network (using Keras)

## Best Performing Model

The **Random Forest Classifier** showed the highest accuracy after tuning for the optimal random state, making it the most reliable for prediction in this notebook.

## Libraries Used

* NumPy
* Pandas
* Matplotlib, Seaborn (for visualization)
* Scikit-learn
* XGBoost
* Keras (for Neural Network)

## How to Run

1. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost keras
   ```
2. Open the notebook:

   ```bash
   jupyter notebook "Heart_disease Predictor.ipynb"
   ```
3. Run all cells sequentially to train models and see visualizations.

   ![Heart Disease Prediction](https://github.com/user-attachments/assets/091d4e62-8aa8-4736-86de-c6d478509aa9)

