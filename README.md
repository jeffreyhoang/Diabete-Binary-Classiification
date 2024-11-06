# Logistic Regression for Binary Classification

This project implements and compares logistic regression models for predicting diabetes in patients using a dataset of health metrics. The model was built as part of an assignment for the 4210 Fall 2024 class and covers three main approaches: using Scikit-learn, a custom implementation with NumPy, and PyTorch.

## Project Overview

The objective of this project is to understand key concepts in machine learning, including logistic regression, gradient descent optimization, and cross-entropy loss, while applying them to a real-world binary classification task. We use a dataset of 768 diabetes patients, each with 8 baseline variables (e.g., glucose levels, blood pressure, BMI) to predict if a patient is positive (1) or negative (0) for diabetes.

## Dataset

The dataset (`diabetes2.csv`) contains the following features:
- **Pregnancies**
- **Glucose Levels**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI**
- **Diabetes Pedigree Function**
- **Age**

## Project Tasks

### Task 1: Data Preprocessing
- Preprocessed the data with Scikit-learn's pipeline, including feature scaling.

### Task 2: Data Splitting
- Split the data into three sets: training (60%), validation (20%), and testing (20%).

### Task 3: Logistic Regression with Scikit-learn
- Used Scikit-learn’s `LogisticRegression()` to fit a logistic regression model.
- Calculated log-loss errors for training, validation, and testing sets.
- Displayed the confusion matrix for the model’s predictions on the testing set.

### Task 4: Logistic Regression from Scratch with NumPy
- Implemented logistic regression with stochastic gradient descent (SGD) in NumPy.
- Implemented cross-entropy loss and its gradient.
- Plotted learning curves showing training and validation errors over batches.
- Tuned hyperparameters to closely match the Scikit-learn model's results.
- Printed cross-entropy errors on training, validation, and testing datasets.

### Task 5: Logistic Regression with PyTorch
- Created and trained a logistic regression model in PyTorch.
- Used available PyTorch methods like `torch.nn.Linear()`, `torch.sigmoid()`, and `torch.nn.BCELoss()`.
- Plotted learning curves showing training and validation errors across epochs.
- Tuned hyperparameters to achieve performance similar to the Scikit-learn model.
- Printed cross-entropy errors on training, validation, and testing datasets.

## How to Run the Project

1. Ensure you have the necessary dependencies:
   - Python 3.7+
   - NumPy
   - Scikit-learn
   - PyTorch
   - Matplotlib
2. Upload the dataset (`diabetes2.csv`) to Google Drive in the `MyDrive/Colab Notebooks/datasets/` directory.
3. Run the iPython notebook `yourLastName_yourFirstName_assignment2.ipynb` in Google Colab or Jupyter Notebook to execute the code for each task.

## Results

This project provides learning curves for each model and prints the cross-entropy errors for training, validation, and testing sets across all three approaches. Each approach was tuned to match or come close to the Scikit-learn model’s performance.

## File Structure

- `diabetes2.csv`: Dataset with diabetes patient information.
- `yourLastName_yourFirstName_assignment2.ipynb`: Main notebook file containing code and explanations for each task.

## Notes

1. Non-executable programs will result in a grade of zero.
2. Regular Python program files (`.py`) are not acceptable; the notebook format (`.ipynb`) must be used.
3. Ensure your notebook is properly commented.
4. Name your submission in the format: `yourLastName_yourFirstName_assignment2.ipynb`.

## License

This project is intended for educational purposes only.
# Diabete-Binary-Classiification
 
