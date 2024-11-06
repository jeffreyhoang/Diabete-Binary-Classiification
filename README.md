# Logistic Regression for Binary Classification

This project implements logistic regression to predict diabetes using health metrics data, covering three approaches: Scikit-learn, custom NumPy implementation, and PyTorch.

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

The goal is to predict whether a patient has diabetes (1) or not (0).

## Project Tasks

1. **Data Preprocessing**: Scaled features using Scikit-learn’s pipeline.
2. **Data Splitting**: Split data into training (60%), validation (20%), and testing (20%) sets.
3. **Logistic Regression with Scikit-learn**: Fit and evaluated a logistic regression model.
4. **Logistic Regression from Scratch with NumPy**: Implemented logistic regression with SGD, custom cross-entropy loss, and gradient.
5. **Logistic Regression with PyTorch**: Created and trained a logistic regression model using PyTorch.

## How to Run the Project

1. Upload `diabetes2.csv` to Google Drive at `MyDrive/Colab Notebooks/datasets/diabetes2.csv`.
2. Run the notebook `yourLastName_yourFirstName_assignment2.ipynb` in Google Colab or Jupyter Notebook.

## Results

Learning curves and cross-entropy errors on training, validation, and testing sets are provided for each model. Hyperparameters were tuned to achieve results close to the Scikit-learn model.

### Logistic Regression from Scikit-Learn Results
Bias: -0.79302711

Weights: [ 0.23022922  0.99633315 -0.19857463 -0.07201953 -0.04436089  0.85422255
   0.1218909   0.39738786] 

Training Log Loss: 0.4752883284278347

Validation Log Loss: 0.47892380385180466

Testing Log Loss: 0.4906038530381513

<img width="478" alt="Screenshot 2024-11-06 at 1 43 20 AM" src="https://github.com/user-attachments/assets/cd64c059-d254-468d-8395-24fcbf3dc935">

### Logistic Regression from Stochastic Gradient Descent Results
Bias: -0.79775049

Weights: [0.25152051, 1.00913526, -0.17242659, -0.1130947, -0.02640905, 0.8715367, 0.1268932, 0.36254751]

Training Error: 0.3488605406535597

Validation Error: 0.476013858478927

Testing Error: 0.4925132822283009

<img width="703" alt="Screenshot 2024-11-06 at 1 45 57 AM" src="https://github.com/user-attachments/assets/ccf241cc-3798-49c4-9cc8-d4b4273a4ddf">

### Logistic Regression from PyTorch
Bias: -0.5407750606536865

Weights: [-0.2187,  0.2810,  0.9510, -0.1168, -0.0457, -0.0172,  0.7122,  0.1306,
          0.3204]

Training Error: 0.4771

Validation Error: 0.4716

Testing Error: 0.4871

<img width="710" alt="Screenshot 2024-11-06 at 1 49 04 AM" src="https://github.com/user-attachments/assets/4a3478f8-52f2-4a6d-9710-ae23c6cf1915">

