# Diabetes Prediction

This project aims to predict whether an individual has diabetes based on their health parameters using machine learning. The model is built using the PIMA Indians Diabetes Database and is implemented in Python with the help of popular libraries like `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction System](#prediction-system)
- [Technologies Used](#technologies-used)


## Installation

To get started, clone this repository and install the required dependencies using the following commands:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
2. Install dependencies:
     ```bash
      pip install -r requirements.txt

## Usage

This project provides a machine learning model for predicting diabetes in a person based on the following health parameters:

Number of pregnancies
Plasma glucose concentration
Diastolic blood pressure
Triceps skinfold thickness
Insulin level
Body mass index (BMI)
Diabetes pedigree function
Age

You can use the model to predict diabetes by providing input data through the Python script.



  ## Data Collection
  
This project uses the PIMA Indians Diabetes Database, which contains medical data of female PIMA Indians who are at least 21 years old. The dataset includes the following columns:

Pregnancies: Number of pregnancies
Glucose: Plasma glucose concentration
BloodPressure: Diastolic blood pressure
SkinThickness: Triceps skinfold thickness
Insulin: 2-Hour serum insulin
BMI: Body mass index
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age of the person
Outcome: 1 if the person has diabetes, 0 otherwise
The dataset is loaded from a CSV file and analyzed using pandas.

## Model Training

We use a Support Vector Machine (SVM) classifier to train the model. The SVM is trained on the features of the dataset (excluding the 'Outcome' column) after the data is standardized using `StandardScaler`. The dataset is split into training and testing sets, and the model is trained on the training data.

```python
from sklearn import svm

# Creating an SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Training the model with the training data
classifier.fit(X_train, Y_train)
```
## Model Evaluation
After training the model, its performance is evaluated using accuracy scores on both the training and test data. The accuracy score is calculated by comparing the model's predictions to the actual labels of the data.
```python
from sklearn.metrics import accuracy_score

# Making predictions on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Making predictions on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Printing accuracy scores
print("Training Data Accuracy:", training_data_accuracy)
print("Test Data Accuracy:", test_data_accuracy)
```
## Prediction System
A prediction system is included that allows the user to input their health parameters and predict whether they are diabetic or not. The input data is standardized and passed through the trained model to generate the prediction.

```python
# Example input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Converting input data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing the input data
std_data = scaler.transform(input_data_reshaped)

# Making a prediction
prediction = classifier.predict(std_data)

# Displaying the result
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")

```
## Technologies Used
Python
scikit-learn (for machine learning)
pandas (for data manipulation)
numpy (for numerical operations)
StandardScaler (for data preprocessing)
