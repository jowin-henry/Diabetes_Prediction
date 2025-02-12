Diabetes Prediction using SVM

This repository contains a machine learning model to predict diabetes using the PIMA Diabetes dataset. The model is built using Python and scikit-learn, employing a Support Vector Machine (SVM) classifier with a linear kernel.

Dataset

The dataset used is the PIMA Diabetes Dataset, which consists of medical predictor variables and an outcome label:

0 - Non-Diabetic

1 - Diabetic

Installation & Setup

Prerequisites

Ensure you have the following installed:

Python 3.x

Jupyter Notebook or Google Colab

Required Python Libraries: numpy, pandas, scikit-learn

Clone Repository

git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

Install Dependencies

pip install numpy pandas scikit-learn

Usage

Running the Model

Load the Dataset - The script reads the diabetes.csv file into a pandas DataFrame.

Preprocessing - Data is standardized using StandardScaler.

Model Training - A Support Vector Machine (SVM) model is trained using a linear kernel.

Evaluation - The model is evaluated on training and test data.

Prediction - A sample input is tested for diabetes prediction.

To run the script:

python diabetes_prediction.py

Model Performance

The model achieves a reasonable accuracy on both training and test datasets.

Accuracy is displayed after training the model.

Sample Prediction

The script includes a test case:

input_data = (5,166,72,19,175,25.8,0.587,51)

After preprocessing and prediction, it will output whether the person is diabetic or not.

Contributions

Feel free to contribute by opening an issue or submitting a pull request.

License

This project is licensed under the MIT License.

