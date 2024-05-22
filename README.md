# Credit Risk Analysis - README

## Overview

This project is designed to perform a credit risk analysis using machine learning techniques. It aims to predict the probability of default on a loan by evaluating various borrower attributes. The analysis includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

1. [Installation]
2. [Dataset]
3. [Project Structure]
4. [Usage]
5. [Model Evaluation]
6. [Contributing]
7. [License]

## Installation

To run this project, you need to have Python 3.x installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Dataset

The dataset used for this analysis should contain the following columns:

- `loan_id`: Unique identifier for the loan.
- `borrower_id`: Unique identifier for the borrower.
- `loan_amount`: The amount of the loan.
- `interest_rate`: The interest rate of the loan.
- `term`: Duration of the loan.
- `employment_length`: Number of years the borrower has been employed.
- `annual_income`: Annual income of the borrower.
- `credit_score`: Credit score of the borrower.
- `default`: Binary variable indicating whether the loan defaulted (1) or not (0).

The dataset should be in CSV format. You can replace `data/credit_risk_data.csv` with the path to your dataset.

## Project Structure

```bash
credit-risk-analysis/
│
├── data/
│   └── credit_risk_data.csv       # Dataset file
│
├── notebooks/
│   ├── data_preprocessing.ipynb   # Notebook for data preprocessing
│   ├── exploratory_analysis.ipynb # Notebook for exploratory data analysis
│   ├── model_training.ipynb       # Notebook for model training
│   └── model_evaluation.ipynb     # Notebook for model evaluation
│
├── scripts/
│   ├── preprocess.py              # Script for data preprocessing
│   ├── train_model.py             # Script for model training
│   └── evaluate_model.py          # Script for model evaluation
│
├── README.md                      # Readme file
└── requirements.txt               # List of required packages
```

## Usage

### Data Preprocessing

Run the `data_preprocessing.ipynb` notebook or the `preprocess.py` script to clean and preprocess the data. This includes handling missing values, encoding categorical variables, and feature scaling.

```bash
python scripts/preprocess.py
```

### Exploratory Data Analysis

The `exploratory_analysis.ipynb` notebook contains visualizations and statistical summaries to understand the dataset better.

### Model Training

Use the `model_training.ipynb` notebook or the `train_model.py` script to train the machine learning model. Various algorithms such as Logistic Regression, Decision Trees, and Random Forests are implemented and compared.

```bash
python scripts/train_model.py
```

### Model Evaluation

Evaluate the trained models using the `model_evaluation.ipynb` notebook or the `evaluate_model.py` script. This step includes calculating metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

```bash
python scripts/evaluate_model.py
```

## Model Evaluation

The performance of each model is assessed using the following metrics:

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Proportion of true positive instances among the instances classified as positive.
- **Recall**: Proportion of true positive instances among the actual positive instances.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

Results and model comparisons are documented in the `model_evaluation.ipynb` notebook.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

For any questions or issues, please contact [bhoi.nikhil2002@gmail.com].

Happy coding!
