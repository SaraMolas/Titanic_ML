# Titanic_ML
Repository to explore the titanic dataset using a few supervised Machine Learning models

## Analytical methods used: 

- Logistic Regression
- Random Forests
- Gradient Boosted Trees

## Python packages used: 

- For data handling and processing: Pandas, Numpy
- For Statistics and Machine Learning: Scikit-Learn
- For plotting: Matplotlib.pyplot, Seaborn

## Description 

The publicly-available Titanic dataset contains the data of the passengers in the Titanic in 1912. The dataset contains 11 variables: Passenger full name, passenger class, sex, age, siblings or spouses on board, parents or children on board, ticket number, passenger fare, cabin number, port of embarkation and survival. The goal of this repository is to use different supervised Machine Learning approaches to predict survival (binary vector 1 = yes, 0 = no) based on some of these variables. The dataset is split into a training set with data from 891 passengers and a test set with 418 passengers. 

1. **LogisticRegressionTitanic**: jupyter notebook (in Python) that performs a visual exploration of the titanic dataset. Followed by some data processing, including the replacement of missing values by an estimate of the most likely value for each group, as well as the conversion of all categorical variables into numerical variables. Finally, a **logistic regression model** is trained with the training dataset, and **model optimization** is carried out by normalization of all variables using a Standard Scaler, and by **hyperparameter tuning** thanks to the GridSearchCV algorithm. This final model is used to predict the outcome (survival) of the test dataset. 
