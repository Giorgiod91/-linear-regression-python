import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

import pandas as pd


data = pd.read_csv('heart_attack_germany.csv')
print(data.head())



X_features = [
    "State", "Age_Group", "Year", "Gender", "BMI", "Smoking_Status", 
    "Alcohol_Consumption", "Physical_Activity_Level", "Diet_Quality", 
    "Family_History", "Hypertension", "Cholesterol_Level", "Diabetes", 
    "Urban_Rural", "Socioeconomic_Status", "Air_Pollution_Index", 
    "Stress_Level", "Healthcare_Access", "Education_Level", 
    "Employment_Status", "Region_Heart_Attack_Rate"
]


y_target = "Heart_Attack_Incidence"


X = data[X_features]
y = data[y_target]

X_encoded = pd.get_dummies(X, drop_first=True)


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Split the dataset into training and testing sets
X_train, X_test , y_train , y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)





