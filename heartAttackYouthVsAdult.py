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



age_group = data["Age_Group"]




plt.scatter(age_group, y, marker="x", c="r")
plt.title("showcase for now")
plt.ylabel("Heart Attacks")
plt.xlabel("Age")
plt.show()








