import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

df[columns] = df[columns].replace(0,np.nan)

df.fillna(df.median(), inplace=True)
