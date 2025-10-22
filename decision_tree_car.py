#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import  required libraries

#For data handling
import pandas as pd

#For splitting Dataset
from sklearn.model_selection import train_test_split

#For ML model
from sklearn.tree import DecisionTreeClassifier

#For evaluation
from sklearn.metrics import accuracy_score, classification_report

#For visualization Decision Tree
from sklearn import tree

#For plotting graphs
import matplotlib.pyplot as plt

#convert categorical text into numbers
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("car_data.csv")
print(df.head())

#encode categorical data
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

#Define features (X) & target (y)
X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']

#Split data into training and testing sets
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and Train Decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state= 42)
model.fit(X_train, y_train)
print("\n Model trained successfully")

#Make predicitions
y_pred = model.predict(X_test)

#Evaluate performance
print("\n Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n:", classification_report(y_test, y_pred))

#Visualization
plt.figure(figsize=(10,6))
tree.plot_tree(
    model,
    feature_names= ['Gender', 'Age', 'AnnualSalary'],
    class_names= ['No', 'Yes'],
    filled= True
)
plt.show()





