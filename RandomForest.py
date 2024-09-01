import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

df= pd.read_csv('bus_maintenance_data.csv')
               
X = df[['Mileage', 'Bus_Age', 'Trips_per_Day', 'Avg_Speed', 'Fuel_Type', 'Road_Type', 'Stops']]
y = df['Maintenance_cost']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

RF = RandomForestClassifier(n_estimators=100, random_state=0)
RF= RF.fit(X_train,y_train)

predicted_values = RF.predict(X_test)
accuracy= accuracy_score(y_test, predicted_values)
print("RF's Accuracy is: ", accuracy)

from sklearn.model_selection import cross_val_score
cross_val_score(RF, X, y, cv=10)
print(cross_val_score(RF, X, y, cv=10))

import pickle

# Save the trained model to a file
with open('bus_maintenance_data.pkl', 'wb') as model_file:
    pickle.dump(RF, model_file)
