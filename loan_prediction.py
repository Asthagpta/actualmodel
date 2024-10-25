import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
data= {
    'Name':['Alice','Bob','Charlie','David','Eva','Frank','Grace','Henry','Isla','Jack']*5,
    'Income':[30000,45000,50000,60000,35000,48000,52000,58000,47000,39000]*5,
    'Credit_Score':[650,700,720,680,620,600,750,710,640,730]*5,
    'Loan_Approved':[1,1,1,1,0,0,1,1,0,1]*5
}
#create datafame
df=pd.DataFrame(data)
#feaatures and target variable
X=df[['Income','Credit_Score']]
Y=df['Loan_Approved']
#split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
#create and train the logistic regression model 
model=LogisticRegression()
model.fit(X_train, Y_train)
#save the model
joblib.dump(model, 'loan_prediction_model.pkl')
print("model trained and saved!")