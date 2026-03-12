import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Age': [25, 45, 35, 50, 23, 40],
    'MonthlySpend': [2000, 5000, 4000, 6000, 1500, 4500],
    'Tenure': [1, 5, 3, 6, 1, 4],
    'Churn': [1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Age','MonthlySpend','Tenure']]
y = df['Churn']

model = LogisticRegression()

model.fit(X,y)

new_customer = [[30,3500,2]]

prediction = model.predict(new_customer)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Customer likely to leave")
else:
    print("Customer likely to stay")