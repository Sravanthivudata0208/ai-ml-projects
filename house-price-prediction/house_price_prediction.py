import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [300000, 450000, 500000, 650000, 800000]
}

df = pd.DataFrame(data)

print(df)

X = df[['Area','Bedrooms']]
y = df['Price']

model = LinearRegression()

model.fit(X,y)

prediction = model.predict([[2000,3]])

print("Predicted House Price:", prediction[0])

plt.scatter(df['Area'],df['Price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()