import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Area"]]
y = df["Price"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new house
area = [[1800]]

predicted_price = model.predict(area)

print("Predicted House Price:", predicted_price[0])