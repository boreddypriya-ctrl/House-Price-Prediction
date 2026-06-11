import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset Preview:")
print(data.head())

# 2. Features (X) and Target (y)
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model creation
model = LinearRegression()

# 5. Train model
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Accuracy check (error)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# 8. Test with custom input
sample_house = [[1600, 3, 10]]
predicted_price = model.predict(sample_house)

print("\nPredicted Price for sample house:", predicted_price[0])