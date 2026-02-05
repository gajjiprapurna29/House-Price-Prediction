# House Price Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")
print("Dataset:")
print(data)

# Features and target
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("\nPredicted Prices:")
print(predictions)

# Predict new house price
new_house = [[1600, 3, 2]]
predicted_price = model.predict(new_house)
print("\nPredicted price for new house:", predicted_price[0])

# Visualization
plt.scatter(data['Area'], data['Price'])
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.title("House Area vs Price")
plt.show()
