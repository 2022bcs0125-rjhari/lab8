import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"])

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# Predictions
pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

metrics = {
    "RMSE": rmse,
    "R2": r2,
    "dataset_size": len(df)
}

print(metrics)

with open("metrics.json", "w") as f:
    json.dump(metrics, f)