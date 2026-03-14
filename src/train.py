import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

df = pd.read_csv("data/housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

metrics = {
    "RMSE": rmse,
    "R2": r2,
    "dataset_size": len(df)
}

print(metrics)

with open("metrics.json", "w") as f:
    json.dump(metrics, f)