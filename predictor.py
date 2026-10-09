import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = fetch_california_housing(as_frame=True)

df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)


print("\nRandom Forest Results:")
print("MAE  :", rf_mae)
print("MSE  :", rf_mse)
print("RMSE :", rf_rmse)
print("R²   :", rf_r2)


#XGBoost model

xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)


xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred)


print("\nXGBoost Results:")
print("MAE  :", xgb_mae)
print("MSE  :", xgb_mse)
print("RMSE :", xgb_rmse)
print("R²   :", xgb_r2)