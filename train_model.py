import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('dataCancer.csv', header=None)

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

X = X.apply(pd.to_numeric, errors='coerce').dropna()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=4,
    random_state=42
)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved successfully")