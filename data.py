import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

cancer_data = pd.read_csv('dataCancer.csv', header=None)

X = cancer_data.iloc[:, 2:]
y = cancer_data.iloc[:, 1]

X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()

y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
rf_conf_matrix = confusion_matrix(y_test, rf_model.predict(X_test_scaled))

print(f"Train Accuracy: {rf_train_acc*100:.4f}")
print(f"Test Accuracy: {rf_test_acc*100:.4f}")
print("Confusion Matrix:\n", rf_conf_matrix)

joblib.dump(rf_model, "models/random_forest_model.pkl")

input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)

def predict_result_RF():
    if len(input_data) == X.shape[1]:
        input_array = np.asarray(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred_rf = rf_model.predict(input_scaled)
        
        if pred_rf[0]=='M':return "Random Forest Prediction: Malignant(Cancerous Cell)"
        else:       return "RandomForest Prediction: Benign(Non-cancerous Cell)"
    else:
        return "Invalid input: expected 32 features only."

if __name__ == "__main__":
    result = print(predict_result_RF())
    
