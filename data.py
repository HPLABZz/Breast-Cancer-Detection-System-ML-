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

# print(f"Train Accuracy: {rf_train_acc*100:.4f}")
# print(f"Test Accuracy: {rf_test_acc*100:.4f}")
# print("Confusion Matrix:\n", rf_conf_matrix)

joblib.dump(rf_model, "models/random_forest_model.pkl")

input_data = (923169,9.683,19.34,61.05,285.7,0.08491,0.0503,0.02337,0.009615,0.158,0.06235,0.2957,1.363,2.054,18.24,0.00744,0.01123,0.02337,0.009615,0.02203,0.004154,10.93,25.59,69.1,364.2,0.1199,0.09546,0.0935,0.03846,0.2552,0.0792
)

def predict_result_RF():
    if len(input_data) == X.shape[1]:
        input_array = np.asarray(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred_rf = rf_model.predict(input_scaled)
        
        if pred_rf[0]=='M':return "Random Forest Prediction: Malignant"
        else:       return "RandomForest Prediction: Benign"
    else:
        return "Invalid input: expected 32 features only."

if __name__ == "__main__":
    result = print(predict_result_RF())
    print(result)
