# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# ---------- Create sample training data ----------
# In real life this comes from field data / datasets
# For prototype we generate realistic synthetic data
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'temperature':  np.random.uniform(15, 40, n),
    'humidity':     np.random.uniform(30, 100, n),
    'rainfall':     np.random.uniform(0, 200, n),
    'soil_moisture':np.random.uniform(10, 80, n),
    'soil_ph':      np.random.uniform(4.5, 8.5, n),
})

# Rule-based labels (simulating real disease conditions)
def label(row):
    # Late Blight favours high humidity + moderate temp
    if row['humidity'] > 70 and 18 < row['temperature'] < 30 and row['rainfall'] > 10:
        return 2  # High Risk
    elif row['humidity'] > 55 or row['rainfall'] > 5:
        return 1  # Medium Risk
    else:
        return 0  # Low Risk

data['risk'] = data.apply(label, axis=1)

# ---------- Train model ----------
X = data.drop('risk', axis=1)
y = data['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- Save model ----------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
print(f"Accuracy: {model.score(X_test, y_test)*100:.1f}%")
