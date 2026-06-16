import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Tanvi\Cardiocare AI\Backend\heart.csv")

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
print(data.columns.tolist())
print(data.shape)





