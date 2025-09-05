import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset (adjust path if needed)
data = pd.read_csv("../student-mat.csv")

# Use a few features to predict final grade (G3)
X = data[["studytime", "absences", "failures"]]
y = data["G3"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
