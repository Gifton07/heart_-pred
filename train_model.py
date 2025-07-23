import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Define features (X) and target (y)
# The order of columns in X is important and must be maintained
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Create and fit the scaler on the entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Save the model and the scaler to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been trained and saved as model.pkl and scaler.pkl")