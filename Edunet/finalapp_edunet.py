import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Shaikh Abdul Mujeeb\Downloads\flood_dataset\flood.csv")

# Prepare data
df['FloodRisk'] = (df['FloodProbability'] > 0.5).astype(int)
X = df.drop(['FloodProbability', 'FloodRisk'], axis=1)
y = df['FloodRisk']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model (optional)
joblib.dump(model, "flood_model.pkl")

# Streamlit UI
st.title("🌊 Flood Risk Prediction App")

st.write("Enter environmental conditions to check if there’s flood risk.")

# Dynamically take input for each feature
user_input = {}
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    risk = "🚨 High Flood Risk" if pred == 1 else "✅ Low Flood Risk"
    st.subheader(risk)
