import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("🍷 Wine Quality Dataset - KNN Classification (Auto Best K)")

# Load dataset
data = pd.read_csv("WineQT.csv")   # make sure filename matches
st.write("Dataset Preview:", data.head())

# Features and target
X = data.drop(columns=['quality'])
y = data['quality']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Find best K automatically
k_values = range(1, 21)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)

best_k = k_values[accuracies.index(max(accuracies))]
best_acc = max(accuracies)

st.write(f"✅ Best K Value: {best_k}")
st.write(f"✅ Best Accuracy: {round(best_acc, 3)}")

# Plot Accuracy vs K graph
st.subheader("📊 Accuracy vs K Value")
fig, ax = plt.subplots()
ax.plot(k_values, accuracies, marker='o')
ax.axvline(best_k, color='red', linestyle='--', label=f"Best K = {best_k}")
ax.set_xlabel("K (Number of Neighbors)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs K")
ax.legend()
st.pyplot(fig)

# Train final model with best K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# User input section
st.header("🔮 Predict Wine Quality")
inputs = {}
for col in X.columns:
    inputs[col] = st.number_input(
        f"{col}",
        float(data[col].min()),
        float(data[col].max()),
        float(data[col].mean())
    )

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    prediction = knn.predict(input_scaled)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
