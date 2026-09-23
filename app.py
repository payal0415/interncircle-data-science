import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Titanic Survival Predictor")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)

    features = ["pclass", "age", "sibsp", "parch", "fare"]

    data = df[features + ["survived"]].copy()
    data["age"] = data["age"].fillna(data["age"].median())

    X = data[features]
    y = data["survived"]

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

st.subheader("Passenger Details")

pclass = st.slider("Passenger Class", 1, 3, 3)
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings / Spouses", 0, 8, 0)
parch = st.slider("Parents / Children", 0, 6, 0)
fare = st.slider("Fare", 0.0, 300.0, 30.0)

if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Prediction: Passenger likely survived.")
    else:
        st.error("Prediction: Passenger likely did not survive.")
