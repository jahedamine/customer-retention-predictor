from fastapi import FastAPI
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("./models/client_classification.pkl")
model_features = model.feature_names_in_

# Mapping des classes
label_map = {
    0: "Non",  # client fidèle
    1: "Oui"   # client à risque de churn
}

app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ Client Classification API is running"}

@app.post("/classify")
def classify_client(age: int, gender: str, last_interaction: int):
    data = pd.DataFrame([[age, gender, last_interaction]],
                        columns=["Age", "Gender", "Last Interaction"])
    data_proc = pd.get_dummies(data)

    for col in model_features:
        if col not in data_proc.columns:
            data_proc[col] = 0
    data_proc = data_proc[model_features]

    prediction = model.predict(data_proc)[0]
    return {"prediction": label_map[int(prediction)]}
