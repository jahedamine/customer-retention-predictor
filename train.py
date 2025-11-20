import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
import os

# Charger le dataset
data = pd.read_csv("./customer_churn_dataset-testing-master.csv")

# Prétraitement : remplacer les valeurs manquantes
data_proc = data.fillna(0)

# Features (on enlève CustomerID qui n'apporte rien)
X = pd.get_dummies(data_proc[["Age", "Gender", "Last Interaction"]])
Y = data_proc["Churn"]

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Modèle RandomForest avec quelques hyperparamètres
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(x_train, y_train)

# Prédictions
y_pred = model.predict(x_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualisation
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Sauvegarde du modèle
os.makedirs("./models", exist_ok=True)
joblib.dump(model, "./models/client_classification.pkl")
print("Modèle sauvegardé dans models/client_classification.pkl")