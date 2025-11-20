# Client Classification API

API intelligente pour prédire si un client est à risque de churn ou fidèle.  
Ce projet illustre comment transformer un modèle ML en service industrialisé, prêt à intégrer dans un CRM ou un dashboard.

---

## Problème

Les entreprises perdent des clients sans comprendre pourquoi.  
Identifier les clients à risque permet d’agir avant qu’ils partent et de lancer des campagnes de fidélisation ciblées.

---

## Solution

- Modèle **RandomForestClassifier** entraîné sur des données clients (âge, genre, dernière interaction).  
- Sauvegarde du modèle en `.pkl` pour réutilisation sans réentraînement.  
- API **FastAPI** exposant un endpoint `/classify` pour prédire en temps réel.  
- Visualisation des performances avec matrice de confusion et métriques (Accuracy, Precision, Recall, F1).
- - Possibilité d’adapter ce pipeline à d’autres cas de classification (fraude bancaire, détection de spam, segmentation clients).
---

##  Démo API

### Lancer l’API
```bash
cd ml_projects_plan/client\ classification
uvicorn main:app --reload
```

Swagger UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Exemple de requête
```bash
curl -X POST http://127.0.0.1:8000/classify \
-H "Content-Type: application/json" \
-d '{"age": 35, "gender": "Male", "last_interaction": 12}'
```

### Réponse
```json
{
  "prediction": "Oui"
}
```

> `Oui` = client à risque de churn  
> `Non` = client fidèle

---

## Architecture

- `train.py` : entraînement du modèle + évaluation + sauvegarde  
- `main.py` : API FastAPI avec endpoint `/classify`  
- `models/` : modèle `.pkl` prêt à déployer  
- `requirements.txt` : installation rapide des dépendances  

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Impact business

- **Identification proactive** des clients à risque  
- **Campagnes ciblées** pour réduire le churn  
- **Augmentation du revenu récurrent** grâce à la fidélisation  
- **API intégrable** dans CRM, chatbot ou dashboard  

---

## Auteur

Made by **Jahed Amine**  
Vision : transformer des projets ML en solutions SaaS industrialisées, prêtes pour le marché.
