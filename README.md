# 🧠 SmartAddict AI

### Smartphone Addiction Risk Prediction with Machine Learning & Explainable AI

> **SmartAddict AI** est un projet de Data Science et d'Intelligence Artificielle visant à prédire le risque d'addiction au smartphone à partir des habitudes numériques, du temps d'écran et de plusieurs indicateurs comportementaux.

Le projet combine **analyse exploratoire des données, feature engineering, Machine Learning, validation croisée, optimisation des hyperparamètres et Explainable AI (SHAP)** afin de construire un modèle prédictif performant et interprétable.

---

## 📌 Aperçu

L'utilisation intensive des smartphones peut être associée à différentes conséquences sur les habitudes quotidiennes : augmentation du temps d'écran, utilisation excessive des réseaux sociaux ou des jeux, perturbation du sommeil et impact sur les activités académiques ou professionnelles.

L'objectif de **SmartAddict AI** est de construire un système capable d'estimer la probabilité qu'un profil présente un **risque élevé d'addiction au smartphone**, à partir de données comportementales.

Le projet ne cherche pas à établir un diagnostic médical.

> ⚠️ **Important :** le modèle fournit une prédiction statistique basée sur les données disponibles. Il ne constitue ni un diagnostic médical ni une preuve de causalité entre une variable et l'addiction.

---

## 🎯 Objectifs

Le projet poursuit plusieurs objectifs :

* analyser les facteurs comportementaux associés au risque d'addiction ;
* explorer et visualiser les données ;
* gérer les valeurs manquantes et les variables catégorielles ;
* créer de nouvelles variables pertinentes grâce au feature engineering ;
* comparer plusieurs algorithmes de Machine Learning ;
* sélectionner un modèle performant et stable ;
* optimiser ses hyperparamètres ;
* interpréter les prédictions grâce à SHAP ;
* générer des prédictions pour de nouvelles observations ;
* préparer une future interface permettant d'explorer les prédictions du modèle.

---

# 📊 Dataset

Le projet utilise le dataset de la compétition Kaggle :

**Playground Series — Smartphone Addiction Prediction**

Les données contiennent des informations relatives aux habitudes d'utilisation du smartphone.

### Dimensions

| Dataset    |  Lignes |        Colonnes |
| ---------- | ------: | --------------: |
| Train      | 691 369 | 14 initialement |
| Test       | 296 302 | 13 initialement |
| Submission | 296 302 |               2 |

La variable cible est :

```text
addicted_label
```

Elle correspond à une classification binaire :

* `0` → profil non classé comme dépendant
* `1` → profil classé comme dépendant

### Principales variables

Le dataset contient notamment :

* `age`
* `daily_screen_time_hours`
* `social_media_hours`
* `gaming_hours`
* `work_study_hours`
* `sleep_hours`
* `notifications_per_day`
* `app_opens_per_day`
* `weekend_screen_time`
* `gender`
* `stress_level`
* `academic_work_impact`

---

# 🔎 Exploratory Data Analysis

L'analyse exploratoire a permis d'étudier :

* la distribution de la variable cible ;
* les distributions des variables numériques ;
* les relations entre les variables ;
* les corrélations avec la variable cible ;
* les valeurs manquantes ;
* la cohérence entre les distributions train/test.

### Distribution de la cible

La variable cible présente un déséquilibre modéré :

```text
Classe 0 : 29.06 %
Classe 1 : 70.94 %
```

Ce déséquilibre a été pris en compte lors de la validation des modèles.

### Variables fortement associées à la cible

Les corrélations observées avec `addicted_label` montrent notamment :

| Variable                  | Corrélation |
| ------------------------- | ----------: |
| `daily_screen_time_hours` |       0.611 |
| `weekend_screen_time`     |       0.590 |
| `social_media_hours`      |       0.532 |
| `work_study_hours`        |       0.251 |
| `gaming_hours`            |       0.205 |

Ces corrélations sont utilisées comme éléments exploratoires et ne doivent pas être interprétées comme des relations causales.

---

# 🛠️ Feature Engineering

Plusieurs variables dérivées ont été créées afin de fournir au modèle des indicateurs comportementaux supplémentaires.

### Variables créées

#### Screen Time Ratio

```text
screen_time_ratio
```

Proportion approximative de la journée consacrée au temps d'écran.

#### Social Media Ratio

```text
social_media_ratio
```

Part du temps d'écran consacrée aux réseaux sociaux.

#### Gaming Ratio

```text
gaming_ratio
```

Part du temps d'écran consacrée aux jeux.

#### Recreational Screen Time

```text
recreational_screen_time
```

Combinaison du temps passé sur les réseaux sociaux et les jeux.

#### Total Leisure Hours

```text
total_leisure_hours
```

Variable construite comme une approximation du temps disponible en dehors du sommeil et du travail/études.

#### Weekend Screen Ratio

```text
weekend_screen_ratio
```

Rapport entre le temps d'écran du week-end et le temps d'écran quotidien.

#### Screen/Sleep Ratio

```text
screen_sleep_ratio
```

Rapport entre le temps d'écran et le temps de sommeil.

#### Usage Intensity

Des variables d'intensité ont également été créées à partir du nombre d'ouvertures d'applications et de notifications rapporté au temps d'écran.

Après feature engineering :

```text
Train : (691369, 22)
Test  : (296302, 21)
```

---

# ⚙️ Preprocessing

Le pipeline de preprocessing comprend notamment :

* imputation des valeurs manquantes numériques ;
* imputation des variables catégorielles ;
* encodage des variables catégorielles ;
* transformation des variables numériques lorsque nécessaire ;
* contrôle des valeurs infinies ;
* séparation stricte entre données d'entraînement et de validation.

Le preprocessing est ajusté uniquement sur les données d'entraînement afin d'éviter toute fuite de données (*data leakage*).

Après transformation :

```text
Train      : 553095 × 44
Validation : 138274 × 44
```

Aucune valeur manquante ou infinie ne subsiste après preprocessing.

---

# 🤖 Modélisation

Plusieurs algorithmes ont été comparés :

* Logistic Regression
* Random Forest
* CatBoost
* XGBoost
* LightGBM

La métrique principale utilisée pour comparer les modèles est le **ROC-AUC**, complétée par :

* PR-AUC
* Accuracy
* Precision
* Recall
* F1-score
* temps d'entraînement

---

# 🏆 Résultats du benchmark

Lors du benchmark initial, XGBoost et LightGBM ont obtenu les meilleures performances.

| Modèle   |    ROC-AUC |
| -------- | ---------: |
| XGBoost  | **0.9541** |
| LightGBM | **0.9537** |

La validation croisée a ensuite montré que les deux modèles étaient pratiquement équivalents :

| Modèle   | Mean ROC-AUC |    Std |
| -------- | -----------: | -----: |
| XGBoost  |       0.9545 | 0.0007 |
| LightGBM |       0.9545 | 0.0008 |

LightGBM a cependant présenté un avantage en temps d'entraînement, ce qui a motivé son choix pour la suite du projet.

---

# 🔬 Cross-Validation

Une validation croisée stratifiée à **3 folds** a été utilisée afin d'obtenir une estimation plus robuste des performances.

Cette étape permet notamment de vérifier que les performances ne dépendent pas uniquement d'une seule séparation train/validation.

Les résultats ont confirmé la stabilité du modèle LightGBM.

---

# 🎚️ Threshold Analysis

Le seuil de classification a également été étudié à partir des prédictions *out-of-fold*.

Le seuil maximisant le F1-score sur ces prédictions était :

```text
threshold = 0.4807
```

Sur le holdout, ce seuil permettait notamment d'augmenter légèrement le recall :

```text
Recall :
0.9292 → 0.9347
```

mais avec une légère diminution du F1-score :

```text
F1 :
0.9219 → 0.9217
```

Le seuil `0.5` reste donc le choix par défaut, tandis que `0.4807` peut être utilisé dans une configuration privilégiant davantage le recall.

---

# 🚀 Hyperparameter Tuning

Après le benchmark et la validation croisée, LightGBM a été optimisé à travers une recherche de **6 configurations évaluées sur 3 folds**.

Le meilleur modèle atteint :

## ⭐ ROC-AUC moyen : 0.9615

Comparaison :

```text
LightGBM baseline : 0.9545
LightGBM tuned    : 0.9615

Gain             : +0.0070
```

L'écart-type obtenu est :

```text
0.0005
```

Le gap moyen entre les performances d'entraînement et de validation est :

```text
0.0082
```

Ces résultats ne montrent pas de signal évident de surapprentissage marqué.

---

# 🥇 Meilleurs hyperparamètres

Le modèle final utilise :

```python
{
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 100,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0,
    "reg_lambda": 1
}
```

Le modèle final a ensuite été réentraîné sur l'intégralité des :

```text
691 369 observations
```

Temps d'entraînement :

```text
37.4 secondes
```

---

# 📦 Submission

Le modèle final a généré :

```text
296 302 prédictions
```

Le fichier :

```text
submission.csv
```

a été créé et vérifié.

Les contrôles effectués comprennent notamment :

* nombre de prédictions ;
* structure du fichier ;
* présence des identifiants ;
* alignement des identifiants avec le fichier de soumission fourni ;
* absence de décalage entre les observations et les prédictions.

---

# 🧠 Explainable AI — SHAP

La performance du modèle n'est pas le seul objectif de SmartAddict AI.

Le projet utilise **SHAP (SHapley Additive exPlanations)** afin d'interpréter les prédictions de LightGBM.

L'analyse globale réalisée sur un échantillon de données de validation met notamment en évidence :

1. `daily_screen_time_hours`
2. `weekend_screen_time`
3. `social_media_hours`

comme variables fortement contributives aux prédictions du modèle.

### Exemple d'explication locale

Pour un profil donné, le modèle attribue une contribution importante à :

* `daily_screen_time_hours`
* `weekend_screen_time`
* `social_media_hours`

alors que certaines variables comme `gaming_ratio` peuvent avoir une contribution négative ou plus faible selon le profil.

L'explication SHAP permet donc de répondre à une question importante :

> **"Pourquoi le modèle considère-t-il que ce profil présente un risque élevé ?"**

⚠️ Les valeurs SHAP expliquent le comportement du modèle. Elles ne démontrent pas qu'une variable est une cause de l'addiction.

---

# 📈 Visualisations

Les principales visualisations du projet seront disponibles dans :

```text
reports/figures/
```

Exemples :

### Distribution de la cible

![Target Distribution](reports/figures/target_distribution.png)

### Importance des variables

![Feature Importance](reports/figures/feature_importance.png)

### SHAP Summary

![SHAP Summary](reports/figures/shap_summary.png)

### SHAP Waterfall

![SHAP Waterfall](reports/figures/shap_waterfall.png)

> Les images seront ajoutées au dépôt lors de la finalisation des rapports.

---

# 🖥️ Dashboard

Une interface **SmartAddict AI** est prévue afin de transformer le modèle Machine Learning en une application interactive.

L'utilisateur pourra renseigner différentes caractéristiques comportementales, puis obtenir :

* une probabilité prédite ;
* un niveau de risque ;
* les principaux facteurs influençant la prédiction ;
* une explication de la décision du modèle.

Architecture cible :

```text
Utilisateur
     │
     ▼
Dashboard
     │
     ▼
Feature Engineering
     │
     ▼
Preprocessing
     │
     ▼
LightGBM
     │
     ├──► Probabilité
     │
     └──► Explication SHAP
```

### Aperçu

> 📸 Les captures d'écran du dashboard seront ajoutées ici après son implémentation.

---

# 🏗️ Architecture du projet

```text
smartaddict-ai/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── index.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
│
├── models/
│   └── lightgbm_final.txt
│
├── app/
│   ├── app.py
│   └── assets/
│
├── reports/
│   ├── figures/
│   └── results/
│
└── tests/
    └── test_features.py
```

---

# 🧰 Technologies utilisées

### Data Science

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* LightGBM
* XGBoost
* CatBoost
* Random Forest
* Logistic Regression

### Explainable AI

* SHAP

### Visualisation

* Matplotlib
* Seaborn

### Application

* Streamlit *(prévu pour le dashboard)*

### Environnement

* Jupyter Notebook
* Git
* GitHub

---

# ⚙️ Installation

## 1. Cloner le repository

```bash
git clone https://github.com/<USERNAME>/smartaddict-ai.git
cd smartaddict-ai
```

## 2. Créer un environnement virtuel

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# ▶️ Utilisation

## Explorer le projet

L'analyse complète est disponible dans :

```text
notebooks/index.ipynb
```

Lancer Jupyter :

```bash
jupyter notebook
```

Puis ouvrir :

```text
notebooks/index.ipynb
```

---

## 🔮 Effectuer une prédiction

Une fois le pipeline de prédiction finalisé :

```bash
python src/predict.py
```

---

## 🖥️ Lancer le dashboard

Après implémentation de l'application :

```bash
streamlit run app/app.py
```

L'application sera alors accessible localement dans le navigateur.

---

# 🧪 Tests

Les tests unitaires sont regroupés dans :

```text
tests/
```

Ils permettent notamment de vérifier :

* les transformations des features ;
* les valeurs produites par le preprocessing ;
* la cohérence des entrées du modèle.

Exécution :

```bash
pytest
```

---

# 🔐 Données et confidentialité

Les données utilisées dans ce projet proviennent du dataset de compétition Kaggle.

Les données brutes ne sont pas nécessairement incluses directement dans le repository.

Si nécessaire, elles doivent être placées dans :

```text
data/raw/
```

Les fichiers contenant des données sensibles ou volumineuses doivent être exclus du repository via `.gitignore`.

---

# ⚠️ Limites du projet

Plusieurs limites doivent être prises en compte.

### 1. Prédiction ≠ causalité

Une forte contribution d'une variable dans le modèle ne signifie pas que cette variable cause l'addiction.

### 2. Prédiction ≠ diagnostic médical

SmartAddict AI est un modèle prédictif expérimental et ne remplace pas une évaluation clinique.

### 3. Dépendance au dataset

Les performances sont directement liées aux caractéristiques et à la qualité du dataset utilisé.

### 4. Variables comportementales

Certaines variables représentent des comportements déclarés ou mesurés dans un contexte spécifique et peuvent ne pas être parfaitement généralisables à toutes les populations.

### 5. Généralisation

Une validation externe sur un dataset indépendant serait nécessaire avant toute utilisation réelle.

---

# 🔮 Perspectives

Plusieurs améliorations peuvent être envisagées :

* validation sur un dataset externe ;
* calibration des probabilités ;
* analyse plus approfondie des erreurs ;
* comparaison avec d'autres méthodes d'ensemble ;
* optimisation du seuil selon différents cas d'utilisation ;
* amélioration de l'interface utilisateur ;
* API REST pour servir le modèle ;
* monitoring du modèle ;
* tests automatisés ;
* déploiement cloud ;
* suivi de la dérive des données (*data drift*) ;
* ajout d'une analyse temporelle des habitudes numériques.

---

# 📚 Méthodologie

Le pipeline complet peut être résumé ainsi :

```text
                    DATASET
                       │
                       ▼
              Exploratory Data Analysis
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
                Preprocessing
                       │
                       ▼
              Model Benchmarking
                       │
                       ▼
             Cross-Validation
                       │
                       ▼
             Hyperparameter Tuning
                       │
                       ▼
                 LightGBM
                       │
                       ▼
              Explainability
                    (SHAP)
                       │
                       ▼
              Final Predictions
                       │
                       ▼
                SmartAddict AI
                  Dashboard
```

---

# 📊 Résumé des performances

| Étape             |    ROC-AUC |
| ----------------- | ---------: |
| LightGBM baseline |     0.9537 |
| CV LightGBM       |     0.9545 |
| LightGBM optimisé | **0.9615** |

### Modèle final

```text
Algorithm      : LightGBM
CV folds       : 3
CV ROC-AUC     : 0.9615
CV Std         : 0.0005
Train samples  : 691369
Training time  : 37.4 s
```

---

# 👨‍💻 Auteur

**[Votre nom]**

Data Science • Machine Learning • Artificial Intelligence

📧 Email : [votre-email]

🔗 GitHub : [votre-profil-GitHub]

---

# ⭐ Contribution

Les contributions et suggestions sont les bienvenues.

Pour proposer une modification :

```bash
git fork
git checkout -b feature/ma-feature
git commit -m "Add: ma feature"
git push origin feature/ma-feature
```

Puis ouvrir une Pull Request.

---

# 📄 Licence

Ce projet est distribué sous licence **MIT**.

Voir le fichier :

```text
LICENSE
```

pour plus d'informations.

---

## 🚀 SmartAddict AI

**From digital behavior data to interpretable AI predictions.**

> *An experimental Data & AI project for understanding and predicting smartphone addiction risk.*
