# 🛍️ Pipeline ML E-commerce : Prédiction de Risque d'Annulation/Retour

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Vue d'ensemble

Pipeline ML complet et production-ready pour **prédire le risque d'annulation et de retour de commandes** dans un contexte e-commerce. Basé sur **250K transactions réelles** avec détection de fuites de données et audit d'équité.

### 🎯 Point Fort Principal

**Détection et correction d'une fuite de données** qui surestimait la performance du modèle :
- ❌ AUC initial (avec fuite) : **0.67**
- ✅ AUC corrigé (sans fuite) : **0.58**
- 📊 Révélation du vrai signal prédictif grâce à une analyse rigoureuse

---

## 🏗️ Architecture du Pipeline (20 Phases)

Le pipeline couvre l'ensemble du cycle de vie ML, structuré en 5 phases principales :

### 1️⃣ **Préparation des Données**
- Nettoyage et gestion des valeurs manquantes
- Détection et traitement des anomalies
- Validation de la qualité des données

### 2️⃣ **Exploration & Analyse (EDA)**
- Analyse descriptive des variables
- Distribution des features
- Identification des patterns et corrélations

### 3️⃣ **Ingénierie des Features**
- Création de features pertinentes
- Encodage des variables catégorielles
- Normalisation et scaling

### 4️⃣ **Modélisation & Évaluation**
- Entraînement de modèles de classification
- Optimisation des hyperparamètres
- Comparaison de performances multi-modèles

### 5️⃣ **Interprétabilité & Monitoring**
- Analyse SHAP (SHapley Additive exPlanations)
- Audit d'équité et détection de biais
- Monitoring de dérive des données

---

## 📊 Résultats Clés

| Métrique | Valeur |
|----------|--------|
| **AUC-ROC** | 0.58 (corrigé) |
| **Observations** | 250K transactions |
| **Features** | 50+ variables d'entrée |
| **Interprétabilité** | ✅ SHAP explainable |
| **Audit d'équité** | ✅ Complété |
| **Détection de dérive** | ✅ Monitorée |

---

## 🛠️ Technologies & Outils

- **Langage** : Python 3.8+
- **Data Science** : Pandas, NumPy, Scikit-learn
- **Modélisation** : XGBoost, LightGBM, Logistic Regression
- **Interprétabilité** : SHAP, Feature Importance
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **Monitoring** : Détection de dérive multi-variée

---

## 📦 Dataset

Le dataset contient **250K transactions e-commerce** avec :
- ✅ Variables comportementales
- ✅ Informations clients
- ✅ Détails de commande
- ✅ Historique de retours/annulations

**Fichier** : `les donnees utilisees .zip`

---

## 🚀 Démarrage Rapide

### Prérequis
```bash
Python >= 3.8
Jupyter Notebook ou Jupyter Lab
```

### Installation des Dépendances
```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn plotly
```

### Exécution du Pipeline
1. Ouvrir `code kaggle+resultat.ipynb` dans Jupyter
2. Dézipper le fichier données : `les donnees utilisees .zip`
3. Exécuter les cellules dans l'ordre

---

## 📈 Workflow Complet

```
1. Chargement données → 2. EDA → 3. Cleaning → 4. Feature Engineering
    ↓
5. Splitting Train/Test → 6. Modélisation → 7. Evaluation → 8. Hyperparameter Tuning
    ↓
9. Comparaison modèles → 10. Sélection best model → 11. SHAP Analysis → 12. Feature Importance
    ↓
13. Bias & Fairness Audit → 14. Data Drift Monitoring → 15. Production Readiness
```

---

## 🔍 Points d'Attention

⚠️ **Fuite de données détectée et corrigée** :
- Variables temporelles causales non disponibles en production
- Impact sur les métriques de validation
- Nécessité de retraining avec données propres

✅ **Solutions implémentées** :
- Suppression des features causales
- Nouvelle baseline établie (AUC 0.58)
- Monitoring en production

---

## 📚 Résultats de l'Analyse SHAP

L'analyse SHAP révèle les features les plus impactantes pour la prédiction :
- Feature Importance (moyenne des valeurs SHAP)
- Dépendance partielle
- Analyse locale par prédiction

---

## 🎓 Audit d'Équité

Validation de l'absence de biais sur des groupes démographiques :
- ✅ Equité démographique
- ✅ Égalité des opportunités
- ✅ Calibration par groupe

---

## 📊 Monitoring & Maintenance

En production, surveiller :
1. **Data Drift** : Changements de distribution des inputs
2. **Model Drift** : Baisse de performance
3. **Prediction Drift** : Changements dans les sorties
4. **Feature Drift** : Evolution des distributions individuelles

---

## 👥 Auteur

**Minyurano**

---

## 📝 License

MIT License - Libre d'utilisation

---

## 📞 Ressources & Documentation

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à laisser une star !**
