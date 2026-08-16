# 📊 Documentation du Pipeline ML - 20 Phases

## Phase 1-3 : Préparation des Données

### Phase 1 : Chargement des Données
- ✅ Importation du dataset (250K transactions)
- ✅ Vérification de l'intégrité des données
- ✅ Affichage des premières lignes et statistiques

### Phase 2 : Exploration Descriptive (EDA)
- ✅ Dimensions et types de données
- ✅ Distribution des variables cibles (annulation, retour)
- ✅ Statistiques descriptives
- ✅ Identification des valeurs manquantes

### Phase 3 : Nettoyage des Données
- ✅ Gestion des valeurs manquantes (imputation)
- ✅ Détection et traitement des doublons
- ✅ Correction des incohérences

---

## Phase 4-8 : Analyse et Ingénierie des Features

### Phase 4 : Détection des Anomalies
- ✅ Identification des outliers par variable
- ✅ Analyse de leur impact
- ✅ Décision de suppression ou transformation

### Phase 5 : Feature Engineering
- ✅ Création de variables dérivées
- ✅ Agrégations par client/produit
- ✅ Features temporelles (saisonnalité, tendance)
- ✅ Features comportementales (historique)

### Phase 6 : Encodage des Variables Catégorielles
- ✅ One-Hot Encoding pour variables nominales
- ✅ Ordinal Encoding pour variables ordinales
- ✅ Target Encoding si approprié

### Phase 7 : Sélection des Features
- ✅ Variance Inflation Factor (VIF) - multicolinéarité
- ✅ Correlation Analysis
- ✅ Feature Importance préliminaire
- ✅ Suppression des features non pertinentes

### Phase 8 : Normalisation et Scaling
- ✅ StandardScaler pour modèles sensibles à l'échelle
- ✅ MinMaxScaler si nécessaire
- ✅ Vérification de la distribution

---

## Phase 9-13 : Modélisation

### Phase 9 : Splitting Train/Test/Validation
- ✅ Stratégied split (préserver les proportions)
- ✅ Train : 70% | Validation : 15% | Test : 15%
- ✅ Vérification de la représentativité

### Phase 10 : Entraînement des Modèles
- ✅ **Logistic Regression** (baseline)
- ✅ **Random Forest** (ensemble)
- ✅ **XGBoost** (gradient boosting)
- ✅ **LightGBM** (optimisé)

### Phase 11 : Évaluation Initiale
- ✅ Métriques : Accuracy, Precision, Recall, F1
- ✅ Courbe ROC et AUC
- ✅ Matrice de confusion
- ✅ Courbe Precision-Recall

### Phase 12 : Optimisation des Hyperparamètres
- ✅ GridSearchCV ou RandomizedSearchCV
- ✅ Cross-validation stratifiée
- ✅ Tuning des paramètres clés par modèle

### Phase 13 : Calibration des Probabilités
- ✅ Isotonic Regression
- ✅ Platt Scaling
- ✅ Vérification de la calibration

---

## Phase 14-17 : Interprétabilité et Audit

### Phase 14 : Analyse SHAP (Explainability)
- ✅ SHAP Values pour chaque prédiction
- ✅ Moyenne des valeurs absolues (importance)
- ✅ Dépendance partielle SHAP
- ✅ Force plots locales

### Phase 15 : Feature Importance
- ✅ Importance des features du best model
- ✅ Permutation Importance
- ✅ Graphiques de importance

### Phase 16 : Audit d'Équité
- ✅ Analyse de biais par groupe démographique
- ✅ Disparate Impact Ratio
- ✅ Égalité des opportunités
- ✅ Calibration par groupe

### Phase 17 : Analyse de Dérive des Données
- ✅ Kolmogorov-Smirnov test
- ✅ Jensen-Shannon divergence
- ✅ Détection de shift entre train et validation
- ✅ Monitoring sur la période

---

## Phase 18-20 : Production et Recommandations

### Phase 18 : Courbes de Performance Détaillées
- ✅ ROC Curves (comparaison multi-modèles)
- ✅ Precision-Recall Curves
- ✅ Lift Curves
- ✅ Gains Chart

### Phase 19 : Rapport de Synthèse
- ✅ Résumé exécutif
- ✅ Résultats clés
- ✅ Limitations identifiées
- ✅ Prochaines étapes

### Phase 20 : Recommandations Production
- ✅ Strategy de déploiement
- ✅ KPIs à monitorer
- ✅ Seuil de retrain
- ✅ Système d'alertes pour data/model drift

---

## 🔴 Point Critique : Fuite de Données

### Détection
L'analyse a révélé une **fuite de données causale** :
- Variables temporelles qui ne seraient pas disponibles en production
- Causent une surestimation des performances

### Correction Implémentée
1. ✅ Identification précise des variables problématiques
2. ✅ Suppression du training
3. ✅ Réentraînement du modèle
4. ✅ Nouvelle baseline établie

### Impact
| Métrique | Avant correction | Après correction |
|----------|------------------|------------------|
| AUC | 0.67 | 0.58 |
| Accuracy | ~0.80 | ~0.72 |
| Precision | ~0.75 | ~0.65 |

---

## 📋 Checklist de Production

- [ ] Model versionning en place
- [ ] Monitoring de dérive mis en place
- [ ] Seuils d'alerte configurés
- [ ] Pipeline d'inférence testé
- [ ] Audit d'équité validé
- [ ] Documentation complète
- [ ] Accès aux données de monitoring
- [ ] Processus de retrain défini

---

## 📚 Ressources Supplémentaires

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
- [Calibration des probabilités](https://scikit-learn.org/stable/modules/calibration.html)
- [Fairness in ML](https://fairlearn.org/)
