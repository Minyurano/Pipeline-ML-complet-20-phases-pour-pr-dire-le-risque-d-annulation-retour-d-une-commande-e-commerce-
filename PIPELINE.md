# Détail des 20 phases du pipeline

Chaque phase correspond à une section du script [`src/pipeline.py`](src/pipeline.py). Les phases sont exécutées dans l'ordre logique des dépendances, si bien que certaines apparaissent dans un ordre différent de leur numéro (par exemple, le découpage, phase 11, est fait avant l'encodage, phase 8, pour éviter toute fuite). Les chiffres cités proviennent de l'exécution de référence sur Kaggle (`notebooks/pipeline_ecommerce_kaggle.ipynb`).

---

## Phase 0 — Cadrage
- **Question :** quelles commandes risquent d'être annulées ou retournées ?
- **Cible :** `is_failed = 1` si `Order_Status ∈ {Cancelled, Returned}`, `0` si `Delivered`.
- **Exclusion :** commandes `Processing` / `Shipped` (issue inconnue).

## Phase 1 — Collecte
- Chargement de `sales` (250 000 × 21), `customers` (40 000 × 15) et `products` (2 000 × 12).
- Fusions `left` sur `Customer_ID` puis `Product_ID` → 250 000 × 46.

## Phase 2 — EDA et détection de fuite
- Distribution de `Order_Status`, doublons (aucun), cohérence des montants (5 montants négatifs).
- **Fuite n° 1 :** `Rating` / `Review_Text` sont 100 % manquants pour les commandes non livrées. Ils ne sont connus qu'après livraison, donc ils sont **exclus**.
- Cardinalité des variables catégorielles.

## Phase 3 — Nettoyage
- Conversion des dates, suppression des 5 lignes à montant négatif (0,002 %).
- Filtrage sur les statuts à issue connue → **taux d'échec de base : 11,1 %**.
- Exclusion des identifiants et des données personnelles (nom, e-mail, téléphone).

## Phase 4 — Valeurs manquantes
- `Coupon_Code` manquant signifie « aucun coupon » (MNAR) : imputation par `NONE` + indicateur `Has_Coupon`.

## Phase 5 — Valeurs aberrantes
- Taux d'outliers par la méthode IQR. Les montants extrêmes sont **légitimes** : ils sont conservés et traités par un `RobustScaler`.
- `Shipping_Ratio` est winsorisé au 99e percentile.

## Phase 6 — Analyse statistique
- **χ²** (catégorielles vs cible) et **Mann-Whitney U** (numériques vs cible).
- **VIF** : `Unit_Price`, `Order_Value` et `Total_Amount` sont redondants → seul `Total_Amount` est conservé. `Shipping_Cost` est remplacé par `Shipping_Ratio`.

## Phase 7 — Feature engineering
- **Variables temporelles :** jour de la semaine, week-end, heure, mois encodé de façon cyclique (sin / cos).
- **Variables dérivées :** délai de livraison estimé, ancienneté du client, taux de remise, part des frais de port.
- **Fuite n° 2 :** `Total_Orders` / `Total_Spent` sont des agrégats calculés sur tout l'historique du client. Ils sont remplacés par des variables **point-in-time** : `Orders_Before` (nombre de commandes précédentes) et `Spend_Before` (dépense cumulée avant la commande).

## Phase 8 — Encodage et mise à l'échelle
- `ColumnTransformer` : `RobustScaler` (17 variables numériques) + `OneHotEncoder(handle_unknown="ignore")` (9 variables catégorielles).
- **Ajusté uniquement sur le train**, puis intégré au `Pipeline`.

## Phase 9 — Déséquilibre des classes
- Ratio d'environ 8:1. Traitement par `class_weight="balanced"` plutôt que SMOTE.
- Métrique principale : **PR-AUC**.

## Phase 10 — Sélection de variables
- Méthode filtre (information mutuelle) et méthode embedded (importance d'un Random Forest).
- `Customer_Tier` domine. Le jeu complet est conservé : les variables faibles ne pénalisent pas les modèles à base d'arbres.

## Phase 11 — Découpage
- Stratifié 70 / 15 / 15 : train 157 593, validation 33 770, test 33 771.

## Phase 12 — Modélisation
| Modèle | ROC-AUC (val.) | PR-AUC (val.) |
|---|---:|---:|
| Baseline (classe majoritaire) | 0.500 | 0.111 |
| Régression logistique | 0.565 | 0.153 |
| Random Forest | 0.580 | 0.226 |
| HistGradientBoosting | **0.588** | **0.237** |

## Phase 13 — Validation croisée et optimisation
- `RandomizedSearchCV` sur HistGradientBoosting : 8 combinaisons × 3 plis stratifiés, critère PR-AUC.
- Meilleurs paramètres : `max_depth=6`, `learning_rate=0.03`, `max_iter=100`, `l2_regularization=0.5`, `min_samples_leaf=50` (PR-AUC en CV : 0.223).

## Phase 14 — Évaluation finale (test)
- ROC-AUC **0.578**, PR-AUC **0.217** (contre 0.111 au hasard), F1 **0.204**.
- 39,7 % des échecs sont détectés, et 13,7 % des alertes émises sont de vrais échecs.

## Phase 15 — Interprétabilité
- **Importance par permutation** (baisse de PR-AUC) : `Customer_Tier`, `Total_Amount` et `Spend_Before` sont les seules variables utiles.
- **SHAP** (TreeExplainer, 300 exemples) : un montant élevé et une dépense passée élevée poussent vers l'échec, tandis que le niveau Platinum pousse vers la livraison.

## Phase 16 — Visualisation
Sept graphiques (dossier `assets/`) : distribution de la cible, taux d'échec par niveau client, courbes ROC et précision-rappel, matrice de confusion, importance par permutation, résumé SHAP, comparaison des modèles.

## Phase 17 — Reproductibilité
- `random_state = 42` partout.
- `metrics.json` contient les métriques, les hyperparamètres, les variables utilisées et les variables exclues pour fuite.
- `model_pipeline.joblib` regroupe le prétraitement et le modèle en un seul objet.

## Phase 18 — Déploiement
- Le pipeline `joblib` est rejouable tel quel en production.
- **Avant une mise en production réelle :** API (par exemple FastAPI), authentification, validation des entrées, journalisation et gestion des erreurs.

## Phase 19 — Monitoring
- Fonction `population_stability_index()` : PSI < 0,1 = stable, 0,1–0,25 = dérive modérée, > 0,25 = réentraînement.
- Exemple : `Total_Amount` entre train et test, PSI = 0,0005.

## Phase 20 — Éthique, biais et gouvernance
- Taux réel et taux prédit par genre, niveau client et tranche d'âge.
- **Constat :** les clients Silver sont fortement sur-ciblés (80 % de prédictions « échec » pour 16 % d'échecs réels).
- **Recommandation :** fixer le seuil selon le coût métier, envisager un ajustement par groupe, et garder une supervision humaine.
