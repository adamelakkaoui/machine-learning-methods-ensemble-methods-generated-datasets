import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%

def initialiser_predictions(y):
    """Initialise les prédictions avec la moyenne des valeurs cibles."""
    return np.full_like(y, np.mean(y), dtype=np.float64)

# %%

def calculer_residus(y, predictions):
    """Calcule les résidus entre la vraie valeur et la prédiction actuelle."""
    return y - predictions

# %%

def diviser_donnees(X, y, variable, seuil):
    """Divise X et y en fonction d'une variable et d'un seuil."""
    masque_gauche = X[:, variable] <= seuil
    masque_droit = ~masque_gauche
    return X[masque_gauche], y[masque_gauche], X[masque_droit], y[masque_droit]

def meilleure_decoupe(X, y):
    """Trouve la meilleure variable et le meilleur seuil pour diviser les résidus."""
    meilleur_mse = float('inf')
    meilleure_variable, meilleur_seuil = None, None

    for variable in range(X.shape[1]):
        seuils = np.unique(X[:, variable])
        for seuil in seuils:
            X_g, y_g, X_d, y_d = diviser_donnees(X, y, variable, seuil)
            if len(y_g) == 0 or len(y_d) == 0:
                continue

            mse_split = (len(y_g) * np.var(y_g) + len(y_d) * np.var(y_d)) / len(y)

            if mse_split < meilleur_mse:
                meilleur_mse, meilleure_variable, meilleur_seuil = mse_split, variable, seuil

    return meilleure_variable, meilleur_seuil

# %%

def construire_arbre(X, y, profondeur_max, profondeur=0):
    """Construit un arbre de décision pour apprendre les résidus."""
    if profondeur >= profondeur_max or len(np.unique(y)) == 1:
        return float(np.mean(y))

    variable, seuil = meilleure_decoupe(X, y)
    if variable is None:
        return float(np.mean(y))

    X_g, y_g, X_d, y_d = diviser_donnees(X, y, variable, seuil)

    return {
        "variable": variable,
        "seuil": seuil,
        "gauche": construire_arbre(X_g, y_g, profondeur_max, profondeur + 1),
        "droit": construire_arbre(X_d, y_d, profondeur_max, profondeur + 1),
    }

# %%

def predire_arbre(arbre, x):
    """Parcourt un arbre et retourne une prédiction (valeur numérique)."""
    while isinstance(arbre, dict):
        if x[arbre["variable"]] <= arbre["seuil"]:
            arbre = arbre["gauche"]
        else:
            arbre = arbre["droit"]
    
    if isinstance(arbre, (list, np.ndarray)):
        arbre = float(np.mean(arbre))
    
    return float(arbre)

# %%

def gradient_boosting(X, y, n_arbres, profondeur_max, taux_apprentissage=0.1):
    """Entraîne un modèle de Gradient Boosting."""
    prediction_initiale = float(np.mean(y))
    predictions = initialiser_predictions(y)
    arbres = []

    for _ in range(n_arbres):
        residus = calculer_residus(y, predictions)
        arbre = construire_arbre(X, residus, profondeur_max)
        arbres.append(arbre)

        predictions += taux_apprentissage * np.array([predire_arbre(arbre, x) for x in X])

    # La moyenne d'entraînement fait partie du modèle et ne doit pas dépendre
    # d'une variable globale lors de la prédiction.
    return prediction_initiale, arbres

# %%

def predire_foret_boosting(modele, X, taux_apprentissage=0.1):
    """Prédit les valeurs de X en utilisant une forêt de Gradient Boosting."""
    prediction_initiale, arbres = modele
    predictions = np.full(X.shape[0], prediction_initiale, dtype=np.float64)
    
    for arbre in arbres:
        prediction_arbre = np.array([predire_arbre(arbre, x) for x in X])
        predictions += taux_apprentissage * prediction_arbre
    
    return predictions

# %%

# Chargement des données
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle manuel
modele_manuel = gradient_boosting(X_train, y_train, n_arbres=50, profondeur_max=3)
y_pred_manuel = predire_foret_boosting(modele_manuel, X_test)

# Entraînement du modèle sklearn
gb_sklearn = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
gb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gb_sklearn.predict(X_test)

# Évaluation
mse_manuel = mean_squared_error(y_test, y_pred_manuel)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

print(f"Erreur quadratique moyenne (manuel) : {mse_manuel:.3f}")
print(f"Erreur quadratique moyenne (sklearn) : {mse_sklearn:.3f}")

# Graphique
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_manuel, label="Prédictions Manuelles", color="red", alpha=0.6)
plt.scatter(y_test, y_pred_sklearn, label="Prédictions Sklearn", color="blue", alpha=0.6)
plt.xlabel("Vraies Valeurs")
plt.ylabel("Valeurs Prédites")
plt.legend()
plt.title("Comparaison des Prédictions : Gradient Boosting")
plt.show()
