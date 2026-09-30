import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%

def impurete_gini(y):
    classes, occurrences = np.unique(y, return_counts=True)
    p = occurrences / occurrences.sum()
    return 1 - np.sum(p ** 2)

# %%

def diviser_donnees(X, y, variable, seuil):
    masque_gauche = X[:, variable] <= seuil
    masque_droit = ~masque_gauche
    return X[masque_gauche], y[masque_gauche], X[masque_droit], y[masque_droit]

# %%

def meilleure_decoupe(X, y, variables):
    meilleur_gini = float('inf')
    meilleure_variable, meilleur_seuil = None, None
    for variable in variables:
        seuils = np.unique(X[:, variable])
        for seuil in seuils:
            X_gauche, y_gauche, X_droit, y_droit = diviser_donnees(X, y, variable, seuil)
            if len(y_gauche) == 0 or len(y_droit) == 0:
                continue
            gini = (len(y_gauche) * impurete_gini(y_gauche) + len(y_droit) * impurete_gini(y_droit)) / len(y)
            if gini < meilleur_gini:
                meilleur_gini, meilleure_variable, meilleur_seuil = gini, variable, seuil
    return meilleure_variable, meilleur_seuil

# %%

def construire_arbre(X, y, profondeur_max, profondeur=0):
    classes, occurrences = np.unique(y, return_counts=True)
    classe_majoritaire = classes[np.argmax(occurrences)]

    if profondeur == profondeur_max or len(np.unique(y)) == 1 or len(y) < 2:
        return classe_majoritaire  # Feuille de l'arbre
    
    # Correction portfolio : une forêt aléatoire sélectionne un sous-ensemble
    # aléatoire de variables à chaque nœud (sqrt du nombre de variables).
    nombre_variables = max(1, int(np.sqrt(X.shape[1])))
    variables = np.random.choice(X.shape[1], size=nombre_variables, replace=False)
    variable, seuil = meilleure_decoupe(X, y, variables)
    if variable is None:
        return classe_majoritaire  # Aucun split possible

    X_gauche, y_gauche, X_droit, y_droit = diviser_donnees(X, y, variable, seuil)
    
    return {
        "variable": variable,
        "seuil": seuil,
        "gauche": construire_arbre(X_gauche, y_gauche, profondeur_max, profondeur + 1),
        "droit": construire_arbre(X_droit, y_droit, profondeur_max, profondeur + 1)
    }

# %%

def construire_foret(X, y, n_arbres, profondeur_max):
    foret = []
    for _ in range(n_arbres):
        indices = np.random.choice(len(X), size=len(X), replace=True)  # Bootstrap
        X_sample, y_sample = X[indices], y[indices]
        foret.append(construire_arbre(X_sample, y_sample, profondeur_max))
    return foret

# %%

def predire_arbre(arbre, x):
    while isinstance(arbre, dict):
        if x[arbre["variable"]] <= arbre["seuil"]:
            arbre = arbre["gauche"]
        else:
            arbre = arbre["droit"]
    return arbre

# %%

def predire_foret(foret, X):
    predictions = np.array([[predire_arbre(arbre, x) for arbre in foret] for x in X])
    return np.array([np.bincount(pred).argmax() for pred in predictions])

# %%

# Importer le dataset Diabetes et préparer les données
diabetes = load_diabetes()
X, y = diabetes.data, (diabetes.target > 140).astype(int)  # Classification binaire (Seuil de 140)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement et test de notre version manuelle
np.random.seed(42)
foret_manuel = construire_foret(X_train, y_train, n_arbres=5, profondeur_max=3)
y_pred_manuel = predire_foret(foret_manuel, X_test)
accuracy_manuel = accuracy_score(y_test, y_pred_manuel)

# Entraînement et test avec sklearn
rf_sklearn = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
rf_sklearn.fit(X_train, y_train)
y_pred_sklearn = rf_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Affichage des résultats
print(f" Précision de notre forêt manuelle: {accuracy_manuel:.3f}")
print(f" Précision de sklearn: {accuracy_sklearn:.3f}")

# Graphique de comparaison des prédictions
plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], 'bo-', label="Vraie Valeur (y_test)")
plt.plot(y_pred_manuel[:50], 'r--', label="Prédiction Forêt Manuelle")
plt.plot(y_pred_sklearn[:50], 'g-.', label="Prédiction Sklearn")
plt.xlabel("Échantillons")
plt.ylabel("Classe")
plt.title("Comparaison des prédictions : Forêt Manuelle vs Sklearn")
plt.legend()
plt.show()
