import numpy as np
import matplotlib.pyplot as plt

def generer_donnees_classification(n_samples=100, n_features=2, n_classes=2, seed=42):
    np.random.seed(seed)
    
    # Création des centres de classes
    centres = np.random.uniform(-5, 5, size=(n_classes, n_features))
    
    # Initialisation des données
    X = []
    y = []
    
    for i in range(n_samples):
        # Choisir aléatoirement une classe
        classe = np.random.randint(0, n_classes)
        
        # Générer un point autour du centre de la classe choisie
        point = centres[classe] + np.random.randn(n_features) * 0.5
        
        X.append(point)
        y.append(classe)
    
    return np.array(X), np.array(y)

# Génération des données
X_manual, y_manual = generer_donnees_classification(n_samples=200, n_features=2, n_classes=2)

# Affichage des données générées
plt.scatter(X_manual[:, 0], X_manual[:, 1], c=y_manual, cmap='coolwarm', edgecolors='k')
plt.title("Données générées manuellement")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %%

from sklearn.datasets import make_classification

# Génération des données avec sklearn
X_sklearn, y_sklearn = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Affichage des données sklearn
plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y_sklearn, cmap='coolwarm', edgecolors='k')
plt.title("Données générées avec sklearn")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
