#A1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# je charge le fichier
df = pd.read_csv("prix_maisons.csv")

print(df.head())
print(df.dtypes)
print(len(df))

# je récupère les colonnes
x = df["surface"].values
y = df["prix"].values

# je calcule la moyenne et l'écart-type pour normaliser
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()

#A2
plt.scatter(x_norm, y_norm)
plt.xlabel("surface (normalisée)")
plt.ylabel("prix (normalisé)")
plt.title("Surface vs Prix")
plt.show()

#partie b

# le modèle quadratique 
def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c

# la loss MSE
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# la RMSE c'est juste la racine de la MSE
def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

# partie d

def backpropagation_quadratic(a, b, c, x, y_true, learning_rate):
    
    # je calcule les prédictions
    y_pred = quadratic_regression(a, b, c, x)
    
    # je calcule l'erreur
    erreur = y_pred - y_true
    
    # je calcule les gradients
    dL_da = (2/len(x)) * np.sum(erreur * x**2)
    dL_db = (2/len(x)) * np.sum(erreur * x)
    dL_dc = (2/len(x)) * np.sum(erreur)
    
    # je mets à jour les paramètres
    a = a - learning_rate * dL_da
    b = b - learning_rate * dL_db
    c = c - learning_rate * dL_dc
    
    # je calcule la RMSE
    score = rmse(y_pred, y_true)
    
    return a, b, c, score