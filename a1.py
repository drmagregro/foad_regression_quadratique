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

# partie e

def gradient_descent_quadratic(x, y, lr=0.01, epochs=1000):
    
    # paramètres de départ au hasard
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    
    historique = []
    
    for epoch in range(epochs):
        a, b, c, score = backpropagation_quadratic(a, b, c, x, y, lr)
        historique.append(score)
        
        if epoch % 100 == 0:
            print(f"epoch {epoch} — RMSE : {score:.4f}")
    
    return a, b, c, historique

# j'entraîne
a, b, c, historique = gradient_descent_quadratic(x_norm, y_norm)

# figure 1
x_ligne = np.linspace(x_norm.min(), x_norm.max(), 100)
y_ligne = quadratic_regression(a, b, c, x_ligne)

plt.scatter(x_norm, y_norm, label="données")
plt.plot(x_ligne, y_ligne, color="red", label="quadratique")
plt.legend()
plt.title("Régression quadratique")
plt.show()

# figure 2
plt.plot(historique)
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Évolution de la RMSE")
plt.show()

# ───── PARTIE F ─────

# modèle linéaire
def linear_regression(a, b, x):
    return a * x + b

def backpropagation_linear(a, b, x, y, lr):
    y_pred = linear_regression(a, b, x)
    err = y_pred - y
    
    dL_da = (2/len(x)) * np.sum(err * x)
    dL_db = (2/len(x)) * np.sum(err)
    
    a -= lr * dL_da
    b -= lr * dL_db
    
    return a, b, rmse(y_pred, y)

def gradient_descent_linear(x, y, lr=0.01, epochs=1000):
    a = np.random.randn()
    b = np.random.randn()
    hist = []
    
    for epoch in range(epochs):
        a, b, score = backpropagation_linear(a, b, x, y, lr)
        hist.append(score)
    
    return a, b, hist

# j'entraîne les deux
a_lin, b_lin, hist_lin = gradient_descent_linear(x_norm, y_norm)
a_q, b_q, c_q, hist_q = gradient_descent_quadratic(x_norm, y_norm)

# RMSE finale des deux
print(f"linéaire   : {hist_lin[-1]:.4f}")
print(f"quadratique : {hist_q[-1]:.4f}")

# je trace les deux courbes
x_ligne = np.linspace(x_norm.min(), x_norm.max(), 100)

plt.scatter(x_norm, y_norm, label="données")
plt.plot(x_ligne, linear_regression(a_lin, b_lin, x_ligne), color="blue", label="linéaire")
plt.plot(x_ligne, quadratic_regression(a_q, b_q, c_q, x_ligne), color="red", label="quadratique")
plt.legend()
plt.title("Linéaire vs Quadratique")
plt.show()

# RMSE des deux au fil des epochs
plt.plot(hist_lin, color="blue", label="linéaire")
plt.plot(hist_q, color="red", label="quadratique")
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()