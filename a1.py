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