import numpy as np
from pathlib import Path

IN = "embeddings_library.npy"      # change si besoin
OUT_LAST = "embedding_last.npy"    # contiendra UNIQUEMENT la dernière ligne (shape (1, d))
OUT_REST = "embeddings_rest.npy"   # contiendra toutes les autres lignes (shape (n-1, d))

# 1) Charge la matrice
X = np.load(IN)  # supposé 2D: (n, d)

if X.ndim != 2:
    raise ValueError(f"Le fichier {IN} doit contenir une matrice 2D, trouvé {X.ndim}D.")

n, d = X.shape
if n == 0:
    raise ValueError("Matrice vide: rien à couper.")

# 2) Sépare
last = X[-1:, :].copy()   # shape (1, d)
rest = X[:-1, :].copy()   # shape (n-1, d)

# 3) Sauvegarde
np.save(OUT_LAST, last)
np.save(OUT_REST, rest)

print(f"Entrée : {IN}  -> shape = {X.shape}, dtype = {X.dtype}")
print(f"Écrit  : {OUT_LAST} -> shape = {last.shape}")
print(f"Écrit  : {OUT_REST} -> shape = {rest.shape}")
