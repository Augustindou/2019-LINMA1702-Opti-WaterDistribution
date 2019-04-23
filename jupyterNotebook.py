# %%
import numpy as np
from scipy.optimize import linprog

# %% [markdown]
# **Valeurs :**
# - A : Matrice d'incidence
# - P : Matrice de position [x,y,z] => On nous donnera peut-être un vecteur de hauteurs et un vecteur de longueurs (pour le moment je calcule les longueurs en faisant sqrt(deltaX^2 + ...), en supposant donc des conduites rectilignes)
# - R : Vecteur contenant les rayons des conduites
# - Alpha : Vecteur contenant les constantes de proportionnalités $\alpha$
# - MaxCons : Vecteur contenant les consommations maximales en chaque point (0 si le point n'est pas un point de consommation)
# - MinCons : Vecteur contenant les consommations minimales en chaque point (0 si le point n'est pas un point de consommation)
# - Prices : Prix de production/de vente pour les points d'approvisionnement/de consommation
# %%
A = np.array([[-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 1, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1],
              [ 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0,-1, 1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 1, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [ 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0]],
dtype=float)

P = np.array([[  4 ,  0 , 12 ], [  9 ,  1 ,  6 ], [  5 ,  3 ,  8 ], [  9 ,  7 ,  3 ], [  4 , 11 ,  1 ], [ 11 , 13 ,  9 ], [ 15 , 10 , 13 ], [ 12 ,  8 , 11 ], [  5 , 14 ,  7 ], [  1 ,  8 ,  2 ], [  3 ,  5 ,  9 ], [ 10 ,  4 ,  9 ], [ 14 ,  2 , 15 ], [  7 , 11 ,  5 ], [ 13 , 15 ,  4 ], [  5 ,  7 ,  4 ]], dtype=float)

R     = np.array([3.0, 1.5, 0.9, 2.0, 0.7, 0.9, 1.3, 0.4, 1.0, 1.2, 0.5, 1.4, 1.0, 0.7, 2.0, 1.4, 1.2, 0.7, 3.2, 1.4], dtype=float)

Alpha = np.array([  3,  7, 13,  1, 15,  9, 10, 13,  6,  8, 12, 10, 14,  8,  9, 12,  4,  2,  2,  3], dtype=float)

MaxCons = np.array([  0,  7,  0,  5, 50,  0,  0,  0,  0, 10,  0,  0,  0,  0,0.9,  0], dtype=float)
MinCons = np.array([  0,  2,  0,  1, 13,  0,  0,  0,  0,  2,  0,  0,  0,  0,0.3,  0], dtype=float)

Prices = np.array([   7,   9,   0,  11,10.5,   0, 4.5,   0,  10,  14,   0,   0,   8,   0,12.5,   0], dtype=float)

# %% [markdown]
# **Calcul de vecteurs/matrices utiles**
# - approvPts : index des points d'approvisionnement
# - consumPts : index des points de consommation
# - intermPts : index des points intermédiaires
# - deltaX, deltaY, deltaZ : différence en x, y et z entre l'arrivée et le départ de chaque conduite
#     - deltaZ est également la différence de hauteur ($\Delta h$)
# - length : longueur des conduites (dans le cas (probable) où la longueur serait donnée à la place de x et y, cette ligne est à supprimer)
# - maxDebit : débit maximal dans chaque conduite ($f_max$ dans le rapport, à modifier dans le code pour avoir une correspondance?)
# - prodPrice : prix de production (0 si le point n'est pas un point d'approvisionnement) => Pas utile pour le moment... à supprimer ?
# - consPrice : prix de vente (0 si le point n'est pas un point de consommation) => Pas utile pour le moment... à supprimer ?

# %%
approvPts = np.arange(len(A))[ np.invert( [( 1 in A[i]) for i in range(len(A))] ) ]
consumPts = np.arange(len(A))[ np.invert( [(-1 in A[i]) for i in range(len(A))] ) ]
intermPts = np.arange(len(A))[ np.logical_and( [( 1 in A[i]) for i in range(len(A))] , [(-1 in A[i]) for i in range(len(A))])]
deltaX = A.T @ P[:,0]
deltaY = A.T @ P[:,1]
deltaZ = A.T @ P[:,2]
length = np.sqrt(deltaX**2 + deltaY**2 + deltaZ**2)
maxDebit = np.abs(Alpha*R**2*deltaZ/length)
prodPrice  = np.zeros(len(P)) ; prodPrice[approvPts] = Prices[approvPts]
consPrice  = np.zeros(len(P)) ; consPrice[consumPts] = Prices[consumPts]

# %% [markdown]
# **Assemblage des différentes matrices et calcul de l'optimum**

# %%
c = Prices.T @ A
A_ub = np.vstack([np.identity(len(A[0])), -np.identity(len(A[0])), A[consumPts], -A[consumPts]])
b_ub = np.concatenate([maxDebit, np.zeros(len(A[0])), MaxCons[consumPts], -MinCons[consumPts]])
A_eq = A[intermPts]
b_eq = np.zeros(len(intermPts))
x = linprog(-c, A_ub, b_ub, A_eq, b_eq) ; x.fun = -x.fun

# %% [markdown]
# **Calcul des valeurs de $\theta_i$ et impression des résultats**

# %%
theta = x.x/maxDebit
print("Vecteur Theta :")
print(theta)
print("Optimum :")
print(x.fun)