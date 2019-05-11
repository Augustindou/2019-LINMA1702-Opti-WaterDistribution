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
A = np.array([
    [ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0,-1, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0,-1, 0, 0,-1, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1,-1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1,-1, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 1],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1]])

P = np.array([
 [-9.6 , 2.7 , 0.08394446],
 [-6.9 , 7.1 , 0.09969025],
 [-9.0 , 3.2 , 0.08757438],
 [-1.7 , 9.4 , 0.07859208],
 [-6.3 , 6.6 , 0.09795454],
 [-8.0 , 4.1 , 0.09254527],
 [-8.3 , 0.0 , 0.06546171],
 [-3.7 , 7.7 , 0.09001772],
 [-2.9 , 7.7 , 0.08740385],
 [-5.8 , 0.0 , 0.06384532],
 [-6.0 , 4.5 , 0.09171610],
 [-1.3 , 7.1 , 0.08500010],
 [-6.1 , 3.2 , 0.08564233],
 [-4.6 , 4.7 , 0.08941448],
 [-4.1 , 4.7 , 0.08835686],
 [-5.2 , 3.3 , 0.08463326],
 [-1.5 , 5.1 , 0.08664609],
 [-0.0 , 5.0 , 0.08933714],
 [-2.6 , 2.1 , 0.07603164],
 [-3.1 , 0.2 , 0.06363058],
 [-1.2 , 1.6 , 0.07571709],
 [-0.9 , 0.0 , 0.06600560],
 [-0.2 , 0.0 , 0.06870617]])

R     = 1

Alpha = 10^6

MinCons   = np.array([200, 200, 200, 200, 200])
MaxCons   = np.array([4000,2000,2000,3000,2000])
MaxApprov = np.array([10000, 5000])

ApprovPrices = np.array([0.2, 0.3])
ConsumPrices = 0.9

approvPts = np.array([2, 6])
approvPts -= np.ones(len(approvPts), dtype=int)
consumPts = np.array([1, 4, 10, 20, 23])
consumPts -= np.ones(len(consumPts), dtype=int)
intermPts = np.array([3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22])
approvPts -= np.ones(len(approvPts), dtype=int)

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
# approvPts = np.arange(len(A))[ np.invert( [( 1 in A[i]) for i in range(len(A))] ) ]
# consumPts = np.arange(len(A))[ np.invert( [(-1 in A[i]) for i in range(len(A))] ) ]
# intermPts = np.arange(len(A))[ np.logical_and( [( 1 in A[i]) for i in range(len(A))] , [(-1 in A[i]) for i in range(len(A))])]
Prices = np.zeros(len(P)) ; Prices[approvPts] += ApprovPrices ; Prices[consumPts] += ConsumPrices
deltaX = A.T @ P[:,0]
deltaY = A.T @ P[:,1]
deltaZ = A.T @ P[:,2]
length = np.sqrt(deltaX**2 + deltaY**2 + deltaZ**2)
maxDebit = np.abs(Alpha*R**2*deltaZ/length)
# prodPrice  = np.zeros(len(P)) ; prodPrice[approvPts] = Prices[approvPts]
# consPrice  = np.zeros(len(P)) ; consPrice[consumPts] = Prices[consumPts]

# %% [markdown]
# **Assemblage des différentes matrices et calcul de l'optimum**

# %%
c = Prices.T @ A
A_ub = np.vstack([np.identity(len(A[0])), -np.identity(len(A[0])), A[consumPts], -A[consumPts], A[approvPts]])
b_ub = np.concatenate([maxDebit, np.zeros(len(A[0])), MaxCons, -MinCons, MaxApprov])
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