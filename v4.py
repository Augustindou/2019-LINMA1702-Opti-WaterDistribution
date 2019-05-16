# %% [markdown]
##### École Polytechnique de Louvain - Université Catholique de Louvain
#### LINMA1702 Modèles et méthodes d'optimisation I
## Optimisation d'un réseau de distribution d'eau <br>
# Année académique 2018-2019 <br>
# 17 Mai 2019 <br>
# __Professeurs et assistants :__ <br>
# Prof. F. Glineur <br>
# Emilie Renard <br>
# <br>
# __Groupe 7 :__ <br>
# Sarah Glume - 2947 1200 <br>
# Ferdinand Hannequart - 7290 1600 <br>
# Augustin d'Oultremont - 2239 1700 <br>
#
### Introduction
# Dans le cadre de ce projet en modèles et méthodes d'optimisation, le fonctionnement d'un réseau de distribution d'eau dans une région montagneuse sera étudié, à l'aide des outils de l'optimisation linéaire.
#
# Dans un premier temps, un réseau existant sera analysé, ainsi que son coût de fonctionnement. Ensuite, des améliorations du réseau seront proposées, au moyen de la construction de châteaux d’eau. La dernière partie traitera de la conception d’un réseau optimal (topologie et dimensionnement des conduites).
#
#### Données utilisées dans le code:
# - __P__ : la position des points (x,y,z) [m]
# - __A__ : la matrice d'incidence [-]
# - __alpha__ : constante [m/h]
# - __R__ : rayon des conduites (identique pour toutes les conduites) [m]
#
#__Approvisionnement__
# - __A_pts__ : les indices des points d'approvisionnement [-]
# - __A_maxDeb__ : le débit maximal extractible en chaque point d'approvisionnement [m$^3$/h]
# - __A_cost__ : le coût d'extraction en chaque point d'approvisionnement [€/m$^3$]
#
#__Consommation__
# - __C_pts__ : les indices des points de consommation [-]
# - __C_minDeb__ : le débit minimal consommable en consommable en chaque point de consommation [m$^3$/h]
# - __C_maxDeb__ : le débit maximal consommable en consommable en chaque point de consommation [m$^3$/h]
# - __C_price__ : le prix facturé (identique pour tous les points de consommation) [€/m$^3$]
#
#__Intermédiaires__
# - __I_pts__ : les indices des points intermédiaires [-]
#
#__Chateaux d'eau__
# - __wt_maxPrice__ : investissement maximal [€]
# - __wt_price__ : prix de construction d'un chateau d'eau [€/m]
# - __wt_rentTime__ : durée pour rentabilisation [années]

# %%
import numpy as np
from scipy.optimize import linprog
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


alpha = 10**6
R = 1

A_pts = np.array([2, 6])
A_maxDeb = np.array([10000, 5000])
A_cost = np.array([0.2, 0.3])

C_pts = np.array([1, 4, 10, 20, 23])
C_minDeb = np.array([200, 200, 200, 200, 200])
C_maxDeb = np.array([4000,2000,2000,3000,2000])
C_price = 0.9

I_pts = np.array([3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22])

wt_maxPrice = 120*10**6
wt_price = 30*10**6
wt_rentTime = 10

# %% [markdown]
##### Calcul de valeurs, vecteurs et matrices utiles :

# %%
m = len(A)
n = len(A[0])

A_pts -= np.ones(len(A_pts), dtype=int)
C_pts -= np.ones(len(C_pts), dtype=int)
I_pts -= np.ones(len(I_pts), dtype=int)
dX = A.T @ P[:,0]
dY = A.T @ P[:,1]
dZ = A.T @ P[:,2]
length = np.sqrt(dX*dX + dY*dY + dZ*dZ)
debFactor = -(alpha*R*R/length)
maxDeb = debFactor*dZ
C_prices = np.full(len(C_pts), C_price)

Prices = np.zeros(m) ; Prices[C_pts] = C_prices ; Prices[A_pts] = A_cost
# %% [markdown]
### Partie 1 : Analyse d'un réseau existant
#### Résolution numérique de la maximisation du bénéfice

# %%
c = Prices.T @ A

A_ub1 =  np.identity(n)
b_ub1 =  maxDeb
A_ub2 = -np.identity(n)
b_ub2 =  np.zeros(n)
A_ub3 =  A[C_pts]
b_ub3 =  C_maxDeb
A_ub4 = -A[C_pts]
b_ub4 = -C_minDeb
A_ub5 = -A[A_pts]
b_ub5 =  A_maxDeb
A_ub  = np.vstack(     [A_ub1,A_ub2,A_ub3,A_ub4,A_ub5])
b_ub  = np.concatenate([b_ub1,b_ub2,b_ub3,b_ub4,b_ub5])

A_eq = A[I_pts]
b_eq = np.zeros(len(I_pts))

x = linprog(-c, A_ub, b_ub, A_eq, b_eq); x.fun = -x.fun

theta = x.x / maxDeb
bilan = A @ x.x
# %% [markdown]
##### Maximum :
print(x.fun)
# %% [markdown]
##### Vecteur Theta (comment positionner les vannes) :
for i in range(len(theta)) :
    print(str(i+1) + " : " + str(theta[i]))
# %% [markdown]
##### Bilan des débits en chaque point d'approvisionnement et de consommation :
# %%
print("Extraction aux points d'approvisionnement")
for i in A_pts :
    print("    " + str(i+1) + " : " + str(np.round(-bilan[i], 1)) + "   [m^3/h]")
print("Consommation aux points de consommation")
for i in C_pts :
    print("    " + str(i+1) + " : " + str(np.round( bilan[i], 1)) + "   [m^3/h]")
# %% [markdown]
### Partie 2 : Améliorations du réseau

# %% [markdown]
#### Résolution numérique du dépassement de la demande conduisant à une diminution du prix

# %%
Prices_surpl = np.copy(Prices) ; Prices_surpl[C_pts] = Prices_surpl[C_pts] / 2
c1 = Prices      .T @ A
c2 = Prices_surpl.T @ A
c = np.concatenate([c1, c2])

A_ub1 =  np.hstack( [ np.identity(n), np.identity(n) ] )
b_ub1 =  maxDeb
A_ub2 = -np.hstack( [ np.identity(n), np.zeros((n,n))] )
b_ub2 =  np.zeros(n)
A_ub3 = -np.hstack( [np.zeros((n,n)), np.identity(n) ] )
b_ub3 =  np.zeros(n)
A_ub4 =  np.hstack( [ A[C_pts]      , np.zeros(A[C_pts].shape) ] )
b_ub4 =  C_maxDeb
A_ub5 =  np.hstack( [ np.zeros(A[C_pts].shape) , A[C_pts]      ] )
b_ub5 =  C_maxDeb*0.25
A_ub6 = -np.hstack( [ A[C_pts]      , A[C_pts]       ] )
b_ub6 = -C_minDeb
A_ub7 = -np.hstack( [ A[A_pts]      , A[A_pts]       ] )
b_ub7 =  A_maxDeb
A_ub  = np.vstack(     [A_ub1, A_ub2, A_ub3, A_ub4, A_ub5, A_ub6, A_ub7])
b_ub  = np.concatenate([b_ub1, b_ub2, b_ub3, b_ub4, b_ub5, b_ub6, b_ub7])

A_eq  =  np.hstack( [ A[I_pts]      , A[I_pts]       ] )
b_eq  =  np.zeros(len(I_pts))

x = linprog(-c, A_ub, b_ub, A_eq, b_eq); x.fun = -x.fun

theta = (x.x[:n] + x.x[n:]) / maxDeb
bilan = A @ (x.x[:n] + x.x[n:])
# %% [markdown]
##### Maximum :
print(x.fun)
# %% [markdown]
##### Vecteur Theta (comment positionner les vannes) :
for i in range(len(theta)) :
    print(str(i+1) + " : " + str(theta[i]))
# %% [markdown]
##### Bilan des débits en chaque point d'approvisionnement et de consommation :
# %%
print("Extraction aux points d'approvisionnement")
for i in A_pts :
    print("    " + str(i+1) + " : " + str(np.round(-bilan[i], 1)) + "   [m^3/h]")
print("Consommation aux points de consommation")
for i in C_pts :
    print("    " + str(i+1) + " : " + str(np.round( bilan[i], 1)) + "   [m^3/h]")

# %% [markdown]
#### Résolution numérique du dépassement de la demande conduisant à une diminution du prix

# %%
Prices_surpl = np.copy(Prices) ; Prices_surpl[C_pts] = Prices_surpl[C_pts] / 2
c1 = Prices      .T @ A
c2 = Prices_surpl.T @ A
c3 = np.full(m, wt_price/(1000*365*24*wt_rentTime))
c = np.concatenate([c1, c2, c3])

A_ub1 =  np.hstack( [ np.identity(n) , np.identity(n) , -A.T*debFactor[:,None] ] )
b_ub1 =  maxDeb
A_ub2 = -np.hstack( [ np.identity(n) , np.zeros((n,n)), np.zeros((n,m)) ] )
b_ub2 =  np.zeros(n)
A_ub3 = -np.hstack( [ np.zeros((n,n)), np.identity(n) , np.zeros((n,m)) ] )
b_ub3 =  np.zeros(n)
A_ub4 =  np.hstack( [ A[C_pts], np.zeros(A[C_pts].shape), np.zeros((len(A[C_pts]),m)) ] )
b_ub4 =  C_maxDeb
A_ub5 =  np.hstack( [ np.zeros(A[C_pts].shape), A[C_pts], np.zeros((len(A[C_pts]),m)) ] )
b_ub5 =  C_maxDeb*0.25
A_ub6 = -np.hstack( [ A[C_pts], A[C_pts]                , np.zeros((len(A[C_pts]),m)) ] )
b_ub6 = -C_minDeb
A_ub7 = -np.hstack( [ A[A_pts], A[A_pts]                , np.zeros((len(A[A_pts]),m)) ] )
b_ub7 =  A_maxDeb
A_ub8 = -np.concatenate( [np.zeros(2*n) , np.ones(m)*wt_price*1000 ])
b_ub8 =  np.array(wt_maxPrice)
A_ub = np.vstack(     [A_ub1,A_ub2,A_ub3,A_ub4,A_ub5,A_ub6,A_ub7,A_ub8.reshape(1,A_ub8.shape[0])])
b_ub = np.concatenate([b_ub1,b_ub2,b_ub3,b_ub4,b_ub5,b_ub6,b_ub7,b_ub8.reshape(1)])

A_eq  =  np.hstack( [ A[I_pts], A[I_pts]                , np.zeros((len(A[I_pts]),m)) ] )
b_eq  =  np.zeros(len(I_pts))

x = linprog(-c, A_ub, b_ub, A_eq, b_eq, options={"disp": True}); x.fun = -x.fun

theta = (x.x[0:n] + x.x[n:2*n]) / maxDeb
bilan = A @ (x.x[0:n] + x.x[n:2*n])
# %% [markdown]
##### Maximum :
print(x.fun)