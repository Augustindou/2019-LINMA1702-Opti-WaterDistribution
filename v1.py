import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Matrice d'incidence : A[ point , conduite ]
#               a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t
A = np.array([[-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
              [ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
              [ 0, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
              [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 3
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], # 4
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 1, 0], # 5
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1], # 6
              [ 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 7
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0], # 8
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 9
              [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 10
              [ 0, 0, 0, 0,-1, 1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 11
              [ 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 12
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 1, 0, 0, 0], # 13
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 14
              [ 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0]],# 15
dtype=float)

# Matrice de position : P[ point, [x,y,z] ]
#                x    y    z
P = np.array([[  4 ,  0 , 12 ], # 0
              [  9 ,  1 ,  6 ], # 1
              [  5 ,  3 ,  8 ], # 2
              [  9 ,  7 ,  3 ], # 3
              [  4 , 11 ,  1 ], # 4
              [ 11 , 13 ,  9 ], # 5
              [ 15 , 10 , 13 ], # 6
              [ 12 ,  8 , 11 ], # 7
              [  5 , 14 ,  7 ], # 8
              [  1 ,  8 ,  2 ], # 9
              [  3 ,  5 ,  9 ], # 10
              [ 10 ,  4 ,  9 ], # 11
              [ 14 ,  2 , 15 ], # 12
              [  7 , 11 ,  5 ], # 13
              [ 13 , 15 ,  4 ], # 14
              [  5 ,  7 ,  4 ]],# 15
dtype=float)

# Rayons des conduites : R[ conduite ]
#                  a    b    c    d    e    f    g    h    i    j    k    l    m    n    o    p    q    r    s    t
R     = np.array([3.0, 1.5, 0.9, 2.0, 0.7, 0.9, 1.3, 0.4, 1.0, 1.2, 0.5, 1.4, 1.0, 0.7, 2.0, 1.4, 1.2, 0.7, 3.2, 1.4], dtype=float)

# Constantes de proportionnalité : alpha[ conduite ]
#                   a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t
Alpha = np.array([  3,  7, 13,  1, 15,  9, 10, 13,  6,  8, 12, 10, 14,  8,  9, 12,  4,  2,  2,  3], dtype=float)

# Maximum de consommation
#                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
MaxCons = np.array([  0,  7,  0,  5, 50,  0,  0,  0,  0, 10,  0,  0,  0,  0,0.9,  0], dtype=float)

# Minimum de consommation
#                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
MinCons = np.array([  0,  2,  0,  1, 13,  0,  0,  0,  0,  2,  0,  0,  0,  0,0.3,  0], dtype=float)

# Prix de production/vente (en fonction du point)
#                     0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
Prices = np.array([   7,   9,   0,  11,10.5,   0, 4.5,   0,  10,  14,   0,   0,   8,   0,12.5,   0], dtype=float)








# Calculate

# INCORRECT
approvPts = np.arange(len(A))[ np.invert( [( 1 in A[i]) for i in range(len(A))] ) ]
consumPts = np.arange(len(A))[ np.invert( [(-1 in A[i]) for i in range(len(A))] ) ]
intermPts = np.arange(len(A))[ np.logical_and( [( 1 in A[i]) for i in range(len(A))] ,
                                               [(-1 in A[i]) for i in range(len(A))]) ]
deltaX = A.T @ P[:,0]
deltaY = A.T @ P[:,1]
deltaZ = A.T @ P[:,2]
length = np.sqrt(deltaX**2 + deltaY**2 + deltaZ**2)
maxDebit = np.abs(Alpha*R**2*deltaZ/length)
prodPrice  = np.zeros(len(P)) ; prodPrice[approvPts] = Prices[approvPts]
consPrice  = np.zeros(len(P)) ; consPrice[consumPts] = Prices[consumPts]

# Optimisation
# à maximiser : c @ x
c = Prices.T @ A
# Conditions : x > 0 ; x < debitMax ; Ax = 0


A_ub = np.vstack     ([np.identity(len(A[0])), -np.identity(len(A[0])),       A[consumPts], -      A[consumPts]])
b_ub = np.concatenate([maxDebit              ,  np.zeros(len(A[0]))   , MaxCons[consumPts], -MinCons[consumPts]])
A_eq = A[intermPts]
b_eq = np.zeros(len(intermPts))


x = linprog(-c, A_ub, b_ub, A_eq, b_eq)
print(x)









# ------------------------------------------------
# ARCHIVE
# ------------------------------------------------

# 3D plot of the network
# def plotNetwork(A, P):
#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection='3d')
#   ax.plot(P[:,0], P[:,1], P[:,2])
#   plt.show()
# plotNetwork(A, P)

# ------ Si je ne me trompe pas, il n'y a pas de max de production ------
# # Maximum de production/consommation (en fonction du point)
# #                         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
# MaxProdCons = np.array([ 20,  7,  0,  5, 50,  0, 15,  0, 20, 10,  0,  0, 10,  0,  1,  0])
# # In case max prod and cons given in 1 array
# maxProd  = np.zeros(len(P)) ; maxProd[approvPts] = MaxProdCons[approvPts]
# maxCons  = np.zeros(len(P)) ; maxProd[consumPts] = MaxProdCons[consumPts]


"""
ALeftInv = np.linalg.inv(A.T@A)@A.T

c = np.zeros(m)
c[Apts] += Acost ; c[Cpts] += Cprices

A_ub1 =  ALeftInv
b_ub1 =  maxDeb
A_ub2 = -ALeftInv
b_ub2 =  np.zeros(n)
A_ub3 =  np.identity(n)[Cpts]
b_ub3 =  Cmaxdeb
A_ub4 = -np.identity(n)[Cpts]
b_ub4 = -Cmindeb
A_ub5 =  np.identity(n)[Apts]
b_ub5 =  Amaxdeb
A_ub  = np.vstack(     [A_ub1,A_ub2,A_ub3,A_ub4,A_ub5])
b_ub  = np.concatenate([b_ub1,b_ub2,b_ub3,b_ub4,b_ub5])

A_eq = np.identity(n)[Ipts]
b_eq = np.zeros(len(Ipts))

x = linprog(-c, A_ub, b_ub, A_eq, b_eq); x.fun = -x.fun

print("Maximum :")
print(x.fun)
"""