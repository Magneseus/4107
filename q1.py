#!/bin/python2.7

import numpy as np

# Functions

def transpose(matrix):
  retMatrix = [[0. for i in range(len(matrix))] for j in range(matrix[0].size)]
  retMatrix = np.matrix(retMatrix)
  
  for i in range(len(matrix)):
    for j in range(matrix[i].size):
      retMatrix[j:j+1, i:i+1] = matrix[i:i+1, j:j+1]
  
  return retMatrix

def avgVec(vec):
  sum = 0
  
  for i in vec:
    sum += i
  
  return sum / len(vec)

# Starting vector and SVD
a = np.matrix([[3,1,2,3],[4,3,4,3],[3,2,1,5],[1,6,5,2]])
V,S,U = np.linalg.svd(a, full_matrices=True)

# Take transpose of U?
U = transpose(U)

# Create the new matrices (2x?)
V2 = V[:,:2]
U2 = U[:,:2]

S2 = []

for i in range(len(S)):
  S2.append([])
  for j in range(len(S)):
    S2[i].append(0.)
  
  S2[i][i] = S[i]

# Sigma2D && Inverse Sigma 2D
S2 = np.matrix(S2)
S2I = np.linalg.inv(S2)
S2 = S2[:2,:2]
S2I = S2I[:2,:2]

# Alice calcs
Alice = [5,3,4,4]

Alice2D = Alice * U2 * S2I

print("Alice2D: ", Alice2D)


# Take avg
AliceAvg = avgVec(Alice)

# U2(Alice) x S2 x V2T(EPL[3])

Rui = AliceAvg + (U2[:1] * S2 * transpose(V2)[:,3:4])

print("Rui: ", Rui)

