import numpy as np
import copy
import math
from numpy import linalg as LA
from numpy.core.fromnumeric import partition

#смена колонок
def swap_columns(a, i, j):
  for k in range(len(a)):
    a[k][i], a[k][j] = a[k][j], a[k][i]

#смена строк
def swap_row(a, i, j):
  temp = copy.copy(A[i])
  A[i] = A[j]
  A[j] = temp

def discrepancy(matrix,vector,vector1): #невязка
  for i in range(4):
    vector1[i] = matrix[i][-1] - sum([matrix[i][j] * vector[j] for j in range(0, 4)])

def norm(matrix):  #норма матрицы 
  x = np.zeros((4, 1))
  for i in range(4):
      x[i]=sum(abs(matrix[i]))
  return(max(x))

def print_array(A):
  for i in range(0, len(A), 1):
    for j in range(0, len(A[i]), 1):
      if (A[i][j] != None):
        print("%.4f     " % A[i][j], end="")
      else:
        print("     ", "nul", end="")
    print("")

def print_vector(A):
  for j in range(0, len(A), 1):
    print("%.4f     " % A[j])
  print("")
  
def print_vector_18(A):
  for j in range(0, len(A), 1):
    print("%.18f     " % A[j])
  print("")

def get_Cij(A, B, C, i, j):
  if (i == 0):
    return A[i][j] / B[0][0]
  else:
    sum = 0
    for k in range(0, i, 1):
      sum += B[i][k]*C[k][j]
    # print(f"C[{i}][{j}] = (1/{B[i][i]}) * ({A[i][j]} - {sum})")
    return (A[i][j] - sum) / B[i][i]

def get_Bij(A, B, C, i, j):
  if (j == 0):
    return A[i][j]
  else:
    sum = 0
    for k in range(0, j, 1):
      sum += B[i][k]*C[k][j]
    return (A[i][j] - sum)

def get_Yi(B, D, Y, i):
  if (i == 0):
    return D[i]/B[i][i]
  else:
    sum = 0
    for k in range(0, i, 1):
      sum += B[i][k]*Y[k]
    return (D[i] - sum)/B[i][i]

def get_Xi(C, X, Y, i):
  if (i == 3):
    return Y[3]
  else:
    sum = 0
    for k in range(i+1, 4, 1):
      sum += C[i][k]*X[k]
    return Y[i] - sum

global n
n = 4
A_orig = np.array([(10, -1, -2,  5), 
                   (-1, 12,  3, -4),
                   (-2, 3,  15,  8), 
                   (5, -4,  8,  18)])
print("Исходная матрица:")
print_array(A_orig)
A = copy.copy(A_orig)

D = np.array([95, -41, 69, 27])

B = np.array([(None, 0, 0,  0), 
              (None, None, 0,  0),
              (None, None, None,  0), 
              (None, None, None,  None)])

C = np.array([(None, None, None,  None), 
              (0, None, None,  None),
              (0, 0, None,  None), 
              (0, 0, 0,  None)])

Y = np.array([None, None, None, None])
X = np.array([None, None, None, None])

for i in range(0, 4, 1):
  for j in range(0, i+1, 1):
    B[i][j] = get_Bij(A, B, C, i, j)
  for j in range(i, 4, 1):
    C[i][j] = get_Cij(A, B, C, i, j)

print("\nНижне треугольная B\n")
print_array(B)
print("\nВерхне треугольная C\n")
print_array(C)

for i in range(0, 4, 1):
  Y[i] = get_Yi(B, D, Y, i)

for i in reversed(range(0, 4, 1)):
  X[i] = get_Xi(C, X, Y, i)

print("Y: ")
print_vector(Y)
print("X: ")
print_vector(X)