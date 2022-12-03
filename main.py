import numpy as np
import copy
import math
from numpy import linalg as LA
from numpy.core.fromnumeric import partition
from collections.abc import Sequence, MutableSequence

n = 4

#вектор невязки
def get_residual_vector(matrix, X):
  discrepancy = np.zeros((4, 1))
  for i in range(n):
    discrepancy[i] = matrix[i][-1] - sum([matrix[i][j] * X[j] for j in range(0, 4)])
  return discrepancy
  
#норма матрицы ||A||_00
# def get_norm_matrix(matrix): 
#   x = np.zeros((n, 1))
#   for i in range(n):
#       x[i]=sum(abs(matrix[i]))
#   return(max(x))

# def get_norm_matrix_1(matrix): 
#   x = np.zeros((n, 1))
#   for j in range(n):
#     for i in range(n):
#       x[j] += abs(matrix[i][j])
#   return(max(x))

def get_norm_matrix_00(matrix): 
  x = np.zeros((n, 1))
  for i in range(n):
    for j in range(n):
      x[i] += abs(matrix[i][j])
  return(max(x))

#норма вектора ||A||_2
def get_norm_vector(x_new, X): 
  norm_X = np.zeros((n, 1))
  for i in range(n):
    norm_X[i] = (abs(x_new[i] - X[i]))**2
  return(math.sqrt(sum(norm_X)))

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
  
def print_vector_18(A, p=18):
  for j in range(0, len(A), 1):
    print(f"%.{p}f     " % A[j])
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
    for k in range(i+1, n, 1):
      sum += C[i][k]*X[k]
    return Y[i] - sum

# def gauss_seidel_method(A, D, eps = 1e-6):
#   X = np.zeros(n)
#   stop_criterion = False
#   while not stop_criterion:
#     x_new = np.copy(X)
#     for i in range(n):
#       s1 = sum(A[i][j] * x_new[j] for j in range(i))
#       s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
#       x_new[i] = (D[i] - s1 - s2) / (A[i][i])
    
#     #норма ||.||_2
#     vector_norm = get_norm_vector(x_new, X)
#     stop_criterion = vector_norm < eps
#     X = x_new
#   return X

def jacobi(A, D, iteration_list, eps = 1e-6):
  X = np.zeros(n)
  stop_criterion = False
  
  while not stop_criterion:
    x_new = np.copy(X)
    for i in range(n):
      s1 = sum(A[i][j] * X[j] for j in range(n))
      # s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
      x_new[i] = X[i] + (D[i] - s1) / (A[i][i])
    
    #норма ||.||_2
    vector_norm = get_norm_vector(x_new, X)
    stop_criterion = vector_norm < eps
    X = x_new
    iteration_list.append(X)
  return X
  
A_orig = np.array([(10, -1, -2,  5), 
                   (-1, 12,  3, -4),
                   (-2, 3,  15,  8), 
                   (5, -4,  8,  18)])

print("Исходная матрица:")
print_array(A_orig)
A = copy.copy(A_orig)

D = np.array([95, -41, 69, 27])

B = np.zeros((n,n))
C = np.zeros((n,n))

Y = np.array([None, None, None, None])
X = np.array([None, None, None, None])

for i in range(0, n, 1):
  for j in range(0, i+1, 1):
    B[i][j] = get_Bij(A, B, C, i, j)
  for j in range(i, n, 1):
    C[i][j] = get_Cij(A, B, C, i, j)

print("\nНижне треугольная B\n")
print_array(B)
print("\nВерхне треугольная C\n")
print_array(C)

for i in range(0, n, 1):
  Y[i] = get_Yi(B, D, Y, i)

for i in reversed(range(0, n, 1)):
  X[i] = get_Xi(C, X, Y, i)

print("\nY: ")
print_vector(Y)
print("X: ")
print_vector(X)

print("\nВектор невязки после м. Холецкого")
A1 = np.column_stack((A, D))
r = get_residual_vector(A1, X)
print_vector_18(r)

# print("\nМетод Гаусса — Зейделя")
# X_1 = gauss_seidel_method(A, D)
# print_vector(X_1)

print("\nМетод Якоби")
iteration_list = []
X_3 = jacobi(A, D, iteration_list)
print_vector(X_3)
print(f"Количество итераций = {len(iteration_list)}")

print("\nВектор невязки после м. Якоби")
# A1 = np.column_stack((A, D))
# r = get_residual_vector(A1, X_2)
# r = np.dot(A, X_1) - D
# print_vector_18(r)
r = np.dot(A, X_3) - D
# print_vector_18(r)
print()
r = np.dot(A, X_3) - D
print_vector_18(r, 8)

print("\nЧисло обусловленности")
norm1=get_norm_matrix_00(A)
norm2=get_norm_matrix_00(np.linalg.inv(A))
print("%2.6f" %(norm1*norm2))

print("\nЧисло обусловленности - питон")
print("%2.6f" % (LA.cond(A, np.inf)))