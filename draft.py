#Давлетшин 13 вариант
#Метод квадратного корня, матрица e, матр норма 1, метод простых итерций с параметром
import numpy as np
from numpy import linalg as LA
import math

import numpy
from numpy.core.fromnumeric import partition
n=4

def print_matrix(a):
    for i in range(len(a)):
        for j in range(len(a[i])):
            print("%2.4f" % (a[i][j]), end=' ')
        print()

def Holets(matrix,holmatrix):#метод квадратного корня
       for i in range (0,n):
              temp=0
              for k in range (0,n):
                     temp+=holmatrix[k][i]*holmatrix[k][i]
              holmatrix[i][i]=math.sqrt(matrix[i][i]-temp)
              for j in range (i,n):
                     temp=0
                     for k in range (0,i):
                            temp+=holmatrix[k][i]*holmatrix[k][j]
                     holmatrix[i][j]=(matrix[i][j]-temp)/holmatrix[i][i]

def backSolveY(matrix,vector,vector1):#обратный ход сверху
       for i in range (0,n):
               vector1[i]= (vector[i]-sum(matrix[i][j]*vector1[j] for j in range(0,i)))/matrix[i][i]   

def backSolveX(matrix,vector,vector1): #обратный ход снизу
       for i in range(n-1, -1, -1):
              vector1[i] = (vector[i] - sum((matrix[i][j] * vector1[j] for j in range(i + 1, n)))) / matrix[i][i]

def discrepancy(matrix,vector,vector1): #невязка
       for i in range(n):
              vector1[i] = matrix[i][-1] - sum([matrix[i][j] * vector[j] for j in range(0, n)])

def vector_norm(vector):
       return LA.norm(vector)


def simplest(matrix, vector, vector1):
       tau = 2/(LA.norm(matrix))
       xk = vector1
       xkp = 0
       count = 0
       vectortr = vector
       vectortr.shape = 1, -1 
       while vector_norm(xk - xkp) > 1e-6:
              xkp = np.copy(xk)
              xk = tau*(vectortr.T - matrix.dot(xk)) + xk
              count += 1
       print(f'Число итераций по простым итерациям: {count}')
       return xk

def relax(matrix, vector,vector1, eps):
    converge = np.zeros((4, 1))
    t=False
    while not t:            
        vector_new = np.copy(vector1)
        for i in range(n):
            s1 = sum(matrix[i][j] * vector_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * vector1[j] for j in range(i + 1, n))
            vector_new[i] = (vector[i] - s1 - s2) / matrix[i][i]
            converge[i]= abs(vector_new[i]-vector1[i])
            if (max(converge)<eps):
                   t=True
        vector1 = vector_new
    return vector1 
 
def norm(matrix):  #норма матрицы 
    x = np.zeros((4, 1))
    for i in range(n):
        x[i]=sum(abs(matrix[i]))
    return(max(x))


matrix = np.array([[22, -3, -8, 7, -24],
                 [-3, 19, -6, 3, 40],
                 [-8, -6, 23, -7, -84],
                 [7, 3, -7, 18, -56]])
matrix2 = np.array([[22, -3, -8, 7],
                 [-3, 19, -6, 3],
                 [-8, -6, 23, -7],
                [7, 3, -7, 18]])
matrix3=np.linalg.inv(matrix2)
vector = np.array([-24,40,-84,-56])
vectorY = np.zeros((4, 1))
vectorX = np.zeros((4, 1))
vectorX2= np.zeros((4, 1))
vectorintX = np.zeros((4, 1))
holmatrix=np.zeros((4,4))
transposematrix=np.zeros((4,4))
discrepancyvector = np.zeros((4, 1))
discrepancyvector1=np.zeros((4, 1))
eps=10**(-6)
Holets(matrix2,holmatrix)
print("Метод квадратного корня")
print_matrix(holmatrix)
transposematrix=holmatrix.transpose()
print("\nТранспонированная матрица")
print_matrix(transposematrix)
backSolveY(transposematrix,vector,vectorY)
backSolveX(holmatrix,vectorY,vectorX)
discrepancy(matrix,vectorX,discrepancyvector)
print("\nРешение")
for i in range (n):
       print("x", i + 1, "=", "%2.4f" %  (vectorX[i]))
       vectorintX[i]=int(vectorX[i])
print("\nВектор невязки")
for i in range (n):
       print("r", i + 1, "=", "%.16f" % (abs(discrepancyvector[i])))
print("\nЧисло обусловленности")
norm1=norm(matrix2)
norm2=norm(matrix3)
print("%2.6f" %(norm1*norm2))
print("\nЧисло обусловленности с помощью команд питона")
print("%2.6f" % (LA.cond(matrix2, numpy.inf)))
print("\nМетод простых итераций")

vectorX2 = simplest(matrix2, vector, np.trunc(vectorX))
for i in range(n):
       print("x", i+1, "=", "%2.4f" % (vectorX2[i]))

print("\nВектор невязки")
for i in range (n):  
       print("r", i + 1, "=", "%.16f" % (abs(discrepancyvector[i])))

