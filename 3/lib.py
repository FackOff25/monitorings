import numpy as np

def ConvertArrFromBinary(binArray):
    arr = np.array([int(numStr,2) for numStr in binArray])
    return arr

def CenterVector(vector):
    m = np.mean(vector)
    return vector - m

def NormalizeVector(vector):
    return vector / np.sum(np.abs(vector))

def FindPlusVectors(vectors):
    W_matrix = np.dot(vectors, vectors.transpose()) 
    A = np.linalg.inv(W_matrix)
    v_plus = []
    for i in range(len(vectors)):
        sum = np.zeros(len(vectors[i]))
        for j in range(len(A)):
            sum += (A[i][j] * vectors[j])
        v_plus.append(sum)
    return np.array(v_plus)
