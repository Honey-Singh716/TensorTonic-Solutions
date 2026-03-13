import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here

    A = np.array(A)
    m,n = A.shape
    trans = np.ones((n,m))

    for row in range(0,len(A)):
        for col in range(0,len(A[0])):
            trans[col][row] = A[row][col]


    return trans
