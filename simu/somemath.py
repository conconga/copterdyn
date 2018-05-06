#!/usr/bin/python
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
# author:      Luciano Augusto Kruk
# website:     www.kruk.eng.br
#
# description: Package with some mathematical functions, as
#              block-matrixes manipulation.
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

import numpy as np
from   numpy           import dot;
#import math  as mt
#from numpy import zeros,sin,cos,empty,sqrt;

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
##WWww=--  calc A (x) B:  --=wwWW##
def fn_elwisemult(A,B):
    """
    Calculates the elementwise product of two matrixes. Each
    element of matrix A is dot-multiplied by all elements in B.
    """

    C = dot(A[0,0],B) # aux result

    # dimensions:
    a = C.shape[0]
    b = C.shape[1]
    r = A.shape[0] * a # C rows
    c = A.shape[1] * b # C columns

    # buffer:
    C = np.zeros((r,c))

    # indexes:
    m = 0 # row in C
    n = 0 # column in C

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            #print "m=%d; m+a=%d; n=%d; n+b=%d" % (m,m+a,n,n+b)
            C[m:(m+a), n:(n+b)] = dot(A[i,j], B)
            n = n+b
        n = 0
        m = m+a

    #print C
    return C

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
def fn_skew(vector):
    """
    This function returns a numpy array with the skew symmetric cross
    product matrix for vector.  the skew symmetric cross product matrix is
    defined such that np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    assert(max(vector.shape) == 3);

    return np.array([[0, -vector[2], vector[1]], 
        [vector[2], 0, -vector[0]], 
        [-vector[1], vector[0], 0]])

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
##WWww=--  block skew:  --=wwWW##
def fn_blockskew(vec):
    """
    Generates a matrix with several skew-matrixes in. Each
    three elements of vec are used to generate one of the 
    skew-matrixes.
    """

    if isinstance(vec, np.ndarray):
        n = max(vec.shape)
    else:
        n = len(vec)

    # requires multiple of 3:
    assert(not (n%3));
    n = n//3

    # outut buffer:
    C = np.zeros((3*n,3*n))

    for i in range(n):
        m = 3*i
        n = 3*(i+1)
        C[m:n, m:n] = fn_skew(np.asarray(vec)[m:n])

    return C

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
##WWww=--  separate [3x1] vectors from a 
#               concatenated set of vectors:  --=wwWW##
def fn_separate3x1(vec):
    aux = vec.squeeze()
    N   = np.max(aux.shape)
    return [aux[(i*3):(3*(i+1))].reshape((3,1)) for i in range(N/3)]

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
##WWww=--  norm of a vector  --=wwWW##
def fn_norm(vec):
    return np.sqrt(np.sum(vec*vec))

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

if (__name__ == "__main__"):
    print
    
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
