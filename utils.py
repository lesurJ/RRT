""" Some useful functions for path smoothing """
""" Authored : Jean Lesur, 2021 """

import math
import numpy as np

def BernsteinPolynomial(t,n,i):
    """ Compute the Bernstein polynomial """
    return math.comb(n,i) * t**i * (1-t)**(n-i)

def BezierCurve(vectors, t):
    """ Given a collection of control points, find the parametric Bezier curve"""
    n = len(vectors)
    polynomial = np.array([BernsteinPolynomial(t, n-1, i) for i in range(n)])
    return np.array(vectors).T @ polynomial