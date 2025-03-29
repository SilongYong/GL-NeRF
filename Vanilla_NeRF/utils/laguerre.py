import numpy as np
import scipy as sp
from scipy.special import sph_harm


def get_lagurre_points(N):
    points, weights = np.polynomial.laguerre.laggauss(N)
    return points, weights

def spherical_harmonics(l, m, theta, phi):
    scalar = sph_harm(m, l, theta, phi)
    return np.concatenate((np.real(scalar).reshape(-1, 1), np.imag(scalar).reshape(-1, 1)), axis=1)

def get_laguerre_points_list(max_N):
    points = []
    weights = []
    for N in range(1, max_N + 1):
        tmp_zero = np.zeros((max_N - N, 1))
        p, w = get_lagurre_points(N)
        p = p.reshape(-1, 1)
        p = np.concatenate((p, tmp_zero), axis=0)
        w = w.reshape(-1, 1)
        w = np.concatenate((w, tmp_zero), axis=0)
        points.append(p)
        weights.append(w)
    points = np.concatenate(points, axis=1)
    points = points.T
    weights = np.concatenate(weights, axis=1)
    weights = weights.T
    return points, weights
