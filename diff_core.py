"""
A module where to store some math functionalities than can be accessed from the main.py file, and get space for
important stuff.
"""
import math
import numpy as np


def partialDerivative_univariate(m, x, F, tol=0.0001, verbose=False):
    """
    Author: Marc Alsina
    It returns the mth derivative of an unknown function with a unique independent variable from a meshgrid data input.
    It uses the Taylor expansion series in order to find the derivative. The routine also adjust the degree of the
    taylor series (aka the number of the finite series) to match a specified tolerance, in order to:
    d^mF(x)/dx^m - deriv[x] = O(dx^p) < tol
    In a specified value of tolerance, for finner meshes, it will reduce the degree of finite differences, and then
    the algorithm will go faster, whereas, in a coarser meshes, the algorithm will increase the number of finite
    differences, and will go slower.
    The actual limitation is the algorithm needs tto have equally-spaced grid data. An improvement of the algorithm will
    be to generalize to non-spaced data.
    :param verbose: Boolean. Whether print parameters or not.
    :param m: Int. The derivative degree.
    :param x: List, tuple or np.array(). The independent variable.
    :param F: List, tuple or np.array(). The function evaluated at data-points x (F(x)).
    :param tol: The desired tolerance, or the maximum error permitted.
    :return: np.array(). The mth derivative of function F(x) evaluated at data points x.
    """
    # TODO investigate methods for non-equally spaced data
    # TODO estimate derivative errors taking into account the round off errors (machine epsilon) and truncation error
    #   (order of Taylor series)
    N = len(F)  # Number of data points
    h = abs(x[1] - x[0])
    deriv = np.ones_like(F) * math.factorial(m) / (h ** m)  # Where to store the values of derivatives
    p = int(math.log(tol, h))
    if ((m + p - 1) % 2) != 0:
        p += 1
    if (m + p - 1) >= N:
        raise ValueError(f"The data points are not sufficient for the specified tolerance. Data points: {N}, finite"
                         f"differences interval: {m + p - 1}. Try reducing the tolerance.")
    e = np.zeros(shape=(m + p))
    e[m] = 1
    i_max, i_min = int((m + p - 1) / 2), int(-(m + p - 1) / 2)
    i_max_f, i_min_f = int(m + p - 1), 0
    i_max_b, i_min_b = int(0), int(1 - m - p)
    W_foward = np.zeros(shape=(m + p, m + p))
    W_centered = np.zeros(shape=(m + p, m + p))
    W_backward = np.zeros(shape=(m + p, m + p))

    for k in range(0, m + p):
        for l1 in range(i_min_f, i_max_f + 1):
            W_foward[k, l1] = float(l1) ** float(k)
        for l2 in range(i_min, i_max + 1):
            W_centered[k, l2 + i_max] = float(l2) ** float(k)
        for l3 in range(i_min_b, i_max_b + 1):
            W_backward[k, l3 + abs(i_min_b)] = float(l3) ** float(k)
    C_forward = np.linalg.solve(a=W_foward, b=e)
    C_centered = np.linalg.solve(a=W_centered, b=e)
    C_backward = np.linalg.solve(a=W_backward, b=e)

    # Computing forward differences
    for indx1 in range(0, i_max):
        buff = 0
        for ifoward in range(0, i_max_f + 1):
            buff += C_forward[ifoward] * F[indx1 + ifoward]
        deriv[indx1] *= buff

    # Computing centered differences
    for indx2 in range(i_max, N - i_max):
        buff = 0
        for icentered in range(0, i_max - i_min + 1):
            buff += C_centered[icentered] * F[indx2 + icentered - i_max]
        deriv[indx2] *= buff

    # Computing backward differences
    for indx3 in range(N - i_max, N):
        buff = 0
        for ibacward in range(0, abs(i_min_b) + 1):
            buff += C_backward[ibacward] * F[indx3 + i_min_b + ibacward]
        deriv[indx3] *= buff

    if verbose:
        print("--Parameters--")
        print(f"p is set to: {p}")
        print(f"i_min_f: {i_min_f}, i_max_f: {i_max_f}")
        print(f"i_min: {i_min}, i_max: {i_max}")
        print(f"i_min_b: {i_min_b}, i_max_b: {i_max_b}")

    return deriv


def partialDerivative_bivariate(m1, m2, x1, x2, F, verbose=False, tol=0.0001):
    """
    Author: Marc Alsina
    A function that evaluates any derivative of a bi-variate function F(x1, x2) (which is unknown) with a given
    specified tolerance. Depending on the tolerance, the algorithm will adjust the degree of differentiation to adjust
    the tolerance, in the way that:
    F^(m1,m2)(x1,x2)-deriv[x1, x2] = max(O(dx1^p1), O(dx2^p2)) < tol
    Where deriv[x1, x2] is the computed derivative. This function works only for equally spaced grid data.
    This function will compute three types of derivatives: forward for initial data points, centered for middle data
    points, and backward for the last data points.
    The given inputs MUST have the data shape of a np.meshgrid(*xi, indexing='ij'), with F being evaluated on these mesh
    grid points, or similar.
    :param verbose: Whether print internal parameters or not
    :param m1: Int, scalar. The derivative degree respect x1
    :param m2: Int, scalar. The derivative degree respect x2
    :param x1: List, Tuple or np.array().
    :param x2: List, Tuple or np.array().
    :param F: List, Tuple or np.array(). The unknown function evaluation at (x1, x2) points.
    :param tol: Minimum tolerance
    :return: The value of function mth derivative at (x1, x2) points.
    """
    # TODO investigate methods for non-equally spaced data
    # TODO estimate derivative errors taking into account the round off errors (machine epsilon) and truncation error
    #   (order of Taylor series)
    if not (len(x1) == len(x2)) and not (len(x1) == len(F)) and not (len(x2) == len(F)):
        raise ValueError(f"Different sizes in the input arrays. Sizes are: x1={len(x1)}, x2={len(x2)}, F={len(F)}")
    h1 = abs(x1[1][0] - x1[0][0])
    h2 = abs(x2[0][1] - x2[0][0])
    N1, N2 = F.shape  # The shape of the mesh grid function points
    deriv = np.ones_like(F) * (math.factorial(m1) * math.factorial(m2)) / ((h1 ** m1) * (h2 ** m2))  # Where to
    # store the values of derivatives
    p = int(math.log(tol, min(h1, h2)))
    p1 = p
    p2 = p
    if ((m1 + p1 - 1) % 2) != 0:
        p1 += 1
    if ((m2 + p2 - 1) % 2) != 0:
        p2 += 1

    if ((m1 + p1 - 1) >= N1) or ((m2 + p2 - 1) >= N2):
        raise ValueError(f"The data points are not sufficient for the specified tolerance. Data points: {(N1, N2)}, "
                         f"finite differences interval: x1 -> {m1 + p1 - 1} | x2 -> {m2 + p2 - 1}. Try reducing the "
                         f"tolerance.")

    # Parameters for x1
    e1 = np.zeros(shape=(m1 + p1))
    e1[m1] = 1
    i_max1, i_min1 = int((m1 + p1 - 1) / 2), int(-(m1 + p1 - 1) / 2)
    i_max_f1, i_min_f1 = int(m1 + p1 - 1), 0
    i_max_b1, i_min_b1 = int(0), int(1 - m1 - p1)
    W1_foward = np.zeros(shape=(m1 + p1, m1 + p1))
    W1_centered = np.zeros(shape=(m1 + p1, m1 + p1))
    W1_backward = np.zeros(shape=(m1 + p1, m1 + p1))

    for k in range(0, m1 + p1):
        for l1 in range(i_min_f1, i_max_f1 + 1):
            W1_foward[k, l1] = float(l1) ** float(k)
        for l2 in range(i_min1, i_max1 + 1):
            W1_centered[k, l2 + i_max1] = float(l2) ** float(k)
        for l3 in range(i_min_b1, i_max_b1 + 1):
            W1_backward[k, l3 + abs(i_min_b1)] = float(l3) ** float(k)
    C1_forward = np.linalg.solve(a=W1_foward, b=e1)
    C1_centered = np.linalg.solve(a=W1_centered, b=e1)
    C1_backward = np.linalg.solve(a=W1_backward, b=e1)

    # Parameters for x2
    e2 = np.zeros(shape=(m2 + p2))
    e2[m2] = 1
    i_max2, i_min2 = int((m2 + p2 - 1) / 2), int(-(m2 + p2 - 1) / 2)
    i_max_f2, i_min_f2 = int(m2 + p2 - 1), 0
    i_max_b2, i_min_b2 = int(0), int(1 - m2 - p2)
    W2_foward = np.zeros(shape=(m2 + p2, m2 + p2))
    W2_centered = np.zeros(shape=(m2 + p2, m2 + p2))
    W2_backward = np.zeros(shape=(m2 + p2, m2 + p2))

    for k in range(0, m2 + p2):
        for l1 in range(i_min_f2, i_max_f2 + 1):
            W2_foward[k, l1] = float(l1) ** float(k)
        for l2 in range(i_min2, i_max2 + 1):
            W2_centered[k, l2 + i_max2] = float(l2) ** float(k)
        for l3 in range(i_min_b2, i_max_b2 + 1):
            W2_backward[k, l3 + abs(i_min_b2)] = float(l3) ** float(k)
    C2_forward = np.linalg.solve(a=W2_foward, b=e2)
    C2_centered = np.linalg.solve(a=W2_centered, b=e2)
    C2_backward = np.linalg.solve(a=W2_backward, b=e2)

    # TODO optimize this for
    for N1_indx in range(0, N1):
        for N2_indx in range(0, N2):
            buff = 0
            # Compute forward differences for N1 and N2
            if (N1_indx < i_max1) and (N2_indx < i_max2):
                for indx_1 in range(0, i_max_f1 + 1):
                    for indx_2 in range(0, i_max_f2 + 1):
                        buff += C1_forward[indx_1] * C2_forward[indx_2] * F[indx_1 + N1_indx, indx_2 + N2_indx]
            # Compute forward differences for N1 and backward differences for N2
            elif (N1_indx < i_max1) and (N2_indx >= N2 - i_max2):
                for indx_1 in range(0, i_max_f1 + 1):
                    for indx_2 in range(0, abs(i_min_b2) + 1):
                        buff += C1_forward[indx_1] * C2_backward[indx_2] * F[
                            indx_1 + N1_indx, N2_indx + i_min_b2 + indx_2]
            # Compute backward differences for N1 and N2
            elif (N1_indx >= N1 - i_max1) and (N2_indx >= N2 - i_max2):
                for indx_1 in range(0, abs(i_min_b1) + 1):
                    for indx_2 in range(0, abs(i_min_b2) + 1):
                        buff += C1_backward[indx_1] * C2_backward[indx_2] * F[
                            N1_indx + i_min_b1 + indx_1, N2_indx + i_min_b2 + indx_2]
            # Compute backward differences for N1 and forward differences for N2
            elif (N1_indx >= N1 - i_max1) and (N2_indx < i_max2):
                for indx_1 in range(0, abs(i_min_b1) + 1):
                    for indx_2 in range(0, i_max_f2 + 1):
                        buff += C1_backward[indx_1] * C2_forward[indx_2] * F[
                            N1_indx + i_min_b1 + indx_1, indx_2 + N2_indx]
            # Compute centered differences for N1 and N2
            elif ((N1_indx >= i_max1) and (N1_indx < N1 - i_max1)) and (
                    (N2_indx >= i_max1) and (N2_indx < N2 - i_max2)):
                for indx_1 in range(0, i_max1 - i_min1 + 1):
                    for indx_2 in range(0, i_max2 - i_min2 + 1):
                        buff += C1_centered[indx_1] * C2_centered[indx_2] * F[
                            N1_indx + indx_1 - i_max1, N2_indx + indx_2 - i_max2]
            # Compute centered differences for N1 and backward differences for N2
            elif ((N1_indx >= i_max1) and (N1_indx < N1 - i_max1)) and (N2_indx >= N2 - i_max2):
                for indx_1 in range(0, i_max1 - i_min1 + 1):
                    for indx_2 in range(0, abs(i_min_b2) + 1):
                        buff += C1_centered[indx_1] * C2_backward[indx_2] * F[
                            N1_indx + indx_1 - i_max1, N2_indx + i_min_b2 + indx_2]
            # Compute centered differences for N1 and forward differences for N2
            elif ((N1_indx >= i_max1) and (N1_indx < N1 - i_max1)) and (N2_indx < i_max2):
                for indx_1 in range(0, i_max1 - i_min1 + 1):
                    for indx_2 in range(0, i_max_f2 + 1):
                        buff += C1_centered[indx_1] * C2_forward[indx_2] * F[
                            N1_indx + indx_1 - i_max1, indx_2 + N2_indx]
            # Compute backward differences for N1 and centered differences for N2
            elif (N1_indx >= N1 - i_max1) and ((N2_indx >= i_max1) and (N2_indx < N2 - i_max2)):
                for indx_1 in range(0, abs(i_min_b1) + 1):
                    for indx_2 in range(0, i_max2 - i_min2 + 1):
                        buff += C1_backward[indx_1] * C2_centered[indx_2] * F[
                            N1_indx + i_min_b1 + indx_1, N2_indx + indx_2 - i_max2]
            # Compute forward differences for N1 and centered differences for N2
            elif (N1_indx < i_max1) and ((N2_indx >= i_max1) and (N2_indx < N2 - i_max2)):
                for indx_1 in range(0, i_max_f1 + 1):
                    for indx_2 in range(0, i_max2 - i_min2 + 1):
                        buff += C1_forward[indx_1] * C2_centered[indx_2] * F[
                            indx_1 + N1_indx, N2_indx + indx_2 - i_max2]

            deriv[N1_indx, N2_indx] *= buff

    if verbose:
        print("--Parameters for x1--")
        print(f"p is set to: {p1}")
        print(f"i_min_f: {i_min_f1}, i_max_f: {i_max_f1}")
        print(f"i_min: {i_min1}, i_max: {i_max1}")
        print(f"i_min_b: {i_min_b1}, i_max_b: {i_max_b1}")
        print()
        print("--Parameters for x2--")
        print(f"p is set to: {p2}")
        print(f"i_min_f: {i_min_f2}, i_max_f: {i_max_f2}")
        print(f"i_min: {i_min2}, i_max: {i_max2}")
        print(f"i_min_b: {i_min_b2}, i_max_b: {i_max_b2}")

    return deriv
