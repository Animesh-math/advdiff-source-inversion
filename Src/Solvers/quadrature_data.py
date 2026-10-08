"""Shared quadrature rules for FEM volume and boundary integrals."""

import numpy as np


def triangle_quadrature(ngauss):
    """Quadrature on the reference triangle for ngauss in {1, 3, 4}."""
    if ngauss == 1:
        weight = np.array([1.0 / 2.0])
        xgauss = np.array([1.0 / 3.0])
        ygauss = np.array([1.0 / 3.0])
    elif ngauss == 3:
        weight = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        xgauss = np.array([1.0 / 2.0, 1.0 / 2.0, 0.0])
        ygauss = np.array([0.0, 1.0 / 2.0, 1.0 / 2.0])
    elif ngauss == 4:
        weight = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        xgauss = np.array([1.0 / 3.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0])
        ygauss = np.array([1.0 / 3.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0])
    else:
        raise ValueError("triangle_quadrature supports ngauss = 1, 3, or 4")

    return weight, xgauss, ygauss


def interval_quadrature_0_to_1(st_ngauss):
    """Quadrature on interval [0,1] for st_ngauss in {1, 2, 3}."""
    if st_ngauss == 1:
        st_weight = np.array([1.0])
        st_xgauss = np.array([0.5])
    elif st_ngauss == 2:
        st_weight = np.array([1.0 / 2.0, 1.0 / 2.0])
        st_xgauss = np.array([0.211324865405187, 0.788675134594813])
    elif st_ngauss == 3:
        st_weight = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        st_xgauss = np.array([0.112701665379258, 0.5, 0.887298334620742])
    else:
        raise ValueError("interval_quadrature_0_to_1 supports st_ngauss = 1, 2, or 3")

    return st_weight, st_xgauss

def interval_quadrature_minus1_to_1(st_ngauss):
    """Quadrature on interval [-1,1] for st_ngauss in {1, 2, 3}."""
    if st_ngauss == 1:
        st_weight = np.array([2.0])
        st_xgauss = np.array([0.0])
    elif st_ngauss == 2:
        st_weight = np.array([1.0, 1.0])
        st_xgauss = np.array([-0.577350269189626, 0.577350269189626])
    elif st_ngauss == 3:
        st_weight = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        st_xgauss = np.array([-0.774596669241483, 0.0, 0.774596669241483])
    else:
        raise ValueError("interval_quadrature_minus1_to_1 supports st_ngauss = 1, 2, or 3")

    return st_weight, st_xgauss