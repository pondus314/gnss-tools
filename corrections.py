import numpy as np
import constants


def angle(vector1, vector2, vector_positive):
    normal_positive = vector_positive / np.linalg.norm(vector_positive)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cross = np.cross(vector1, vector2) / (norm1 * norm2)
    cross = np.dot(cross, normal_positive)
    dot = vector1.dot(vector2) / (norm1 * norm2)
    return np.arctan2(cross, dot)


def ecef_to_eci(sat_ecef_positions, sats_pr_metres):
    n = sat_ecef_positions.shape[0]
    sat_eci_positions = sat_ecef_positions
    for i in range(n):
        theta = - constants.OMEGA_E_DOT * sats_pr_metres[i] / constants.C
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        sat_eci_positions[i, 0:2] = sat_ecef_positions[i, 0:2].dot(rotation)
    return sat_eci_positions


def klobuchar_correction_nanos(user_xyz, user_t, user_lla, sat_xyz, ion_alpha, ion_beta):

    user_to_sat = sat_xyz - user_xyz
    earth_axis = np.array([0., 0., 1000.])
    north_bound = earth_axis - user_xyz * user_xyz.dot(earth_axis) / np.linalg.norm(user_xyz)**2
    sat_bound = user_to_sat - user_xyz * user_xyz.dot(user_to_sat) / np.linalg.norm(user_xyz)**2

    sat_elevation_sc = np.abs(angle(sat_bound, user_to_sat, np.cross(sat_bound, user_to_sat))) / np.pi
    sat_azimuth = angle(sat_bound, north_bound, user_xyz)

    psi = 0.0137/(sat_elevation_sc + 0.11) - 0.022
    phi_i = user_lla[0] / 180. + psi * np.cos(sat_azimuth)
    phi_i = np.clip(phi_i, -.416, .416)
    lambda_i = user_lla[1] / 180. + psi * np.sin(sat_azimuth) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)
    t = np.mod(lambda_i * 43200 + user_t, 86400)

    p = np.poly1d(ion_beta[::-1])
    P = p(phi_m)
    P = np.clip(P, -np.inf, 72000)

    X = 2*np.pi*(t - 50400) / P
    F = 1. + 16. * (.53 - sat_elevation_sc) ** 3

    if np.abs(X) > np.pi / 2:
        return F * 5
    else:
        a = np.poly1d(ion_alpha[::-1])
        A = a(phi_m)
        A = np.clip(A, 0, np.inf)
        return F * (5 + A * (1 - (X ** 2) / 2 + (X ** 4) / 24))
