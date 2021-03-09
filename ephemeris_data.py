import datetime
import numpy as np
import constants
import sys

A_0 = 0
A_1 = 1
A_2 = 2
IODE = 3
C_RS = 4
DELTA_N = 5
M_0 = 6
C_UC = 7
E = 8
C_US = 9
SQRT_A = 10
TOE = 11
C_IC = 12
OMEGA_0 = 13
C_IS = 14
I_0 = 15
C_RC = 16
OMEGA_LOWER = 17
OMEGA_DOT = 18
IDOT = 19
L2_CODES = 20
WEEK = 21
L2_P_FLAG = 22
SV_ACC = 23
SV_HEALTH = 24
TGD = 25
IODC = 26
TRANSMISSION_T = 27
INTERVAL_H = 28


class EphemerisData:
    def __init__(self, toc, data):
        self.toc: datetime.datetime = toc  # this is in GPS time, not UTC time,
        self.data = data[:28]
        if data[INTERVAL_H] != 0:
            self.fit_interval = data[INTERVAL_H] * 1800  # the time interval is symmetric, so multiply by 0.5h
        else:
            self.fit_interval = 2 * 3600  # same here, default time interval is 4 hours total, 2 before, 2 after

    def get_ek(self, gps_time):
        def inner(iters):
            if iters == 0:
                return m_k
            else:
                ek = inner(iters - 1)
                return ek + (m_k - ek + self.data[E] * np.sin(ek)) / (1. - self.data[E] * np.cos(ek))

        a = self.data[SQRT_A] ** 2
        n_0 = np.sqrt(constants.MU / a ** 3)
        t_k = np.mod((gps_time - self.data[TOE] + 302400.), 604800.) - 302400.
        n = n_0 + self.data[DELTA_N]
        m_k = self.data[M_0] + n * t_k

        return np.mod(inner(10) + np.pi, 2 * np.pi) - np.pi

    def get_dt(self, gps_time, gps_week):
        time = datetime.datetime(1980, 1, 6, 0, 0, 0) + datetime.timedelta(weeks=gps_week, seconds=gps_time)
        dt = (time - self.toc).total_seconds()
        return dt

    def get_dtsvs(self, gps_time, gps_week):
        dt = self.get_dt(gps_time, gps_week)
        dtsv = self.data[A_0] + self.data[A_1] * dt + self.data[A_2] * dt * dt
        return dtsv - self.data[TGD]

    def get_dtsv_relativistic(self, gps_time, gps_week):
        dt_rel = constants.F_REL * self.data[E] * self.data[SQRT_A] * np.sin(self.get_ek(gps_time))
        return self.get_dtsvs(gps_time, gps_week) + dt_rel

    def get_position(self, gps_time):
        a = self.data[SQRT_A] ** 2
        t_k = np.mod((gps_time - self.data[TOE] + 302400.), 604800.) - 302400.
        if np.abs(t_k) > self.fit_interval:
            print("Requested time", np.abs(t_k) - self.fit_interval, "outside of fit interval", file=sys.stderr)
        e_k = self.get_ek(gps_time)
        nu_k = 2 * np.arctan(np.sqrt((1 + self.data[E]) / (1 - self.data[E])) * np.tan(e_k/2))
        if e_k * nu_k < 0:
            nu_k += np.pi
        phi_k = nu_k + self.data[OMEGA_LOWER]
        sin_2phi_k = np.sin(2 * phi_k)
        cos_2phi_k = np.cos(2 * phi_k)

        delu_k = self.data[C_US] * sin_2phi_k + self.data[C_UC] * cos_2phi_k
        delr_k = self.data[C_RS] * sin_2phi_k + self.data[C_RC] * cos_2phi_k
        deli_k = self.data[C_IS] * sin_2phi_k + self.data[C_IC] * cos_2phi_k
        u_k = phi_k + delu_k
        r_k = a * (1 - self.data[E] * np.cos(e_k)) + delr_k
        i_k = self.data[I_0] + deli_k + self.data[IDOT] * t_k

        x_k_prime = r_k * np.cos(u_k)
        y_k_prime = r_k * np.sin(u_k)
        omega_k = self.data[OMEGA_0] + (self.data[OMEGA_DOT] - constants.OMEGA_E_DOT) * t_k - constants.OMEGA_E_DOT * \
                  self.data[TOE]
        x_k = x_k_prime * np.cos(omega_k) - y_k_prime * np.cos(i_k) * np.sin(omega_k)
        y_k = x_k_prime * np.sin(omega_k) + y_k_prime * np.cos(i_k) * np.cos(omega_k)
        z_k = y_k_prime * np.sin(i_k)

        return [x_k, y_k, z_k]
