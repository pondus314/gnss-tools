import numpy as np
import datetime

MU = 398600500000000
OMEGA_E_DOT = 7.2921151467e-5
PI = 3.1415926535898
C = 299792458
F_REL = -2 * np.sqrt(MU) / (C**2)
WEEK_SECONDS = 3600*24*7
CONSTELL_CODES = ["_", "G", "?", "R", "J", "C", "E", "I"]
GPS_EPOCH = datetime.datetime(1980, 1, 6)
MAX_REF_DIF = datetime.timedelta(minutes=2)