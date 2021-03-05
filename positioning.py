import numpy as np
from scipy import constants


def pos_solution(sat_xyzt, pr_metres, user_xyzt_guess = np.array([0, 0, 0, 0], np.float64), type ="LS", weights = None):
    n = sat_xyzt.shape[0]
    user_xyzt_guess = user_xyzt_guess[np.newaxis].T
    while True:
        r_hat_vec = sat_xyzt[:,0:3]-user_xyzt_guess[0:3]
        r_hat = np.linalg.norm(r_hat_vec, axis=1)[np.newaxis]
        H = np.ones([n, 4], np.float64)
        H[:, :-1] = r_hat_vec/r_hat.T
        del_rho = r_hat - pr_metres
        if type == "LS":
            guess_delta = np.linalg.lstsq(H, del_rho)
        elif type == "SPS":
            guess_delta = np.linalg.solve(H, del_rho)
        else:
            return None
        user_xyzt_guess[0:3] += guess_delta[0:3]
        time_delta = guess_delta[3] / constants.c
        if np.linalg.norm(guess_delta) < 0.01:
            break
    user_xyzt_guess[3] += time_delta
    return user_xyzt_guess

