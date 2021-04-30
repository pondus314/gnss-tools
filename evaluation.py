import statsmodels.tsa.stattools as stattools
import numpy as np


def main():
    data = np.genfromtxt('results/gnss_log_2021_04_08_12_46_29_WLS.csv', delimiter=',', dtype=np.float64, skip_header=1)
    data_pieces = []
    last_skip = 0
    for i in range(len(data) - 1):
        if data[i + 1, 3] - data[i, 3] > 10:
            print(data[i+1, 3], data[i, 3])
            data_pieces.append(data[last_skip:i+1, :])
            last_skip = i + 1
    print(len(data_pieces))


if __name__ == '__main__':
    main()