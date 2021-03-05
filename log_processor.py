import rinex_processing
import numpy as np
import scipy
import copy


def main():
    refname = "newr336.20o"
    filename = "uhm file name yes"
    observation_data = rinex_processing.ObservationData.process_rinex(refname)
    observ_data, ref_timestamps = rinex_processing.ObservationData.get_observable_data(observation_data, only_gps=True)
    ref_c_one = rinex_processing.ObservationData.extract_pr_ttx(observ_data, ref_timestamps)


if __name__ == '__main__':
    main()