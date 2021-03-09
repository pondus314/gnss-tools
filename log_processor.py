from contextlib import redirect_stdout

import rinex_processing
import numpy as np
import constants
import datetime
import positioning
import pyproj

REFERENCE= False

class RawMeasurementData:
    def __init__(self, ttx_nanos, ttx_uncertainty_nanos, snr_db, gps_week, gps_sec):
        self.ttx_secs = ttx_nanos*10**(-9)
        self.ttx_uncertainty_nanos = ttx_uncertainty_nanos
        self.snr_db = snr_db
        self.gps_week = gps_week
        self.gps_sec = gps_sec
        self.timestamp = datetime.datetime(1980, 1, 6) + datetime.timedelta(weeks=gps_week, seconds=gps_sec)
        self.sat_timestamp = self.timestamp - datetime.timedelta(seconds=ttx_nanos**(-9))


def process_google_data(filename):
    goo_f = open(filename, "r")
    line = goo_f.readline()
    while line.startswith("#"):
        line = goo_f.readline()
    i_fb_n = 0
    measurements = []
    measurement_set = dict()
    first = True
    while first or (line := goo_f.readline()):
        first = False
        line_data = line.split(',')
        if line_data[0] != 'Raw':
            continue
        time_nanos = int(line_data[2])
        leap_second = line_data[3]
        full_bias_nanos = int(line_data[5])
        svid = int(line_data[11])
        state = int(line_data[13])
        ttx_nanos = int(line_data[14])
        ttx_uncert_nanos = int(line_data[15])
        snr_db = float(line_data[27])
        constell = int(line_data[28])
        if not (state & 1 and state & 8):
            continue
        elif constell != 1:
            continue
        elif ttx_uncert_nanos >= 500:
            continue
        if i_fb_n != full_bias_nanos:
            i_fb_n = full_bias_nanos
            measurements.append(measurement_set)
            measurement_set = dict()
        gps_total_sec = 10 ** (-9) * (time_nanos - full_bias_nanos)
        gps_week, gps_time = np.divmod(gps_total_sec, constants.WEEK_SECONDS)
        sat_id = "{}{:02d}".format(constants.CONSTELL_CODES[constell], svid)
        measurement_data = RawMeasurementData(ttx_nanos, ttx_uncert_nanos, snr_db, gps_week, gps_time)
        measurement_set[sat_id] = measurement_data
    else:
        measurements.append(measurement_set)
    return measurements


def main():
    refname = "ref/newr336o.20o"
    filename = "measurements/gnss_logger_2020_12_01_14_31_25.txt"
    ephname = "eph/hour3360.20n"
    if REFERENCE:
        print("Loading reference data from", refname)
        observation_data = rinex_processing.ObservationData.process_rinex(refname)
        observ_data = rinex_processing.ObservationData.get_observable_data(observation_data, only_gps=True)
        ref_c_one_ttx = rinex_processing.ObservableData.extract_pr_ttx(observ_data)
    else:
        print("Reference data not being used in this run")
    print("Reference data loaded, now loading ephemeris data from", ephname)
    eph_data = rinex_processing.NavigationData.process_gps_rinex(ephname)
    print("Ephemeris data loaded, now loading measured data from", filename)
    record_data = process_google_data(filename)
    positions_xyz = []
    print("Measurement data loaded, now calculating positions")
    for i in range(len(record_data)):
        sat_positions = []
        sat_pr_metres = []
        user_t = 0
        if len(record_data[i]) < 4:
            continue
        for sat in record_data[i].keys():
            raw_data = record_data[i][sat]
            relevant_eph = eph_data.get_relevant_ephemeris(raw_data.sat_timestamp, [sat])
            if len(relevant_eph) < 1:
                continue
            eph = relevant_eph[sat]
            dtsv = eph.get_dtsv_relativistic(raw_data.ttx_secs, raw_data.gps_week)
            sat_pos = eph.get_position(raw_data.ttx_secs - dtsv)
            sat_positions.append(sat_pos + [raw_data.ttx_secs - dtsv])
            sat_pr_metres.append(constants.C*(raw_data.gps_sec - (raw_data.ttx_secs - dtsv)))
            user_t = raw_data.gps_sec
        if len(sat_positions) < 4:
            continue
        position = positioning.pos_solution(np.array(sat_positions),
                                            np.array(sat_pr_metres),
                                            method="LS",
                                            user_xyzt_guess=np.array([0, 0, 0, user_t]))
        positions_xyz.append(position.squeeze())
    pos_array = np.array(positions_xyz)
    print("positions calculated, now transforming to lat long")
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(proj_from=ecef, proj_to=lla)
    positions = np.array(transformer.transform(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], radians=False)).T[:, [1, 0, 2]]
    np.savetxt(filename[0:-3] + "csv", positions, delimiter=",", header="lat, lon, alt", fmt=["%.8f", "%.8f", "%2d"])
    print("Lat Long positions calculated, saved in file", filename[0:-3] + "csv")


if __name__ == '__main__':
    main()