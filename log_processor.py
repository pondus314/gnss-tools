import sys
from typing import List, Dict

import rinex_processing
import numpy as np
import constants
import datetime
import positioning
import pyproj
import bisect

REFERENCE = True
METHOD = "WLS"
REJECT_OUTLIERS = True


class RawMeasurementData:
    def __init__(self, ttx_nanos, ttx_uncertainty_nanos, gps_week, gps_sec, leap_second, correction=0):
        self.ttx_secs = ttx_nanos * 10 ** (-9)
        self.ttx_uncertainty_nanos = ttx_uncertainty_nanos
        self.gps_week = gps_week
        self.gps_sec = gps_sec
        self.timestamp = constants.GPS_EPOCH + datetime.timedelta(weeks=gps_week, seconds=gps_sec)
        self.sat_timestamp = constants.GPS_EPOCH + datetime.timedelta(weeks=gps_week,
                                                                      microseconds=ttx_nanos * 10 ** (-3))
        self.leap_second = leap_second
        self.correction = correction


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
        # snr_db = float(line_data[27])
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
        measurement_data = RawMeasurementData(ttx_nanos, ttx_uncert_nanos, gps_week, gps_time, leap_second)
        measurement_set[sat_id] = measurement_data
    else:
        measurements.append(measurement_set)
    return measurements


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return np.array(range(len(data)))[s < m]


def get_positioning_solution(raw_data: List[Dict[str, RawMeasurementData]], eph_data, method):

    n = len(raw_data)
    positions_xyz = []
    for i in range(n):
        sat_positions = []
        sat_weights = []
        sat_pr_metres = []
        user_t = 0
        if len(raw_data[i]) < 4:
            continue
        sats = raw_data[i].keys()
        sat_timestamps = list(map(lambda x: raw_data[i][x].sat_timestamp, sats))
        relevant_ephs = eph_data.get_relevant_ephemeris(dict(zip(sats, sat_timestamps)), sats)
        for sat in raw_data[i].keys():
            measurement_data = raw_data[i][sat]
            eph = relevant_ephs[sat]
            dtsv = eph.get_dtsv_relativistic(measurement_data.ttx_secs, measurement_data.gps_week)
            ttx_corrected = measurement_data.ttx_secs - dtsv
            sat_pos = eph.get_position(ttx_corrected)
            sat_positions.append(sat_pos + [ttx_corrected])
            sat_pr_metres.append(constants.C * (measurement_data.gps_sec - ttx_corrected + measurement_data.correction))
            sat_weights.append(1. / measurement_data.ttx_uncertainty_nanos)
            user_t = measurement_data.gps_sec
        if len(sat_positions) < 4:
            continue
        position = positioning.pos_solution(np.array(sat_positions),
                                            np.array(sat_pr_metres),
                                            method=method,
                                            user_xyzt_guess=np.array([0, 0, 0, user_t]),
                                            weights=sat_weights)
        position = position.squeeze()
        if REJECT_OUTLIERS:
            offsets = np.array([np.linalg.norm(sat_positions[i][0:3] - position[0:3]) - sat_pr_metres[i] for i in range(len(raw_data[i]))])
            m = 2.
            to_reuse = reject_outliers(offsets)
            while len(to_reuse) < 4:
                m = m * 1.2
                to_reuse = reject_outliers(offsets, m)
            sat_positions_rejected = np.array(sat_positions)[to_reuse]
            sat_pr_metres_rejected = np.array(sat_pr_metres)[to_reuse]
            sat_weights_rejected = np.array(sat_weights)[to_reuse]
            position_improved = positioning.pos_solution(np.array(sat_positions_rejected),
                                                         np.array(sat_pr_metres_rejected),
                                                         method=method,
                                                         user_xyzt_guess=np.array([0, 0, 0, user_t]),
                                                         weights=sat_weights_rejected)
            positions_xyz.append(position_improved.squeeze())
        else:
            positions_xyz.append(position)
    return positions_xyz


def main():
    refname = "ref/sneo336o.20o"
    filename = "measurements/gnss_logger_2020_12_01_14_31_25.txt"
    ephname = "eph/hour3360.20n"

    print("Beginning processing, loading ephemeris data from \"{}\"".format(ephname))
    eph_data = rinex_processing.NavigationData.process_gps_rinex(ephname)

    print("Ephemeris data loaded, now loading measured data from \"{}\"".format(filename))
    uncorrected_data = process_google_data(filename)
    n = len(uncorrected_data)
    print("Measurement data loaded, now", end=" ")

    if REFERENCE:
        print("Loading reference data from\"{}\"".format(refname))
        observation_data = rinex_processing.ObservationData.process_rinex(refname)
        observ_data = rinex_processing.ObservationData.get_observable_data(observation_data, only_gps=True)
        ref_c_one_ttx = rinex_processing.ObservableData.extract_pr_ttx(observ_data)
        coord_array = np.array(observation_data.station_coord)

        print("Reference data loaded, now running corrections")
        for i in range(n):
            to_remove = []
            for sat in uncorrected_data[i].keys():  # Go through all of the uncorrected data sats
                timestamp = uncorrected_data[i][sat].sat_timestamp  # ttx as a datetime
                if sat not in observ_data.sat_timestamps.keys():
                    continue
                sat_timestamps = observ_data.sat_timestamps[sat]
                sat_indices = observ_data.sat_indices[sat]

                i_sat = bisect.bisect_left(sat_timestamps, timestamp)  # Get index of the closest larger timestamp
                if i_sat == len(sat_timestamps):
                    if timestamp - sat_timestamps[-1] > constants.MAX_REF_DIF:
                        print(
                            "data from satellite {} at time {}, {}s outside reference interval".format(sat, timestamp, (
                                        timestamp - sat_timestamps[-1]).total_seconds()),
                            file=sys.stderr
                        )
                        to_remove.append(sat)
                        continue
                    else:
                        i_sat -= 1
                if i_sat != 0:  # present in the reference data
                    if sat_timestamps[i_sat] - timestamp > timestamp - sat_timestamps[i_sat - 1]:
                        i_sat -= 1

                i_ref = sat_indices[i_sat]
                ref_timestamp = observ_data.timestamps[i_ref]
                eph_dict = eph_data.get_relevant_ephemeris({sat: ref_timestamp}, sats=[sat])

                if len(eph_dict) == 0:
                    continue

                ttx_secs = ref_c_one_ttx[i_ref][sat] * 10 ** (-9)
                eph = eph_dict[sat]
                dtsv = eph.get_dtsv_relativistic(ttx_secs, uncorrected_data[i][sat].gps_week)
                sat_pos = np.array(eph.get_position(ttx_secs - dtsv))
                pr_real = np.linalg.norm(sat_pos - coord_array)
                pr_obs = observ_data.observable_data[i_ref][sat] + dtsv * constants.C
                dif_m = pr_real - pr_obs
                uncorrected_data[i][sat].correction += dif_m / constants.C

            for sat in to_remove:
                uncorrected_data[i].pop(sat)

        used_data = uncorrected_data
        print("Differential correction finished, now calculating positions")

    else:
        used_data = uncorrected_data
        print("calculating positions")

    positions_xyz = get_positioning_solution(used_data, eph_data, METHOD)

    pos_array = np.array(positions_xyz)
    n_after = len(positions_xyz)
    print("Positions calculated, now transforming to lat long")

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(proj_from=ecef, proj_to=lla)
    positions_lla = transformer.transform(pos_array[:, 0],
                                          pos_array[:, 1],
                                          pos_array[:, 2],
                                          radians=False)
    positions = np.zeros([n_after, 4])
    positions[:, :3] = np.array(positions_lla).T[:, [1, 0, 2]]
    positions[:, 3] = np.array(pos_array[:, 3])
    outname = "results" + filename[12:-4] + "_" + METHOD + ("_ref" if REFERENCE else "") \
              + ("_rej" if REJECT_OUTLIERS else "")+ ".csv"
    np.savetxt(outname,
               positions,
               delimiter=",",
               header="lat,lon,alt,time",
               fmt=["%.8f", "%.8f", "%.2f", "%2d"],
               comments="")
    print("Lat Long positions calculated, saved in file \"", outname, "\"", sep="")


if __name__ == '__main__':
    main()
