import datetime
from contextlib import redirect_stdout
from typing import List, Union, Dict

import constants
import ephemeris_data
import sys
import numpy as np


# TODO: add documentation to functions and classes here

def parse_rinex_decimal(rinex_decimal: str):
    rinex_decimal = rinex_decimal.strip()
    return float(rinex_decimal.replace("D", "E"))


class ObservableData:
    def __init__(self,
                 timestamps: List[datetime.datetime],
                 sat_timestamps: Dict[str, List[datetime.datetime]],
                 sat_indices: Dict[str, List[int]],
                 obs_data: List[Dict[str, Union[int, float]]],
                 obs_name: str,
                 pr_type: bool):
        self.timestamps = timestamps
        self.sat_timestamps = sat_timestamps
        self.sat_indices = sat_indices
        self.observable_data = obs_data
        self.obs_name = obs_name
        self.pr_type = pr_type

    @staticmethod
    def extract_pr_ttx(obs_data):
        if not obs_data.pr_type:
            raise NotImplemented
        obs_ttx = []
        timestamps = obs_data.timestamps
        pr_data = obs_data.observable_data
        for i in range(len(pr_data)):
            date = timestamps[i]
            week_nanos = (date.weekday() + 1 % 7) * 86400 * (10 ** 9)
            week_nanos += (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 10 ** 9
            ttx = dict()
            for sat in pr_data[i].keys():
                pr = pr_data[i][sat]
                pr_nanos = (pr * 10 ** 9) / constants.C
                ttx_nanos = week_nanos - pr_nanos
                ttx[sat] = ttx_nanos
            obs_ttx.append(ttx)
        return obs_ttx

    @staticmethod
    def extract_pr_correction(obs_data, station_coord, ephemeris_all):
        obs_ttx = ObservableData.extract_pr_ttx(obs_data)
        corrections = []
        for i in range(len(obs_data.timestamps)):
            corrections_i = {}
            date = obs_data.timestamps[i]
            sats = list(obs_data.observable_data.keys())
            timestamps = dict(zip(sats, [obs_data.timestamps[i] for _ in range(len(sats))]))
            relevant_ephs = ephemeris_all.get_relevant_ephemeris(timestamps, sats)
            for sat in sats:
                sat_eph = relevant_ephs[sat]
                gps_week = (timestamps[i] - constants.GPS_EPOCH).days // 7
                dtsv = sat_eph.get_dtsv_relativistic(obs_ttx, gps_week)
                sat_pos = sat_eph.get_position(obs_ttx[i][sat] - dtsv)
                physical_range = np.linalg.norm(np.array(sat_pos) - np.array(station_coord))
                pr_m = obs_data.observable_data[i][sat] + constants.C * dtsv
                corrections_i[sat] = physical_range - pr_m
            corrections.append(corrections_i)
        return corrections


class ObservationData:
    def __init__(self, station_coord, timestamps, sat_data, observ_types):
        self.timestamps = timestamps
        self.measurement_count = len(timestamps)
        self.sat_data = sat_data
        self.observ_types = observ_types
        self.station_coord = station_coord

    @classmethod
    def process_rinex(cls, filename: str):
        rinf = open(filename, "r")
        header = True
        station_coord = []
        observ_count = -1
        observ_left = -1
        observ_types = []
        while header:
            line = rinf.readline().rstrip()
            if not station_coord:
                if line.endswith("APPROX POSITION XYZ"):
                    station_coord = list(map(float, [line[i * 14:i * 14 + 14] for i in range(3)]))

            if line.endswith("TYPES OF OBSERV"):
                if observ_count == -1:
                    observ_count = int(line[0:6])
                    observ_left = observ_count
                observ_types += [line[(i + 1) * 6:(i + 2) * 6].strip() for i in range(min(9, observ_left))]
                observ_left -= min(9, observ_left)

            elif line.endswith("END OF HEADER"):
                header = False

        timestamps = []
        sat_data = []
        while line := rinf.readline():
            line = line.rstrip()
            timestamp: List[Union[int, float]] = list(map(int, [line[i * 3:i * 3 + 3] for i in range(5)]))
            timestamp.append(float(line[15:26]))
            status = int(line[26:29])
            if status != 0:
                continue
            date = datetime.datetime(timestamp[0] + 2000, timestamp[1], timestamp[2], timestamp[3], timestamp[4],
                                     int(timestamp[5]), int((timestamp[5] - (timestamp[5])) * 1000))
            timestamps.append(date)

            sat_count = int(line[29:32])
            sats_left = sat_count
            sats = [line[32 + 3 * i:35 + 3 * i] for i in range(min(sats_left, 12))]
            while sats_left > 12:
                line = rinf.readline().rstrip()
                sats_left -= 12
                sats += [line[32 + 3 * i:35 + 3 * i] for i in range(min(sats_left, 12))]
            store = dict()
            for sat in sats:
                store[sat] = dict()
            for sat_num in range(sat_count):
                sat = sats[sat_num]
                for observ_num in range(observ_count):
                    observ_type = observ_types[observ_num]
                    obsmod5 = observ_num % 5
                    if obsmod5 == 0:
                        line = rinf.readline()
                    observ = line[obsmod5 * 16:obsmod5 * 16 + 16].strip()
                    if observ == "":
                        continue
                    if observ_type.startswith("L"):
                        [l, h] = [observ[0:-1], observ[-1]]
                        store[sat][observ_type] = float(l)
                        store[sat]["H" + observ_type[1]] = int(h)
                    else:
                        store[sat][observ_type] = float(observ)
            sat_data.append(store)
        rinf.close()

        types_to_add = []
        for observ_type in observ_types:
            if observ_type.startswith("L"):
                types_to_add.append("H" + observ_type[1:])
        return cls(station_coord, timestamps, sat_data, observ_types)

    @staticmethod
    def get_observable_data(obs_data, observ="C1", only_gps=False):
        # returns observable data as a list of dicts from Satellite identifier to list of observations of the observable
        # also returns the list of relevant timestamps for the returned data
        observ_data = []
        timestamps_out = []
        sat_data = obs_data.sat_data
        sats_timestamps = dict()
        sats_indices = dict()
        index = 0
        for i in range(obs_data.measurement_count):
            obs = dict()
            date = obs_data.timestamps[i]

            for sat in sat_data[i].keys():
                if only_gps and not sat.startswith("G"):
                    continue
                if observ in sat_data[i][sat]:
                    sat_indices = sats_indices.get(sat, [])
                    sat_timestamps = sats_timestamps.get(sat, [])
                    sats_indices[sat] = sat_indices + [index]
                    sats_timestamps[sat] = sat_timestamps + [date]

                    data = sat_data[i][sat][observ]
                    obs[sat] = data

            if len(obs) > 0:
                index += 1
                observ_data.append(obs)
                timestamps_out.append(date)
        return ObservableData(timestamps_out,
                              sats_timestamps,
                              sats_indices,
                              observ_data,
                              observ,
                              (observ.startswith("C") or observ.startswith("P")))


class NavigationData:
    def __init__(self, sats_data, ion_data=None, delta_utc=None, leap_secs=None):
        self.ion_data: List[List[np.float64]] = ion_data
        self.sats_data: Dict[str, List[ephemeris_data.EphemerisData]] = sats_data
        self.delta_utc = delta_utc
        self.leap_secs = leap_secs

    @classmethod
    def process_gps_rinex(cls, filename: str):
        rinf = open(filename, "r")
        header = True
        ion_data = None
        delta_utc = None
        leap_secs = None
        while header:
            line = rinf.readline().rstrip()
            if line.endswith("ION ALPHA"):
                if ion_data is None:
                    ion_data = [[], []]
                ion_data[0] = list(map(parse_rinex_decimal, [line[i * 12 + 2:i * 12 + 14] for i in range(4)]))
            elif line.endswith("ION BETA"):
                if ion_data is None:
                    ion_data = [[], []]
                ion_data[1] = list(map(parse_rinex_decimal, [line[i * 12 + 2:i * 12 + 14] for i in range(4)]))
            elif line.endswith("DELTA-UTC: A0,A1,T,W"):
                delta_utc = list(map(parse_rinex_decimal, [line[3:22], line[22:41]])) + [int(line[41:50]), int(line[50:59])]
            elif line.endswith("LEAP SECONDS"):
                leap_secs = int(line[4:10])
            elif line.endswith("END OF HEADER"):
                header = False

        sats_data = dict()
        timestamps = dict()
        while line := rinf.readline():
            line = line.rstrip()
            sat_id = "G{:02d}".format(int(line[:2]))
            timestamp: List[Union[int, float]] = list(map(int, [line[i * 3 + 2:i * 3 + 5] for i in range(5)]))
            timestamp.append(float(line[17:22]))
            date = datetime.datetime(timestamp[0] + 2000, timestamp[1], timestamp[2], timestamp[3], timestamp[4],
                                     int(timestamp[5]), int((timestamp[5] - (timestamp[5])) * 1000))
            if not sat_id in sats_data:
                sats_data[sat_id] = []
                timestamps[sat_id] = []
            timestamps[sat_id].append(date)
            eph_data = []
            for i in range(29):
                if i % 4 == 3:
                    line = rinf.readline()
                data = parse_rinex_decimal(line[3 + 19 * ((i + 1) % 4):22 + 19 * ((i + 1) % 4)])
                eph_data.append(data)

            eph = ephemeris_data.EphemerisData(date, eph_data)
            sats_data[sat_id].append(eph)

        rinf.close()

        return cls(sats_data, ion_data=ion_data, delta_utc=delta_utc, leap_secs=leap_secs)

    def get_relevant_ephemeris(self, timestamps: Dict[str, datetime.datetime], sats: List[str]):
        relevant_eph = dict()
        for sat in sats:
            timestamp = timestamps[sat]
            if sat not in self.sats_data.keys():
                print("Data for satellite", sat, "not found in ephemeris", file=sys.stderr)
                continue
            sat_data = self.sats_data[sat]
            n = len(sat_data)
            min_dt = None
            min_i = None
            suitable = None
            for i in range(n):
                eph = sat_data[i]
                toc = eph.toc
                fit = eph.fit_interval
                dt = (timestamp - toc).total_seconds()
                if min_dt is None or min_dt > np.abs(dt):
                    min_dt = np.abs(dt)
                    min_i = i
                    if np.abs(dt) < fit:
                        suitable = eph
                if np.abs(dt) > fit:
                    continue
            if suitable is None:
                print("No suitable fit found for satellite", sat, ", closest fit at time", sat_data[min_i].toc, min_dt / 3600, "hours away.")
                continue
            relevant_eph[sat] = suitable
        return relevant_eph


def main():
    filename = input()
    observation_data = ObservationData.process_rinex(filename)
    observable_data = ObservationData.get_observable_data(observation_data, only_gps=True)
    c_one_ttx_nanos = ObservableData.extract_pr_ttx(observable_data)

    observ_timestamps = observable_data.timestamps
    with open(filename[0:-3] + "txt", "w") as f:
        with redirect_stdout(f):
            for i in range(len(c_one_ttx_nanos)):
                for sat in c_one_ttx_nanos[i].keys():
                    data = [0 for i in range(31)]
                    gps_time = observ_timestamps[i] - constants.GPS_EPOCH
                    print('Raw,', end="")
                    data[12] = 9
                    data[27] = 1 if sat[0] == 'G' else 5 if sat[0] == 'R' else 0
                    data[4] = int(-gps_time.total_seconds() * 10 ** 9)
                    data[10] = int(sat[1:])
                    data[13] = c_one_ttx_nanos[i][sat]
                    print(*data, sep=",")


if __name__ == '__main__':
    main()
