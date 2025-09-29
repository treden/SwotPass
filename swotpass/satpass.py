#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
This module contains tools to compute satellite past and future passes on a given area.
Originally developed by C. Chupin under the PyGoat toolbox (CC, YTT, VB, LT)
01/2024 (YTT): Adaptation to SWOT altimetry and extraction from the PyGoat toolbox as an independent module

.. Warning:: This code is still under development, and significant changes may take place in future versions.

.. todo::
    - Address the issue with Sentinel tracks!!
    - Add CFOSAt
    - Optimize
    - Check time time_unit
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging
import os

# Define the resources path
resources = f'{os.path.dirname(os.path.realpath(__file__))}/../resources'
metadata_file = f'{resources}/altimeters.meta'

# Define data types for metadata CSV columns
dtypes = {'short_name': str, 'full_name': str, 'start_date_cycle_min': str, 'end_of_mission': str,
          'repeatability': float, 'cycle_min': pd.Int64Dtype(), 'cycle_max': pd.Int64Dtype(),
          'nominal_track_filename': str}
parse_dates = ['start_date_cycle_min', 'end_of_mission']

# Read metadata CSV
meta = pd.read_csv(metadata_file, sep=',', comment='#', skipinitialspace=True, dtype=dtypes,
                   parse_dates=parse_dates, index_col=['short_name'])

# Define logger
logger = logging.getLogger(__name__)

# Define NominalTrack class
class NominalTrack(object):
    """
    Nominal Track object

    :param mission: satellite mission name (default is 'en', could be 'sw', 's3a', 'j2', ...)
    :type mission: str

    >>> j3 = NominalTrack(mission='j3')
    """

    def __init__(self, mission='en', **kwargs):
        # Set paths and attributes
        self.source_file = resources / Path(f"nominal_tracks/{meta.loc[mission].nominal_track_filename}")
        self.mission = mission
        self.meta = meta.loc[mission]

        # Read nominal track data from CSV
        if self.mission in ['s3a', 's3b', 'cfo']:
            self.nominal = pd.read_csv(self.source_file, header=None,
                                       names=['track', 'lon', 'lat', 'time_c06'])  
        else:
            self.nominal = pd.read_csv(self.source_file, header=None, names=['track', 'lon', 'lat'])

    def select_in_area(self, area):
        """
        For a given area, return a DataFrame with tracks number and coordinates

        :param area: given area (format: [lon_min, lon_max, lat_min, lat_max] in degree in [-180°:180°]/[-90°:90°])
        :type area: list
        :returns: DataFrame with track number, lon, and lat
        """

        area = list(area)
        if area:
            mask = (self.nominal.lon >= area[0]) & (self.nominal.lon <= area[1]) & \
                   (self.nominal.lat >= area[2]) & (self.nominal.lat <= area[3])
        else:
            mask = [True] * len(self.nominal.index)

        return self.nominal[mask]

    def create_tracks_dict(self, numbers, nominal_cut):
        """
        Create a dict with satellite_tracks_number & list of coordinates

        :param numbers: list of satellite tracks numbers
        :param nominal_cut: DataFrame with points on the given area

        :returns: dict {track_number: list of coordinates}
        """

        track_dict = {}

        for num in numbers:
            _dt = pd.Timedelta(self.meta["repeatability"], unit='D') / len(set(self.nominal.track)) / \
                  len(self.nominal.lon[self.nominal.track == num])

            _points = [(lon, lat, t) for lon, lat, t in zip(nominal_cut.lon[nominal_cut.track == num],
                                                            nominal_cut.lat[nominal_cut.track == num],
                                                            nominal_cut.lon[
                                                                nominal_cut.track == num].index * _dt)]
            track_dict[num] = _points

        return track_dict

    def extract(self, area, ocean=True):
        """
        Extract the tracks in a given area

        :param area: a area in the [lon_min, lon_max, lat_min, lat_max] in decimal degree [-180°,180°,-90.,90.])
        :type area: list
        :param ocean: set to True to select only the ocean tracks (default: True)
        :type ocean: bool
        :return: a dict with the track number as key and its list of coordinates in the area
        """
        selection = self.select_in_area(area=area)
        tracks_num = list(set(selection['track']))
        track_dict = self.create_tracks_dict(numbers=tracks_num, nominal_cut=selection)

        if ocean:
            return track_dict
        else:
            return track_dict

    def get_tracks(self, area):
        """
        For a given area, select tracks numbers and create dict with associated list of coordinates

        :param area: given area (format: [lon_min, lon_max, lat_min, lat_max] in degree in [-180°:180°]/[-90°:90°])

        :returns: dict {track_number: list of coordinates}
        """
        nominal_masked_cut = self.select_in_area(area=area)
        tracks_num = list(set(nominal_masked_cut['track']))
        _points = self.create_tracks_dict(numbers=tracks_num, nominal_cut=nominal_masked_cut)

        return tracks_num, _points

# Ephemeris computation
def ephemeris(nt, date, **kwargs):
    """
    Compute complete ephemeris depending on the NominalTrack object.

    :param nt: NominalTrack object
    :param date: list of 1/2 dates (e.g., ['2020-02-30', '2020-06-21'])
    :rtype: pandas.DatetimeIndex
    """
    date = pd.DatetimeIndex(date)

    if isinstance(nt.meta["end_of_mission"], pd.Timestamp):
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           end=nt.meta["end_of_mission"]).to_frame()
    elif isinstance(nt.meta["cycle_max"], np.int64):
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           periods=(nt.meta["cycle_max"] - nt.meta["cycle_min"] + 1)).to_frame()
    else:
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           end=date[-1]).to_frame()

    if len(date) == 1:
        selected_ephemeris = complete_ephemeris[date[0].strftime('%Y-%m'):date[0].strftime('%Y-%m')].index.tz_localize("UTC")
    elif len(date) == 2:
        selected_ephemeris = complete_ephemeris[(date[0] - pd.Timedelta(nt.meta["repeatability"], unit='D')):date[1]].index
    else:
        print('--- Error date interval format => return complete ephemeris')
        selected_ephemeris = complete_ephemeris

    return selected_ephemeris

# SatPass main launcher
# SatPass main launcher
def sat_pass(mission, date, area=None, **kwargs):
    """
    Function to compute satellite ephemeris for a given mission and date interval.

    :param mission: list of missions to consider
    :param date: list of dates (e.g., [datetime(2020, 3, 2), datetime(2020, 6, 2)])
    :param area: a 4-number list [xmin, xmax, ymin, ymax]
    :return: DataFrame of the tracks and time in the area
    """

    if isinstance(mission, str):
        nt = NominalTrack(mission=mission)
    elif isinstance(mission, NominalTrack):
        nt = mission

    dct_eph = {}

    eph = ephemeris(nt, date)

    DF = []
    
    if area:
        n_tracks, dct = nt.get_tracks(area=area)

        for n, trace in dct.items():
            _x, _y, _t = np.array(trace).T
            if len(_x) > 1:
                if np.diff(_y)[0] > 0:
                    direction = 'asc'
                else:
                    direction = 'des'

                _i = int(len(_x) / 2)
                _x, _y, _t = _x[_i], _y[_i], (_t[_i] + eph).round('60S')

                cycle = (_t - nt.meta['start_date_cycle_min']).total_seconds() / 86400 // nt.meta[
                    'repeatability'] + 1

                df = pd.DataFrame(np.array([[n] * len(eph), [direction] * len(eph), cycle]).T, index=_t,
                                columns=['track', 'direction', 'cycle'])

                DF.append(df)
    else:
        n_tracks, dct, dct_masked = nt.get_tracks(area=None)

    DF = pd.concat(DF).sort_index()
    DF.index = pd.DatetimeIndex(DF.index)
    DF = DF[date[0]:date[-1]]
    DF.track = DF.track.astype(int)
    DF['time'] = DF.index

    DF.cycle = DF.cycle.astype(float) 

    return DF

if __name__ == '__main__':
    print('SAT PASS')