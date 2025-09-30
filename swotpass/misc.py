import pandas as pd
# from tqdm import tqdm
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
# from scipy.interpolate import griddata

import swotpass.satpass as satpass
import numpy as np

geod = Geod(ellps="WGS84") 

def assign_subcycles(df, threshold = pd.Timedelta(days = 2), remove_incomplete = True):

    df = df.sort_values(by=['direction', 'time'])

    df['time_diff'] = df.groupby('direction')['time'].diff()

    def assign_subcycle(group):
        subcycle_numbers = (group['time_diff'] > threshold).cumsum()
        if group['direction'].iloc[0] == 'desc':
            subcycle_numbers += subcycle_numbers.max() + 1
        return subcycle_numbers

    # Apply the custom function within each direction group
    df['subcycle'] = df.groupby('direction', group_keys=False).apply(assign_subcycle)

    # Drop the temporary time_diff column if needed
    df = df.drop(columns=['time_diff'])
#     subcycle[np.isnan(subcycle)] = 0.
    # df.loc[:, 'subcycle'] = df.loc[:, 'subcycle']%2

    if remove_incomplete:
        df = remove_incomplete_subcycles
    df.index = df.time

    df.cycle = df.cycle.astype(float)
    return df

def remove_incomplete_subcycles(df):
        
    counts = df.groupby(['direction', 'cycle', 'subcycle']).size()
    max_counts = counts.groupby(['direction', 'subcycle']).transform('max')
    # Identify the combinations to remove
    combinations_to_remove = counts[counts != max_counts].index
    
    # Remove the identified combinations
    df = df.set_index(['direction', 'cycle', 'subcycle']).drop(combinations_to_remove).reset_index()
    return df

def construct_index_swot(satpass_swot, SWOT_files):

    filepath_swot = []
    
    for file in SWOT_files:
        t0, t1 = pd.Timestamp(file.split('SWOT_')[-1].split('_')[6]), pd.Timestamp(file.split('SWOT_')[-1].split('_')[7])
        t = pd.DatetimeIndex([t0, t1]).mean()
        cycle, track = int(file.split('SWOT_')[-1].split('_')[4]), int(file.split('SWOT_')[-1].split('_')[5])
    
        df = pd.DataFrame(np.array([cycle, track, file]).T, index = ['cycle', 'track', 'filepath']).T
        df.index = [t]
        filepath_swot.append(df)
    
    filepath_swot = pd.concat(filepath_swot).sort_index()
    filepath_swot.cycle = filepath_swot.cycle.astype(int)
    filepath_swot.track = filepath_swot.track.astype(int)
    filepath_swot.index = filepath_swot.index.round('60S')

    df_swot = pd.merge(satpass_swot, filepath_swot, how = 'left')
    df_swot = df_swot.sort_values('time')
    # df_swot = remove_incomplete_subcycles(df_swot)
    index, _ = pd.factorize(df_swot['direction'].astype(str) + df_swot['cycle'].astype(str) + df_swot['subcycle'].astype(str))
    df_swot.index = index
    df_swot = df_swot.dropna()
    return df_swot

def associate_swot_passage(x, y, t, mission = 'sw', verbose = True):


    t = pd.DatetimeIndex(t)

    date_range = [t.min() - pd.Timedelta(days = 3), t.max() + pd.Timedelta(days = 3)]
    domain = [x.min() -1,x.max()+1,y.min() -1,y.max()+1]

    sw = satpass.NominalTrack(mission=mission)
    swot_passes_in_area = sw.select_in_area(domain)
    
    valid_tracks = np.unique(swot_passes_in_area.track)[(swot_passes_in_area.groupby('track').count().lat > 1).values]
    
    swot_passes_in_area = swot_passes_in_area[swot_passes_in_area['track'].isin(valid_tracks)]
    swot_passes = satpass.sat_pass(sw, date_range, domain)

    # print(len(np.unique(swot_passes.track)))
    # print(len(np.unique(swot_passes_in_area.track)))

    swot_pass, swot_cycle, swot_time = [], [], []

    # Attempt to import tqdm for progress tracking
    try:
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(zip(x, y, t), total=len(x))
        else:
            iterator = zip(x, y, t)
    except ImportError:
        print("TQDM library not available. Running without progress bar.")
        verbose = False
        iterator = zip(x, y, t)

    # Wrap the loop with tqdm to display the progress bar
    print('Identify and associate closest SWOT passes [passage, cycle and time] to each x, y, t')
    for xx, yy, tt in iterator:
        valid_ids = identify_close_swot_passes((xx, yy), swot_passes_in_area, max_distance_from_nadir=60e3)
        cl_pass, cl_cycle, cl_time = extract_closest_swot_pass(swot_passes, tt, valid_ids)
        
        swot_pass.append(cl_pass)
        swot_cycle.append(cl_cycle)
        swot_time.append(cl_time)

    swot_time = pd.DatetimeIndex(swot_time)
    swot_dt = (t - swot_time).total_seconds().values.astype(float)/86400

    result = np.array([x, y]).T
    result = pd.DataFrame(result, columns = ['x', 'y'])
    
    result.loc[:, 't'] = t
    result.loc[:, 't_swot'] = swot_time
    
    result.loc[:, 'swot_dt'] = swot_dt
    result.loc[:, 'swot_pass'] = swot_pass
    result.loc[:, 'swot_cycle'] = swot_cycle

    return result


def identify_close_swot_passes(xy_point, tracks_swot, max_distance_from_nadir = 60e3):
    point = Point(xy_point[0],xy_point[1])
    valid_ids = []
    for ids, track in tracks_swot.groupby('track'):
        line = LineString(np.array([track.lon.values, track.lat.values]).T)
        nearest_point = nearest_points(line, point)
        distance = geod.geometry_length(LineString(nearest_point))  
        if distance <= max_distance_from_nadir:
            valid_ids.append(ids)
    return valid_ids

def identify_overlapping_swot_passes(xy_point, tracks_swot, max_distance_from_nadir = 60e3):
    point = Point(xy_point[0],xy_point[1])
    valid_ids = []
    dis2nadir = []
    for ids, track in tracks_swot.groupby('track'):
        line = LineString(np.array([track.lon.values, track.lat.values]).T)
        nearest_point = nearest_points(line, point)
        distance = geod.geometry_length(LineString(nearest_point))  
        if distance <= max_distance_from_nadir:
            valid_ids.append(ids)
            dis2nadir.append(distance)
    return valid_ids, dis2nadir

def extract_closest_swot_pass(satpass_swot, t, valid_passes):
    nearest_passes = satpass_swot[satpass_swot['track'].isin(valid_passes)]
    shorter_pass = nearest_passes.loc[(np.abs(t - nearest_passes.time)).idxmin()]
    
    return shorter_pass.track, int(float(shorter_pass.cycle)), shorter_pass.time
    
def associate_swot_passage_parallel(x, y, t, mission='sw', n_passes=1, verbose = True, parallel = True):
    from joblib import Parallel, delayed


    x, y, t = np.hstack([x]), np.hstack([y]), np.hstack([t])

    #### Mooring situation
    if (len(x) == 1)&(len(t)>1):
        x, y = np.repeat(x, len(t)), np.repeat(y, len(t))
    
    t = pd.DatetimeIndex(t)

    date_range = [t.min() - pd.Timedelta(days=10), t.max() + pd.Timedelta(days=10)]
    domain = [x.min() - 4, x.max() + 4, y.min() - 4, y.max() + 4]

    sw = satpass.NominalTrack(mission=mission)
    swot_passes_in_area = sw.select_in_area(domain)

    valid_tracks = np.unique(swot_passes_in_area.track)[(swot_passes_in_area.groupby('track').count().lat > 1).values]
    
    swot_passes_in_area = swot_passes_in_area[swot_passes_in_area['track'].isin(valid_tracks)]
    swot_passes = satpass.sat_pass(sw, date_range, domain)

    swot_pass, swot_cycle, swot_time = [], [], []

    # Attempt to import tqdm for progress tracking
    try:
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(zip(x, y, t), total=len(x))
        else:
            iterator = zip(x, y, t)
    except ImportError:
        print("TQDM library not available. Running without progress bar.")
        verbose = False
        iterator = zip(x, y, t)

    print('Identify and associate closest SWOT passes [passage, cycle, and time] to each x, y, t')

    def process_subset(xx, yy, tt):
        valid_passes, dis2nadir = identify_overlapping_swot_passes((xx, yy), swot_passes_in_area, max_distance_from_nadir=65e3)
        closest_passes = []
        for i in range(n_passes):
            if len(valid_passes) > i:
                closest_pass, closest_cycle, closest_time  = extract_closest_swot_pass(swot_passes, tt, valid_passes)

                ind = np.where(valid_passes == closest_pass)[0][0]
                _dis2nadir = dis2nadir[ind]

                closest_passes.append((closest_pass, closest_cycle, closest_time, _dis2nadir/1000))

                valid_passes.remove(closest_pass)
                dis2nadir.remove(_dis2nadir)
            else:
                closest_passes.append((np.nan, np.nan, np.nan, np.nan))
        return np.vstack(closest_passes).T

    if parallel:
        results = Parallel(n_jobs=-1)(delayed(process_subset)(xx, yy, tt) for xx, yy, tt in iterator)
    
    else:
        results = []
        for xx, yy, tt in tqdm(zip(x, y, t), total=len(x)):
            results.append(process_subset(xx, yy, tt))
            

    result = np.array([x, y]).T
    result = pd.DataFrame(result, columns = ['x', 'y'])
    result.loc[:, 't'] = pd.DatetimeIndex(t)
    

    swot_passes = [[] for _ in range(n_passes)]
    swot_cycles = [[] for _ in range(n_passes)]
    swot_times = [[] for _ in range(n_passes)]
    dis2nadir = [[] for _ in range(n_passes)]

    for i in range(n_passes):
        swot_passes[i] = np.array([result[0][i] for result in results])
        swot_cycles[i] = np.array([result[1][i] for result in results])
        swot_times[i] = np.array([result[2][i] for result in results])
        dis2nadir[i] = np.array([result[3][i] for result in results])

    swot_times = [pd.DatetimeIndex(swot_time) for swot_time in swot_times]
    swot_dts = [(t - swot_time).total_seconds().values.astype(float) / 86400 for swot_time in swot_times]

    for i in range(n_passes):
        result.loc[:, f't_swot_{i+1}'] = swot_times[i]
        result.loc[:, f'swot_dt_{i+1}'] = swot_dts[i]
        result.loc[:, f'swot_pass_{i+1}'] = swot_passes[i]
        result.loc[:, f'swot_cycle_{i+1}'] = swot_cycles[i]
        result.loc[:, f'dis2nadir_{i+1}'] = dis2nadir[i]


    if n_passes == 1:
        result = result.rename(
            columns={col: col[:-2] for col in result.columns if col.endswith('_1')}
        )


    return result