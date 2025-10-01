import pandas as pd
# from tqdm import tqdm
from pyproj import Geod
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.interpolate import griddata

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

def assign_filepath(satpass_swot, SWOT_files):

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
    df_swot = df_swot.sort_values('t_swot')

    df_swot.index = pd.DatetimeIndex(df_swot.t_swot)

    return df_swot

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

def associate_swot_passage_xy(x, y, date_range, mission='sw'):

    domain = [x - 1, x + 1, y - 1, y + 1]

    sw = satpass.NominalTrack(mission=mission)
    swot_passes_in_area = sw.select_in_area(domain)
    valid_tracks = np.unique(swot_passes_in_area.track)[(swot_passes_in_area.groupby('track').count().lat > 1).values]

    swot_passes_in_area = swot_passes_in_area[swot_passes_in_area['track'].isin(valid_tracks)]
    swot_passes = satpass.sat_pass(sw, date_range, domain)

    print('Identify and associate closest SWOT passes at position x, y [passage, cycle, and time] during the given period')

    valid_passes, dis2nadir = identify_overlapping_swot_passes([x, y], swot_passes_in_area, max_distance_from_nadir=65e3)

    swot_passes = swot_passes[swot_passes["track"].isin(valid_passes)]
    # swot_passes['dis2nadir'] = dis2nadir
    
    track2dist = dict(zip(valid_tracks, np.array(dis2nadir)/1e3))
    swot_passes['dis2nadir'] = swot_passes['track'].map(track2dist)

    swot_passes['x'] = x
    swot_passes['y'] = y

    return swot_passes.rename(columns = dict(time = 't_swot'))

def associate_swot_passage_xyt(x, y, t, mission='sw', n_passes=1, verbose = True, parallel = True):
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
        result.loc[:, f'dt_swot_{i+1}'] = swot_dts[i]
        result.loc[:, f'track_{i+1}'] = swot_passes[i]
        result.loc[:, f'cycle_{i+1}'] = swot_cycles[i]
        result.loc[:, f'dis2nadir_{i+1}'] = dis2nadir[i]

    if n_passes == 1:
        result = result.rename(
            columns={col: col[:-2] for col in result.columns if col.endswith('_1')}
        )

    return result

def read_and_preprocess_SWOT(ds, swot_var_names = dict(ssha = 'ssha_unfiltered', mdt = 'mdt'), kwargs_diagnosis = dict(compute_diagnosis = False, derivative = 'fit', n = 5, verbose = False)):
    kwargs_diagnosis = kwargs_diagnosis.copy()

    ssha = ds[swot_var_names['ssha']]
    ssh = ssha + ds[swot_var_names['mdt']]

    compute_diagnosis = kwargs_diagnosis.pop('compute_diagnosis', False)
    SwotDiag = kwargs_diagnosis.pop('SwotDiag', None)

    for k, var in swot_var_names.items():
        ds = ds.rename({var:k})

    ds = ds.assign(ssh = ssh)

    if compute_diagnosis:

        diag = SwotDiag.diagnosis.compute_ocean_diagnostics_from_eta(ssh, ssh.longitude, ssh.latitude, **kwargs_diagnosis)
        for k, d in diag.items():
            ds = ds.assign(**{k : (ssh.dims, d)})
    
    return ds

def interpolate_swot_passages(swot_table, interp_vars = ['ssha'], swot_var_names = dict(ssha = 'ssha_unfiltered', mdt = 'mdt'), parallel = True, n_jobs = -1,
                              kwargs_diagnosis = dict(compute_diagnosis = False, derivative = 'fit', n_stencil = 5, verbose = False, SwotDiag = None)):

    results = []
    print(f'Interpolating SWOT {interp_vars} field on each location')
    
    def process_subset(index, group):
        file = swot_table.filepath[(swot_table.track == index[0])&(swot_table.cycle == index[1])].values
        # print(file)
        if np.any(file):
            try:
                ds = xr.open_dataset(file[0])
                ind = np.any((ds.latitude < group.y.max() +1)&(ds.latitude > group.y.min() -1), axis = 1)
                ds = ds.sel(num_lines = ind)
                interp = interpolate_swot(ds, group.x, group.y, interp_vars=interp_vars, swot_var_names = swot_var_names, kwargs_diagnosis=kwargs_diagnosis)
                interp.index = group.index
                ds.close()
                del ds
            except Exception as e:
                # Handle the error
                print("An error occurred while processing the file for pass", index[0], 'cycle', index[1], ":", e, "Skipping the file.")
                interp = pd.DataFrame(np.ones((len(group.x), len(interp_vars))) * np.nan, index=group.index, columns=interp_vars)
 
        else:
            interp = pd.DataFrame(np.ones((len(group.x), len(interp_vars)))*np.nan, index = group.index, columns = interp_vars)

        return interp#pd.concat([group, interp], axis = 1)


    # Attempt to import tqdm for progress tracking
    try:
        from tqdm import tqdm
        iterator = tqdm(swot_table.groupby(['track', 'cycle']), total=len(swot_table.groupby(['track', 'cycle'])))

    except ImportError:
        print("TQDM library not available. Running without progress bar.")
        iterator = swot_table.groupby(['track', 'cycle'])

    if parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("joblib library not available. Running without serialization.")
            parallel = False

    if parallel:
        results = Parallel(n_jobs=n_jobs)(delayed(process_subset)(index, group) for index, group in iterator)
    
    else:
        for index, group in iterator:
            results.append(process_subset(index, group))
        
    return pd.concat([swot_table, pd.concat(results).sort_index()], axis = 1)#.rename(columns = dict(t_swot_1 = 't_swot', swot_dt_1 = 'swot_dt', swot_pass_1 = 'swot_pass', swot_cycle_1 = 'swot_cycle', dis2nadir_1 = 'dis2nadir'))


def interpolate_swot(ds, x, y, interp_vars = ['ssha'], swot_var_names = dict(ssha = 'ssha_unfiltered', mdt = 'mdt'), 
                     kwargs_diagnosis = dict(compute_diagnosis = False, derivative = 'fit', n_stencil = 5, verbose = False)):

    var = np.hstack([interp_vars])

    interp = {}

    ds = ds.where((ds.latitude<y.max() + 0.5)&(ds.latitude>y.min() - 0.5))
    ds = read_and_preprocess_SWOT(ds, swot_var_names = swot_var_names, kwargs_diagnosis = kwargs_diagnosis)

    vars = {v : ds[v] for v in var}
            
    #     vars['zeta'] = ds.zeta

    xy_from = np.array([np.ravel(ds.longitude), np.ravel(ds.latitude)]).T
    xy_to = np.array([x, y]).T
    for k, v in vars.items():    
        v = np.ravel(v)    
        interp[k] = griddata(xy_from, v, xy_to)
    interp = pd.DataFrame(interp)

    ds.close()
    del ds

    return interp