import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr 
from scipy.ndimage import morphology 
from scipy.ndimage import maximum_filter
import math
import funpy.eddy_track_id as eddy_track
import funpy.model_utils as mod_utils
from joblib import Parallel, delayed
import multiprocessing
import timeit
import os
from scipy.signal import butter, filtfilt

dx = 0.05; dy = 0.1; dt = 0.2
tstart = 1500
max_iters = 150
num_cores = multiprocessing.cpu_count()

fdir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/lab_netcdfs')
#fdir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/lab_netcdfs')
#fdir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/lab_netcdfs')

u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
v_psi = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')['v_psi']

x = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['y']
x = x.values
y = y.values

t = np.arange(0, len(u_psi)*dt, dt)

filt_t = 2*4

def butterworth_filter(data, cutoff_freq, fs, axis=0, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=axis)
    return filtered_data

fs = 5 
cutoff_freq = 1 / filt_t 

u_psi = butterworth_filter(u_psi, cutoff_freq, fs, axis=0)
v_psi = butterworth_filter(v_psi, cutoff_freq, fs, axis=0)

vort = mod_utils.curl(u_psi, v_psi, dx, dy) 
vort_abs = np.abs(vort)

Nx = vort.shape[-1]
Ny = vort.shape[1]

start = 0
for t in range(start, (max_iters*num_cores), num_cores):
    trun = tstart + t 
    inputs = np.arange(trun, trun+num_cores)
    print(inputs)
    starttime = timeit.default_timer()

    eddy_track_map_all = np.zeros(vort[:len(inputs),:,:].shape, int)
    leng_all = []
    spin_all = []
    xc_all = []
    yc_all = []    

    print("Iteration# ", (t-1)/num_cores + 1, "t = ", trun)

    results = Parallel(n_jobs=num_cores)(delayed(eddy_track.eddy_id_singletime)(x, y, vort[i,:,:], szthresh=0.4, delt=0.05) for i in inputs)
    elapsed = timeit.default_timer() - starttime
    print('Run time = ',elapsed)

    for i in range(num_cores):
        #eddy_track_map_all[trun+i-tstart, :, :] = results[i][0]
        eddy_track_map_all[i, :, :] = results[i][0]
        leng_all.append(results[i][1])
        spin_all.append(results[i][2])
        xc_all.append(results[i][3])
        yc_all.append(results[i][4])

    time = np.linspace(inputs[0]*dt, inputs[-1]*dt, len(eddy_track_map_all))
    dim = ["time", "y", "x"]
    coords = [time, y, x]
    dat = xr.DataArray(eddy_track_map_all, coords, dims=dim, name='eddy_id') 
    dat.to_netcdf(os.path.join(fdir, 'eddy_id', 'eddy_track_map_all_averaged_4tp_sz04_%04d_%04d.nc' % (t, t+num_cores)))

    f = open(os.path.join(fdir, 'eddy_id', 'eddy_stats_all_averaged_4tp_sz04_%d_%d.txt' % (t, t+num_cores)), 'w')
    for i in range(len(leng_all)):
        for j in range(len(leng_all[i])):
            f.write('%f, ' % i)
            f.write('%f, ' % leng_all[i][j])
            f.write('%f, ' % spin_all[i][j])
            f.write('%f, ' % xc_all[i][j])
            f.write('%f\n' % yc_all[i][j])
    f.close()



