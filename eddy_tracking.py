import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr 
from scipy.ndimage import morphology 
from scipy.ndimage import maximum_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import math
import shapely.geometry
import shapely.ops
import pickle
import time 
import os
from joblib import Parallel, delayed
import multiprocessing
import glob
import cmocean.cm as cmo 
import funpy.model_utils as mod_utils
from scipy.signal import butter, filtfilt
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

dx = 0.05; dy = 0.1; dt = 0.2; start = 1500

fdir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/lab_netcdfs')

eddy_stats = pd.read_csv(os.path.join(fdir, 'eddy_id', 'eddy_stats.csv'))

eddy_ids = xr.open_mfdataset(os.path.join(fdir, 'eddy_id', 'eddy_track_map_all_averaged_4tp_*.nc'), combine='nested', concat_dim='time')['eddy_id']
u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
v_psi = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')['v_psi']
u_psi = u_psi[start:,:,:]
v_psi = v_psi[start:,:,:]
x = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['y']
x = x.values
y = y.values

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

eddy_time = eddy_stats['time']
leng = eddy_stats['length']
circ = eddy_stats['circulation']
size = eddy_stats['size']
xc = eddy_stats['x']
yc = eddy_stats['y']


def find_tind(eddy_time, t):
    return np.where(eddy_time == t)[0]

def get_next_eddy_char(xt, yt, eddy_circ, tind, xc, yc, size, circ, dist_thresh):
    # get x, y, and length information for eddies at next time step
    xts = xc[tind].values
    yts = yc[tind].values

    # compute distances away from current eddy
    dists = np.sqrt((xts - xt) ** 2 + (yts - yt) ** 2)     

    # find indices of closest eddies
    dinds = np.argwhere(dists<dist_thresh)[:,0]

    # new eddy characteristics 
    new_eddy_size = size[tind[dinds]].values
    new_eddy_circ = circ[tind[dinds]].values 

    # subset based on same circulaiton sign 
    sind = np.where(np.sign(new_eddy_circ)==np.sign(eddy_circ))[0] 

    # subset eddy characteristics 
    new_eddy_size = new_eddy_size[sind]
    new_eddy_circ = new_eddy_circ[sind] 
    return sind, dinds, dists, new_eddy_size, new_eddy_circ, xts, yts

def find_new_single_eddy(eddy_size, new_eddy_size, eddy_circ, new_eddy_circ, dists, dinds, sind):
    if len(sind) > 1:
        dsize = np.abs(eddy_size - new_eddy_size)/eddy_size 
        dcirc = np.abs(np.abs(eddy_circ) - np.abs(new_eddy_circ))/eddy_circ 
        
        cost = dists[dinds[sind]] + dsize + dcirc
        minind = np.argmin(cost)
        dind = dinds[minind]
        dsize = dsize[minind]
        dcirc = dcirc[minind]
        new_eddy_size = new_eddy_size[minind]
        new_eddy_circ = new_eddy_circ[minind]
        eddy_id = dind + 1 
        sind = sind[minind]
    elif len(sind) == 1:
        dind = dinds[sind] 
        dsize = np.abs(eddy_size - new_eddy_size)/eddy_size 
        dcirc = np.abs(np.abs(eddy_circ) - np.abs(new_eddy_circ))/eddy_circ 
        eddy_id = dind + 1
    else: 
        dind = np.argmin(dists) 
        dsize = np.nan 
        dcirc = np.nan  
        eddy_id = np.nan           
    return dsize, dcirc, eddy_id, new_eddy_circ, new_eddy_size, sind 

def check_eddy(xts, yts, dists, dinds, sind, dsize, dcirc, new_eddy_size, new_eddy_circ, dist_thresh, size_thresh, circ_thresh, tind, eddy_track):
    # check that eddy is close enough and sufficiently similar in size and strength 
    # print(dists[dinds[sind]], dists[dinds[sind]] < dist_thresh, dsize, dsize < size_thresh, dcirc, dcirc < circ_thresh, tind[dinds][sind], eddy_track[tind[dinds][sind]] == False)
    if dists[dinds[sind]] < dist_thresh and dsize < size_thresh and dcirc < circ_thresh and eddy_track[tind[dinds][sind]] == False:
        # new x & y positions
        xt = xts[dinds[sind]]
        yt = yts[dinds[sind]]
        eddy_size = new_eddy_size
        eddy_circ = new_eddy_circ
        eddy_track[tind[dinds][sind]] = True 
        
    else:
        xt = np.nan
        yt = np.nan 
        eddy_size = np.nan 
        eddy_circ = np.nan 
    return xt, yt, eddy_size, eddy_circ, eddy_track

def find_eddy_track(eddy_id, tstart, eddy_time, size, circ, xc, yc, eddy_track, dist_thresh=0.1, size_thresh=0.3, circ_thresh=1.0):
    # set index of eddies to start tracking at start time
    tind = find_tind(eddy_time, tstart)

    # reset number of eddies at start time for keeping track of path ind
    N = len(tind)

    # get x, y information for eddies at start time
    xts = xc[tind].values
    yts = yc[tind].values

    # get x, y information for chosen eddy at start time
    xt = xts[eddy_id]
    yt = yts[eddy_id]
    
    if eddy_track[tind[eddy_id]] == False:     
        # mark eddy as tracked
        eddy_track[tind[eddy_id]] = True 

        # get eddy characteristics 
        eddy_size = size[tind[eddy_id]]
        eddy_circ = circ[tind[eddy_id]]

        # define max time of eddy tracks
        T = int(np.max(eddy_time))

        # define arrays for eddy track info
        eddy_track_x = np.zeros(T)*np.nan
        eddy_track_y = np.zeros(T)*np.nan
        eddy_track_size = np.zeros(T)*np.nan
        eddy_track_circ = np.zeros(T)*np.nan
        eddy_track_eddy_id = np.zeros(T)*np.nan

        # fill in nans for tracks that start after t = 0
        if tstart > 0:
            eddy_track_x[:tstart] = np.nan
            eddy_track_y[:tstart] = np.nan
            eddy_track_size[:tstart] = np.nan
            eddy_track_circ[:tstart] = np.nan
            eddy_track_eddy_id[:tstart] = np.nan

        # set current eddy characteristics 
        eddy_track_x[tstart] = xt
        eddy_track_y[tstart] = yt
        eddy_track_size[tstart] = eddy_size
        eddy_track_circ[tstart] = eddy_circ
        eddy_track_eddy_id[tstart] = eddy_id + 1

        # loop through time and track eddies
        tcount = tstart
        for t in range(tstart, T):
            # check that eddy track is still valid
            if np.isfinite(xt) and np.isfinite(yt) and tcount+1<T:
                # find indices of all eddies at next time step
                tind = find_tind(eddy_time, int(tcount+1))
                sind, dinds, dists, new_eddy_size, new_eddy_circ, xts, yts = get_next_eddy_char(xt, yt, eddy_circ, tind, xc, yc, size, circ, dist_thresh)
                dsize, dcirc, eddy_id, new_eddy_circ, new_eddy_size, sind = find_new_single_eddy(eddy_size, new_eddy_size, eddy_circ, new_eddy_circ, dists, dinds, sind) 
                xt, yt, eddy_size, eddy_circ, eddy_track = check_eddy(xts, yts, dists, dinds, sind, dsize, dcirc, new_eddy_size, new_eddy_circ, dist_thresh, size_thresh, circ_thresh, tind, eddy_track)

                # put eddy info into tracks
                eddy_track_x[tcount+1] = xt
                eddy_track_y[tcount+1] = yt
                eddy_track_size[tcount+1] = eddy_size
                eddy_track_circ[tcount+1] = eddy_circ
                eddy_track_eddy_id[tcount+1] = eddy_id

                if np.isfinite(xt) == False: 
                    ## check track continuing with a gap
                    t_gap = 1
                    maxgap = 10
                    while (t_gap < maxgap) and (tcount+1+t_gap<T):
                        tind = find_tind(eddy_time, int(tcount+1+t_gap))
                        sind, dinds, dists, new_eddy_size, new_eddy_circ, xts, yts = get_next_eddy_char(eddy_track_x[tcount], eddy_track_y[tcount], circ[tcount], tind, xc, yc, size, circ, dist_thresh*2)
                        dsize, dcirc, eddy_id, new_eddy_circ, new_eddy_size, sind = find_new_single_eddy(size[tcount], new_eddy_size, circ[tcount], new_eddy_circ, dists, dinds, sind)
                        xt, yt, eddy_size, eddy_circ, eddy_track = check_eddy(xts, yts, dists, dinds, sind, dsize, dcirc, new_eddy_size, new_eddy_circ, dist_thresh*2, size_thresh, circ_thresh, tind, eddy_track)
                        if np.isfinite(xt) == True and eddy_track[tind[sind]] == False:
                            tcount = tcount + t_gap
                            t_gap = maxgap                         
                            eddy_track_x[tcount+1] = xt
                            eddy_track_y[tcount+1] = yt
                            eddy_track_size[tcount+1] = eddy_size
                            eddy_track_circ[tcount+1] = eddy_circ
                            eddy_track_eddy_id[tcount+1] = eddy_id  
                        else: 
                            t_gap += 1    

                tcount += 1 
            elif tcount >= T:
                break
            else:
                break 
    else:
        # define max time of eddy tracks
        T = int(np.max(eddy_time))

        # define arrays for eddy track info
        eddy_track_x = np.zeros(T)*np.nan
        eddy_track_y = np.zeros(T)*np.nan
        eddy_track_size = np.zeros(T)*np.nan
        eddy_track_circ = np.zeros(T)*np.nan
        eddy_track_eddy_id = np.zeros(T)*np.nan
    return eddy_track_x, eddy_track_y, eddy_track_size, eddy_track_circ, eddy_track_eddy_id, eddy_track

dist_thresh = 0.1; size_thresh = 1.0; circ_thresh = 1.0

eddies_per_time = np.bincount(eddy_time.astype(int))
tstart = 0
Neddies = eddies_per_time[tstart]
eddy_tracks_x = []
eddy_tracks_y = []
eddy_sizes = []
eddy_circs = []
eddy_ids_track = [] 
track_len = []
eddy_track = np.full(len(eddy_time), dtype=bool, fill_value=False)

T = len(eddy_ids)

t0 = time.time()
for tstart in range(T-1):
    Neddies = eddies_per_time[tstart]
        
    for e in range(Neddies):
        #if eddy_track[int(np.sum(eddies_per_time[:tstart]))+ e]==False:
        eddy_count = int(np.sum(eddies_per_time[:tstart]))
        track_x, track_y, sizes, circs, ids, eddy_track = find_eddy_track(e, tstart, eddy_time, \
                                                        size, circ, xc, yc, eddy_track, \
                                                        dist_thresh = dist_thresh, \
                                                        size_thresh = size_thresh, \
                                                        circ_thresh = circ_thresh) 
        tlen = len(np.where(np.isfinite(track_x)==True)[0])
        if tlen > 0:
            eddy_tracks_x.extend(track_x)
            eddy_tracks_y.extend(track_y)
            eddy_sizes.extend(sizes)
            eddy_circs.extend(circs)
            eddy_ids_track.extend(ids)
            track_len.append(tlen)
                    
t1 = time.time()

print(t1-t0)

T = 5999
eddy_tracks_x = np.asarray(eddy_tracks_x)
eddy_tracks_y = np.asarray(eddy_tracks_y)
eddy_sizes = np.asarray(eddy_sizes)
eddy_circs = np.asarray(eddy_circs)
eddy_ids_track = np.asarray(ids)

eddy_tracks_x = np.reshape(eddy_tracks_x, (-1, T))
eddy_tracks_y = np.reshape(eddy_tracks_y, (-1, T))
eddy_sizes = np.reshape(eddy_sizes, (-1, T))
eddy_circs = np.reshape(eddy_circs, (-1, T))
eddy_ids_track = np.reshape(ids, (-1, T)) 



with open(os.path.join(fdir, 'eddy_track', 'eddy_tracks_x_tracked_dist03_size1_circ1.npy'), 'wb') as f:
    np.save(f, eddy_tracks_x)

with open(os.path.join(fdir, 'eddy_track', 'eddy_tracks_y_tracked_dist03_size1_circ1.npy'), 'wb') as f:
    np.save(f, eddy_tracks_y)

with open(os.path.join(fdir, 'eddy_track', 'eddy_sizes_tracked_dist03_size1_circ1.npy'), 'wb') as f:
    np.save(f, eddy_sizes)

with open(os.path.join(fdir, 'eddy_track', 'eddy_circs_tracked_dist03_size1_circ1.npy'), 'wb') as f:
    np.save(f, eddy_circs)

with open(os.path.join(fdir, 'eddy_track', 'eddy_ids_track_tracked_dist03_size1_circ1.npy'), 'wb') as f:
    np.save(f, eddy_ids_track)        

### 


with open(os.path.join(fdir, 'eddy_track', 'eddy_tracks_x_tracked_dist03_size1_circ1.npy'), 'rb') as f:
    eddy_tracks_x = np.load(f)

with open(os.path.join(fdir, 'eddy_track', 'eddy_tracks_y_tracked_dist03_size1_circ1.npy'), 'rb') as f:
    eddy_tracks_y = np.load(f)    

with open(os.path.join(fdir, 'eddy_track', 'eddy_sizes_tracked_dist03_size1_circ1.npy'), 'rb') as f:
    eddy_sizes = np.load(f)

with open(os.path.join(fdir, 'eddy_track', 'eddy_circs_tracked_dist03_size1_circ1.npy'), 'rb') as f:
    eddy_circs = np.load(f)

with open(os.path.join(fdir, 'eddy_track', 'eddy_ids_track_tracked_dist03_size1_circ1.npy'), 'rb') as f:
    eddy_ids_track = np.load(f)   


plotdir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/', 'plots', 'eddy_track_animations_tracked')
xx, yy = np.meshgrid(x, y)

for i in range(1000):
    fig, ax = plt.subplots(figsize=(4,8))
    p = ax.pcolormesh(xx, yy, vort[i,:,:], shading='gouraud', cmap=cmo.curl)   
    fig.colorbar(p, ax=ax)
    p.set_clim(-0.8,0.8) 
    for j in range(1,int(np.max(eddy_ids[i,:,:]))):
        geoms = []
        for yidx, xidx in zip(*np.where(eddy_ids[i,:,:]==j)):
            geoms.append(shapely.geometry.box(x[xidx], y[yidx], x[xidx+1], y[yidx+1]))
        full_geom = shapely.ops.unary_union(geoms)
        ax.plot(*full_geom.exterior.xy, linewidth=1, color='black') 

    for e in range(len(eddy_tracks_x)):
        if np.isfinite(eddy_tracks_x[e,i]) == True:
            ax.plot(eddy_tracks_x[e,:i], eddy_tracks_y[e,:i], '-o', markersize=2, color='grey', alpha=0.5)
            ax.plot(eddy_tracks_x[e,i], eddy_tracks_y[e,i], 'o', markersize=8, markerfacecolor='white', markeredgecolor='black', alpha=0.5)
    ax.set_xlim((np.min(xx), np.max(xx)))
    ax.set_ylim((np.min(yy), np.max(yy)))
    fig.savefig(os.path.join(plotdir, '%05d.png' % i))
    plt.close('all') 

