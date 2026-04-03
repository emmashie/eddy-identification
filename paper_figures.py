import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr 
import shapely.geometry
import shapely.ops
import os 
import cmocean.cm as cmo 
import funpy.model_utils as mod_utils
from scipy.signal import butter, filtfilt
import pandas as pd 
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import numpy.ma as ma 
import funpy.wave_functions as wf 
from scipy.signal import welch
from scipy.stats import circmean 
import cv2 
import matplotlib.colors as mcolors

plt.style.use('classic')

dx = 0.05; dy = 0.1; dt = 0.2; lwidth = 2; fsize = 12; shore = 32.5+22; lab_offset = 22; sz_edge = (25.9+27.5+27.4)/3 + lab_offset
ymin = 0; ymax = 55; start = 1500

def load_eddy_vort(fdir):
    eddy_id = xr.open_mfdataset(os.path.join(fdir, 'eddy_id', 'eddy_track_map_all_averaged_4tp_*.nc'), combine='nested', concat_dim='time')['eddy_id']
    u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
    v_psi = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')['v_psi']
    u_psi = u_psi.values 
    v_psi = v_psi.values

    x = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['x']
    y = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['y']
    x = x.values
    y = y.values

    filt_t = 2*4

    fs = 5 
    cutoff_freq = 1 / filt_t 

    u_psi = butterworth_filter(u_psi, cutoff_freq, fs, axis=0)
    v_psi = butterworth_filter(v_psi, cutoff_freq, fs, axis=0)

    vort = mod_utils.curl(u_psi, v_psi, dx, dy)  
    return  eddy_id, vort, x, y 

def butterworth_filter(data, cutoff_freq, fs, axis=0, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=axis)
    return filtered_data


def split_sz_off(dat, shore, sz_edge):
    dat_sz = dat.copy().iloc[np.where((dat.avg_x<shore)&(dat.avg_x>sz_edge))[0]]
    dat_off = dat.copy().iloc[np.where((dat.avg_x<sz_edge))[0]]
    return dat_sz, dat_off

def load_data(eddy_stats_file, shore, y):
    dat = pd.read_csv(eddy_stats_file)
    dat_xfilt = dat.copy().iloc[np.where((dat.avg_x<shore)&(dat.avg_x>37))[0]].reset_index()
    dat_xyfilt = dat_xfilt.copy().iloc[np.where((dat_xfilt.y<np.max(y))&(dat_xfilt.y>np.min(y)))[0]]
    dat_pos = dat_xyfilt.copy().iloc[np.where(dat_xyfilt.pole==1.0)[0]].reset_index()
    dat_neg = dat_xyfilt.copy().iloc[np.where(dat_xyfilt.pole==-1.0)[0]].reset_index()
    return dat_pos, dat_neg 

def open_tracks(fdir):
    with open(os.path.join(fdir, 'lab_netcdfs', 'eddy_track', 'eddy_tracks_x_tracked_dist03_size1_circ1.npy'), 'rb') as f:
        eddy_tracks_x = np.load(f)

    with open(os.path.join(fdir, 'lab_netcdfs', 'eddy_track', 'eddy_tracks_y_tracked_dist03_size1_circ1.npy'), 'rb') as f:
        eddy_tracks_y = np.load(f)    

    with open(os.path.join(fdir, 'lab_netcdfs', 'eddy_track', 'eddy_sizes_tracked_dist03_size1_circ1.npy'), 'rb') as f:
        eddy_sizes = np.load(f)

    with open(os.path.join(fdir, 'lab_netcdfs', 'eddy_track', 'eddy_circs_tracked_dist03_size1_circ1.npy'), 'rb') as f:
        eddy_circs = np.load(f)

    with open(os.path.join(fdir, 'lab_netcdfs', 'eddy_track', 'eddy_ids_track_tracked_dist03_size1_circ1.npy'), 'rb') as f:
        eddy_ids_track = np.load(f)     
    return eddy_tracks_x, eddy_tracks_y, eddy_sizes, eddy_circs, eddy_ids_track  
 

def find_sz_edge(fdir, dt=0.2, WL=128, OL=64):
    dep = np.loadtxt(os.path.join(fdir, 'dep.out'))
    eta_flist = [os.path.join(fdir, 'eta_1.nc'), os.path.join(fdir, 'eta_2.nc'), os.path.join(fdir, 'eta_3.nc'), os.path.join(fdir, 'eta_4.nc')]
    eta_dat = xr.open_mfdataset(eta_flist, combine='nested', concat_dim='time')
    eta = eta_dat['eta']
    x = eta_dat['x']
    
    freq, Sf = welch(eta, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    Hs = mod_utils.compute_Hsig_spectrally(freq, Sf, np.min(freq), np.max(freq))
    Hs_alongmean = np.nanmean(Hs, axis=0)

    Sf_alongmean = np.nanmean(Sf, axis=1)
    Tp_off = 1/freq[np.where(Sf_alongmean[:,xind]==np.max(Sf_alongmean[:,xind]))[0][0]]

    energy_density = wf.energy_density(Hs_alongmean/2)
    k = wf.wavenum(1/Tp_off*np.ones(len(dep[0,:])), dep[0,:])
    cg = wf.group_speed(2*np.pi/k, Tp_off, dep[0,:])
    energy_flux = wf.energy_flux(energy_density, cg)
    sz_ind = np.nanargmax(np.abs(np.diff(energy_flux))) 
    return x[sz_ind]

def get_eddy_char(dat_sz, T=6000):
    eddy_sizes_sz = [] 
    eddy_count_sz = [] 
    eddy_circ_sz = [] 
    eddy_x_sz = []
    eddy_y_sz = []
    for i in range(T):
        dat_tmp = dat_sz.copy().iloc[np.where(dat_sz.time==i)[0]]
        eddy_count_sz.append(len(dat_tmp))
        eddy_sizes_sz.append(dat_tmp['size'].values)
        eddy_circ_sz.append(dat_tmp.circulation.values)
        eddy_x_sz.append(dat_tmp.avg_x.values)
        eddy_y_sz.append(dat_tmp.avg_y.values)
    return eddy_count_sz, eddy_sizes_sz, eddy_circ_sz, eddy_x_sz, eddy_y_sz 

def rolling_cumsum(arr, window_size, direction='forward'):
    """
    Compute a forward or backward-looking cumulative sum for a given array and window size.

    Parameters:
    arr (np.array): Input array for which to compute the cumulative sum.
    window_size (int): The size of the window for the cumulative sum.
    direction (str): The direction for the cumulative sum, either 'forward' or 'backward'.

    Returns:
    np.array: An array of cumulative sums.
    """
    result = np.full(len(arr), np.nan)
    
    if direction == 'forward':
        for i in range(len(arr) - window_size + 1):
            result[i] = np.nansum(arr[i:i + window_size])
    elif direction == 'backward':
        for i in range(window_size - 1, len(arr)):
            result[i] = np.nansum(arr[i - window_size + 1:i + 1])
    else:
        raise ValueError("Direction must be either 'forward' or 'backward'")
    
    return result

def split_sz_edge(dat, offshore, inshore):
    dat = dat.copy().iloc[np.where((dat.x<inshore)&(dat.x>offshore))[0]]
    return dat


def hex_to_sequential_cmap(hex_color, name='custom_sequential', n_colors=256):
    """
    Create a sequential colormap from white to the specified hex color
    """
    # Convert hex to RGB
    rgb = mcolors.hex2color(hex_color)
    
    # Create colormap from white to the target color
    colors = ['white', rgb]
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=n_colors)
    
    return cmap

##### CODE BEGIN ######
basedir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/')
fdir = os.path.join(basedir, 'lab_netcdfs')
plotdir = os.path.join(basedir, 'plots')
eddy_stats_file = os.path.join(fdir, 'eddy_id', 'eddy_stats.csv')

dep = np.loadtxt(os.path.join(fdir, 'dep.out'))

u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
u_psi5 = u_psi[start:,:,:]

eddy_id5, vort5, x, y = load_eddy_vort(fdir)

dat_pos5, dat_neg5 = load_data(eddy_stats_file, shore, y)

eddy_tracks_x5, eddy_tracks_y5, eddy_sizes5, eddy_circs5, eddy_ids_track5 = open_tracks(basedir)

dat = pd.read_csv(eddy_stats_file)
dat_sz5 = dat.copy().iloc[np.where((dat.avg_x<shore)&(dat.avg_x>sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat_off5 = dat.copy().iloc[np.where((dat.avg_x<sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat5 = dat.copy()

basedir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/')
fdir = os.path.join(basedir, 'lab_netcdfs')
plotdir = os.path.join(basedir, 'plots')
eddy_stats_file = os.path.join(fdir, 'eddy_id', 'eddy_stats.csv')

u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
u_psi10 = u_psi[start:,:,:]

eddy_id10, vort10, x, y = load_eddy_vort(fdir) 

dat_pos10, dat_neg10 = load_data(eddy_stats_file, shore, y)

eddy_tracks_x10, eddy_tracks_y10, eddy_sizes10, eddy_circs10, eddy_ids_track10 = open_tracks(basedir)

dat = pd.read_csv(eddy_stats_file)
dat_sz10 = dat.copy().iloc[np.where((dat.avg_x<shore)&(dat.avg_x>sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat_off10 = dat.copy().iloc[np.where((dat.avg_x<sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat10 = dat.copy()

basedir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/')
fdir = os.path.join(basedir, 'lab_netcdfs')
plotdir = os.path.join(basedir, 'plots')
eddy_stats_file = os.path.join(fdir, 'eddy_id', 'eddy_stats.csv')

u_psi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
u_psi40 = u_psi[start:,:,:]

eddy_id40, vort40, x, y = load_eddy_vort(fdir)

dat_pos40, dat_neg40 = load_data(eddy_stats_file, shore, y)

eddy_tracks_x40, eddy_tracks_y40, eddy_sizes40, eddy_circs40, eddy_ids_track40 = open_tracks(basedir)

dat = pd.read_csv(eddy_stats_file)
dat_sz40 = dat.copy().iloc[np.where((dat.avg_x<shore)&(dat.avg_x>sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat_off40 = dat.copy().iloc[np.where((dat.avg_x<sz_edge)&(dat.avg_y>ymin)&(dat.avg_y<ymax))[0]]
dat40 = dat.copy()

def find_bins(eddy_x):
    x_bins = np.arange(37, 54.5, 0.5)
    x_bin_cent = x_bins[:-1] + 0.5/2
    inds = [np.where((eddy_x>=x_bins[i]) & (eddy_x<x_bins[i+1]))[0] for i in range(len(x_bins)-1)] 
    return x_bin_cent, inds

def cross_var(dat):
    x_bin_cent, inds = find_bins(dat.x) 
    eddypt = np.zeros((len(x_bin_cent), len(np.unique(dat['time']))))
    for i in range(len(inds)):
        times = dat['time'][inds[i]]
        for t in range(len(np.unique(dat['time']))):
            eddypt[i,t] = len(times[times==t])
    Neddies = np.mean(eddypt, axis=-1)
    sizes = np.asarray([np.median(dat['size'][inds[i]]) for i in range(len(inds))])
    circs = np.asarray([np.median(np.abs(dat['circulation'][inds[i]])) for i in range(len(inds))])
    return x_bin_cent, Neddies, sizes, circs 

def summary_stats(dat, sz_edge=23.2+22, width=1):
    dat = dat.copy().iloc[np.where((dat.avg_x<sz_edge+width)&(dat.avg_x>sz_edge-width))[0]]
    eddypt = np.zeros(len(np.unique(dat['time']))) 
    for t in range(len(eddypt)):
        eddypt[t] = len(np.where(dat['time']==t)[0])
    size = dat['size']
    circ = np.abs(dat['circulation'])
    return eddypt, size, circ

def compute_uex(u, dy, yaxis=1):
    Ly = u.shape[yaxis]*dy
    uex = np.sum(u, axis=yaxis)*dy/Ly
    return uex


def rolling_cumsum(arr, window_size):
    cumsum = np.nancumsum(arr)
    rolling_cumsum = np.zeros(len(arr) - window_size + 1)
    rolling_cumsum[0] = cumsum[window_size - 1]
    rolling_cumsum[1:] = cumsum[window_size:] - cumsum[:-window_size]
    return rolling_cumsum 

def find_bin_ind(eddy_tracks_x, xbins, bin_width=2):
    trackx = np.ravel(eddy_tracks_x)
    inds = []
    for i in range(len(xbins)-1):
        tmp = np.where((trackx >= xbins[i]) & (trackx < xbins[i] + bin_width))[0] 
        inds.append(tmp)
    return inds

def eddies_per_time(eddy_tracks_x, xbins):
    eddy_pt = np.zeros((len(xbins), len(eddy_tracks_x[0,:])))
    for t in range(len(eddy_tracks_x10[0,:])):
        for i in range(len(xbins)-1):
            tmp = np.where((eddy_tracks_x[:,t]>=xbins[i]) & (eddy_tracks_x[:,t]<xbins[i+1]))[0] 
            eddy_pt[i, t] = len(tmp)
    return eddy_pt

def find_cross_var(inds, var):
    var_cross = np.zeros(len(inds))
    for i in range(len(inds)):
        var_cross[i] = np.median(var[inds[i]])
    return var_cross 

def find_bins(eddy_x, bin_width=1, bin_step=0.5):
    x_bins = np.arange(37, 54.5, bin_step)
    x_bin_cent = x_bins[:-1] + bin_width / 2  
    inds = [np.where((eddy_x >= x_bins[i]) & (eddy_x < x_bins[i] + bin_width))[0] for i in range(len(x_bins) - 1)] 
    return x_bin_cent, inds

def cross_var(dat):
    x_bin_cent, inds = find_bins(dat.x) 
    eddypt = np.zeros((len(x_bin_cent), len(np.unique(dat['time']))))
    for i in range(len(inds)):
        times = dat['time'][inds[i]]
        for t in range(len(np.unique(dat['time']))):
            eddypt[i,t] = len(times[times==t])
    Neddies = np.mean(eddypt, axis=-1)
    sizes = np.asarray([np.median(dat['size'][inds[i]]) for i in range(len(inds))])
    circs = np.asarray([np.median(np.abs(dat['circulation'][inds[i]])) for i in range(len(inds))])
    return x_bin_cent, Neddies, sizes, circs 

#################################################################################
################ plotting #######################################################
#################################################################################
xx, yy = np.meshgrid(x-lab_offset,y-55/2)

############## FIGURE 1 #####################
xind5 = np.where(dat5['x']<shore)[0]
xind10 = np.where(dat10['x']<shore)[0]
xind40 = np.where(dat40['x']<shore)[0]

upsi_comp5 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/lab_netcdfs/', 'eddy_comps', 'eddy_composites_upsi.nc'))['eddy_composites']
vpsi_comp5 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vpsi.nc'))['eddy_composites']
vort_comp5 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vort.nc'))['eddy_composites']

upsi_comp10 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_upsi.nc'))['eddy_composites']
vpsi_comp10 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vpsi.nc'))['eddy_composites']
vort_comp10 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir10_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vort.nc'))['eddy_composites']

upsi_comp40 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_upsi.nc'))['eddy_composites']
vpsi_comp40 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vpsi.nc'))['eddy_composites']
vort_comp40 = xr.open_dataset(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir40_tp2/lab_netcdfs/', 'eddy_comps', 'eddy_composites_vort.nc'))['eddy_composites']

xc = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))['x']
yc = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))['y']

xxc, yyc = np.meshgrid(xc,yc)

vort_comp5 = vort_comp5.values
vort_comp10 = vort_comp10.values 
vort_comp40 = vort_comp40.values

upsi_comp5 = upsi_comp5.values
upsi_comp10 = upsi_comp10.values 
upsi_comp40 = upsi_comp40.values

vpsi_comp5 = vpsi_comp5.values
vpsi_comp10 = vpsi_comp10.values 
vpsi_comp40 = vpsi_comp40.values

tmp = dat5['circulation'][xind5]
tmp_ind = np.where(tmp>0)[0]

pos_vort_mean5= np.nanmean(vort_comp5[:,:,xind5[tmp_ind]], axis=-1)
pos_upsi_mean5 = np.nanmean(upsi_comp5[:,:,xind5[tmp_ind]], axis=-1)
pos_vpsi_mean5 = np.nanmean(vpsi_comp5[:,:,xind5[tmp_ind]], axis=-1)

tmp = dat5['circulation'][xind5]
tmp_ind = np.where(tmp<0)[0]

neg_vort_mean5= np.nanmean(vort_comp5[:,:,xind5[tmp_ind]], axis=-1)
neg_upsi_mean5 = np.nanmean(upsi_comp5[:,:,xind5[tmp_ind]], axis=-1)
neg_vpsi_mean5 = np.nanmean(vpsi_comp5[:,:,xind5[tmp_ind]], axis=-1)


tmp = dat10['circulation'][xind10]
tmp_ind = np.where(tmp>0)[0]

pos_vort_mean10 = np.nanmean(vort_comp10[:,:,xind10[tmp_ind]], axis=-1)
pos_upsi_mean10 = np.nanmean(upsi_comp10[:,:,xind10[tmp_ind]], axis=-1)
pos_vpsi_mean10 = np.nanmean(vpsi_comp10[:,:,xind10[tmp_ind]], axis=-1)

tmp = dat10['circulation'][xind10]
tmp_ind = np.where(tmp<0)[0]

neg_vort_mean10 = np.nanmean(vort_comp10[:,:,xind10[tmp_ind]], axis=-1)
neg_upsi_mean10 = np.nanmean(upsi_comp10[:,:,xind10[tmp_ind]], axis=-1)
neg_vpsi_mean10 = np.nanmean(vpsi_comp10[:,:,xind10[tmp_ind]], axis=-1)


tmp = dat40['circulation'][xind40]
tmp_ind = np.where(tmp>0)[0]

pos_vort_mean40 = np.nanmean(vort_comp40[:,:,xind40[tmp_ind]], axis=-1)
pos_upsi_mean40 = np.nanmean(upsi_comp40[:,:,xind40[tmp_ind]], axis=-1)
pos_vpsi_mean40 = np.nanmean(vpsi_comp40[:,:,xind40[tmp_ind]], axis=-1)

tmp = dat40['circulation'][xind40]
tmp_ind = np.where(tmp<0)[0]

neg_vort_mean40 = np.nanmean(vort_comp40[:,:,xind40[tmp_ind]], axis=-1)
neg_upsi_mean40 = np.nanmean(upsi_comp40[:,:,xind40[tmp_ind]], axis=-1)
neg_vpsi_mean40 = np.nanmean(vpsi_comp40[:,:,xind40[tmp_ind]], axis=-1)

width = 1
dat5_sz_edge = split_sz_edge(dat5, sz_edge-width, sz_edge+width)
dat10_sz_edge = split_sz_edge(dat10, sz_edge-width, sz_edge+width)
dat40_sz_edge = split_sz_edge(dat40, sz_edge-width, sz_edge+width)

eddy_count_sz5, eddy_sizes_sz5, eddy_circ_sz5, eddy_x_sz5, eddy_y_sz5 = get_eddy_char(dat5_sz_edge) 
eddy_count_sz10, eddy_sizes_sz10, eddy_circ_sz10, eddy_x_sz10, eddy_y_sz10 = get_eddy_char(dat10_sz_edge) 
eddy_count_sz40, eddy_sizes_sz40, eddy_circ_sz40, eddy_x_sz40, eddy_y_sz40 = get_eddy_char(dat40_sz_edge) 


n = 4; tplot = 2750 
tmp = np.argmin(np.abs(x-sz_edge))
t = 2700; clim = 0.6
fig, ax = plt.subplots(figsize=(10,10), ncols=9)
xmin = 0.1; h1 = 0.45; h2 = 0.05; h3 = 0.25; ymin = 0.1; w = 0.23; off = 0.03 
ax[0].set_position([xmin, ymin+h2+h3+off*2.5, w, h1])
ax[1].set_position([xmin+w+off, ymin+h2+h3+off*2.5, w, h1])
ax[2].set_position([xmin+w*2+off*2, ymin+h2+h3+off*2.5, w, h1])

ax[3].set_position([xmin, ymin+h3+off*2.5, w, h2])
ax[4].set_position([xmin+w+off, ymin+h3+off*2.5, w, h2])
ax[5].set_position([xmin+w*2+off*2, ymin+h3+off*2.5, w, h2])

ax[6].set_position([xmin, ymin, w, h3])
ax[7].set_position([xmin+w+off, ymin, w, h3])
ax[8].set_position([xmin+w*2+off*2, ymin, w, h3])

p0 = ax[0].pcolormesh(xx, yy, vort5[1500+tplot,:,:], shading='gouraud', cmap=cmo.curl,
                      vmin=-clim, vmax=clim)
p0.set_clim(-clim, clim)
ax[0].set_xlim(np.min(xx), np.max(xx))
ax[0].set_ylim(np.min(yy), np.max(yy))


for i in range(1,int(np.max(eddy_id5[tplot,:,:]))):
    geoms = []
    for yidx, xidx in zip(*np.where(eddy_id5[tplot,:,:]==i)):
        geoms.append(shapely.geometry.box(x[xidx]-lab_offset, y[yidx]-55/2, x[xidx+1]-lab_offset, y[yidx+1]-55/2))
    full_geom = shapely.ops.unary_union(geoms)
    ax[0].plot(*full_geom.exterior.xy, linewidth=1, color='black') 
ax[0].set_ylabel(r'$y\ \mathrm{(m)}$')
ax[0].text(16, -26, r'$\mathrm{(a)}$', fontsize=fsize)
ax[0].fill_betweenx(y-55/2, np.ones(len(y))*(sz_edge-lab_offset), np.ones(len(y))*(shore-lab_offset), color='grey', alpha=0.5)
ax[0].set_yticks([-20, -10, 0, 10, 20])
ax[0].set_yticklabels([r'$-20$', r'$-10$', r'$0$', r'$10$', r'$20$'])
ax[0].set_xticks([15, 20, 25, 30, 35])
ax[0].set_xticklabels(['', '', '', '', ''])
ax[0].set_title(r'$\sigma_\theta = 3.8^\circ$')

t = 2750
p1 = ax[1].pcolormesh(xx, yy, vort10[1500+tplot,:,:], shading='gouraud', cmap=cmo.curl,
                      vmin=-clim, vmax=clim)
p1.set_clim(-clim, clim)
ax[1].set_xlim(np.min(xx), np.max(xx))
ax[1].set_ylim(np.min(yy), np.max(yy))

for i in range(1,int(np.max(eddy_id10[tplot,:,:]))):
    geoms = []
    for yidx, xidx in zip(*np.where(eddy_id10[tplot,:,:]==i)):
        geoms.append(shapely.geometry.box(x[xidx]-lab_offset, y[yidx]-55/2, x[xidx+1]-lab_offset, y[yidx+1]-55/2))
    full_geom = shapely.ops.unary_union(geoms)
    ax[1].plot(*full_geom.exterior.xy, linewidth=1, color='black') 
ax[1].text(16, -26, r'$\mathrm{(b)}$', fontsize=fsize)
ax[1].fill_betweenx(y-55/2, np.ones(len(y))*(sz_edge-lab_offset), np.ones(len(y))*(shore-lab_offset), color='grey', alpha=0.5)
ax[1].set_yticks([0])
ax[1].set_yticklabels([''])
ax[1].set_xticks([15, 20, 25, 30, 35])
ax[1].set_xticklabels(['', '', '', '', ''])
ax[1].set_title(r'$\sigma_\theta = 11.2^\circ$')

t = 4600
p2 = ax[2].pcolormesh(xx, yy, vort40[1500+tplot,:,:], shading='gouraud', cmap=cmo.curl,
                      vmin=-clim, vmax=clim)
p2.set_clim(-clim, clim)
ax[2].set_xlim(np.min(xx), np.max(xx))
ax[2].set_ylim(np.min(yy), np.max(yy))

for i in range(1,int(np.max(eddy_id40[tplot,:,:]))):
    geoms = []
    for yidx, xidx in zip(*np.where(eddy_id40[tplot,:,:]==i)):
        geoms.append(shapely.geometry.box(x[xidx]-lab_offset, y[yidx]-55/2, x[xidx+1]-lab_offset, y[yidx+1]-55/2))
    full_geom = shapely.ops.unary_union(geoms)
    ax[2].plot(*full_geom.exterior.xy, linewidth=1, color='black') 
cbar_ax = fig.add_axes([0.88, ymin+h2+h3+off*2.5, 0.0125, h1])
cb = fig.colorbar(p0, ax=ax[2], cax=cbar_ax, label=r'$\zeta \ \mathrm{(s^{-1})}$')
cb.ax.set_ylim([-clim,clim])
cb.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cb.set_ticklabels([r'$-0.6$', r'$-0.4$', r'$-0.2$', r'$0$', r'$0.2$', r'$0.4$', r'$0.6$'])
ax[2].text(16, -26, r'$\mathrm{(c)}$', fontsize=fsize)
ax[2].fill_betweenx(y-55/2, np.ones(len(y))*(sz_edge-lab_offset), np.ones(len(y))*(shore-lab_offset), color='grey', alpha=0.5)
ax[2].set_yticks([0])
ax[2].set_yticklabels([''])
ax[2].set_xticks([15, 20, 25, 30, 35])
ax[2].set_xticklabels(['', '', '', '', ''])
ax[2].set_title(r'$\sigma_\theta = 24.9^\circ$')

tmp2 = np.argmin(np.abs(x-shore))
ax[3].plot(x-lab_offset, -dep[0,:], color='black', linewidth=lwidth)
ax[3].fill_between(x[tmp:tmp2]-lab_offset, np.ones(len(dep[0,tmp:tmp2]))*0.5, -dep[0,tmp:tmp2], color='tab:grey', alpha=0.3)
ax[3].fill_between(x-lab_offset, -dep[0,:], np.ones(len(dep[0,:]))*(-1.2), color='black', alpha=0.3)
ax[3].set_ylim(-1.2, 0.2)
ax[3].set_xticks([15, 20, 25, 30, 35])
ax[3].set_xticklabels([r'$15$', r'$20$', r'$25$', r'$30$', r'$35$'])
ax[3].set_xlabel(r'$x\ \mathrm{(m)}$')
ax[3].set_yticks([-1, -0.5, 0.0])
ax[3].set_yticklabels([r'$-1.0$', r'$-0.5$', r'$-0.0$'])
ax[3].set_ylabel(r'$h\ \mathrm{(m)}$')

ax[4].plot(x-lab_offset, -dep[0,:], color='black', linewidth=lwidth)
ax[4].fill_between(x[tmp:tmp2]-lab_offset, np.ones(len(dep[0,tmp:tmp2]))*0.5, -dep[0,tmp:tmp2], color='tab:grey', alpha=0.3)
ax[4].fill_between(x-lab_offset, -dep[0,:], np.ones(len(dep[0,:]))*(-1.2), color='black', alpha=0.3)
ax[4].set_ylim(-1.2, 0.2)
ax[4].set_xticks([15, 20, 25, 30, 35])
ax[4].set_xticklabels([r'$15$', r'$20$', r'$25$', r'$30$', r'$35$'])
ax[4].set_xlabel(r'$x\ \mathrm{(m)}$')
ax[4].set_yticks([-1, -0.5, -0])
ax[4].set_yticklabels(['', '', ''])

ax[5].plot(x-lab_offset, -dep[0,:], color='black', linewidth=lwidth)
ax[5].fill_between(x[tmp:tmp2]-lab_offset, np.ones(len(dep[0,tmp:tmp2]))*0.5, -dep[0,tmp:tmp2], color='tab:grey', alpha=0.3)
ax[5].fill_between(x-lab_offset, -dep[0,:], np.ones(len(dep[0,:]))*(-1.2), color='black', alpha=0.3)
ax[5].set_ylim(-1.2, 0.2)
ax[5].set_xticks([15, 20, 25, 30, 35])
ax[5].set_xticklabels([r'$15$', r'$20$', r'$25$', r'$30$', r'$35$'])
ax[5].set_xlabel(r'$x\ \mathrm{(m)}$')
ax[5].set_yticks([-1, -0.5, -0])
ax[5].set_yticklabels(['', '', ''])

times = np.arange(0, 6000*dt, dt)/60
tt, yy_hov = np.meshgrid(times, y-55/2)
S = 25; skip = 10; alpha = 0.5; n = 10; Tmax = len(times); xmax = 10
color2 = '#000000'
color1 = 'tab:grey'
color3 = '#ffa600'
clim = 0.3 
ssize = 50

p6 = ax[6].pcolormesh(tt, yy_hov, u_psi5[:,:,tmp].T, cmap=cmo.balance)
p6.set_clim(-clim, clim)
ax[6].set_ylim(-55/2, 55/2)
ax[6].set_xlim(xmax, xmax*2)
ax[6].set_yticks([-20, -10, 0, 10, 20])
ax[6].set_yticklabels([r'$-20$', r'$-10$', r'$0$', r'$10$', r'$20$'])
ax[6].set_xticks([10,12,14,16,18,20])
ax[6].set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$'])

# Compile x (time) and y (y position) coordinates for individual eddies
eddy_times_5 = []
eddy_y_positions_5 = []
for i in range(0, Tmax, skip):
    for y_pos in eddy_y_sz5[i]:
        eddy_times_5.append(times[i])
        eddy_y_positions_5.append(y_pos - 55/2)

# Subsample by individual eddies instead of time steps
subsample_indices = np.arange(0, len(eddy_times_5), skip)
ax[6].scatter(np.array(eddy_times_5)[subsample_indices], 
              np.array(eddy_y_positions_5)[subsample_indices], 
              s=ssize, c=np.zeros(len(subsample_indices)), 
              edgecolors=color2, cmap='Greys', alpha=alpha)

#for i in range(0,Tmax,skip):
#    ax[6].scatter(np.ones(len(eddy_y_sz5[i]))*times[i], eddy_y_sz5[i]-55/2, s=ssize, c=np.zeros(len(eddy_y_sz5[i])), edgecolors=color2, cmap='Greys', alpha=alpha)

ax[6].text(10, 57-55/2, r'$\mathrm{(d)}$', fontsize=fsize)
ax[6].set_xlabel(r'$\mathrm{Time\ (min)}$')
ax[6].set_ylabel(r'$y\ \mathrm{(m)}$')
ax[6].axvline(x=(tplot+1500)*dt/60, color='black', linestyle='--', linewidth=2)

p7 = ax[7].pcolormesh(tt, yy_hov, u_psi10[:,:,tmp].T, cmap=cmo.balance)
p7.set_clim(-clim, clim)
ax[7].set_ylim(-55/2, 55/2)
ax[7].set_xlim(xmax, xmax*2)
ax[7].set_yticks([-20, -10, 0, 10, 20])
ax[7].set_yticklabels(['', '', '', '', ''])
ax[7].set_xticks([10,12,14,16,18,20])
ax[7].set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$'])

# Compile x (time) and y (y position) coordinates for individual eddies
eddy_times_10 = []
eddy_y_positions_10 = []
for i in range(0, Tmax, skip):
    for y_pos in eddy_y_sz10[i]:
        eddy_times_10.append(times[i])
        eddy_y_positions_10.append(y_pos - 55/2)

# Subsample by individual eddies instead of time steps
subsample_indices = np.arange(0, len(eddy_times_10), skip)
ax[7].scatter(np.array(eddy_times_10)[subsample_indices], 
              np.array(eddy_y_positions_10)[subsample_indices], 
              s=ssize, c=np.zeros(len(subsample_indices)), 
              edgecolors=color2, cmap='Greys', alpha=alpha)

#for i in range(0,Tmax,skip):
#    ax[7].scatter(np.ones(len(eddy_y_sz10[i]))*times[i], eddy_y_sz10[i]-55/2, s=ssize, c=np.zeros(len(eddy_y_sz10[i])), edgecolors=color2, cmap='Greys', alpha=alpha)

ax[7].text(10, 57-55/2, r'$\mathrm{(e)}$', fontsize=fsize)
ax[7].set_xlabel(r'$\mathrm{Time\ (min)}$')
ax[7].axvline(x=(tplot+1500)*dt/60, color='black', linestyle='--', linewidth=2)

p8 = ax[8].pcolormesh(tt, yy_hov, u_psi40[:,:,tmp].T, cmap=cmo.balance)
p8.set_clim(-clim, clim)
ax[8].set_ylim(-55/2, 55/2)
ax[8].set_xlim(xmax, xmax*2)
ax[8].set_yticks([-20, -10, 0, 10, 20])
ax[8].set_yticklabels(['', '', '', '', ''])
ax[8].set_xticks([10,12,14,16,18,20])
ax[8].set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$'])

# Compile x (time) and y (y position) coordinates for individual eddies
eddy_times_40 = []
eddy_y_positions_40 = []
for i in range(0, Tmax, skip):
    for y_pos in eddy_y_sz40[i]:
        eddy_times_40.append(times[i])
        eddy_y_positions_40.append(y_pos - 55/2)

# Subsample by individual eddies instead of time steps
subsample_indices = np.arange(0, len(eddy_times_40), skip)
ax[8].scatter(np.array(eddy_times_40)[subsample_indices], 
              np.array(eddy_y_positions_40)[subsample_indices], 
              s=ssize, c=np.zeros(len(subsample_indices)), 
              edgecolors=color2, cmap='Greys', alpha=alpha)

#for i in range(0,Tmax,skip):
#    ax[8].scatter(np.ones(len(eddy_y_sz40[i]))*times[i], eddy_y_sz40[i]-55/2, s=ssize, c=np.zeros(len(eddy_y_sz40[i])), edgecolors=color2, cmap='Greys', alpha=alpha)

ax[8].text(10, 57-55/2, r'$\mathrm{(f)}$', fontsize=fsize)
ax[8].set_xlabel(r'$\mathrm{Time\ (min)}$')
ax[8].axvline(x=(tplot+1500)*dt/60, color='black', linestyle='--', linewidth=2)

cbar_ax2 = fig.add_axes([0.88, ymin, 0.0125, h3])
cb2 = fig.colorbar(p8, ax=ax[8], cax=cbar_ax2, label=r'$u_\psi \ \mathrm{(m s^{-1})}$')
cb2.ax.set_ylim([-clim,clim])
cb2.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
cb2.set_ticklabels([r'$-0.3$', r'$-0.2$', r'$-0.1$', r'$0$', r'$0.1$', r'$0.2$', r'$0.3$'])

fig.savefig(os.path.join(plotdir, 'eddy-stats', 'grl-paper-figs', 'Figure1.png'), dpi=600)

def extract_eddy_velocity_values(u_psi, eddy_times, eddy_x_pos, eddy_y_pos, x, y, dt):
    """
    Extract velocity values at eddy locations from u_psi data.
    
    Parameters:
    u_psi: 3D velocity array [time, y, x]
    eddy_times: list of times for each eddy (in time steps or minutes)
    eddy_x_pos: list of x-positions for each eddy (can contain empty arrays)
    eddy_y_pos: list of y-positions for each eddy (can contain empty arrays)
    x: 1D array of x-coordinates from model
    y: 1D array of y-coordinates from model
    dt: time step (0.2 seconds)
    
    Returns:
    Lists of times, x-positions, y-positions, and corresponding velocity values
    """
    
    velocity_times = []
    velocity_x_pos = []
    velocity_y_pos = []
    velocity_values = []
    
    for i in range(len(eddy_times)):
        # Skip if no eddies at this time step
        if (i >= len(eddy_x_pos) or i >= len(eddy_y_pos) or 
            len(eddy_x_pos[i]) == 0 or len(eddy_y_pos[i]) == 0):
            continue
            
        # Convert time to time index if needed
        if isinstance(eddy_times[i], (list, np.ndarray)):
            # If eddy_times contains lists for each time step
            for j, time_val in enumerate(eddy_times[i]):
                if j < len(eddy_x_pos[i]) and j < len(eddy_y_pos[i]):
                    t_idx = int(time_val) if time_val < u_psi.shape[0] else None
                    x_pos = eddy_x_pos[i][j]
                    y_pos = eddy_y_pos[i][j]
                    
                    if t_idx is not None:
                        # Find nearest grid indices
                        x_idx = np.argmin(np.abs(x - x_pos))
                        y_idx = np.argmin(np.abs(y - y_pos))
                        
                        # Extract velocity value
                        if (x_idx < u_psi.shape[2] and y_idx < u_psi.shape[1]):
                            vel_val = u_psi[t_idx, y_idx, x_idx]
                            
                            velocity_times.append(time_val)
                            velocity_x_pos.append(x_pos)
                            velocity_y_pos.append(y_pos)
                            velocity_values.append(vel_val)
        else:
            # If eddy_times is a single time value per eddy
            t_idx = int(eddy_times[i]) if eddy_times[i] < u_psi.shape[0] else None
            
            if t_idx is not None:
                # Loop through all eddies at this time step
                for j in range(len(eddy_x_pos[i])):
                    if j < len(eddy_y_pos[i]):  # Make sure y position exists too
                        x_pos = eddy_x_pos[i][j]
                        y_pos = eddy_y_pos[i][j]
                        
                        # Find nearest grid indices
                        x_idx = np.argmin(np.abs(x - x_pos))
                        y_idx = np.argmin(np.abs(y - y_pos))
                        
                        # Extract velocity value
                        if (x_idx < u_psi.shape[2] and y_idx < u_psi.shape[1]):
                            vel_val = u_psi[t_idx, y_idx, x_idx]
                            
                            velocity_times.append(eddy_times[i])
                            velocity_x_pos.append(x_pos)
                            velocity_y_pos.append(y_pos)
                            velocity_values.append(vel_val)
    
    return velocity_times, velocity_x_pos, velocity_y_pos, velocity_values

# For your specific data, you can use it like this:

# Extract velocity values for eddies in the surf zone edge region for case 5
vel_times_5, vel_x_5, vel_y_5, vel_values_5 = extract_eddy_velocity_values(
    u_psi5.values,  # velocity data
    range(len(eddy_x_sz5)),  # time indices (0 to len-1)
    eddy_x_sz5,     # x positions for each time step
    eddy_y_sz5,     # y positions for each time step  
    x, y, dt
)

# Same for cases 10 and 40
vel_times_10, vel_x_10, vel_y_10, vel_values_10 = extract_eddy_velocity_values(
    u_psi10.values, range(len(eddy_x_sz10)), eddy_x_sz10, eddy_y_sz10, x, y, dt
)

vel_times_40, vel_x_40, vel_y_40, vel_values_40 = extract_eddy_velocity_values(
    u_psi40.values, range(len(eddy_x_sz40)), eddy_x_sz40, eddy_y_sz40, x, y, dt
)

vel_values_5 = np.array(vel_values_5)
vel_values_10 = np.array(vel_values_10)
vel_values_40 = np.array(vel_values_40)


################## FIGURE 2 #################### 
fig = plt.figure(figsize=(8,5.2))  # Increased height slightly to prevent x-label cutoff
# Define dimensions - tighter spacing with room for colorbar on right
xmin = 0.12; ymin = 0.14; w = 0.20; h = 0.35; off = 0.06  # Increased ymin to prevent cutoff
skinny_inches = 0.3  # physical size in inches for skinny dimension
gap = 0.0      # no gap between main and skinny subplots (changed from 0.002)
cmax = 0.55
n = 4
# Convert skinny dimension from inches to figure fraction
figwidth, figheight = fig.get_size_inches()
skinny_w = skinny_inches / figwidth  # width in figure fraction
skinny_h = skinny_inches / figheight  # height in figure fraction
# Create main axes first
ax = []
positions = [
    [xmin, ymin+h+off],  # ax[0]
    [xmin+w+off, ymin+h+off],  # ax[1]
    [xmin+w*2+off*2, ymin+h+off],  # ax[2]
    [xmin, ymin],  # ax[3]
    [xmin+w+off, ymin],  # ax[4]
    [xmin+w*2+off*2, ymin],  # ax[5]
]
for i, (x, y) in enumerate(positions):
    ax_main = fig.add_axes([x, y, w, h])
    ax.append(ax_main)
# Store vorticity and velocity data for each subplot
# Top row (0-2): negative vorticity, Bottom row (3-5): positive vorticity
vort_data = [neg_vort_mean5, neg_vort_mean10, neg_vort_mean40,
             pos_vort_mean5, pos_vort_mean10, pos_vort_mean40]
u_data = [neg_upsi_mean5, neg_upsi_mean10, neg_upsi_mean40,
          pos_upsi_mean5, pos_upsi_mean10, pos_upsi_mean40]
v_data = [neg_vpsi_mean5, neg_vpsi_mean10, neg_vpsi_mean40,
          pos_vpsi_mean5, pos_vpsi_mean10, pos_vpsi_mean40]
# Original plotting code (removed ylabel and xlabel from main axes)
p0 = ax[0].pcolormesh(xxc, yyc, neg_vort_mean5, cmap=cmo.curl)
ax[0].quiver(xxc[::n,::n], yyc[::n,::n], neg_upsi_mean5[::n,::n], neg_vpsi_mean5[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[0].add_artist(circle)
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_ylim(-1.5, 1.5)
ax[0].set_xlim(-1.5, 1.5)
p0.set_clim(-cmax, cmax)
ax[0].set_xticks([0])
ax[0].set_xticklabels([''])
ax[0].set_yticks([-1, 0, 1])
ax[0].set_yticklabels([r'$-0.5$', r'$0$', r'$0.5$']) # change labels to diameter rather than radius units
ax[0].text(-1.45, -1.45, r'$\mathrm{(a)}$', fontsize=fsize)
ax[0].set_title(r'$\sigma_\theta = 3.8^\circ$')
p1 = ax[1].pcolormesh(xxc, yyc, neg_vort_mean10, cmap=cmo.curl)
ax[1].quiver(xxc[::n,::n], yyc[::n,::n], neg_upsi_mean10[::n,::n], neg_vpsi_mean10[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[1].add_artist(circle)
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_ylim(-1.5, 1.5)
ax[1].set_xlim(-1.5, 1.5)
p1.set_clim(-cmax, cmax)
ax[1].set_xticks([0])
ax[1].set_xticklabels([''])
ax[1].set_yticks([0])
ax[1].set_yticklabels([''])
ax[1].text(-1.45, -1.45, r'$\mathrm{(b)}$', fontsize=fsize)
ax[1].set_title(r'$\sigma_\theta = 11.2^\circ$')
p2 = ax[2].pcolormesh(xxc, yyc, neg_vort_mean40, cmap=cmo.curl)
q = ax[2].quiver(xxc[::n,::n], yyc[::n,::n], neg_upsi_mean40[::n,::n], neg_vpsi_mean40[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[2].add_artist(circle)
ax[2].set_aspect('equal', adjustable='box')
ax[2].set_ylim(-1.5, 1.5)
ax[2].set_xlim(-1.5, 1.5)
p2.set_clim(-cmax, cmax)
ax[2].set_xticks([0])
ax[2].set_xticklabels([''])
ax[2].set_yticks([0])
ax[2].set_yticklabels([''])
# Moved colorbar closer to subplots
cbar_ax = fig.add_axes([xmin+w*3+off*2.5, ymin + h + off, 0.0125, h])
cb = fig.colorbar(p2, ax=ax[2], cax=cbar_ax, label=r'$\zeta \ \mathrm{(s^{-1})}$')
cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])
cb.set_ticklabels([r'$-0.4$', r'$-0.2$', r'$0$', r'$0.2$', r'$0.4$'])
ax[2].text(-1.45, -1.45, r'$\mathrm{(c)}$', fontsize=fsize)
# Fixed quiver key - moved further up and to the right
ax[2].quiverkey(q, X=0.95, Y=1.22, U=0.1, label=r'$0.1\ \mathrm{(m\ s^{-1})}$', labelpos='E')
ax[2].set_title(r'$\sigma_\theta = 24.9^\circ$')
p3 = ax[3].pcolormesh(xxc, yyc, pos_vort_mean5, cmap=cmo.curl)
ax[3].quiver(xxc[::n,::n], yyc[::n,::n], pos_upsi_mean5[::n,::n], pos_vpsi_mean5[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[3].add_artist(circle)
ax[3].set_aspect('equal', adjustable='box')
ax[3].set_ylim(-1.5, 1.5)
ax[3].set_xlim(-1.5, 1.5)
p3.set_clim(-cmax, cmax)
ax[3].set_yticks([-1, 0, 1])
ax[3].set_yticklabels([r'$-0.5$', r'$0$', r'$0.5$']) # change labels to diameter rather than radius units
ax[3].set_xticks([-1, 0, 1])
ax[3].set_xticklabels([r'$-0.5$', r'$0$', r'$0.5$'])
ax[3].text(-1.45, -1.45, r'$\mathrm{(d)}$', fontsize=fsize)
ax[3].text(-1.65, -2.7, r'$\mathrm{Offshore}$', fontsize=fsize-5)
ax[3].text(1.65, -2.7, r'$\mathrm{Onshore}$', fontsize=fsize-5, ha='right')

p4 = ax[4].pcolormesh(xxc, yyc, pos_vort_mean10, cmap=cmo.curl)
ax[4].quiver(xxc[::n,::n], yyc[::n,::n], pos_upsi_mean10[::n,::n], pos_vpsi_mean10[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[4].add_artist(circle)
ax[4].set_aspect('equal', adjustable='box')
ax[4].set_ylim(-1.5, 1.5)
ax[4].set_xlim(-1.5, 1.5)
p4.set_clim(-cmax, cmax)
ax[4].set_yticks([0])
ax[4].set_yticklabels([''])
ax[4].set_xticks([-1, 0, 1])
ax[4].set_xticklabels([r'$-0.5$', r'$0$', r'$0.5$'])
ax[4].text(-1.45, -1.45, r'$\mathrm{(e)}$', fontsize=fsize)
ax[4].text(-1.65, -2.7, r'$\mathrm{Offshore}$', fontsize=fsize-5)
ax[4].text(1.65, -2.7, r'$\mathrm{Onshore}$', fontsize=fsize-5, ha='right')

p5 = ax[5].pcolormesh(xxc, yyc, pos_vort_mean40, cmap=cmo.curl)
ax[5].quiver(xxc[::n,::n], yyc[::n,::n], pos_upsi_mean40[::n,::n], pos_vpsi_mean40[::n,::n])
circle = plt.Circle((0, 0), 1, fill=False)
ax[5].add_artist(circle)
ax[5].set_aspect('equal', adjustable='box')
ax[5].set_ylim(-1.5, 1.5)
ax[5].set_xlim(-1.5, 1.5)
p5.set_clim(-cmax, cmax)
ax[5].set_yticks([0])
ax[5].set_yticklabels([''])
ax[5].set_xticks([-1, 0, 1])
ax[5].set_xticklabels([r'$-0.5$', r'$0$', r'$0.5$'])
ax[5].text(-1.65, -2.7, r'$\mathrm{Offshore}$', fontsize=fsize-5)
ax[5].text(1.65, -2.7, r'$\mathrm{Onshore}$', fontsize=fsize-5, ha='right')

# Moved colorbar closer to subplots
cbar_ax = fig.add_axes([xmin+w*3+off*2.5, ymin, 0.0125, h])
cb = fig.colorbar(p5, ax=ax[5], cax=cbar_ax, label=r'$\zeta \ \mathrm{(s^{-1})}$')
cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])
cb.set_ticklabels([r'$-0.4$', r'$-0.2$', r'$0$', r'$0.2$', r'$0.4$'])
ax[5].text(-1.45, -1.45, r'$\mathrm{(f)}$', fontsize=fsize)
# Now add skinny subplots based on actual rendered positions
fig.canvas.draw()  # Force a draw to get actual positions
ax_left = []
ax_bottom = []
# Find center indices for transects
ny, nx = vort_data[0].shape
center_y = ny // 2
center_x = nx // 2
for i in range(6):
    # Get the actual position after aspect adjustment
    pos = ax[i].get_position()
    # Get transect data for this subplot (correctly matched to vort_data)
    vort = vort_data[i]
    u = u_data[i]
    v = v_data[i]
    # Transects through center
    vort_x_transect = vort[center_y, :]  # horizontal slice through center (for bottom plot)
    vort_y_transect = vort[:, center_x]  # vertical slice through center (for left plot)
    u_x_transect = u[center_y, :]  # u velocity along x (for bottom plot)
    v_x_transect = v[center_y, :]  # v velocity along x (for bottom plot)
    u_y_transect = u[:, center_x]  # u velocity along y (for left plot)
    v_y_transect = v[:, center_x]  # v velocity along y (for left plot)
    
    # Calculate velocity magnitude for transects
    vel_mag_x_transect = np.sqrt(u_x_transect**2 + v_x_transect**2)
    vel_mag_y_transect = np.sqrt(u_y_transect**2 + v_y_transect**2)
    
    # Apply sign: negative for top row (i=0,1,2), positive for bottom row (i=3,4,5)
    if i < 3:  # Top row - negative vorticity
        vel_mag_x_transect = -vel_mag_x_transect
        vel_mag_y_transect = -vel_mag_y_transect
    
    # Normalize by INDIVIDUAL max values for each transect
    max_vort_y = np.abs(vort_y_transect).max()
    max_vort_x = np.abs(vort_x_transect).max()
    max_vel_y = np.abs(vel_mag_y_transect).max()
    max_vel_x = np.abs(vel_mag_x_transect).max()
    
    # Avoid division by zero
    if max_vort_y == 0:
        max_vort_y = 1
    if max_vort_x == 0:
        max_vort_x = 1
    if max_vel_y == 0:
        max_vel_y = 1
    if max_vel_x == 0:
        max_vel_x = 1
    
    vort_y_norm = vort_y_transect / max_vort_y
    vort_x_norm = vort_x_transect / max_vort_x
    vel_mag_y_norm = vel_mag_y_transect / max_vel_y
    vel_mag_x_norm = vel_mag_x_transect / max_vel_x
    
    # Calculate limits centered on ZERO with padding
    padding_factor = 0.2
    
    # For vorticity - centered on zero
    vort_y_absmax = max(abs(vort_y_norm.min()), abs(vort_y_norm.max()))
    vort_y_lim = vort_y_absmax * (1 + padding_factor)
    
    vort_x_absmax = max(abs(vort_x_norm.min()), abs(vort_x_norm.max()))
    vort_x_lim = vort_x_absmax * (1 + padding_factor)
    
    # For velocity magnitude - centered on zero
    vel_y_absmax = max(abs(vel_mag_y_norm.min()), abs(vel_mag_y_norm.max()))
    vel_y_lim = vel_y_absmax * (1 + padding_factor)
    
    vel_x_absmax = max(abs(vel_mag_x_norm.min()), abs(vel_mag_x_norm.max()))
    vel_x_lim = vel_x_absmax * (1 + padding_factor)
    
    print(f"Subplot {i}: vort_y=[{-vort_y_lim:.3f}, {vort_y_lim:.3f}], " +
          f"vel_y=[{-vel_y_lim:.3f}, {vel_y_lim:.3f}]")
    
    # Get the x and y coordinate arrays
    if xxc.ndim == 2:
        x_coords = xxc[center_y, :]
        y_coords = yyc[:, center_x]
    else:
        x_coords = xxc
        y_coords = yyc
    
    # Left skinny subplot - shows y-transect (vorticity on left axis, velocity magnitude on right axis)
    ax_l = fig.add_axes([pos.x0 - skinny_w - gap, pos.y0, skinny_w, pos.height])
    ax_left.append(ax_l)
    ax_l.sharey(ax[i])
    
    # Plot normalized vorticity on left axis (solid line)
    ax_l.plot(vort_y_norm, y_coords, 'k:', linewidth=1.5, label='vorticity')
    # Add dashed line at zero
    ax_l.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.8)
    # Set limits centered on zero
    ax_l.set_xlim(-vort_y_lim, vort_y_lim)
    # Remove ticks on horizontal (short) axis for vorticity
    ax_l.set_xticks([])
    ax_l.tick_params(axis='y', labelright=False, right=False, left=True)
    ax_l.yaxis.set_label_position('left')
    ax_l.yaxis.tick_left()
    
    # Create twin axis for normalized velocity magnitude (dotted line)
    ax_l_vel = ax_l.twiny()
    ax_l_vel.plot(vel_mag_y_norm, y_coords, 'k-', linewidth=1.5, label='velocity mag')
    # Set limits centered on zero
    ax_l_vel.set_xlim(-vel_y_lim, vel_y_lim)
    # Remove ticks on horizontal (short) axis for velocity
    ax_l_vel.set_xticks([])
    # Remove ticks on shared y-axis
    ax_l_vel.tick_params(axis='y', labelright=False, right=False, labelleft=False, left=False)
    
    # Bottom skinny subplot - shows x-transect (vorticity on bottom axis, velocity magnitude on top axis)
    ax_b = fig.add_axes([pos.x0, pos.y0 - skinny_h - gap, pos.width, skinny_h])
    ax_bottom.append(ax_b)
    ax_b.sharex(ax[i])
    
    # Plot normalized vorticity on bottom axis (solid line)
    ax_b.plot(x_coords, vort_x_norm, 'k:', linewidth=1.5, label='vorticity')
    # Add dashed line at zero
    ax_b.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.8)
    # Set limits centered on zero
    ax_b.set_ylim(-vort_x_lim, vort_x_lim)
    # Remove ticks on vertical (short) axis for vorticity
    ax_b.set_yticks([])
    ax_b.tick_params(axis='x', labeltop=False, top=False, bottom=True)
    ax_b.xaxis.set_label_position('bottom')
    ax_b.xaxis.tick_bottom()
    
    # Create twin axis for normalized velocity magnitude (dotted line)
    ax_b_vel = ax_b.twinx()
    ax_b_vel.plot(x_coords, vel_mag_x_norm, 'k-', linewidth=1.5, label='velocity mag')
    # Set limits centered on zero
    ax_b_vel.set_ylim(-vel_x_lim, vel_x_lim)
    # Remove ticks on vertical (short) axis for velocity
    ax_b_vel.set_yticks([])
    # Remove ticks on shared x-axis
    ax_b_vel.tick_params(axis='x', labeltop=False, top=False, labelbottom=False, bottom=False)
    
# Add labels to appropriate skinny subplots
# Y-labels on left skinny subplots for first column (indices 0 and 3)
ax_left[0].set_ylabel(r'$y/D_\epsilon$', rotation=90, ha='right', va='center')
ax_left[3].set_ylabel(r'$y/D_\epsilon$', rotation=90, ha='right', va='center')
# X-labels on bottom skinny subplots for bottom row (indices 3, 4, 5)
ax_bottom[3].set_xlabel(r'$x/D_\epsilon$')
ax_bottom[4].set_xlabel(r'$x/D_\epsilon$')
ax_bottom[5].set_xlabel(r'$x/D_\epsilon$')
fig.savefig(os.path.join(plotdir, 'eddy-stats', 'grl-paper-figs', 'Figure2.png'), dpi=300)

 
############## FIGURE 3 ###################
bin_step = 0.5; bin_width = 1
x_bins = np.arange(37, 54.5-bin_step, bin_step)
x_bin_cent = x_bins + bin_step/2

ind5 = find_bin_ind(eddy_tracks_x5, x_bins)
ind10 = find_bin_ind(eddy_tracks_x10, x_bins)
ind40 = find_bin_ind(eddy_tracks_x40, x_bins)

sizes5 = find_cross_var(ind5, np.ravel(eddy_sizes5))
sizes10 = find_cross_var(ind10, np.ravel(eddy_sizes10))
sizes40 = find_cross_var(ind40, np.ravel(eddy_sizes40))

circs5 = find_cross_var(ind5, np.ravel(eddy_circs5))
circs10 = find_cross_var(ind10, np.ravel(eddy_circs10))
circs40 = find_cross_var(ind40, np.ravel(eddy_circs40))

xbins5, Neddies5, sizes5, circs5 =  cross_var(dat5)
xbins10, Neddies10, sizes10, circs10 =  cross_var(dat10)
xbins40, Neddies40, sizes40, circs40 =  cross_var(dat40)

h = [np.mean(dep[0,np.argmin(np.abs(x-xbins5[i]))]) for i in range(len(xbins5))]

### rotational velocity 
eddy_len5 = np.sqrt(eddy_sizes5/np.pi)*2
eddy_U5 = eddy_circs5/(eddy_len5*np.pi)

eddy_len10 = np.sqrt(eddy_sizes10/np.pi)*2
eddy_U10 = eddy_circs10/(eddy_len10*np.pi)

eddy_len40 = np.sqrt(eddy_sizes40/np.pi)*2
eddy_U40 = eddy_circs40/(eddy_len40*np.pi) 

### speed of eddy 
dist5 = np.sqrt((eddy_tracks_x5[:,1:] - eddy_tracks_x5[:,:-1])**2 + (eddy_tracks_y5[:,1:] - eddy_tracks_y5[:,:-1])**2)
speed5 = dist5/dt

dist10 = np.sqrt((eddy_tracks_x10[:,1:] - eddy_tracks_x10[:,:-1])**2 + (eddy_tracks_y10[:,1:] - eddy_tracks_y10[:,:-1])**2)
speed10 = dist10/dt

dist40 = np.sqrt((eddy_tracks_x40[:,1:] - eddy_tracks_x40[:,:-1])**2 + (eddy_tracks_y40[:,1:] - eddy_tracks_y40[:,:-1])**2)
speed40 = dist40/dt

nonlin_cross5 = []
speed_cross5 = []
U_cross5 = []
nonlin_cross10 = []
speed_cross10 = []
U_cross10 = []
nonlin_cross40 = []
speed_cross40 = []
U_cross40 = []
for i in range(1,len(xbins5)):
    tmp_x_ind = np.where((np.ravel(eddy_tracks_x5[:,:-1])>=xbins5[i-1])&(np.ravel(eddy_tracks_x5[:,:-1])<xbins5[i] + bin_width))[0] 
    tmp_U = np.ravel(eddy_U5[:,:-1])[tmp_x_ind] 
    tmp_speed = np.ravel(speed5)[tmp_x_ind]
    sind = np.where(tmp_speed<5)[0]
    nonlin_cross5.append(np.nanmedian(np.abs(tmp_U[sind])/tmp_speed[sind]))
    speed_cross5.append(np.nanmedian(tmp_speed[sind]))
    U_cross5.append(np.nanmedian(np.abs(tmp_U[sind])))
    ###
    tmp_x_ind = np.where((np.ravel(eddy_tracks_x10[:,:-1])>=xbins10[i-1])&(np.ravel(eddy_tracks_x10[:,:-1])<xbins10[i] + bin_width))[0] 
    tmp_U = np.ravel(eddy_U10[:,:-1])[tmp_x_ind] 
    tmp_speed = np.ravel(speed10)[tmp_x_ind]
    sind = np.where(tmp_speed<5)[0]
    nonlin_cross10.append(np.nanmedian(np.abs(tmp_U[sind])/tmp_speed[sind]))
    speed_cross10.append(np.nanmedian(tmp_speed[sind]))
    U_cross10.append(np.nanmedian(np.abs(tmp_U[sind])))
    ###
    tmp_x_ind = np.where((np.ravel(eddy_tracks_x40[:,:-1])>=xbins40[i-1])&(np.ravel(eddy_tracks_x40[:,:-1])<xbins40[i] + bin_width))[0] 
    tmp_U = np.ravel(eddy_U40[:,:-1])[tmp_x_ind] 
    tmp_speed = np.ravel(speed40)[tmp_x_ind]
    sind = np.where(tmp_speed<5)[0]
    nonlin_cross40.append(np.nanmedian(np.abs(tmp_U[sind])/tmp_speed[sind]))
    speed_cross40.append(np.nanmedian(tmp_speed[sind]))
    U_cross40.append(np.nanmedian(np.abs(tmp_U[sind])))

maj5 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_maj5.npy')
min5 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_min5.npy')
maj10 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_maj10.npy')
min10 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_min10.npy')
maj40 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_maj40.npy')
min40 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_min40.npy')
theta5 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_theta5.npy')
theta10 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_theta10.npy')
theta40 = np.load('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/maj_min/elip_theta40.npy')

xbins5, tmp_ind5 = find_bins(dat5['x'].values)
majmin_cross5 = find_cross_var(tmp_ind5, maj5/min5)
xbins10, tmp_ind10 = find_bins(dat10['x'].values)
majmin_cross10 = find_cross_var(tmp_ind10, maj10/min10)
xbins40, tmp_ind40 = find_bins(dat40['x'].values)
majmin_cross40 = find_cross_var(tmp_ind40, maj40/min40)

color1 = '#003f5c'; color2 = '#bc5090'; color3 = '#ffa600'
whis = 3
Wsz = shore - sz_edge
x = xx[0,:]+lab_offset

fig, ax = plt.subplots(figsize=(6,9), nrows=6)
ymin = 0.08
h1 = 0.13
off = 0.02
w = 0.6
w1 = 0.75 
w2 = 0.15
pos1 = 1; pos2 = 3; pos3 = 5; widths = 0.8; xlim1 = -0.2; xlim2 = 6.2
tmp = np.argmin(np.abs(x-sz_edge))
xmin = 0.15

ax[0].set_position([xmin, ymin+(4*h1)+(5*off), w1, h1])
ax[1].set_position([xmin, ymin+(3*h1)+(4*off), w1, h1])
ax[2].set_position([xmin, ymin+(2*h1)+(3*off), w1, h1])
ax[3].set_position([xmin, ymin+(1*h1)+(2*off), w1, h1])
ax[4].set_position([xmin, ymin+(0*h1)+(1*off), w1, h1])
ax[5].set_position([xmin, ymin+(0*h1)+(0*off), w1, h1])

# Subplot 0
ax0 = ax[0].twinx() 
ax0.set_position([xmin, ymin+(5*h1)+(5*off), w1, h1])
ax0.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax0.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax0.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax0.set_ylim(-0.24, 0.5)
ax0.set_yticks([-0.24])
ax0.set_yticklabels([''])
ax0.patch.set_visible(False)
ax0.set_zorder(ax[0].get_zorder() - 1)

ax[0].patch.set_visible(False)
ax[0].plot(x_bin_cent-lab_offset, Neddies5, color=color1, linestyle='dotted', 
           linewidth=lwidth+1, zorder=3)
ax[0].plot(x_bin_cent-lab_offset, Neddies10, color=color2, linestyle='dashed', 
           linewidth=lwidth, zorder=3)
ax[0].plot(x_bin_cent-lab_offset, Neddies40, color=color3, linestyle='solid', 
           linewidth=lwidth, zorder=3)
ax[0].set_ylabel(r'$N_{\epsilon}$')
ax[0].grid(True)
custom_lines = [Line2D([0], [0], color=color1, lw=lwidth, linestyle='dotted'),
                Line2D([0], [0], color=color2, lw=lwidth, linestyle='dashed'),
                Line2D([0], [0], color=color3, lw=lwidth, linestyle='solid')]
ax[0].legend(custom_lines, [r'$\sigma_\theta = 3.8^\circ$', r'$\sigma_\theta = 11.2^\circ$', 
             r'$\sigma_\theta = 24.9^\circ$'], fontsize=fsize-2, loc='upper left')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[0].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax[0].text(30.5, 8.6, r'$\mathrm{(a)}$', fontsize=fsize)
ax[0].set_ylim(-3,10)
ax[0].set_yticks([-2, 0,2,4,6,8,10])
ax[0].set_yticklabels(['', r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$'])
ax[0].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[0].set_xlim(15, 32.5)
ax[0].set_xticklabels(['', '', '', '', '', '', '', '', ''])

# Subplot 1
ax1 = ax[1].twinx() 
ax1.set_position([xmin, ymin+(4*h1)+(4*off), w1, h1])
ax1.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax1.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax1.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax1.set_ylim(-0.24, 0.5)
ax1.set_yticks([-0.24])
ax1.set_yticklabels([''])
ax1.patch.set_visible(False)
ax1.set_zorder(ax[1].get_zorder() - 1)

ax[1].patch.set_visible(False)
ax[1].plot(x_bin_cent-lab_offset, np.sqrt(sizes5/np.pi)*2/Wsz, color=color1, 
           linestyle='dotted', linewidth=lwidth+1, zorder=3)
ax[1].plot(x_bin_cent-lab_offset, np.sqrt(sizes10/np.pi)*2/Wsz, color=color2, 
           linestyle='dashed', linewidth=lwidth, zorder=3)
ax[1].plot(x_bin_cent-lab_offset, np.sqrt(sizes40/np.pi)*2/Wsz, color=color3, 
           linestyle='solid', linewidth=lwidth, zorder=3)
ax[1].set_ylabel(r'$D_\epsilon / W_\mathrm{sz}$')
ax[1].set_ylim((0,0.5))
ax[1].grid(True)
formatter = ticker.ScalarFormatter(useMathText=True)
ax[1].yaxis.set_major_formatter(formatter)
ax[1].text(30.5, 0.4, r'$\mathrm{(b)}$', fontsize=fsize)
ax[1].set_yticks([0,0.1,0.2,0.3,0.4,0.5])
ax[1].set_yticklabels([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'])
ax[1].set_xlim(15, 32.5)
ax[1].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[1].set_xticklabels(['', '', '', '', '', '', '', '', ''])

# Subplot 2
ax2 = ax[2].twinx() 
ax2.set_position([xmin, ymin+(3*h1)+(3*off), w1, h1])
ax2.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax2.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax2.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax2.set_ylim(-0.24, 0.5)
ax2.set_yticks([-0.24])
ax2.set_yticklabels([''])
ax2.patch.set_visible(False)
ax2.set_zorder(ax[2].get_zorder() - 1)

ax[2].patch.set_visible(False)
ax[2].plot(x_bin_cent-lab_offset, majmin_cross5, color=color1, linestyle='dotted', 
           linewidth=lwidth+1, zorder=3)
ax[2].plot(x_bin_cent-lab_offset, majmin_cross10, color=color2, linestyle='dashed', 
           linewidth=lwidth, zorder=3)
ax[2].plot(x_bin_cent-lab_offset, majmin_cross40, color=color3, linestyle='solid', 
           linewidth=lwidth, zorder=3)
ax[2].set_ylabel(r'$\alpha_\epsilon$')
ax[2].set_ylim((1,4))
ax[2].grid(True)
formatter = ticker.ScalarFormatter(useMathText=True)
ax[2].yaxis.set_major_formatter(formatter)
ax[2].text(30.5, 3.5, r'$\mathrm{(c)}$', fontsize=fsize)
ax[2].set_yticks([1,2,3,4])
ax[2].set_yticklabels([r'$1$', r'$2$', r'$3$', r'$4$'])
ax[2].set_xlim(15, 32.5)
ax[2].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[2].set_xticklabels(['', '', '', '', '', '', '', '', ''])

# Subplot 3
ax3 = ax[3].twinx() 
ax3.set_position([xmin, ymin+(2*h1)+(2*off), w1, h1])
ax3.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax3.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax3.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax3.set_ylim(-0.24, 0.5)
ax3.set_yticks([-0.24])
ax3.set_yticklabels([''])
ax3.patch.set_visible(False)
ax3.set_zorder(ax[3].get_zorder() - 1)

ax[3].patch.set_visible(False)
ax[3].plot(x_bin_cent-lab_offset, circs5, color=color1, linestyle='dotted', 
           linewidth=lwidth+1, zorder=3)
ax[3].plot(x_bin_cent-lab_offset, circs10, color=color2, linestyle='dashed', 
           linewidth=lwidth, zorder=3)
ax[3].plot(x_bin_cent-lab_offset, circs40, color=color3, linestyle='solid', 
           linewidth=lwidth, zorder=3)
ax[3].set_ylabel(r'$\Gamma_\epsilon$ $(\mathrm{m^2 s^{-1}})$')
ax[3].grid(True)
formatter = ticker.ScalarFormatter(useMathText=True)
ax[3].yaxis.set_major_formatter(formatter)
ax[3].set_xlim((15, 32.5))
ax[3].text(30.5, 0.65, r'$\mathrm{(d)}$', fontsize=fsize)
ax[3].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax[3].set_yticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'])
ax[3].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[3].set_xticklabels(['', '', '', '', '', '', '', '', ''])

# Subplot 4
ax4 = ax[4].twinx() 
ax4.set_position([xmin, ymin+(1*h1)+(1*off), w1, h1])
ax4.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax4.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax4.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax4.set_ylim(-0.24, 0.5)
ax4.set_yticks([-0.24])
ax4.set_yticklabels([''])
ax4.patch.set_visible(False)
ax4.set_zorder(ax[4].get_zorder() - 1)

ax[4].patch.set_visible(False)
ax[4].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], speed_cross5, 
           color=color1, linestyle='dotted', linewidth=lwidth+1, zorder=3)
ax[4].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], speed_cross10, 
           color=color2, linestyle='dashed', linewidth=lwidth, zorder=3)
ax[4].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], speed_cross40, 
           color=color3, linestyle='solid', linewidth=lwidth, zorder=3)
ax[4].set_ylabel(r'$c_\epsilon$ $\mathrm{(m s^{-1})}$')
ax[4].grid(True)
formatter = ticker.ScalarFormatter(useMathText=True)
ax[4].yaxis.set_major_formatter(formatter)
ax[4].set_xlim((15, 32.5))
ax[4].set_ylim((0, 0.4))
ax[4].text(30.5, 0.35, r'$\mathrm{(e)}$', fontsize=fsize)
ax[4].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax[4].set_yticklabels([r'$0.0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$'])
ax[4].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[4].set_xticklabels(['', '', '', '', '', '', '', '', ''])

# Subplot 5
ax5 = ax[5].twinx() 
ax5.set_position([xmin, ymin+(0*h1)+(0*off), w1, h1])
ax5.plot(x-lab_offset, -dep[0,:]/6-0.04, color='black', linewidth=lwidth, zorder=1)
ax5.fill_between(x[tmp:]-lab_offset, np.ones(len(dep[0,tmp:]))*0.5, -dep[0,tmp:]/6-0.04, 
                 color='tab:grey', alpha=0.3, zorder=0)
ax5.fill_between(x-lab_offset, -dep[0,:]/6-0.04, np.ones(len(dep[0,:]))*(-1.2)/6-0.04, 
                 color='black', alpha=0.3, zorder=0)
ax5.set_ylim(-0.24, 0.5)
ax5.set_yticks([-0.24])
ax5.set_yticklabels([''])
ax5.patch.set_visible(False)
ax5.set_zorder(ax[5].get_zorder() - 1)

ax[5].patch.set_visible(False)
ax[5].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], nonlin_cross5, 
           color=color1, linestyle='dotted', linewidth=lwidth+1, zorder=3)
ax[5].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], nonlin_cross10, 
           color=color2, linestyle='dashed', linewidth=lwidth, zorder=3)
ax[5].plot(x_bin_cent[:-1]-lab_offset+x_bin_cent[1]-x_bin_cent[0], nonlin_cross40, 
           color=color3, linestyle='solid', linewidth=lwidth, zorder=3)
ax[5].set_ylabel(r'$I_c$')
ax[5].grid(True)
ax[5].set_xlabel(r'$x\ \mathrm{(m)}$')
formatter = ticker.ScalarFormatter(useMathText=True)
ax[5].yaxis.set_major_formatter(formatter)
ax[5].set_xlim((15, 32.5))
ax[5].set_ylim((0.2, 1.4))
ax[5].text(30.5, 1.25, r'$\mathrm{(f)}$', fontsize=fsize)
ax[5].set_yticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
ax[5].set_yticklabels([r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$', r'$1.2$', r'$1.4$'])
ax[5].set_xticks([16, 18, 20, 22, 24, 26, 28, 30, 32])
ax[5].set_xticklabels([r'$16$', r'$18$', r'$20$', r'$22$', r'$24$', r'$26$', r'$28$', r'$30$', r'$32$'])

fig.savefig(os.path.join(plotdir, 'eddy-stats', 'grl-paper-figs', 'Figure3.png'), dpi=300)


############# FIGURE 4 ###################
Ne_sz_edge5, size_sz_edge5, circ_sz_edge5 = summary_stats(dat5, sz_edge, width)
Ne_sz_edge10, size_sz_edge10, circ_sz_edge10 = summary_stats(dat10, sz_edge, width)
Ne_sz_edge40, size_sz_edge40, circ_sz_edge40 = summary_stats(dat40, sz_edge, width)

Ne_off = np.array([np.median(Ne_sz_edge5), np.median(Ne_sz_edge10), np.median(Ne_sz_edge40)]) / (width*2)
Ne_off_std = np.array([np.std(Ne_sz_edge5/(width*2)), np.std(Ne_sz_edge10/(width*2)), np.std(Ne_sz_edge40/(width*2))]) 

circ_off = np.array([np.median(circ_sz_edge5), np.median(circ_sz_edge10), np.median(circ_sz_edge40)])
circ_off_std = np.array([np.std(circ_sz_edge5), np.std(circ_sz_edge10), np.std(circ_sz_edge40)])

## uex from JGR paper
u_ex = np.array([0.00091303, 0.01969376, 0.04097266, 0.03879731, 0.03618826, 0.03301637])
u_ex_std = np.array([9.53820303e-05, 2.40090179e-03, 6.27906415e-03, 4.70010331e-03, 2.31939267e-03, 2.11039398e-03])

dirspread = np.array([0.3, 3.8, 11.2, 16.5, 21.6, 24.9]) 
dirspread_sub = np.array([3.8, 11.2, 24.9])

indx5, indy5 = np.where((eddy_tracks_x5[:,:-1]>sz_edge-1) & (eddy_tracks_x5[:,:-1]<sz_edge+1))
indx10, indy10 = np.where((eddy_tracks_x10[:,:-1]>sz_edge-1) & (eddy_tracks_x10[:,:-1]<sz_edge+1))
indx40, indy40 = np.where((eddy_tracks_x40[:,:-1]>sz_edge-1) & (eddy_tracks_x40[:,:-1]<sz_edge+1))

T = int(5/dt)
xind = np.argmin(np.abs(x-sz_edge))

u_psi_off5 = ma.masked_where(u_psi5[:,:,xind]>=0, u_psi5[:,:,xind])
uex5 = compute_uex(u_psi_off5, dy) 
del u_psi_off5

u_psi_off10 = ma.masked_where(u_psi10[:,:,xind]>=0, u_psi10[:,:,xind])
uex10 = compute_uex(u_psi_off10, dy) 
del u_psi_off10

u_psi_off40 = ma.masked_where(u_psi40[:,:,xind]>=0, u_psi40[:,:,xind])
uex40 = compute_uex(u_psi_off40, dy) 
del u_psi_off40


speed_ts5 = np.nanmedian(np.abs(speed5[indx5]), axis=0)
speed_ts10 = np.nanmedian(np.abs(speed10[indx10]), axis=0)
speed_ts40 = np.nanmedian(np.abs(speed40[indx40]), axis=0)

u5_ts5 = np.nanmedian(np.abs(eddy_U5[indx5]), axis=0)
u10_ts10 = np.nanmedian(np.abs(eddy_U10[indx10]), axis=0)
u40_ts40 = np.nanmedian(np.abs(eddy_U40[indx40]), axis=0)


from scipy.stats import gaussian_kde, spearmanr
import matplotlib.lines as mlines

fig, ax = plt.subplots(ncols=2, figsize=(6,3.6))
msize = 5; nbins = 25; alpha = 0.1

# Function to create density contours (without labels)
def add_density_contours(ax, x, y, color, levels=3, alpha_contour=0.8):
    # Convert to numpy arrays and flatten
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Remove NaN and infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:  # Need minimum points for KDE
        print(f"Warning: Only {len(x_clean)} valid points, skipping contours")
        return
    
    try:
        # Create KDE
        xy = np.vstack([x_clean, y_clean])
        kde = gaussian_kde(xy)
        
        # Create grid for contour plot
        x_min, x_max = x_clean.min(), x_clean.max()
        y_min, y_max = y_clean.min(), y_clean.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Handle case where range is zero
        if x_range == 0:
            x_range = 0.1 * abs(x_min) if x_min != 0 else 0.1
        if y_range == 0:
            y_range = 0.1 * abs(y_min) if y_min != 0 else 0.1
        
        # Extend grid slightly beyond data range
        x_grid = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 50)
        y_grid = np.linspace(y_min - 0.1*y_range, y_max + 0.1*y_range, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        # Convert to density percentage
        Z_relative = (Z / Z.max()) * 100  # Percentage of maximum density
        
        # Create BLACK contours without labels, original thickness
        contours = ax.contour(X, Y, Z_relative, levels=levels, colors='white',
                            alpha=alpha_contour, linewidths=2)
        
    except Exception as e:
        print(f"Error creating contours: {e}")

def add_quantile_regression(ax, x, y, q=0.15, line_color='k', line_style='--', line_width=2):
    """
    Add a robust linear quantile-regression fit and return fit statistics.
    Lower q values track the low end of the data cloud.
    """
    # Local imports keep this helper runnable even when only this block is executed.
    from scipy.optimize import linprog
    from scipy.stats import spearmanr

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return np.nan, np.nan, np.nan, np.nan, 0

    n = len(x_clean)

    # Solve quantile regression with linear programming:
    # min q*sum(u) + (1-q)*sum(v)
    # s.t. y - (b0 + b1*x) = u - v, u>=0, v>=0
    c = np.concatenate([
        np.array([0.0, 0.0]),
        np.full(n, q),
        np.full(n, 1.0 - q)
    ])

    A_eq = np.zeros((n, 2 + 2 * n))
    A_eq[:, 0] = 1.0
    A_eq[:, 1] = x_clean
    A_eq[:, 2:2 + n] = np.eye(n)
    A_eq[:, 2 + n:] = -np.eye(n)
    b_eq = y_clean

    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return np.nan, np.nan, np.nan, np.nan, 0

    intercept = res.x[0]
    slope = res.x[1]

    xfit = np.linspace(np.min(x_clean), np.max(x_clean), 200)
    yfit = intercept + slope * xfit
    ax.plot(xfit, yfit, color=line_color, linestyle=line_style, linewidth=line_width)

    yhat = intercept + slope * x_clean
    fit_corr = np.corrcoef(y_clean, yhat)[0, 1]
    rho, _ = spearmanr(x_clean, y_clean)
    return slope, intercept, fit_corr, rho, len(x_clean)

# Left subplot - Plot each scatter with its contours immediately after (NO LABELS)
ax[0].plot(eddy_count_sz5, np.abs(uex5), 'o', color=color1, markersize=msize, markeredgecolor=color1, alpha=alpha)
add_density_contours(ax[0], eddy_count_sz5, np.abs(uex5), color1)

ax[0].plot(eddy_count_sz10, np.abs(uex10), 'o', color=color2, markersize=msize, markeredgecolor=color2, alpha=alpha)
add_density_contours(ax[0], eddy_count_sz10, np.abs(uex10), color2)

ax[0].plot(eddy_count_sz40, np.abs(uex40), 'o', color=color3, markersize=msize, markeredgecolor=color3, alpha=alpha)
add_density_contours(ax[0], eddy_count_sz40, np.abs(uex40), color3)

ax[0].plot(np.median(eddy_count_sz5), np.median(np.abs(uex5)), 's', color=color1, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)
ax[0].plot(np.median(eddy_count_sz10), np.median(np.abs(uex10)), 's', color=color2, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)
ax[0].plot(np.median(eddy_count_sz40), np.median(np.abs(uex40)), 's', color=color3, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)

# Create manual legend with less transparent points
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color1, 
                  markeredgecolor=color1, markersize=msize, alpha=0.8, 
                  label=r'$\sigma_\theta = 3.8^\circ$', linestyle='None'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color2, 
                  markeredgecolor=color2, markersize=msize, alpha=0.8, 
                  label=r'$\sigma_\theta = 11.2^\circ$', linestyle='None'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color3, 
                  markeredgecolor=color3, markersize=msize, alpha=0.8, 
                  label=r'$\sigma_\theta = 24.9^\circ$', linestyle='None')
]

ax[0].grid(True)
ax[0].set_ylabel(r'$U_{ex}$ $\mathrm{(m\ s^{-1})}$')
ax[0].set_xlabel(r'$N_\epsilon$')
ax[0].set_yticks([0, 0.02, 0.04, 0.06])
ax[0].set_yticklabels([r'$0$', r'$0.02$', r'$0.04$', r'$0.06$'])
ax[0].set_xticks([0, 5, 10, 15, 20, 25])
ax[0].set_xticklabels([r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$'])
ax[0].text(22, 0.055, r'$(a)$')
ax[0].legend(handles=legend_elements, loc='best', fontsize=10)

# Right subplot - Plot each scatter with its contours immediately after (NO LABELS)
ax[1].plot(u5_ts5, np.abs(uex5[:-1]), 'o', color=color1, markersize=msize, markeredgecolor=color1, alpha=alpha)
add_density_contours(ax[1], u5_ts5, np.abs(uex5[:-1]), color1)

ax[1].plot(u10_ts10, np.abs(uex10[:-1]), 'o', color=color2, markersize=msize, markeredgecolor=color2, alpha=alpha)
add_density_contours(ax[1], u10_ts10, np.abs(uex10[:-1]), color2)

ax[1].plot(u40_ts40, np.abs(uex40[:-1]), 'o', color=color3, markersize=msize, markeredgecolor=color3, alpha=alpha)
add_density_contours(ax[1], u40_ts40, np.abs(uex40[:-1]), color3)

ax[1].plot(np.median(u5_ts5[np.isfinite(u5_ts5)]), np.median(np.abs(uex5[:-1][np.isfinite(u5_ts5)])), 's', color=color1, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)
ax[1].plot(np.median(u10_ts10), np.median(np.abs(uex10[:-1])), 's', color=color2, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)
ax[1].plot(np.median(u40_ts40), np.median(np.abs(uex40[:-1])), 's', color=color3, markersize=msize+3, markeredgecolor='k', markeredgewidth=1.5)

ax[1].grid(True)
ax[1].set_xlabel(r'$|U_\epsilon|$ $\mathrm{(m s^{-1})}$')
ax[1].set_yticks([0, 0.02, 0.04, 0.06])
ax[1].set_yticklabels(['', '', '', ''])
ax[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[1].set_xticklabels([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$'])
ax[1].text(0.35, 0.055, r'$(b)$')

fig.tight_layout()
fig.savefig(os.path.join(plotdir, 'eddy-stats', 'grl-paper-figs', 'Figure4.png'), dpi=300)
