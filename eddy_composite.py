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
from scipy.stats import circmean 
from scipy.stats import circstd 
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
import multiprocessing
plt.style.use('classic')

dx = 0.05; dy = 0.1; dt = 0.2; lwidth = 2; fsize = 12; shore = 31.5+22; lab_offset = 22; sz_edge = 23.2 + 22; start = 1500
basedir = os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/compiled_output_hmo25_dir5_tp2_ntheta15/')
fdir = os.path.join(basedir, 'lab_netcdfs')
plotdir = os.path.join(basedir, 'plots')

def load_model_output(fdir, tstart):
    eddy_id = xr.open_mfdataset(os.path.join(fdir, 'eddy_id', 'eddy_track_map_all_averaged_4tp_*.nc'), combine='nested', concat_dim='time')['eddy_id']
    fbrx = xr.open_mfdataset(os.path.join(fdir, 'fbrx_*.nc'), combine='nested', concat_dim='time')['fbrx']
    fbry = xr.open_mfdataset(os.path.join(fdir, 'fbry_*.nc'), combine='nested', concat_dim='time')['fbry']    
    upsi = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')['u_psi']
    vpsi = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')['v_psi']
    nubrk = xr.open_mfdataset(os.path.join(fdir, 'nubrk_*.nc'), combine='nested', concat_dim='time')['nubrk']
    eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']

    x = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['x']
    y = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['y']
    x = x.values
    y = y.values

    curl_fbr = mod_utils.curl(fbrx, fbry, x[1]-x[0], y[1]-y[0])  
    vort = mod_utils.curl(upsi, vpsi, x[1]-x[0], y[1]-y[0])
    return  eddy_id, upsi[tstart:,:,:], vpsi[tstart:,:,:], vort[tstart:,:,:], curl_fbr[tstart:,:,:], nubrk[tstart:,:,:], eta[tstart:,:,:], x, y 

def interp_weights(xy, uv, d=2):
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def composite_eddy_var(t, eddy_id, var, x, y, xnew, ynew, Neddies_pertime, Nrad=2):
    Neddies = Neddies_pertime[t]

    varnews = np.zeros((len(xnew), len(ynew), Neddies-1))
    for Neddy in range(1,Neddies):
        eddy_indY, eddy_indX = np.where(eddy_id[t,:,:]==Neddy)
        eddyY = y[eddy_indY]
        eddyX = x[eddy_indX]
        eddyR = np.sqrt((len(eddyY)*dx*dy)/np.pi)

        Xc = np.mean(eddyX)
        Yc = np.mean(eddyY)

        Xcomp_max = Xc + eddyR*Nrad 
        Xcomp_min = Xc - eddyR*Nrad
        Ycomp_max = Yc + eddyR*Nrad 
        Ycomp_min = Yc - eddyR*Nrad 

        Xind_max = np.argmin(np.abs(x-Xcomp_max))
        Xind_min = np.argmin(np.abs(x-Xcomp_min))
        Yind_max = np.argmin(np.abs(y-Ycomp_max))
        Yind_min = np.argmin(np.abs(y-Ycomp_min))

        x_eddy = x[Xind_min:Xind_max]
        y_eddy = y[Yind_min:Yind_max]
        var_eddy = var[t,Yind_min:Yind_max,Xind_min:Xind_max] 

        xraw = (x_eddy-Xc) / eddyR  
        yraw = (y_eddy-Yc) / eddyR 
        [XRAW, YRAW] = np.meshgrid(xraw, yraw)

        ## Interpolate all the original data sets into the new, rotated coordinate system

        # Find the real values in the observed data sets
        [yinds,xinds] = np.where(~np.isnan(var_eddy))

        # Get the x and y coordinates of the real data points from the unrotated coordinates,
        # and create an xyold coordinate pair system of this data
        xyold = np.zeros((len(yinds),2))
        xyold[:,0] = YRAW[yinds,xinds]
        xyold[:,1] = XRAW[yinds,xinds]    

        # Create an intepolation matrix vertex points and weights
        [XNEW,YNEW] = np.meshgrid(xnew,ynew)
        XN = len(xnew)
        YN = len(ynew)
        xynew = np.zeros(((YN*XN),2))
        xynew[:,0] = YNEW.flatten()
        xynew[:,1] = XNEW.flatten()
        vtx, wts = interp_weights(xyold, xynew)

        # Interpolate the data for the chlorophyll anomaly data
        # Using the real data, interpolate onto the new, rotated coordinate system
        valuesi = interpolate(var_eddy[yinds,xinds], vtx, wts)

        # Reshape the interpolated values into an array
        varnew = valuesi.reshape(YN,XN)
        varnews[:,:,Neddy-1] = varnew 
    return varnews

eddy_id, upsi, vpsi, vort, curl_fbr, nubrk, eta, x, y = load_model_output(fdir, start)
Neddies_pertime = np.asarray([np.max(eddy_id[i,:,:]).values for i in range(len(eddy_id))]) 
Ncomp = np.sum(Neddies_pertime)-len(eddy_id)

max_iters = 150
num_cores = multiprocessing.cpu_count()

Nrad = 2; step = 0.1
xnew = np.arange(-Nrad, Nrad+step, step)
ynew = np.arange(-Nrad, Nrad+step, step)


#### curl fbr
eddy_composites = np.zeros((len(xnew), len(ynew), Ncomp)) 
eddy_times = np.zeros(Ncomp)

eddy_count = 0
for t in range(0, (max_iters*num_cores), num_cores):
    inputs = np.arange(t, t+num_cores)
    #print(inputs)
    results = Parallel(n_jobs=num_cores)(delayed(composite_eddy_var)(i, eddy_id, curl_fbr, x, y, xnew, ynew, Neddies_pertime, Nrad) for i in inputs)
    for i in range(len(inputs)):
        eddy_composites[:,:,eddy_count:eddy_count+results[i].shape[-1]] = results[i]
        #print(eddy_count, eddy_count+results[i].shape[-1])
        eddy_times[eddy_count:eddy_count+results[i].shape[-1]] = inputs[i]
        eddy_count += results[i].shape[-1]
    if np.mod(t,1000) == 0:
        print("On iteration: %d" % t)


dim = ["y", "x", "eddy_time"]
coords = [ynew, xnew, eddy_times]
dat = xr.DataArray(eddy_composites, coords, dims=dim, name='eddy_composites') 
dat.to_netcdf(os.path.join(fdir, 'eddy_comps', 'eddy_composites_curlfbr.nc'))

print("Done with Fbr")

#### eta
eta = eta.values 
eddy_composites = np.zeros((len(xnew), len(ynew), Ncomp)) 
eddy_times = np.zeros(Ncomp)

eddy_count = 0
for t in range(0, (max_iters*num_cores), num_cores):
    inputs = np.arange(t, t+num_cores)
    #print(inputs)
    results = Parallel(n_jobs=num_cores)(delayed(composite_eddy_var)(i, eddy_id, eta, x, y, xnew, ynew, Neddies_pertime, Nrad) for i in inputs)
    for i in range(len(inputs)):
        eddy_composites[:,:,eddy_count:eddy_count+results[i].shape[-1]] = results[i]
        #print(eddy_count, eddy_count+results[i].shape[-1])
        eddy_times[eddy_count:eddy_count+results[i].shape[-1]] = inputs[i]
        eddy_count += results[i].shape[-1]
    if np.mod(t,1000) == 0:
        print("On iteration: %d" % t)

dim = ["y", "x", "eddy_time"]
coords = [ynew, xnew, eddy_times]
dat = xr.DataArray(eddy_composites, coords, dims=dim, name='eddy_composites') 
dat.to_netcdf(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))

print("Done with Eta")

#### u
upsi = upsi.values 
eddy_composites = np.zeros((len(xnew), len(ynew), Ncomp)) 
eddy_times = np.zeros(Ncomp)

eddy_count = 0
for t in range(0, (max_iters*num_cores), num_cores):
    inputs = np.arange(t, t+num_cores)
    #print(inputs)
    results = Parallel(n_jobs=num_cores)(delayed(composite_eddy_var)(i, eddy_id, upsi, x, y, xnew, ynew, Neddies_pertime, Nrad) for i in inputs)
    for i in range(len(inputs)):
        eddy_composites[:,:,eddy_count:eddy_count+results[i].shape[-1]] = results[i]
        #print(eddy_count, eddy_count+results[i].shape[-1])
        eddy_times[eddy_count:eddy_count+results[i].shape[-1]] = inputs[i]
        eddy_count += results[i].shape[-1]
    if np.mod(t,1000) == 0:
        print("On iteration: %d" % t)

dim = ["y", "x", "eddy_time"]
coords = [ynew, xnew, eddy_times]
dat = xr.DataArray(eddy_composites, coords, dims=dim, name='eddy_composites') 
dat.to_netcdf(os.path.join(fdir, 'eddy_comps', 'eddy_composites_upsi.nc'))

print("Done with u")

#### v
vpsi = vpsi.values 
eddy_composites = np.zeros((len(xnew), len(ynew), Ncomp)) 
eddy_times = np.zeros(Ncomp)

eddy_count = 0
for t in range(0, (max_iters*num_cores), num_cores):
    inputs = np.arange(t, t+num_cores)
    #print(inputs)
    results = Parallel(n_jobs=num_cores)(delayed(composite_eddy_var)(i, eddy_id, vpsi, x, y, xnew, ynew, Neddies_pertime, Nrad) for i in inputs)
    for i in range(len(inputs)):
        eddy_composites[:,:,eddy_count:eddy_count+results[i].shape[-1]] = results[i]
        #print(eddy_count, eddy_count+results[i].shape[-1])
        eddy_times[eddy_count:eddy_count+results[i].shape[-1]] = inputs[i]
        eddy_count += results[i].shape[-1]
    if np.mod(t,1000) == 0:
        print("On iteration: %d" % t)

dim = ["y", "x", "eddy_time"]
coords = [ynew, xnew, eddy_times]
dat = xr.DataArray(eddy_composites, coords, dims=dim, name='eddy_composites') 
dat.to_netcdf(os.path.join(fdir, 'eddy_comps', 'eddy_composites_vpsi.nc'))

print("Done with v")


#### vort
eddy_composites = np.zeros((len(xnew), len(ynew), Ncomp)) 
eddy_times = np.zeros(Ncomp)

eddy_count = 0
for t in range(0, (max_iters*num_cores), num_cores):
    inputs = np.arange(t, t+num_cores)
    #print(inputs)
    results = Parallel(n_jobs=num_cores)(delayed(composite_eddy_var)(i, eddy_id, vort, x, y, xnew, ynew, Neddies_pertime, Nrad) for i in inputs)
    for i in range(len(inputs)):
        eddy_composites[:,:,eddy_count:eddy_count+results[i].shape[-1]] = results[i]
        #print(eddy_count, eddy_count+results[i].shape[-1])
        eddy_times[eddy_count:eddy_count+results[i].shape[-1]] = inputs[i]
        eddy_count += results[i].shape[-1]
    if np.mod(t,1000) == 0:
        print("On iteration: %d" % t)

dim = ["y", "x", "eddy_time"]
coords = [ynew, xnew, eddy_times]
dat = xr.DataArray(eddy_composites, coords, dims=dim, name='eddy_composites') 
dat.to_netcdf(os.path.join(fdir, 'eddy_comps', 'eddy_composites_vort.nc'))

print("Done with vort")

############# ANALYSIS #################
eddy_stats_file = os.path.join(fdir, 'eddy_id', 'eddy_stats.csv')
dat = pd.read_csv(eddy_stats_file)

fbr_comp = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_curlfbr.nc'))['eddy_composites']
eta_comp = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))['eddy_composites']
x = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))['x']
y = xr.open_dataset(os.path.join(fdir, 'eddy_comps', 'eddy_composites_eta.nc'))['y']
fbr_comp = fbr_comp.values 
eta_comp = eta_comp.values

xx, yy = np.meshgrid(x,y)


for e in range(fbr_comp.shape[-1]):
    fig, ax = plt.subplots(figsize=(5,4))
    p = ax.pcolormesh(xx, yy, eta_comp[:,:,e], cmap=cmo.balance)
    fig.colorbar(p, ax=ax)
    circle = plt.Circle( (0, 0), 1, fill = False)
    ax.add_artist(circle)
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)
    p.set_clim(-0.1,0.1)
    fig.savefig(os.path.join('/gscratch/nearshore/enuss/lab_runs_y550/postprocessing/plots/ml-testing', 'comp_eta', '%06d.png' % e))
    plt.close('all')


if 0:
    cmax = 5
    fig, ax = plt.subplots(ncols=3, figsize=(11,3))
    xx, yy = np.meshgrid(x_eddy, y_eddy)
    p0 = ax[0].pcolormesh(xx, yy, var_eddy, cmap=cmo.balance) 
    fig.colorbar(p0, ax=ax[0])
    p0.set_clim((-cmax,cmax))
    ax[0].set_ylim((np.min(yy), np.max(yy)))
    ax[0].set_xlim((np.min(xx), np.max(xx)))

    xx, yy = np.meshgrid(xraw, yraw)
    p1 = ax[1].pcolormesh(xx, yy, var_eddy, cmap=cmo.balance) 
    fig.colorbar(p1, ax=ax[1])
    p1.set_clim((-cmax,cmax))
    ax[1].set_ylim((-2, 2))
    ax[1].set_xlim((-2, 2))

    xx, yy = np.meshgrid(xnew, ynew)
    p2 = ax[2].pcolormesh(xx, yy, varnew, cmap=cmo.balance) 
    fig.colorbar(p2, ax=ax[2])
    p2.set_clim((-cmax,cmax))
    ax[2].set_ylim((-2, 2))
    ax[2].set_xlim((-2, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'interp_test.png'))