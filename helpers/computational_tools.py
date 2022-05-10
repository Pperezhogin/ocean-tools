import numpy as np
import math
import xrft
import numpy.fft as npfft
from scipy import signal
import xarray as xr
import os

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh'
    as xarray
    '''
    try:
        coord = array.xq
    except:
        coord = array.xh
    return coord

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    try:
        coord = array.yq
    except:
        coord = array.yh
    return coord

def rename_coordinates(xr_dataset):
    '''
    in-place change of coordinate names to Longitude and Latitude.
    For convenience of plotting with xarray.plot()
    '''
    for key in ['xq', 'xh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Longitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

    for key in ['yq', 'yh']:
        try:
            xr_dataset[key].attrs['long_name'] = 'Latitude'
            xr_dataset[key].attrs['units'] = ''
        except:
            pass

def select_LatLon(array, Lat, Lon):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def remesh(input, target):
    '''
    Input and target should be xarrays of any type (u-array, v-array, q-array, h-array).
    Datasets are prohibited.
    Horizontal mesh of input changes according to horizontal mesh of target.
    Other dimensions are unchanged!

    If type of arrays is different:
        - Interpolation to correct points occurs
    If input is Hi-res:
        - Coarsening with integer grain and subsequent interpolation to correct mesh if needed
    if input is Lo-res:
        - Interpolation to Hi-res mesh occurs

    Input and output Nan values are treates as zeros (see "fillna")
    '''

    # Define coordinates
    x_input  = x_coord(input)
    y_input  = y_coord(input)
    x_target = x_coord(target)
    y_target = y_coord(target)

    # ratio of mesh steps
    ratiox = np.diff(x_target)[0] / np.diff(x_input)[0]
    ratiox = math.ceil(ratiox)

    ratioy = np.diff(y_target)[0] / np.diff(y_input)[0]
    ratioy = math.ceil(ratioy)
    
    # B.C.
    result = input.fillna(0)
    
    if (ratiox > 1 or ratioy > 1):
        # Coarsening; x_input.name returns 'xq' or 'xh'
        result = result.coarsen({x_input.name: ratiox, y_input.name: ratioy}, boundary='pad').mean()

    # Coordinate points could change after coarsegraining
    x_result = x_coord(result)
    y_result = y_coord(result)

    # Interpolate if needed
    if not x_result.equals(x_target) or not y_result.equals(y_target):
        result = result.interp({x_result.name: x_target, y_result.name: y_target}).fillna(0)

    # Remove unnecessary coordinates
    if x_target.name != x_input.name:
        result = result.drop_vars(x_input.name)
    if y_target.name != y_input.name:
        result = result.drop_vars(y_input.name)
    
    return result

def compute_isotropic_KE(u, v, dx, dy, Lat, Lon, window, nfactor, truncate, detrend, window_correction):
    '''
    u, v, dx, dy - arrays defined in the center of the cells
    dx, dy - grid spacing in metres
    Default options: window correction + linear detrending
    Output:
    mean(u^2+v^2)/2 = int(E(k),dk)
    This equality is expected for detrend=None, window='boxcar'
    freq_r - radial wavenumber, m^-1
    window = 'boxcar' or 'hann'
    '''
    # Select desired Lon-Lat square
    u = select_LatLon(u,Lat,Lon)
    v = select_LatLon(v,Lat,Lon)

    # mean grid spacing in metres
    dx = select_LatLon(dx,Lat,Lon).mean().values
    dy = select_LatLon(dy,Lat,Lon).mean().values

    # define uniform grid
    x = dx*np.arange(len(u.xh))
    y = dy*np.arange(len(u.yh))
    u['xh'] = x
    u['yh'] = y
    v['xh'] = x
    v['yh'] = y

    Eu = xrft.isotropic_power_spectrum(u, dim=('xh','yh'), window=window, nfactor=nfactor, truncate=truncate, detrend=detrend, window_correction=window_correction)
    Ev = xrft.isotropic_power_spectrum(v, dim=('xh','yh'), window=window, nfactor=nfactor, truncate=truncate, detrend=detrend, window_correction=window_correction)

    E = (Eu+Ev) / 2 # because power spectrum is twice the energy
    E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
    
    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2+v**2)/2).mean(dim=('Time', 'xh', 'yh')).values)
    #spacing = np.diff(E.freq_r).mean()
    #print('int(E(k),dk)=', (E.sum(dim='freq_r').mean(dim='Time') * spacing).values)
    #print(f'Max wavenumber={E.freq_r.max().values} [1/m], \n x-grid-scale={np.pi/dx} [1/m], \n y-grid-scale={np.pi/dy} [1/m]')
    
    return E

def compute_KE_time_spectrum(u, v, Lat, Lon, Time, window, nchunks, detrend, window_correction):
    '''
    Returns KE spectrum with normalization:
    mean(u^2+v^2)/2 = int(E(nu),dnu),
    where nu - time frequency in 1/day (not "angle frequency")
    E(nu) - energy density, i.e. m^2/s^2 * day
    '''

    # Select range of Lat-Lon-time
    u = select_LatLon(u,Lat,Lon).sel(Time=Time)
    v = select_LatLon(v,Lat,Lon).sel(Time=Time)

    # Let integer division by nchunks
    nTime = len(u.Time)
    chunk_length = math.floor(nTime / nchunks)
    nTime = chunk_length * nchunks

    # Divide time series to time chunks
    u = u.isel(Time=slice(nTime)).chunk({'Time': chunk_length})
    v = v.isel(Time=slice(nTime)).chunk({'Time': chunk_length})

    # compute spatial-average time spectrum
    ps_u = xrft.power_spectrum(u, dim='Time', window=window, window_correction=window_correction, detrend=detrend, chunks_to_segments=True).mean(dim=('xq','yh'))
    ps_v = xrft.power_spectrum(v, dim='Time', window=window, window_correction=window_correction, detrend=detrend, chunks_to_segments=True).mean(dim=('xh','yq'))

    ps = ps_u + ps_v

    # in case of nchunks > 1
    try:
        ps = ps.mean(dim='Time_segment')
    except:
        pass

    # Convert 2-sided power spectrum to one-sided
    ps = ps[ps.freq_Time>=0]
    freq = ps.freq_Time
    ps[freq==0] = ps[freq==0] / 2

    # Drop zero frequency for better plotting
    ps = ps[ps.freq_Time>0]

    ############## normalization tester #############
    #print('Energy balance:')
    #print('mean(u^2+v^2)/2=', ((u**2)/2).mean(dim=('Time', 'xq', 'yh')).values + ((v**2)/2).mean(dim=('Time', 'xh', 'yq')).values)
    #print('int(E(nu),dnu)=', (ps.sum(dim='freq_Time') * ps.freq_Time.spacing).values)
    #spacing = np.diff(u.Time).mean()
    #print(f'Minimum period {2*spacing} [days]')
    #print(f'Max frequency={ps.freq_Time.max().values} [1/day], \n Max inverse period={0.5/spacing} [1/day]')

    return ps

def mass_average(KE, h, dx, dy):
    return (KE*h*dx*dy).mean(dim=('xh', 'yh')) / (h*dx*dy).mean(dim=('xh', 'yh'))