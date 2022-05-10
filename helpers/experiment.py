import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property
from helpers.computational_tools import rename_coordinates, remesh, compute_isotropic_KE, compute_KE_time_spectrum
from helpers.netcdf_cache import netcdf_property

Averaging_Time = slice(3650,7300)

class Experiment:
    '''
    Imitates xarray. All variables are
    returned as @property. Compared to xarray, allows
    additional computational tools and initialized instantly (within ms)
    '''
    def __init__(self, folder, key=''):
        '''
        Initializes with folder containing all netcdf files corresponding
        to a given experiment.
        Xarray datasets are read only by demand within @property decorator
        @cached_property allows to read each netcdf file only ones

        All fields needed for plotting purposes are suggested to be
        registered with @cached_property decorator (for convenience)
        '''
        self.folder = folder
        self.key = key # for storage of statistics
        self.recompute = False # default value of recomputing of cached on disk properties

        if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
            print('Error, cannot find files in folder'+self.folder)

    def remesh(self, target, key, compute=False):
        '''
        Returns object "experiment", where "Main variables"
        are coarsegrained according to resolution of the target experiment
        '''

        # The coarsegrained experiment is no longer attached to the folder
        result = Experiment(folder=self.folder, key=key)

        # Coarsegrain "Main variables" explicitly
        for key in ['RV', 'RV_f', 'PV', 'e', 'h', 'u', 'v', 'ua', 'va', 'ea']:
            if compute:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)).compute())
            else:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)))

        result.param = target.param # copy coordinates from target experiment

        return result        
    
    ################### Getters for netcdf files as xarrays #####################
    @cached_property
    def series(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean.stats.nc'), decode_times=False)
        return result

    @cached_property
    def param(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean_geometry.nc')).rename(
                {'latq': 'yq', 'lonq': 'xq', 'lath': 'yh', 'lonh': 'xh'} # change coordinates notation as in other files
            )
        rename_coordinates(result)
        return result

    @cached_property
    def prog(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'prog_*.nc'), decode_times=False, concat_dim='Time', parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result

    @cached_property
    def energy(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'energy_*.nc'), decode_times=False, concat_dim='Time', parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result

    @cached_property
    def ave(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'ave_*.nc'), decode_times=False, concat_dim='Time', parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result
    
    ############################### Main variables  #########################
    # These variables are suggested to be used in computations, i.e. coarsegraining,
    # spectra and so on. They are recommended to be registered
    # The use of @property decorator allows not to load data at initialization
    # They must be registered in "coarsegrain" method
    @cached_property
    def RV(self):
        return self.prog.RV

    @cached_property
    def RV_f(self):
        return self.RV / self.param.f

    @cached_property
    def PV(self):
        return self.prog.PV

    @cached_property
    def e(self):
        return self.prog.e

    @cached_property
    def h(self):
        return self.prog.h

    @cached_property
    def u(self):
        return self.prog.u

    @cached_property
    def v(self):
        return self.prog.v

    @cached_property
    def ua(self):
        return self.ave.u

    @cached_property
    def va(self):
        return self.ave.v

    @cached_property
    def ea(self):
        return self.ave.e

    ######################## Auxiliary variables #########################
    

    ########################  Statistical tools  #########################
    #-----------------------  Spectral analysis  ------------------------#
    def compute_KE_spectrum(self, Lat=(35,45), Lon=(5,15), window='hann', nfactor=2, truncate=True, detrend='linear', window_correction=True):
        u = remesh(self.u, self.e)
        v = remesh(self.v, self.e)
        dx = self.param.dxT
        dy = self.param.dyT
        return compute_isotropic_KE(u, v, dx, dy, Lat, Lon, window, nfactor, truncate, detrend, window_correction)

    def compute_KE_time_spectrum(self, Lat=(35,45), Lon=(5,15), Time=Averaging_Time, window='hann', nchunks=4, detrend='linear', window_correction=True):
        return compute_KE_time_spectrum(self.ua, self.va, Lat, Lon, Time, window, nchunks, detrend, window_correction)

    @netcdf_property
    def KE_spectrum(self):
        return self.compute_KE_spectrum()

    @netcdf_property
    def KE_spectrum_global(self):
        return self.compute_KE_spectrum(Lat=(30,50), Lon=(0,22))

    @netcdf_property
    def KE_spectrum_mean(self):
        return self.KE_spectrum.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def KE_spectrum_global_mean(self):
        return self.KE_spectrum_global.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def KE_time_spectrum(self):
        return self.compute_KE_time_spectrum(nchunks=2)

    #-------------------  Mean flow and variability  --------------------#
    @netcdf_property
    def ssh_mean(self):
        return self.ea.isel(zi=0).sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def ssh_var(self):
        return self.ea.isel(zi=0).sel(Time=Averaging_Time).var(dim='Time')

    @netcdf_property
    def u_mean(self):
        return self.ua.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def v_mean(self):
        return self.va.sel(Time=Averaging_Time).mean(dim='Time')

    #-------------------------  KE, MKE, EKE  ---------------------------#        
    @netcdf_property
    def KE(self):
        return 0.5 * (remesh(self.u**2, self.e) + remesh(self.v**2, self.e))

    @netcdf_property
    def KE_series(self):
        return (self.KE*self.h).mean(dim=('xh','yh')) / self.h.mean(dim=('xh','yh'))

    @netcdf_property
    def MKE(self):
        return 0.5 * (remesh(self.u_mean**2, self.e) + remesh(self.v_mean**2, self.e))  

    @netcdf_property
    def EKE(self):
        eke = self.KE.sel(Time=Averaging_Time).mean(dim='Time') - self.MKE
        eke = eke.where(eke>0).fillna(0) # 
        return eke

    @property
    def EKE_old(self):
        try:
            eke = self.energy.KE.sel(Time=Averaging_Time).mean(dim='Time') - self.MKE
            eke = eke.where(eke>0).fillna(0) #
        except:
            eke = self.EKE
        return eke

    @netcdf_property
    def KE_mean(self):
        return self.MKE+self.EKE