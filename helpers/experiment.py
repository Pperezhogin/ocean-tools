import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property
from helpers.computational_tools import remesh, compute_2dfft, rename_coordinates, compute_isotropic_KE
from helpers.netcdf_cache import netcdf_property

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
        for key in ['RV', 'RV_f', 'PV', 'e', 'u', 'v']:
            if compute:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)).compute())
            else:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)))

        result.param = target.param # copy coordinates from target experiment

        return result        
    
    ################### Getters for netcdf files as xarrays #####################
    @cached_property
    def param(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean_geometry.nc')).rename(
                {'latq': 'yq', 'lonq': 'xq', 'lath': 'yh', 'lonh': 'xh'} # change coordinates notation as in other files
            )
        rename_coordinates(result)
        return result

    @cached_property
    def prog(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'prog_*.nc'), decode_times=False, concat_dim='Time', parallel=True, chunks={'Time': 1, 'zl': 1})
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
    def u(self):
        return self.prog.u

    @cached_property
    def v(self):
        return self.prog.v

    ############## Computational tools. Spectra, and so on #################
    @netcdf_property
    def KE(self):
        return 0.5 * (remesh(self.u**2, self.e) + remesh(self.v**2, self.e))

    #@netcdf_property
    @property
    def KE_spectrum(self):
        u = remesh(self.u, self.e)
        v = remesh(self.v, self.e)
        dx = self.param.dxT
        dy = self.param.dyT
        return compute_isotropic_KE(u, v, dx, dy, window='hann', Lat=(35,45), Lon=(5,15), nfactor=2, truncate=False)
        #return 2*np.pi*xrft.isotropize((np.abs(self.fft_u)**2+np.abs(self.fft_v)**2)/2, fftdim=('kx','ky'), nfactor=2, truncate=True)

    def compute_KE_spectrum_oneline(self, Lat=(35,45), Lon=(5,15), window='hann', nfactor=2, truncate=True):
        from helpers.computational_tools import select_LatLon
        u = select_LatLon(remesh(self.u.isel(Time=-1), self.e),Lat,Lon)
        v = select_LatLon(remesh(self.v.isel(Time=-1), self.e),Lat,Lon)
        dx = select_LatLon(self.param.dxT,Lat,Lon).mean().values
        dy = select_LatLon(self.param.dyT,Lat,Lon).mean().values

        x = dx*np.arange(len(u.xh))
        y = dy*np.arange(len(u.yh))
        u['xh'] = x
        u['yh'] = y
        v['xh'] = x
        v['yh'] = y
        
        Eu = xrft.isotropic_power_spectrum(u, dim=('xh','yh'), detrend='linear', window=window, window_correction=True, nfactor=nfactor, truncate=truncate)
        Ev = xrft.isotropic_power_spectrum(v, dim=('xh','yh'), detrend='linear', window=window, window_correction=True, nfactor=nfactor, truncate=truncate)

        E = Eu+Ev
        E['freq_r'] = E['freq_r']*2*np.pi
        E = E / 2

        print(f'kmax = ({np.pi/dx}, {np.pi/dy}, {np.sqrt((np.pi/dx)**2+(np.pi/dx)**2)})')
        print(f'freq_r max = {E.freq_r.max().values}')

        return E

    def compute_KE_spectrum(self, Lat=(35,45), Lon=(5,15), window='hann', nfactor=2, truncate=True):
        from helpers.computational_tools import select_LatLon
        u = remesh(self.u.isel(Time=-1), self.e)
        v = remesh(self.v.isel(Time=-1), self.e)
        fftu = compute_2dfft(u, self.param.dxT, self.param.dyT, Lat=Lat, Lon=Lon, window=window)
        fftv = compute_2dfft(v, self.param.dxT, self.param.dyT, Lat=Lat, Lon=Lon, window=window)

        E = 2*np.pi*xrft.isotropize((np.abs(fftu)**2+np.abs(fftv)**2)/2, fftdim=('kx','ky'), nfactor=nfactor, truncate=truncate)

        print('Parseval equality:')
        E_spatial = ((select_LatLon(u**2 + v**2, Lat,Lon))/2).mean(dim=('xh','yh'))
        print('Mean spatial energy:', E_spatial.values)
        E_spectral = (E * np.diff(E.freq_r).mean()).sum(dim='freq_r').values
        print('Integral over spectrum:', E_spectral)
        return E

    def compute_KE_spectrum_old(self, Lat=(35,45), Lon=(5,15), window='hanning', nfactor=2, truncate=True, averaging=True):
        import dask
        from helpers.old_tools import compute_spectrum
        tstart = 7280
        prog = self.prog
        param = self.param
        t = prog.Time
        xq = prog.xq
        yq = prog.yq
        xh = prog.xh
        yh = prog.yh
        dxT = param.dxT
        dyT = param.dyT
        lonh = param.xh
        lath = param.yh
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            u = np.array(prog.u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
            v = np.array(prog.v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
            dx = float(dxT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
            dy = float(dyT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())

        k, Eu = compute_spectrum(u[:,0,:,:], window, dx, dy, nfactor=nfactor, truncate=truncate, averaging=averaging)
        k, Ev = compute_spectrum(v[:,0,:,:], window, dx, dy, nfactor=nfactor, truncate=truncate, averaging=averaging)

        E_upper = Eu+Ev

        k, Eu = compute_spectrum(u[:,1,:,:], window, dx, dy, nfactor=nfactor, truncate=truncate, averaging=averaging)
        k, Ev = compute_spectrum(v[:,1,:,:], window, dx, dy, nfactor=nfactor, truncate=truncate, averaging=averaging)

        E_lower = Eu+Ev

        E = np.vstack([E_upper, E_lower])
        EE = xr.DataArray(E, dims=('zl', 'freq_r'))
        EE['freq_r'] = k

        print('Parseval equality:')
        E_spatial = ((u**2 + v**2)/2).mean(axis=(0,2,3))
        print('Mean spatial energy:', E_spatial)
        E_spectral = (EE * np.diff(EE.freq_r).mean()).sum(dim='freq_r').values
        print('Integral over spectrum:', E_spectral)

        return EE

