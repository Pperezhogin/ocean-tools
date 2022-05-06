import xarray as xr
import os
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np
from helpers.experiment import Experiment
from celluloid import Camera

class CollectionOfExperiments:
    '''
    This class extend dictionary of experiments by additional
    tools for plotting and comparing experiments
    '''
    def __init__(self, exps, experiments_dict, names_dict):
        '''
        experiments_dict - "experiment" objects labeled by keys
        names_dict - labels for plotting
        '''
        self.exps = exps
        self.experiments = experiments_dict
        self.names = names_dict

    def __getitem__(self, q):
        ''' 
        Access experiments with key values directly
        '''
        try:
            return self.experiments[q]
        except:
            print('item not found')
    
    def __add__(self, otherCollection):
        # merge dictionaries and lists
        exps = [*self.exps, *otherCollection.exps]
        experiments_dict = {**self.experiments, **otherCollection.experiments}
        names_dict = {**self.names, **otherCollection.names}

        return CollectionOfExperiments(exps, experiments_dict, names_dict)

    def remesh(self, input, target, exp=None, name=None, compute=False):
        '''
        input  - key of experiment to coarsegrain
        target - key of experiment we want to take coordinates from
        '''
        result = self[input].remesh(self[target], compute) # call experiment method

        if exp is None:
            exp = input+'_'+target
        if name is None:
            name = input+' coarsegrained to '+target

        print('Experiment '+input+' coarsegrained to '+target+
            ' is created. Its identificator='+exp)
        self.exps.append(exp)
        self.experiments[exp] = result
        self.names[exp] = name
    
    def print_exps(self):
        print(*self.exps)
    
    @classmethod
    def init_folder(cls, common_folder, exps=None, exps_names=None, additional_subfolder=''):
        '''
        Scan folders in common_folder and returns class instance with exps given by these folders
        exps - list of folders can be specified
        exps_names - list of labels can be specified
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder 
        '''

        if exps is None:
            exps = sorted(os.listdir(common_folder))

        if exps_names is None:
            exps_names = exps

        # Construct dictionary of experiments, where keys are given by exps
        experiments_dict = {}
        names_dict = {}
        for i in range(len(exps)):
            folder = os.path.join(common_folder,exps[i],additional_subfolder)
            experiments_dict[exps[i]] = Experiment(folder)
            names_dict[exps[i]] = exps_names[i] # convert array to dictionary

        return cls(exps, experiments_dict, names_dict)      

    ######################### service plotting functions #################
    def get_axes(self, nfig, ncol=3, size=4, ratio=1.15):
        if nfig > ncol:
            xfig=ncol
            yfig=math.ceil(nfig/ncol)
        else:
            xfig = nfig
            yfig = 1
        
        figsize_x = size * ratio * xfig
        figsize_y = size * yfig
        fig, ax = plt.subplots(yfig, xfig, figsize=(figsize_x,figsize_y), constrained_layout=True)
        try:
            ax = ax.flat # 1d array of subplots
        except:
            ax = np.array([ax,]) # to make work only one picture
        return fig, ax

    def animate(self, plot_function, nfig, ncol=3, ratio=1.15, Time=range(-50,0), videoname='my_movie.mp4'):
        '''
        Decorator for animation. 
        Time - range of indices to plot
        plot_function must have Time and ax argument,
        and return list of "matplotlib.Artist" objects
        '''
        def new_plot_function(*args, **kwargs):
            fig, ax = self.get_axes(nfig=nfig, ncol=ncol, ratio=ratio)
            p=[]
            N = len(Time)
            n = 0
            use_colorbar=True
            for j in Time:
                try:
                    p_ = plot_function(*args, **kwargs, Time=j, ax=ax, use_colorbar=use_colorbar)
                except:
                    p_ = plot_function(*args, **kwargs, Time=j, ax=ax)
                n += 1
                print(f"{n} Images of {N} are plotted",end="\r")
                use_colorbar=False
                p.append(p_)

            print("Converting list of figures to animation object...",end="\r")
            ani = animation.ArtistAnimation(fig, p, interval=100, blit=True, repeat_delay=0)
            print("Saving animation as videofile...                 ",end="\r")
            ani.save(videoname)
            print("Done                                             ",end="\r")
            plt.close()
            return videoname
        return new_plot_function

    #########################  snapshot plotters #########################
    # Basic pcolor viewer. Differs from specialized functions 
    # by lack of specialized colorbar
    # Formally, it is wrapper of standard method xarray.pcolormesh()
    # across different experiments with default selector (Time=-1,zl=0)
    def pcolormesh(self, key, exps, Time=-1, zl=0, names=None, vmin=None, vmax=None, 
        cmap=None, cbar_title=None, ax=None, use_colorbar=True):
        '''
        exps - list of experiments
        key - name of variable to plot
        Time - index of time
        zl - index in vertical
        names - labels of experiments
        vmin, vmax - range of colorbar
        cmap - name of colormap
        cbar_title - obviously
        ax - Optionally takes axes. This allows to construct facet figure
        use_colorbar - necessary evil to make movies (colorbar is plotted at first time index)

        Returns list of "matplotlib.Artist" objects. Can be used later for movies
        '''
        plt.rcParams.update({'font.size': 14})
        nfig = len(exps)
        if ax is None:
            fig, ax = self.get_axes(nfig)

        if names is None:
            names = [self.names[exp] for exp in exps]

        p = []
        for ifig, exp in enumerate(exps):
            try:
                field = self[exp].__getattribute__(key).isel(zl=zl,Time=Time)
            except:
                field = self[exp].__getattribute__(key).isel(zi=zl,Time=Time)
            p.append(field.plot.pcolormesh(vmin=vmin,vmax=vmax, # standard xarray plotter
                ax=ax[ifig],add_colorbar=False,cmap=cmap))
            ax[ifig].set_title(names[ifig])
        if use_colorbar:
            plt.colorbar(p[0],ax=ax,label=cbar_title)
        return p

    def plot_RV(self, exps, Time=-1, zl=0, names=None, ax=None, use_colorbar=True):
        return self.pcolormesh('RV_f', exps, Time, zl, names, -0.2, 0.2, 'bwr', 
            'Relative vorticity / local Coriolis ($\zeta/f$)', ax, use_colorbar)

    def plot_PV(self, exps, Time=-1, zl=0, names=None, ax=None, use_colorbar=True):
        return self.pcolormesh('PV', exps, Time, zl, names, 0, 2e-7, 'seismic', 
            'Potential vorticity, $m^{-1} s^{-1}$', ax, use_colorbar)
    
    def plot_KE(self, exps, Time=-1, zl=0, names=None, vmax=0.05, ax=None, use_colorbar=True):
        return self.pcolormesh('KE', exps, Time, zl, names, 0, vmax, 'inferno', 
            'Kinetic energy, $m^2/s^2$', ax, use_colorbar)

    def plot_KE_spectrum(self, exps, Time=-1, ax=None):
        
        p = []
        for exp in exps:
            KE = self[exp].KE_spectrum
            k = KE.freq_r

            KE_upper = KE.isel(zl=0,Time=Time)
            KE_lower = KE.isel(zl=0,Time=Time)

            p.extend(ax[0].loglog(k, KE_upper, label=self.names[exp]))
            ax[0].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[0].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[0].set_title('Upper layer')
            ax[0].legend(prop={'size': 14})
            ax[0].grid(which='both',linestyle=':')

            p.extend(ax[1].loglog(k, KE_lower, label=self.names[exp]))
            ax[1].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[1].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[1].set_title('Lower layer')
            ax[1].legend(prop={'size': 14})
            ax[1].grid(which='both',linestyle=':')

        k = [5e-5, 1e-3]
        E = [1.5e+2, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[0].loglog(k,E,'--k')
        ax[0].text(2e-4,1e+1,'$k^{-3}$')
        ax[0].set_xlim([5e-6, 2e-3])
        
        ax[1].set_xlim([5e-6, 2e-3])
        k = [5e-5, 1e-3]
        E = [3e+1, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[1].loglog(k,E,'--k')
        ax[1].text(2e-4,1e+1,'$k^{-3}$')

        return p