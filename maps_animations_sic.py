import numpy as np
import h5py
import glob
from pathlib import Path 
import scipy.io
import xarray as xr
import cartopy
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import warnings; warnings.simplefilter('ignore')
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyresample import kd_tree, geometry
from pyresample.geometry import GridDefinition

nws_dir = Path('/Users/severinf/Data/raw/sic_nws/') 
smos_dir=Path('/Users/severinf/Data/raw/smos_locean_arctic/')
oisst_dir=Path('/Users/severinf/Data/raw/oisst/')
fig_dir = Path('/Users/severinf/Figures/SASSIE/SASSIE-satellite/')

lonmin=-180
lonmax=-135
latmin=65
latmax=80
lon0=-150

latstep=10
lonstep=30
land=True

######################### Functions #########################

def convert_wg_time_to_td64(wg_jd_apres_ref_date, wg_ref = np.datetime64('0000-01-01')):
    """
    Input: wg_jd_apres_ref_date ==> julian date to convert
           wg_ref ==> ref date
    Output: wg_date ==> standard date
    """
    # must be integer
    wg_day_apres_ref_date = int(np.floor(wg_jd_apres_ref_date))
    # must be integer
    wg_seconds  = int(3600*24*( wg_jd_apres_ref_date - wg_day_apres_ref_date))
    wg_date = np.timedelta64(wg_day_apres_ref_date-1, 'D') + np.timedelta64(wg_seconds, 's') +  wg_ref
    return wg_date
    

for year in range(2010,2023):
    
    ####################### Prepare xarrays ###########################
    
    mat = h5py.File(str(nws_dir) + '/nws_'+str(year)+'.mat')
    mat.keys()
    lat_sic=np.array(mat['LAT'])
    lon_sic=np.array(mat['LON'])
    time_sic=np.array(mat['date'])
    date_sic=[]
    for itime in np.arange(np.array(mat['date']).size):
        tmp=convert_wg_time_to_td64(time_sic[itime],wg_ref=np.datetime64('0000-01-01'))
        date_sic.append(tmp)                      
    
    #change lon to -180,180
    ind=np.where(np.array(mat['LON'][:,0].squeeze())>180)
    ind1=np.where(np.array(mat['LON'][:,0].squeeze())<=180)
    lon = np.array(mat['LON'][:,0].squeeze()) + 0
    lon[ind] = lon[ind]-360
    lat=mat['LAT'][0,:].squeeze()
    
    sic_all = xr.DataArray(np.transpose(np.array(mat['iceconc']), (2, 1, 0)), 
    coords={'lat': np.array(mat['LAT'])[0,:].squeeze(),'lon': np.array(mat['LON'])[:,0].squeeze(),'time': time_sic.squeeze()}, 
    dims=["lat", "lon", "time"])
    
    #mask land temporary, just use "else" when new files from Astrid
    if year==2015 or year==2017 or year==2021:
        ind=np.where(~np.isfinite(np.nanmean(sic_all[:,:,0:60],axis=2)))
        mask_land_tmp=np.ones(sic_all[:,:,363].shape)
        mask_land_tmp[ind]=0
        mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
        mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
    elif year==2016:
        ind=np.where(~np.isfinite(np.nanmean(sic_all[:,:,-60:],axis=2)))
        mask_land_tmp=np.ones(sic_all[:,:,363].shape)
        mask_land_tmp[ind]=0
        mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
        mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
    else:
        ind=np.where(~np.isfinite(np.nanmean(sic_all,axis=2)))
        mask_land_tmp=np.ones(sic_all[:,:,363].shape)
        mask_land_tmp[ind]=0
        mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
        mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
    
    # put 0 in all open ocean except when maps are all empty
    sic_all2=[]
    for i in range(0,sic_all.shape[2]):
        tmp=sic_all.data[:,:,i].squeeze()+0
        ind=np.where(np.isfinite(tmp))
        if len(ind[0])>1:
            ind=np.where(~np.isfinite(tmp))
            tmp[ind]=0
        try:
            sic_all2=np.dstack((sic_all2,tmp))
        except:
            sic_all2=tmp
    
    sic_all = xr.DataArray(sic_all2, 
    coords={'lat': lat,'lon': lon,'time': time_sic.squeeze()}, 
    dims=["lat", "lon", "time"])
    sic_all=sic_all.where(mask_land)
    
    #before 2015 has sic only every 4 days, some other years have data missing
    sic_all=sic_all.interpolate_na(dim=('time'), method='nearest')
    
    sic = xr.DataArray(sic_all.data, 
    coords={'lat': sic_all.lat.data,'lon': sic_all.lon.data,'time': date_sic}, 
    dims=["lat", "lon", "time"]).sel(time=slice(str(year)+'-08-31',str(year)+'-12-30'),lat=slice(latmin,latmax))#,lon=slice(lonmin,lonmax))


    # raw Astrid data
    # ffmpeg -r 4 -i animation/map_raw_sic_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_raw_sic_2011.mp4
    d=0
    for i in range(0,mat['iceconc'].shape[0]):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
        # ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
        ax.coastlines('10m',zorder=2)
        ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gl.xlocator = mticker.FixedLocator(np.arange(lonmin,lonmax,15))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'k','rotation':0}
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylocator = mticker.FixedLocator(np.arange(latmin,latmax,5))
        gl.ylabel_style = {'size': 20, 'color': 'gray','rotation':0}
        palette = plt.cm.jet
        pp=ax.pcolormesh(mat['LON'][:,0].squeeze(), mat['LAT'][0,:].squeeze(), np.transpose(mat['iceconc'][i,:,:].squeeze(), (1, 0)),cmap=palette,vmin=0,vmax=10,transform=cartopy.crs.PlateCarree())
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
        h.ax.tick_params(labelsize=20)
        h.set_label(str(date_sic[i]),fontsize=20)
        cmin,cmax = h.mappable.get_clim()
        ticks = np.linspace(cmin,cmax,6)
        h.set_ticks(ticks)
        plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
        fig_name='animation/map_raw_sic_'+str(year)+'_'+str(d).zfill(3)+'.png'
        plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')
        d=d+1
        
    #sic array I made
    #ffmpeg -r 4 -i animation/map_sic_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_sic_2011.mp4
    for i in range(0,sic.shape[2]):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
        # ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
        ax.coastlines('10m',zorder=2)
        ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gl.xlocator = mticker.FixedLocator(np.arange(lonmin,lonmax,15))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'k','rotation':0}
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylocator = mticker.FixedLocator(np.arange(latmin,latmax,5))
        gl.ylabel_style = {'size': 20, 'color': 'gray','rotation':0}
        palette = plt.cm.jet
        pp=ax.pcolormesh(sic.lon,sic.lat,sic[:,:,i].squeeze(),cmap=palette,vmin=0,vmax=10,transform=cartopy.crs.PlateCarree())
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
        h.ax.tick_params(labelsize=20)
        h.set_label(str(sic.time[i].data),fontsize=20)
        cmin,cmax = h.mappable.get_clim()
        ticks = np.linspace(cmin,cmax,6)
        h.set_ticks(ticks)
        plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
        fig_name='animation/map_sic_'+str(year)+'_'+str(i).zfill(3)+'.png'
        plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')