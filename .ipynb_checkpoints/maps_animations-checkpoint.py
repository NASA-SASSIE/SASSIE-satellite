import numpy as np
import h5py
import glob
from pathlib import Path 
import xarray as xr
import cartopy
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import warnings; warnings.simplefilter('ignore')
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

nws_dir = Path('/Users/severinf/Data/raw/sic_nws/') 
nsidc_dir = Path('/Users/severinf/Data/raw/sic_nsidc/') 
smos_dir=Path('/Users/severinf/Data/raw/smos_locean_arctic/')
oisst_dir=Path('/Users/severinf/Data/raw/oisst/')
fig_dir = Path('/Users/severinf/Figures/SASSIE/SASSIE-satellite/')

latstep=10
lonstep=30
land=True

#define lat/lon region below (different for each parameter)


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

def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    """
    for data in projection axe.projection
    find and mask the overlaps (more 1/2 the axe.projection range)

    X, Y either the coordinates in axe.projection or longitudes latitudes
    Z the data
    operation one of 'pcorlor', 'pcolormesh', 'countour', 'countourf'

    if source_projection is a geodetic CRS data is in geodetic coordinates
    and should first be projected in axe.projection

    X, Y are 2D same dimension as Z for contour and contourf
    same dimension as Z or with an extra row and column for pcolor
    and pcolormesh

    return ptx, pty, Z
    """
    if not hasattr(axe, 'projection'):
        return Z
    if not isinstance(axe.projection, cartopy.crs.Projection):
        return Z

    if len(X.shape) != 2 or len(Y.shape) != 2:
        return Z

    if (source_projection is not None and
            isinstance(source_projection, cartopy.crs.Geodetic)):
        transformed_pts = axe.projection.transform_points(
            source_projection, X, Y)
        ptx, pty = transformed_pts[..., 0], transformed_pts[..., 1]
    else:
        ptx, pty = X, Y


    with numpy.errstate(invalid='ignore'):
        # diagonals have one less row and one less columns
        diagonal0_lengths = numpy.hypot(
            ptx[1:, 1:] - ptx[:-1, :-1],
            pty[1:, 1:] - pty[:-1, :-1]
        )
        diagonal1_lengths = numpy.hypot(
            ptx[1:, :-1] - ptx[:-1, 1:],
            pty[1:, :-1] - pty[:-1, 1:]
        )
        to_mask = (
            (diagonal0_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            numpy.isnan(diagonal0_lengths) |
            (diagonal1_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            numpy.isnan(diagonal1_lengths)
        )

        # TODO check if we need to do something about surrounding vertices

        # add one extra colum and row for contour and contourf
        if (to_mask.shape[0] == Z.shape[0] - 1 and
                to_mask.shape[1] == Z.shape[1] - 1):
            to_mask_extended = numpy.zeros(Z.shape, dtype=bool)
            to_mask_extended[:-1, :-1] = to_mask
            to_mask_extended[-1, :] = to_mask_extended[-2, :]
            to_mask_extended[:, -1] = to_mask_extended[:, -2]
            to_mask = to_mask_extended
        if numpy.any(to_mask):

            Z_mask = getattr(Z, 'mask', None)
            to_mask = to_mask if Z_mask is None else to_mask | Z_mask

            Z = ma.masked_where(to_mask, Z)

        return ptx, pty, Z
        

######################### SIC, SSS, SST #########################
lonmin=-150 #-180
lonmax=-119 #-135
latmin=67
latmax=80
lon0=-125 #-150

for year in range(2010,2023):
    
    ####################### Prepare xarrays ###########################
    
    #sic nsidc
    #ffmpeg -r 4 -i animation/map_sicnsidc_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_sicnsidc_2011.mp4
    filename_grid='/Users/severinf/Data/raw/grid_sic_nsidc.nc'
    files=list(nsidc_dir.glob('sic_psn25_'+str(year)+'*.nc'))
    grid = xr.open_dataset(filename_grid)
    grid = grid.rename({'xgrid': 'x','ygrid': 'y'})
    sic = xr.open_mfdataset(paths=np.sort(files)).sel(time=slice(str(year)+'-08-31',str(year)+'-11-30'))
    sic=sic.where(grid.longitude<=-110) #otherwise contours are messed up
    sic=sic.where(grid.longitude>=-180)
    for i in range(0,sic.cdr_seaice_conc.shape[0]):
        fig = plt.figure(figsize=(8,10))
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
        ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
        ax.coastlines('10m',zorder=2)
        ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gl.xlocator = mticker.FixedLocator(np.arange(lonmin,lonmax,15))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'k','rotation':0}
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylocator = mticker.FixedLocator(np.arange(latmin,latmax,5))
        gl.ylabel_style = {'size': 20, 'color': 'gray','rotation':0}
        palette = plt.cm.Blues_r
        pp=ax.pcolormesh(grid.longitude,grid.latitude,np.array(sic.cdr_seaice_conc[i,:,:].squeeze().data),cmap=palette,vmin=0,vmax=1,transform=cartopy.crs.PlateCarree())
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        cc=ax.contour(grid.longitude,grid.latitude,np.array(sic.cdr_seaice_conc[i,:,:].squeeze().data),levels=[0.15],colors='m',linewidth=3,transform=cartopy.crs.PlateCarree(), source_projection=cartopy.crs.Geodetic())
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
        h.ax.tick_params(labelsize=20)
        h.set_label('SIC - '+str(sic.time[i].data),fontsize=20)
        cmin,cmax = h.mappable.get_clim()
        ticks = np.linspace(cmin,cmax,6)
        h.set_ticks(ticks)
        plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
        fig_name='animation/map_sicnsidc_'+str(year)+'_'+str(i).zfill(3)+'.png'
        plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')
        
    #oisst
    #ffmpeg -r 4 -i animation/map_oisst_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_oisst_2011.mp4
    files=list(oisst_dir.glob('oisst_'+str(year)+'*.nc'))
    tmp = xr.open_mfdataset(paths=np.sort(files)).sel(time=slice(str(year)+'-08-31',str(year)+'-11-30'),lat=slice(latmin,latmax),zlev=0)
    tmp=tmp.drop('zlev')
    #a bunch of stuff to put back lon in right order
    new_lon = tmp.lon.values + 0
    new_lon[new_lon > 180] = new_lon[new_lon > 180]-360
    tmp=tmp.drop('lon')
    tmp=tmp.assign_coords({'lon': new_lon})
    D = tmp.anom[:,:,180:]
    C = tmp.anom[:,:,0:180:]
    F = tmp.ice[:,:,180:]
    E = tmp.ice[:,:,0:180:]
    H = tmp.err[:,:,180:]
    G = tmp.err[:,:,0:180:]
    
    B = tmp.sst[:,:,180:]
    A = tmp.sst[:,:,0:180:]
    oisst = xr.merge([B, A]).transpose('lat','lon','time')
    oisst = oisst.assign(
    anom=(['lat','lon','time'],xr.merge([D, C]).transpose('lat','lon','time').anom.data),
    ice=(['lat','lon','time'],xr.merge([F, E]).transpose('lat','lon','time').ice.data), 
    err=(['lat','lon','time'],xr.merge([H, G]).transpose('lat','lon','time').err.data),
                    )
    oisst.anom.attrs['long_name'] = tmp.anom.attrs['long_name']
    oisst.anom.attrs['units'] = tmp.anom.attrs['units']
    oisst.anom.attrs['valid_min'] = tmp.anom.attrs['valid_min']
    oisst.anom.attrs['valid_max'] = tmp.anom.attrs['valid_max']
    
    oisst.err.attrs['long_name'] = tmp.err.attrs['long_name']
    oisst.err.attrs['units'] = tmp.err.attrs['units']
    oisst.err.attrs['valid_min'] = tmp.err.attrs['valid_min']
    oisst.err.attrs['valid_max'] = tmp.err.attrs['valid_max']
    if len(oisst.time.values)==len(sic.time.values):
        print('sic contours ok for oisst '+str(year))
    else:
        print('sic different size, attention contours for oisst '+str(year))
    for i in range(0,oisst.sst.shape[2]):
        fig = plt.figure(figsize=(8,10))
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
        ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
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
        pp=ax.pcolormesh(oisst.lon,oisst.lat,np.array(oisst.sst[:,:,i].squeeze().data),cmap=palette,vmin=-2,vmax=8,transform=cartopy.crs.PlateCarree())
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        cc=ax.contour(grid.longitude,grid.latitude,np.array(sic.cdr_seaice_conc[i,:,:].squeeze().data),levels=[0.15],colors='m',linewidth=3,transform=cartopy.crs.PlateCarree(), source_projection=cartopy.crs.Geodetic())
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
        h.ax.tick_params(labelsize=20)
        h.set_label('SST (degC) - '+str(oisst.time[i].data),fontsize=20)
        cmin,cmax = h.mappable.get_clim()
        ticks = np.linspace(cmin,cmax,6)
        h.set_ticks(ticks)
        plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
        fig_name='animation/map_oisst_'+str(year)+'_'+str(i).zfill(3)+'.png'
        plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')
        
    #smos sss
    #ffmpeg -r 4 -i animation/map_smos_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_smos_2011.mp4
    files=list(smos_dir.glob('SMOS_L3_DEBIAS_LOCEAN_AD_'+str(year)+'*.nc'))
    smos = xr.open_mfdataset(paths=np.sort(files)).sel(time=slice(str(year)+'-08-31',str(year)+'-11-30')) 
    if len(smos.time.values)==len(sic.time.values):
        print('sic contours ok for smos '+str(year))
    else:
        print('sic different size, attention contours for smos '+str(year))
    for i in range(0,smos.SSS.shape[0]):
        fig = plt.figure(figsize=(8,10))
        ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
        ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
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
        pp=ax.pcolormesh(smos.lon[0,:,:],smos.lat[0,:,:],np.array(smos.SSS[i,:,:].squeeze().data),cmap=palette,vmin=20,vmax=35,transform=cartopy.crs.PlateCarree())
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        cc=ax.contour(grid.longitude,grid.latitude,np.array(sic.cdr_seaice_conc[i,:,:].squeeze().data),levels=[0.15],colors='m',linewidth=3,transform=cartopy.crs.PlateCarree(), source_projection=cartopy.crs.Geodetic())
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
        h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
        h.ax.tick_params(labelsize=20)
        h.set_label('SSS - '+str(smos.time[i].data),fontsize=20)
        cmin,cmax = h.mappable.get_clim()
        ticks = np.linspace(cmin,cmax,6)
        h.set_ticks(ticks)
        plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
        fig_name='animation/map_smos_'+str(year)+'_'+str(i).zfill(3)+'.png'
        plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')


# ######################### SIC ASTRID #########################
# #Different domain for sic astrid
# lonmin=-180
# lonmax=-135
# latmin=65
# latmax=80
# lon0=-150

# for year in range(2010,2023):
    
#     ####################### Prepare xarrays ###########################
    
#     mat = h5py.File(str(nws_dir) + '/nws_'+str(year)+'.mat')
#     mat.keys()
#     lat_sic=np.array(mat['LAT'])
#     lon_sic=np.array(mat['LON'])
#     time_sic=np.array(mat['date'])
#     date_sic=[]
#     for itime in np.arange(np.array(mat['date']).size):
#         tmp=convert_wg_time_to_td64(time_sic[itime],wg_ref=np.datetime64('0000-01-01'))
#         date_sic.append(tmp)                      
    
#     #change lon to -180,180
#     ind=np.where(np.array(mat['LON'][:,0].squeeze())>180)
#     ind1=np.where(np.array(mat['LON'][:,0].squeeze())<=180)
#     lon = np.array(mat['LON'][:,0].squeeze()) + 0
#     lon[ind] = lon[ind]-360
#     lat=mat['LAT'][0,:].squeeze()
    
#     sic_all = xr.DataArray(np.transpose(np.array(mat['iceconc']), (2, 1, 0)), 
#     coords={'lat': np.array(mat['LAT'])[0,:].squeeze(),'lon': np.array(mat['LON'])[:,0].squeeze(),'time': time_sic.squeeze()}, 
#     dims=["lat", "lon", "time"])
    
#     #mask land temporary, just use "else" when new files from Astrid
#     if year==2015 or year==2017 or year==2021:
#         ind=np.where(~np.isfinite(np.nanmean(sic_all[:,:,0:60],axis=2)))
#         mask_land_tmp=np.ones(sic_all[:,:,363].shape)
#         mask_land_tmp[ind]=0
#         mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
#         mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
#     elif year==2016:
#         ind=np.where(~np.isfinite(np.nanmean(sic_all[:,:,-60:],axis=2)))
#         mask_land_tmp=np.ones(sic_all[:,:,363].shape)
#         mask_land_tmp[ind]=0
#         mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
#         mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
#     else:
#         ind=np.where(~np.isfinite(np.nanmean(sic_all,axis=2)))
#         mask_land_tmp=np.ones(sic_all[:,:,363].shape)
#         mask_land_tmp[ind]=0
#         mask_land_tmp=mask_land_tmp.reshape((mask_land_tmp.shape[0], mask_land_tmp.shape[1], 1))
#         mask_land=np.tile(mask_land_tmp,(1,1,sic_all.shape[2]))
    
#     # put 0 in all open ocean except when maps are all empty
#     sic_all2=[]
#     for i in range(0,sic_all.shape[2]):
#         tmp=sic_all.data[:,:,i].squeeze()+0
#         ind=np.where(np.isfinite(tmp))
#         if len(ind[0])>1:
#             ind=np.where(~np.isfinite(tmp))
#             tmp[ind]=0
#         try:
#             sic_all2=np.dstack((sic_all2,tmp))
#         except:
#             sic_all2=tmp
    
#     sic_all = xr.DataArray(sic_all2, 
#     coords={'lat': lat,'lon': lon,'time': time_sic.squeeze()}, 
#     dims=["lat", "lon", "time"])
#     sic_all=sic_all.where(mask_land)
    
#     #before 2015 has sic only every 4 days, some other years have data missing
#     sic_all=sic_all.interpolate_na(dim=('time'), method='nearest')
    
#     sic = xr.DataArray(sic_all.data, 
#     coords={'lat': sic_all.lat.data,'lon': sic_all.lon.data,'time': date_sic}, 
#     dims=["lat", "lon", "time"]).sel(time=slice(str(year)+'-08-31',str(year)+'-12-30'),lat=slice(latmin,latmax))#,lon=slice(lonmin,lonmax))


#     # raw Astrid data
#     # ffmpeg -r 4 -i animation/map_raw_sicastrid_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_raw_sicastrid_2011.mp4
#     d=0
#     for i in range(0,mat['iceconc'].shape[0]):
#         fig = plt.figure(figsize=(10,8))
#         ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
#         ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
#         # ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
#         ax.coastlines('10m',zorder=2)
#         ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
#         gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
#         gl.xlocator = mticker.FixedLocator(np.arange(lonmin,lonmax,15))
#         gl.xformatter = LONGITUDE_FORMATTER
#         gl.xlabel_style = {'size': 20, 'color': 'k','rotation':0}
#         gl.yformatter = LATITUDE_FORMATTER
#         gl.ylocator = mticker.FixedLocator(np.arange(latmin,latmax,5))
#         gl.ylabel_style = {'size': 20, 'color': 'gray','rotation':0}
#         palette = plt.cm.jet
#         pp=ax.pcolormesh(mat['LON'][:,0].squeeze(), mat['LAT'][0,:].squeeze(), np.transpose(mat['iceconc'][i,:,:].squeeze(), (1, 0)),cmap=palette,vmin=0,vmax=10,transform=cartopy.crs.PlateCarree())
#         cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
#         h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
#         h.ax.tick_params(labelsize=20)
#         h.set_label(str(date_sic[i]),fontsize=20)
#         cmin,cmax = h.mappable.get_clim()
#         ticks = np.linspace(cmin,cmax,6)
#         h.set_ticks(ticks)
#         plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
#         fig_name='animation/map_raw_sicastrid_'+str(year)+'_'+str(d).zfill(3)+'.png'
#         plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')
#         d=d+1
        
#     #sic array I made
#     #ffmpeg -r 4 -i animation/map_sicastrid_2011_%03d.png -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" map_sicastrid_2011.mp4
#     for i in range(0,sic.shape[2]):
#         fig = plt.figure(figsize=(10,8))
#         ax = plt.axes(projection=cartopy.crs.NorthPolarStereo(central_longitude=lon0))
#         ax.set_extent([lonmin, lonmax, latmin, latmax], crs=cartopy.crs.PlateCarree()) 
#         # ax.add_feature(cfeature.LAND, facecolor = '0.75',zorder=1)
#         ax.coastlines('10m',zorder=2)
#         ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
#         gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
#         gl.xlocator = mticker.FixedLocator(np.arange(lonmin,lonmax,15))
#         gl.xformatter = LONGITUDE_FORMATTER
#         gl.xlabel_style = {'size': 20, 'color': 'k','rotation':0}
#         gl.yformatter = LATITUDE_FORMATTER
#         gl.ylocator = mticker.FixedLocator(np.arange(latmin,latmax,5))
#         gl.ylabel_style = {'size': 20, 'color': 'gray','rotation':0}
#         palette = plt.cm.jet
#         pp=ax.pcolormesh(sic.lon,sic.lat,sic[:,:,i].squeeze(),cmap=palette,vmin=0,vmax=10,transform=cartopy.crs.PlateCarree())
#         cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.04])
#         h=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal',ax=ax)
#         h.ax.tick_params(labelsize=20)
#         h.set_label(str(sic.time[i].data),fontsize=20)
#         cmin,cmax = h.mappable.get_clim()
#         ticks = np.linspace(cmin,cmax,6)
#         h.set_ticks(ticks)
#         plt.subplots_adjust(right=0.9,left=0.1,top=0.95,bottom=0.18)
#         fig_name='animation/map_sicastrid_'+str(year)+'_'+str(i).zfill(3)+'.png'
#         plt.savefig(fig_dir / fig_name,dpi=200,transparent=False,facecolor='white')