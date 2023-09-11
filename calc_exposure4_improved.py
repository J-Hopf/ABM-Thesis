#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:55:57 2023
@author: janhopfer
"""

###########################################################################################
# please mind the reference system to use, in gethw() and the parameters in getcon()
# free-time activity garden: 100m (0.001 degree), take a walk (2000 m) (0.02 degree)
##############################################################################################
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from rasterstats import zonal_stats, point_query
import pyproj
from shapely.ops import transform
import model_functions3_3_shop as m
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show_hist
from math import modf
import osmnx as ox
from scipy import signal
from IPython import get_ipython
import statistics
import random
import time

random.seed(10)
os.chdir("core")

filedir = "C:/Users/Jan/PycharmProjects/Thesis2/"
results_dir = f"{filedir}results1/r1_improved/"  # use same folder as streamline_osmnx
savedir = f"{results_dir}Uni/"  # each profile a savedir.
savedir2 = f"{results_dir}Uni_2/"  # each profile a savedir.
preddir = f"{filedir}input/hr_pred/"

# name of the concentration raster .tifs (otherwise it has to be changed in a lot of functions)
name_conrast = "hr_pred_utrecht_X"

# # OSM input data
# Bus stops
OSMstops = pd.read_csv(f"{filedir}input/OSM/bus4_highway.csv")  # Open csv with coordinates of bus stops
geom_bus = gpd.GeoSeries.from_wkt(OSMstops['geometry'])  # Get geometry from csv
busstops = gpd.GeoDataFrame(OSMstops, geometry=geom_bus, crs="EPSG:4326")  # Make GeoDataFrame
u_busstops = busstops.unary_union  # Get MultiPoint shape from GeoDataFrame

def wgs2laea(p):
    rd = pyproj.CRS('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    project = pyproj.Transformer.from_crs("EPSG:4326", rd, always_xy=True)
    p = transform(project.transform, p)
    return p


def plot_raster():
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(55, 15))

    for i, ax in enumerate(axs.flat):
        src = rasterio.open(f"{preddir}{name_conrast}{i}.tif")

        ax.set_axis_off()
        a = ax.imshow(src.read(1), cmap='pink')
        ax.set_title(f' {i:02d}:00')

    cbar = fig.colorbar(a, ax=axs.ravel().tolist())
    cbar.set_label(r'$NO_2$', rotation=270)
    plt.show()


def gethw(df):
    ph = Point(float(df.home_lon), float(df.home_lat))  # home location
    pw = Point(float(df.work_lon), float(df.work_lat))  # work location
    return ph, pw


def buffermean(p, ext, rasterfile):
    pbuf = p.buffer(ext)
    z = zonal_stats(pbuf, rasterfile, stats='mean')[0]['mean']
    return z


def gkern(kernlen=21, sd=3.0):  # sd=0.1
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=sd).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


# extgaus=2, sd=0.1
def gaussKernconv(p, extgaus, rasterfile, sd=1.0):  # extgaus = 2, sd = 0.1
    """Extract pollution map in the buffer and convolve with gaussian (smoothing) kernel."""
    pbuf = p.buffer(extgaus)
    get_ras = zonal_stats(pbuf, rasterfile, stats="count", raster_out=True)[0]['mini_raster_array']
    ag = gkern(get_ras.data.shape[0], sd)
    ag = ag / np.sum(ag)
    z = signal.convolve2d(get_ras.data, ag, mode='valid')  # valid does not pad.
    return z[0][0]  # the [0][0] is necessary to return a Float not an array as an array leads to problems


def value_extract(p, rasterfile):
    p_q = point_query(p, rasterfile)[0]
    if p_q is not None:
        return p_q
    else:
        return -99


def getcon2(act_num, rasterfile, df, routegeom, s_route, ext=0.001, extgaus=2, sd=0.1, indoor_ratio=0.7):
    # ph = homelocation, pw = worklocation
    ph, pw = gethw(df)  # homework.loc[j]
    # extracts value of point locations in rasterfile
    con_ph = point_query(ph, rasterfile)[0]
    con_pw = point_query(pw, rasterfile)[0]
    if con_ph is not None and con_pw is not None:
        if act_num < 10:
            if act_num == 1:  # This line is not called in this configuration
                con = indoor_ratio * con_ph  # value_extract(ph, rasterfile)
            elif act_num == 2:
                p_qu = point_query(routegeom, rasterfile)
                if p_qu[0] is not None:  # routegeom = route.loc[j]['geometry']
                    # returns the mean of the values of all route points (nan_to_num not needed as it is checked above)
                    con = np.nanmean(p_qu)  # Add "indoor" factor for bus travel?
                else:
                    con = -99
            # ext=0.001, extgaus=2, gaussd/sd=0.1
            elif act_num == 3:  # indoor_ratio = 0.7
                con = indoor_ratio * con_pw  # value_extract(pw, rasterfile)  # work_indoor
            elif act_num == 4:  # ext=0.001 means 300 m ?
                con = buffermean(ph, ext, rasterfile)  # freetime 1 will change into to sport (second route)
            elif act_num == 5:  # extgaus=2
                con = gaussKernconv(ph, extgaus, rasterfile, sd=0.1)  # freetime 2, distance decay, outdoor.
            elif act_num == 6:
                con = buffermean(ph, ext, rasterfile)  # freetime 3, in garden or terras
        elif act_num < 30:
            # Returns nearest bus stops as Points
            busstop_home, busstop_work = m.get_busstops(u_busstops, ph, pw)
            if act_num == 21:  # walk from home to bus stop and back
                con = (con_ph + value_extract(busstop_home, rasterfile)) / 2
            elif act_num == 23:  # walk from bus stop to work and back
                con = (value_extract(busstop_work, rasterfile) + con_pw) / 2
            elif act_num == 25:
                p_qu = point_query(routegeom, rasterfile)
                if p_qu[0] is not None:  # routegeom = route.loc[j]['geometry']
                    # returns the mean of the values of all route points (nan_to_num not needed as it is checked above)
                    con = np.nanmean(p_qu) * 1.1  # Multiply indoor-outdoor ratio for bus travel
                else:
                    con = -99
            elif act_num == 22:  # wait at bus stop home
                con = value_extract(busstop_home, rasterfile)
            elif act_num == 24:  # wait at bus stop work
                con = value_extract(busstop_work, rasterfile)
        else:
            if act_num == 72:  # shopping from home
                p_qu = point_query(s_route, rasterfile)
                if p_qu[0] is not None:  # routegeom = route.loc[j]['geometry']
                    # returns the mean of the values of all route points (nan_to_num not needed as it is checked above)
                    con = np.nanmean(p_qu)
                else:
                    con = -99
            elif act_num == 71:  # shop location
                shop = Point(float(df.shop_lon), float(df.shop_lat))
                con = indoor_ratio * value_extract(shop, rasterfile)
            elif act_num == 73:  # Shopping on route from work to home
                # Could also do closest point to shop of the route
                # shop = Point(float(df.shop_lon), float(df.shop_lat))
                # shopcon = value_extract(shop, rasterfile)
                # con = (value_extract(shop, rasterfile) + np.nanmean(point_query(routegeom, rasterfile))) / 2
                routecon = np.nanmean(point_query(routegeom, rasterfile))
                con = routecon
        return con
    else:
        print("all -99")
        return {1: -99,
                2: -99,
                3: -99,
                4: -99,
                5: -99,
                6: -99
                }[act_num]


def remove_none(lst):
    # Add all values to list that are not None
    lst = [i for i in lst if i is not None]
    return lst


def cal_exp2(filedir, savedir, iteration, real=False, ext=0.001, extgaus=2, gaussd=0.1, save_csv=True):
    ODdir = savedir + "genloc/"
    if real:  # 1/2 Uni_2
        ODfile = f'h2w_real_{iteration}.csv'
    else:  # 2/2 Uni
        ODfile = f'h2w_{iteration}.csv'
    homework = gpd.read_file(ODdir + ODfile)  # for comparison #gpd can read csv as well, just geom as None.

    routedir = savedir + 'genroute/'
    routefile = f'route_{iteration}.gpkg'  # get route file for all people, only one route file, geodataframe
    shoproutefile = f'route_{iteration}_shop.gpkg'
    route = gpd.read_file(routedir + routefile)
    s_route = gpd.read_file(routedir + shoproutefile)
    # route = route.to_crs('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')

    schedir = savedir + 'gensche/'
    # exp is concentration weighted by time duration
    exp_each_act = []
    exp_each_person = []
    exp_each_act_df = []
    n = len(homework)
    for j in range(n):  # iterate over each person
        # each person has a schedule, only schedule is file per person.
        sched = pd.read_csv(f'{schedir}ws_iter_{iteration}_id_{j}.csv')
        start = sched['start_time']
        end = sched['end_time']
        # dur = sched['duration']
        start_int = np.floor(start).astype(int)
        # for using range,this value should plus 1 as range always omit the last value.
        end_int = np.floor(end).astype(int)
        act_num = sched['activity_code']

        for k in range(sched.shape[0]):  # iterate over each activity in schedule
            conh = 0  # hourly concentration for each activity
            missingtimeh = 0
            missingtime = 0
            if end[k] - start[k] < 1:  # less than 1 hour trip, will just use the concentration hour of the starttime (start_int)
                if start_int[k] == end_int[k]:  # if in the same hour
                    rasterfile = f'{preddir}{name_conrast}{start_int[k]}.tif'
                    con = getcon2(act_num[k], rasterfile, homework.loc[j], route.loc[j]['geometry'],
                                 s_route.loc[j]['geometry'], extgaus=extgaus, sd=gaussd)
                    conh = con * (end[k] - start[k])  # percentage multiply by concentration of the hour
                    # next hour will get the rest of the percentage
                    missingtimeh = 0

                else:  # if not in the same hour
                    constart = getcon2(act_num[k], f'{preddir}{name_conrast}{start_int[k]}.tif', homework.loc[j],
                                      route.loc[j]['geometry'], s_route.loc[j]["geometry"], ext=ext, extgaus=extgaus,
                                      sd=gaussd)
                    conend = getcon2(act_num[k], f'{preddir}{name_conrast}{end_int[k]}.tif', homework.loc[j],
                                    route.loc[j]['geometry'], s_route.loc[j]["geometry"], ext=ext, extgaus=extgaus,
                                    sd=gaussd)
                    # conh = con * (end[k]-start[k])
                    # start percentage multiply by concentration of the hour, next hour will get the rest of the percentage
                    conh = constart * (1 - (start[k] - start_int[k])) + conend * (end[k] - end_int[k])
                    # same as using modf, start percentage multiply by concentration of the hour, end will get the rest of the percentage
                    missingtimeh = 0

            else:  # more than one hour
                for i in range(start_int[k], end_int[k]):  # iterate over activity
                    con = getcon2(act_num[k], f'{preddir}{name_conrast}{i}.tif', homework.loc[j],
                                 route.loc[j]['geometry'], s_route.loc[j]["geometry"], ext=ext, extgaus=extgaus,
                                 sd=gaussd)
                    # control at the beginning and in the end.
                    if i == start_int[k]:  # first hour may be from e.g. 7:20 instead of 7:00
                        # start percentage multiply by concentration of the hour
                        cons = con * (1 - modf(start[k])[0])
                        missingtime = 0
                    elif i == end_int[k]:  # last hour may be to e.g. 9:20 instead of 9:00
                        cons = con * modf(end[k])[0]  # end percentage
                        missingtime = 0
                    else:
                        cons = con  # middle times
                        missingtime = 0

                    # summing exposures
                    conh = conh + cons
                    missingtimeh = missingtimeh + missingtime
            exp = conh / (end[k] - start[k] - missingtimeh + 0.01)  # average exp per activity
            # if not np.isscalar(exp):
            #     exp = exp.item()
            exp_each_act.append(exp)
            # con_each_person.append(np.nanmean(remove_none(con_each_act[k*j : (k+1)*j ]) ))
        # mean_act_exp = np.nanmean(remove_none(exp_each_act[j*sched.shape[0]:(j+1)*sched.shape[0]]), keepdims =False)

        s = sched.shape[0]  # number of rows in schedule
        # print(sum(1 for x in exp_each_act[-s:] if x is None))  # Count None and see if remove_none is needed?
        # # mean_act_exp = statistics.mean(remove_none(exp_each_act[-s:]))  # get the last s items with [-s:]
        mean_act_exp = statistics.mean(exp_each_act[-s:])  # get the last s items with [-s:]
        # Saves the the exposures as list in list not just as a long list
        exp_each_act_df.append(exp_each_act[-s:])
        # add this because sometimes it returns a nested array like [[1]], a strange behaviour of remove_none(nparray):
        # mean_act_exp = np.where(np.isscalar(mean_act_exp), mean_act_exp, mean_act_exp.item()).item()
        exp_each_person.append(mean_act_exp)
        print(f"person: {j}")  # Print Person Number
    if save_csv:
        exposuredir = f"{savedir}exposure/"
        m.makefolder(exposuredir)
        # The lists in list get saved as a Dataframe which behaves like a table
        pd.DataFrame(exp_each_act_df).to_csv(f'{exposuredir}iter_{iteration}_act.csv', index=False)
        # pd.DataFrame(exp_each_act).to_csv(f'{exposuredir}iter_{iteration}_act.csv')
        pd.DataFrame(exp_each_person).to_csv(f'{exposuredir}iter_{iteration}_person.csv')
    return exp_each_act, exp_each_person


# plot
def formattime(timeinput):
    minute, hour = modf(timeinput)
    minute = np.floor(minute * 60)
    return "%02d:%02d" % (hour, minute)


# plot activity exposure
def plotact2(sub1, sub2, act, savename="1", select_start=1, save=False):
    schedir = savedir + 'gensche/'

    fig, ax = plt.subplots(sub1, sub2, figsize=(10, 8),  # (20, 5)
                           sharey=True)
    axs = ax.flatten()
    for i1 in range(sub1 * sub2):
        id1 = i1 + select_start
        sch = pd.read_csv(f'{schedir}ws_iter_{iteration}_id_{id1}.csv')
        st = sch['start_time']
        et = sch['end_time']
        shp = sch.shape[0]
        a = []
        b = []
        # Add starting time and end time into one list with the corresponding exposure values
        for s in range(shp):
            a.append(st[s])
            a.append(et[s])
            b.append(act.iloc[id1, 0:shp][s])
            b.append(act.iloc[id1, 0:shp][s])
        # Plots the exposure graph
        axs[i1].plot(a, b, "ko-", label="Exposure")  # act[i * r:(i + 1) * r]
        axs[i1].set_title(f'person ID: {id1}')
        xlabels = list(range(25))  # To get one label for each hour
        axs[i1].set_xticks(xlabels)
        axs[i1].set_xticklabels(xlabels)
        axs[i1].set_xlabel('hour')
        axs[i1].set_ylabel("Exposure: NO2 in µg/m3", fontsize=10)
        # axs[i1].legend(loc="upper right")
    fig.tight_layout()
    # plt.xlim((14.5, 18.0))
    if save:
        fig.savefig(f'{savedir}{savename}.png')


# change also in streamline_osmnx
rangestart = 30
rangeend = 41

start = time.time()

# simulated locations
for iteration in range(rangestart, rangeend):
    cal_exp2(filedir, savedir, iteration, real=False, save_csv=True)
    print(f"--- iteration {iteration} done 1/2 ---\n")

end = time.time()
print(f"It took {(end - start)/60:.1f} minutes to complete 1/2.\n")

# # known locations
# for iteration in range(rangestart, rangeend):
#     cal_exp2(filedir, savedir2, iteration, real=True, save_csv=False)
#     print(f"--- iteration {iteration} done 2/2 ---\n")

# end2 = time.time()
# print(f"It took {(end2 - end)/60:.1f} minutes to complete 2/2 and {(end - start)/60:.1f} minutes for 1/2.\n")


####################################################################################################
# plotting...


# still doing hourly
ext = 0.001  # 300 m
iteration = 31
ODdir = savedir + "genloc/"
ODfile = f'h2w_{iteration}.csv'
homework = gpd.read_file(ODdir + ODfile)  # for comparison #gpd can read csv as well, just geom as None.

lat = np.array(homework.home_lat).astype(float)
lon = np.array(homework.home_lon).astype(float)

act = pd.read_csv(f"{savedir}exposure/iter_{iteration}_act.csv")  # Because every row is a list we don't need .iloc
person = pd.read_csv(f"{savedir}exposure/iter_{iteration}_person.csv").iloc[:, 1]

df2 = pd.DataFrame({"personal_exposure": person, "lat": lat, "lon": lon})
exp_gdf = gpd.GeoDataFrame(df2["personal_exposure"], crs=4326, geometry=[Point(xy) for xy in zip(df2.lon, df2.lat)])
exp_gdf.to_file(f'{savedir}person_iter{iteration}.gpkg')

# visualise
fig, ax = plt.subplots()
ax.set_aspect('equal')
utrecht = ox.geocode_to_gdf('Utrecht')
utrecht.plot(ax=ax, color="gray", alpha=0.5)
ax.set_ylabel("Latitude")
ax.set_xlabel("Longitude")
exp_gdf.plot(ax=ax, column='personal_exposure', legend=True)

# plot activity
plotact2(sub1=2, sub2=2, act=act, savename="more", select_start=20, save=False)
plt.show()
plt.close('all')


# Save all exposure values and routes in one file
import glob

files = glob.glob(f"{savedir}exposure/*person*")

df_from_each_file = (pd.read_csv(f, sep=",").iloc[:, 1] for f in files)
df_merged = pd.concat(df_from_each_file, ignore_index=True, axis=1)
df_merged.columns = [str(x) for x in range(rangestart, rangeend)]
df_merged["lat"] = lat
df_merged["lon"] = lon

sh = df_merged.shape
print(f"shape of df_merged: {sh[0]} rows x {sh[1]} columns")
df_merged.to_csv(f"{savedir}/allperson.csv")

exp_gdf = gpd.GeoDataFrame(df_merged, crs=4326, geometry=[Point(xy) for xy in zip(df_merged.lon, df_merged.lat)])
repr(exp_gdf).encode('utf_8')
exp_gdf.to_file(f'{savedir}person_all.gpkg', driver="GPKG")
# exp_gdf.plot()
# plt.show()
