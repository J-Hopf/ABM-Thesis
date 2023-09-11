#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:55:57 2023
@author: janhopfer
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from matplotlib import pyplot as plt
import osmnx as ox
import random


random.seed(10)
os.chdir("core")

filedir = "C:/Users/Jan/PycharmProjects/Thesis2/"
results_dir = f"{filedir}results1/r1_improved/"  # use same folder as streamline_osmnx
savedir = f"{results_dir}Uni/"  # each profile a savedir.

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

# still doing hourly
iteration = 31
start_number = 20
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

# plot activity schedule
plotact2(sub1=2, sub2=2, act=act, savename="more", select_start=start_number, save=False)

plt.show()
plt.close('all')
