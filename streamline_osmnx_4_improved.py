#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:55:06 2023
@author: janhopfer
"""
import model_functions4_improved as m
import pandas as pd
from pandas import DataFrame
import geopandas as gpd
import time
import os
import osmnx as ox
import osmnx.io
import random
import numpy as np
import scipy.stats
import networkx as nx
from shapely.geometry import Point, shape, LineString

os.chdir("core")

##############
#  please enter the directory, and specify a directory for saving your files
################

filedir = "C:/Users/Jan/PycharmProjects/Thesis2/"  # directory
savedir0 = "C:/Users/Jan/PycharmProjects/Thesis2/results1/r1_improved/"  # save your results

# change values also in calc_exposure
rangestart = 30
rangeend = 41
shopbool = True  # Simulate with shopping
busbool = True  # Simulate with bus travel
shop_perc = 0.8  # probability of going shopping # 1 for _all # 0.3 for other simulations
shop_home = 0.5  # normally 0.5 (50 %) # 1 = all go from home
shop_route = 0.3  # normally 0.3 (30 %) # 1 = all try on route

# initiate result folders
resultfolder = "Uni"  # result folder
resultfolder_2 = "Uni_2"  # for comparison
m.makefolder(f"{savedir0}{resultfolder}/")
m.makefolder(f"{savedir0}{resultfolder_2}/")
savedir = f"{savedir0}{resultfolder}/"  # each profile a savedir.
savedir2 = f"{savedir0}{resultfolder_2}/"  # savedir2 for comparison.

Gw = ox.io.load_graphml(filepath=f'{filedir}input/osmnx_graph/ut10kwalk.graphml')
# note: add speed only works for freeflow travel speeds, osm highway speed
Gb = ox.io.load_graphml(filepath=f'{filedir}input/osmnx_graph/ut10bike.graphml')
Gd = ox.io.load_graphml(filepath=f'{filedir}input/osmnx_graph/ut10drive.graphml')

Ovin = pd.read_csv(f'{filedir}input/Ovin_mini.csv', usecols=["MaatsPart", "Doel", "age_lb", "KAf_mean"])  # Ovin_mini is Ovin with just the columns used in the module

# travel probability for university students
# _new.csv has the same probabilities like travelmean_from_distance2work for NL_student
# otherwise a person could have walked 30 km to work which seems unlikely
f_d_wo = pd.read_csv(f"{filedir}input/distprob/Uni_stu_uni_new.csv")
# general travel probability for university students going also by bus
f_d_bus = pd.read_csv(f"{filedir}input/distprob/Uni_stu_uni_bus2.csv")
# travel probability for university students going from home to the grocery shop (without bus)
f_d_shop = pd.read_csv(f"{filedir}input/distprob/Uni_stu_uni_shop1.csv")

""" generate destination locations """
# _new.csv contains only locations inside the extend of the concentration raster
Uni_ut_home_new = pd.read_csv(f"{filedir}input/locationdata/Uni_Ut_home_new.csv")

# all possible work locations (all the possible uni locations).
# in the university student scenario we used all the university locations instead of this one.
Uni_ut_work = pd.read_csv(f"{filedir}input/locationdata/Ut_Uni_coll.csv")
# _new3.csv contains only locations inside the extend of the concentration raster
Uni_ut_homework_new3 = pd.read_csv(f"{filedir}input/locationdata/Uni_Ut_homework_new3.csv")

busdir = f"{filedir}input/OSM/bus4_highway.csv"
shopdir = f"{filedir}input/OSM/con_sup_1.csv"

# The function check_in_extend() can evaluate if all location points are inside the raster extend
rast = f"{filedir}input/hr_pred/hr_pred_utrecht_X0.tif"
m.check_in_extend(rast, Uni_ut_home_new, [""])


nr_locations = Uni_ut_homework_new3.shape[0]  # Counts how many rows the table has
print(f"nr_locations (n): {str(nr_locations)}\n")

ite = 1  # first iteration


# This function is run for each iteration
def generate_activity_all(Gw, Gb, Gd, savedir, bustravel, shopping, ite, busdir, shopdir, real_od=None, ori=None,
                          des=None, n=nr_locations, dist_var=None, des_type=None, sopa=None, age_from=None, age_to=None,
                          shop_percent=0.8):
    """ Generates the activity schedules with routes.
    :param Gw, Gb, Gd: graphs for walk, bike, drive
    :param str savedir: directory to save mobiresults
    :param bool bustravel: if True agent could travel by bus
    :param bool shopping: if True agent could go shopping
    :param int ite: iteration id
    :param Dataframe real_od: real od matrix (for validation), if is None, no simulation. Origin and destination need to be provided, as a dataframe with: home_lon, home_lat, work_lon, work_lat
    :param Dataframe ori: home location
    :param Dataframe des: all the potential work location
    :param int n: number of agents (home locations) to calculate, sometimes we don't want to calculate for each home location.
    :param Dataframe dist_var: survey data used: Ovin = pd.read_csv(f'{filedir}human_data/dutch_activity/Ovin.csv')
    :param str des_type: destination type (in Ovin), e.g. "work"
    :param str sopa: socioeconomics status (in Ovin) e.g. "scholar"
    :param int age_from: age range #age_from=18
    :param int age_to: age range #age_to=99
    :param shop_percent: percentage of people going shopping (0.3 means 30%)
    """
    # make new folders only if the directory (dir) doesnt exist.
    m.makefolder(f"{savedir}genloc")
    m.makefolder(f"{savedir}genroute")
    m.makefolder(f"{savedir}gensche")

    # 1. generate OD locations
    if real_od is None:  # Uni
        # csvname = f'{savedir}genloc/h2w_{ite}'
        df = m.storedf(homedf=ori, goal=des, dist_var=dist_var, des_type=des_type, sopa=sopa,
                       age_from=age_from, age_to=age_to, n=n)  # des = Uni_ut_work.csv
    else:  # Uni_2
        df = real_od  # = Uni_ut_homework.csv or Uni_ut_homework_new3.csv

    if shopping:
        allshoproute = []
        allshoptravel_time = []
        allshoptravel_distance = []
        allshoptravel_mean = []
        allshop_num_candi = []
        shoplon = []
        shoplat = []

    allroutes = []
    alltravel_time = []
    alltravel_distance = []
    alltravel_mean = []

    # 2. generate activity schedules
    for id in range(n):  # here n = nr_locations
        if bustravel:  # travel_mean can be bus
            # Get travel mean and home, work locations
            travel_mean, start, end, G, speed, start_bus, end_bus, mindist_start, mindist_end\
                = m.travelmean_bus(id, df, f_d_bus, Gw, Gb, Gd, m.input_bus_stops(busdir))
        else:  # travel_mean is never bus
            # Get travel mean and home, work locations
            travel_mean, start, end, G, speed = m.travelmean(id, df, f_d_wo, Gw, Gb, Gd)
        shop_num_candi = -9
        shop = Point(0.0, 0.0)
        if shopping and random.random() < shop_percent:
            shops = m.input_shops(shopdir)  # Dataframe with OSM shops
            if travel_mean not in ["bus"]:  # foot, bike, car try route
                if random.random() < shop_route:
                    try:
                        route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start, end, G, speed)
                    except:
                        print(f"no path {id} beforeBuffer")  # Specify no path
                    # Radius for buffer dependent on travel mean
                    bufferradius = m.get_bufferradius(travel_mean)
                    # check number of shops in route buffer
                    shop_num_candi, center_shop = m.checkroutebuffer(route, shops, bufferradius)
                    if shop_num_candi > 0 and route != start:  # Check if at least one shop is on route
                        # Get shop for shopping on route
                        shop = center_shop
                        # schedule for shopping with route # w_sr_s = work_shop-on-route_sport
                        m.schedule_general_shop_r(travel_time, travel_mean, savedir + "gensche",
                                                  name=f"ws_iter_{ite}_id_{id}")  # ws: work and sport
                        s_route, s_travel_time, s_travel_distance, s_travel_mean = None, 0, 0, None
                    else:  # No shop on route, therefore shopping from home
                        try:
                            # route for shopping
                            s_route, s_travel_time, s_travel_distance, s_travel_mean, shop =\
                                m.getroute_shop(id, f_d_shop, Gw, Gb, Gd, shops, home=start)
                        except:
                            print(f"no path to shop {id} NotonRoute")
                            # Do schedule without shop??
                        else:
                            # schedule for shopping from home
                            m.schedule_general_shop_h(travel_time, s_travel_time, travel_mean, s_travel_mean,
                                                      savedir + "gensche", name=f"ws_iter_{ite}_id_{id}")
                elif random.random() < shop_home:  # 50 % go directly from home and don't try on route
                    try:
                        route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start, end, G, speed)
                        s_route, s_travel_time, s_travel_distance, s_travel_mean, shop = \
                            m.getroute_shop(id, f_d_shop, Gw, Gb, Gd, shops, home=start)
                    except:
                        print(f"no path {id} fromHome")
                    else:
                        # schedule for shopping from home # w_sh_s = work_shop-from-home_sport
                        m.schedule_general_shop_h(travel_time, s_travel_time, travel_mean, s_travel_mean,
                                                  savedir + "gensche", name=f"ws_iter_{ite}_id_{id}")
                else:  # Rest does not go shopping
                    try:
                        s_route, s_travel_time, s_travel_distance, s_travel_mean = None, 0, 0, None
                        if travel_mean == "bus":  # Is that needed????
                            route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start_bus, end_bus, G,
                                                                               speed)
                        else:
                            route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start, end, G, speed)
                    except:
                        print(f"no path {id} noShop")
                    else:
                        if travel_mean == "bus":  # Is that needed????
                            m.schedule_general_bus(travel_time, savedir + "gensche", mindist_start, mindist_end,
                                                   name=f"ws_iter_{ite}_id_{id}")
                        else:
                            m.schedule_general_wo(travel_time, travel_mean, savedir + "gensche",
                                                  name=f"ws_iter_{ite}_id_{id}")  # ws: work and sport
            else:  # bus and train go shopping from home because route can not be "detoured"
                try:
                    route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start_bus, end_bus, G, speed)
                    s_route, s_travel_time, s_travel_distance, s_travel_mean, shop = \
                        m.getroute_shop(id, f_d_shop, Gw, Gb, Gd, shops, home=start)
                except:
                    print(f"no path {id} bus")
                else:
                    if travel_mean == "bus":
                        # schedule for shopping from home with bus
                        m.schedule_general_bus_shop_h(travel_time, s_travel_mean, s_travel_time, savedir + "gensche",
                                                      mindist_start, mindist_end, name=f"ws_iter_{ite}_id_{id}")
                    else:  # For train or if bus stops are too far from home or destination
                        # schedule for shopping from home
                        m.schedule_general_shop_h(travel_time, s_travel_time, travel_mean, s_travel_mean,
                                                  savedir + "gensche", name=f"ws_iter_{ite}_id_{id}")
        else:  # People who don't go shopping -> use same functions as before
            try:
                s_route, s_travel_time, s_travel_distance, s_travel_mean = None, 0, 0, None
                if travel_mean == "bus":
                    route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start_bus, end_bus, G, speed)
                else:
                    route, travel_time, travel_distance = m.getroute_2(id, travel_mean, start, end, G, speed)
            except:
                print(f"no path {id} noShop")
            else:
                if travel_mean == "bus":
                    m.schedule_general_bus(travel_time, savedir + "gensche", mindist_start, mindist_end,
                                           name=f"ws_iter_{ite}_id_{id}")
                else:
                    m.schedule_general_wo(travel_time, travel_mean, savedir + "gensche",
                                          name=f"ws_iter_{ite}_id_{id}")  # ws: work and sport

        if shopping:
            shoplon.append(shop.x)
            shoplat.append(shop.y)

            # generated activities and save the tables
            allshoproute.append(s_route)
            allshoptravel_time.append(s_travel_time)
            allshoptravel_distance.append(s_travel_distance)
            allshoptravel_mean.append(s_travel_mean)
            allshop_num_candi.append(shop_num_candi)

        # generated activities and save the tables
        alltravel_time.append(travel_time)
        alltravel_distance.append(travel_distance)
        alltravel_mean.append(travel_mean)
        allroutes.append(route)

    # 3. generate OD routes.
    d = {'duration_s': m.roundlist(alltravel_time), 'distance_m': m.roundlist(alltravel_distance),
         'travel_mean': alltravel_mean, 'geometry': allroutes}
    gpd1 = gpd.GeoDataFrame(d, crs=4326)

    if shopping:
        d2 = {'duration_s': m.roundlist(allshoptravel_time), 'distance_m': m.roundlist(allshoptravel_distance),
              'travel_mean': allshoptravel_mean, 'geometry': allshoproute}
        gpd2 = gpd.GeoDataFrame(d2, crs=4326)

        # Save shop locations to df
        df["shop_lon"] = np.array(shoplon)  # np.array(mylist)
        df["shop_lat"] = np.array(shoplat)

    # Save df
    if real_od is None:  # Uni
        df.to_csv(f'{savedir}genloc/h2w_{ite}.csv')
    if savedir is not None:
        # df.to_csv(f'{savedir}genloc/h2w_real.csv')
        df.to_csv(f'{savedir}genloc/h2w_real_{ite}.csv')  # not generating the route but save the real location files

    exception = []
    try:
        gpd1.to_file(f'{savedir}genroute/route_{ite}.gpkg')
        if shopping:
            gpd2.to_file(f'{savedir}genroute/route_{ite}_shop.gpkg')
    except RuntimeError:
        exception.append({ite: ite, id: id})
        print("skip:", exception)
    return exception


start = time.time()

# generated activities and get the routes

# exc = exception
for ite in range(rangestart, rangeend):
    exc = generate_activity_all(ori=Uni_ut_home_new, des=Uni_ut_work, real_od=None, n=nr_locations, ite=ite,
                                dist_var=Ovin, des_type="work", sopa="Scholier/student",
                                busdir=busdir, shopdir=shopdir,
                                age_from=18, age_to=99, Gw=Gw, Gb=Gb, Gd=Gd,
                                savedir=savedir, bustravel=busbool, shopping=shopbool, shop_percent=shop_perc)
    print(f"----  iteration {ite} done  1/2 ----\n")

end = time.time()
print(f"It took {(end - start)/60:.1f} minutes to complete 1/2\n")

# # if real locations are known
# for ite in range(rangestart, rangeend):
#     exc = generate_activity_all(ori=Uni_ut_home_new, real_od=Uni_ut_homework_new3, n=nr_locations, ite=ite,
#                                 dist_var=Ovin, des_type="work", sopa="Scholier/student",
#                                 age_from=18, age_to=99, Gw=Gw, Gb=Gb, Gd=Gd,
#                                 savedir=savedir2, bustravel=busbool, shopping=shopbool, shop_percent=shop_perc)
#     print(f"----  iteration {ite} done  2/2 ----\n")
#
# end2 = time.time()
# print(f"It took {(end2 - end)/60:.1f} minutes to complete 2/2\n")
