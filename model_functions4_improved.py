#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:55:20 2023
@author: janhopfer
"""

import pandas as pd
from pandas.core import frame
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Point, LineString, Polygon, shape
from shapely.ops import transform, nearest_points
import shapely.speedups
import scipy
import scipy.stats
import networkx as nx
import os
import osmnx as ox
import osmnx.distance
import pyproj
import random
import rasterio


##### The following functions were adapted with minor changes from Men Lu #####

def mean_sd(df):
    """ Get the mean and standard deviation (SD) given a dataframe column.
    :param df: DataFrame
    :return: *mu, *sd
    """
    mu = df.apply(lambda x: np.mean(x))
    sd = df.apply(lambda x: np.std(x))
    return *mu, *sd


def lognormal_mean_sd(df):
    mu = df.apply(lambda x: x + 1).apply(lambda x: np.log(x)).apply(lambda x: np.mean(x))
    sd = df.apply(lambda x: x + 1).apply(lambda x: np.log(x)).apply(lambda x: np.std(x))
    return *mu, *sd


def lognormal_mean_sd_scipy(df):
    """ Log transform and then get mean and standard deviation (SD) given a dataframe column,
    using scipy, same as log_mean_sd
    :return: mu, sd
    """

    shape, loc, scale = scipy.stats.lognorm.fit(df.apply(lambda x: x + 1), floc=0)
    sd = shape
    mu = np.log(scale)
    return mu, sd


def gen_lognormal(df, method="manual"):
    """ generate lognormal distribution, "manual" use log_mean_sd, else use scipy to get mean and sd.
    :param df: mean distance
    :param method: "manual" -> lognormal_mean_sd(), else lognormal_mean_sd_scipy()
    :return: sim_dist: Picks from lognormal distribution of distance input
    """
    if method == "manual":
        mu, sd = lognormal_mean_sd(df)
    else:
        mu, sd = lognormal_mean_sd_scipy(df)
        # mu1, sd1 = log_mean_sd(df)
        # print(mu-mu1,sd-sd1) for checking purpose. same mobiresults
    sim_dist = np.random.lognormal(mu, sd, 1)[0] - 1
    if sim_dist < 0:
        sim_dist = 0.1
    return sim_dist


def pot_dest(goal):
    """ calculate once to make it a function.
    :param goal:  a dataframe of lon, lat
    :returns: geopandas dataframe (w_gdf) and a union (u) of its elements.
    """
    if type(goal) is pd.core.frame.DataFrame:
        w_gdf = gpd.GeoDataFrame(crs=4326, geometry=[Point(xy) for xy in zip(goal.lon, goal.lat)])
    else:  # geopandas
        w_gdf = goal
    u = w_gdf.unary_union
    return w_gdf, u


def distanceOvin(Ovin, socialpart="Scholier/student", age_from=18, age_to=50):
    """ Returns distance based on social occupation and travel goal,
    Output: work, outdoor, shopping distances,
    Based on the Ovin dataset paprameter name
    :param Ovin: Input dataset
    :param socialpart: Filters Ovin dataset based on social occupation/partition
    :param age_from: Filters Ovin for age (min)
    :param age_to: Filters Ovin for age (max)
    :return: all distances from column KAf_mean (km) in the Ovin database as work_dist, outdoor_dist and shopping_dist
    """
    if socialpart == "Scholier/student":  # only consider eductation for students
        work_dist = Ovin.query('Doel == "Onderwijs/cursus volgen" & MaatsPart=="{0}" & {1} <= age_lb <={2}'
                               .format(socialpart, age_from, age_to))[['KAf_mean']]  # Why is initially no age here?
    else:
        work_dist = Ovin.query('Doel == "Werken" & MaatsPart=="{0}" & {1} <= age_lb <={2}'
                               .format(socialpart, age_from, age_to))[['KAf_mean']]
    outdoor_dist = Ovin.query('Doel == "Sport/hobby" &  MaatsPart=="{0}" & {1} <= age_lb <={2} '
                              .format(socialpart, age_from, age_to))[['KAf_mean']]
    shopping_dist = Ovin.query('Doel =="Winkelen/boodschappen doen"& MaatsPart=="{0}" & {1} <= age_lb <={2}'
                               .format(socialpart, age_from, age_to))[['KAf_mean']]
    return work_dist, outdoor_dist, shopping_dist


# for students, "work" -> education
# getdestloc(p, w_gdf, u, dist_var=Ovin, des_type="work", sopa="Scholier/student", age_from=18, age_to=99)
def getdestloc(homep, w_gdf, u, Ovin, goal="work", sopa="Scholier/student", age_from=18, age_to=50):
    """
    Get destination location point, this one needs the nongeneral distanceOvin function above. If you have the
    distance (lognormal distributed) as input, use the function getdestloc_simple. Note for the destination point
    selection, only a variable of potential distances is needed (for characterising the distribution for simulation).
    :param homep: home point
    :param w_gdf: destination (e.g. work) geopandas
    :param u: union geopandas
    :param Ovin:
    :param goal: activity for getting the distance (input as des_type), sopa: social participation for getting the distance
    :param sopa: social partition for instance "Scholier/student"
    :param age_from:
    :param age_to:
    :return: des_p: destinaton point,  num_p: number of candidate points (to sample a destination point from)
    """
    nearestpoint = nearest_points(homep, u)[1]  # get nearest point
    mindist = homep.distance(nearestpoint)  # find the distance from home point to the nearest point
    work_dist, outdoor_dist, shopping_dist = distanceOvin(Ovin=Ovin, socialpart=sopa, age_from=age_from,
                                                          age_to=age_to)  # get KAf_mean distances
    # calculate distance (radius)  # gets one distance value from a lognorm distribution fitted to the work_dist values
    if goal == "work":  # goal = ges_type
        sim_dist = gen_lognormal(work_dist, "")  # gen_lnormal returns float
        sim_distdeg = sim_dist / 110.8  # convert to degree
    elif goal == "sport":
        sim_dist = gen_lognormal(outdoor_dist, "")
        sim_distdeg = sim_dist / 110.8
    elif goal == "shopping":
        sim_dist = gen_lognormal(shopping_dist, "")
        sim_distdeg = sim_dist / 110.8
    else:
        print("define variable goal as work, sport or shopping")
        sim_distdeg = 0

    # if the distance is shorter than the distance to the nearest points, take the nearest point.
    if sim_distdeg < mindist:
        # sim_distdeg = mindist
        des_p = nearestpoint
        num_points = 0
        print(f'use nearest point as simulated distance is too short.')
    else:  # calculate a buffer around home point and select point inside buffer.
        point_buffer = homep.buffer(sim_distdeg)  # distance to degree`maybe better to project.
        worklocset = w_gdf[w_gdf.within(point_buffer)]
        num_points = len(worklocset)  # number of points inside the buffer
        print(f'sample from {num_points} points')
        workloc = worklocset.sample(n=1, replace=True, random_state=1)  # chose one point
        des_p = workloc.iloc[0]["geometry"]  # get point out of the geopandas dataframe
    return des_p, num_points


def getdestloc_simple(homep, w_gdf, u, dist_var):  # for students, work -> eductation
    """
    input a distance dataframe (single-columned or select a column) instead of using the Ovin for it.
    otherwise same as getdestloc
    :param homep: home point
    :param w_gdf: destination (e.g. work) geopandas
    :param u: union geopandas
    :param dist_var:
    :return: des_p, num_points
    """
    nearestpoint = nearest_points(homep, u)[1]  # get nearest point
    min_dist = homep.distance(nearestpoint)  # find the distance to the nearest point
    # print(mindist) degree
    sim_dist = gen_lognormal(dist_var, "")  # dist_var = Ovin
    sim_distdeg = sim_dist / 110.8

    if sim_distdeg < min_dist:  # if the distance is shorter than the distance to the nearest points. take the nearest point
        # sim_distdeg = min_dist
        des_p = nearestpoint
        num_points = 0
        print(f'use nearest point as simulated distance is too short.')
    else:  # else calculate a buffer and select points.
        pbuf = homep.buffer(sim_distdeg)  # distance to degree`maybe better to project.
        # pbuf.crs={'init': 'epsg:4326', 'no_defs': True}
        # pbuf.crs= 4326
        worklocset = w_gdf[w_gdf.within(pbuf)]
        num_points = len(worklocset)  # number of points inside the buffer
        print(f'sample from {num_points} points')
        workloc = worklocset.sample(n=1, replace=True, random_state=1)
        des_p = workloc.iloc[0]["geometry"]  # get point out of the geopandas dataframe
    return des_p, num_points


def storedf(homedf, goal, dist_var, des_type="work", sopa="Scholier/student", age_from=18, age_to=50, n=50):
    """ main function calculating all the destination points to dataframe.

    output: return the dataframe of lat lon of the original and destination locations.
    :param homedf: dataframe of home (original) locations, names have to be "lat, lon"
    :param goal: (des) dataframe or geopandas dataframe of work (destination) locations. names have to be "lat lon"
    :param dist_var: here it is the Ovin dataframe
    :param des_type:
    :param sopa: social participation
    :param age_from:
    :param age_to:
    :param n: number of first "n" points to calculate, e.g. n = 4, only calculate for the first 4 points
    :return: totalre: pandas dataframe with [home_lon, home_lat, work_lon, work_lat, number of candicate destinations]
    """
    totalarray = []
    w_gdf, u = pot_dest(goal)  # get work(destination location)  # goal (des) = Uni_ut_work.csv
    for id in range(n):  # n = nr_locations
        # get one location from Uni_ut_home.csv
        h = homedf.loc[id]
        homep = Point(h.lon, h.lat)
        # Only if dist_var has only one column
        if dist_var.shape[1] == 1:  # dist_var = Ovin
            # input p and output op got dropped as it is the same and remains unchanged inside the function
            des_p, num_p = getdestloc_simple(homep, w_gdf, u, dist_var)
        else:  # With Ovin es input it is always else
            des_p, num_p = getdestloc(homep, w_gdf, u, dist_var, des_type, sopa, age_from=age_from, age_to=age_to)

        # Add shopping location and num_candi_shop into array?
        totalarray.append(pd.DataFrame({"home_lon": homep.centroid.x, "home_lat": homep.centroid.y,
                                        "work_lon": des_p.centroid.x, "work_lat": des_p.centroid.y, "num_candi": num_p,
                                        }, index=[id]))
    totalre = pd.concat(totalarray)  # Merge list of dataframes into one dataframe. Faster than doing it several times
    return totalre


# Not in use right now
def travelmean_from_distance2work(dis, param=None):
    tra_mode = ['car', 'bicycle', 'foot', "train"]
    if param is None:
        if dis < 1000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.001, 0.1, 0.899, 0])[0]
        elif dis < 6000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.05, 0.9, 0.05, 0])[0]
        elif dis < 10000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.8, 0.2, 0.000, 0])[0]
        else:
            cls_ = np.random.choice(tra_mode, 1, p=[1, 0, 0, 0])[0]

    elif param == "NL":  # parameterise from the Ovin2014 dataset
        if dis < 1000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.14, 0.53, 0.33, 0])[0]
        elif dis < 2500:
            cls_ = np.random.choice(tra_mode, 1, p=[0.2, 0.6, 0.2, 0])[0]
        elif dis < 3700:
            cls_ = np.random.choice(tra_mode, 1, p=[0.3, 0.65, 0.05, 0])[0]
        elif dis < 5000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.4, 0.55, 0.05, 0])[0]
        elif dis < 7500:
            cls_ = np.random.choice(tra_mode, 1, p=[0.6, 0.4, 0.00, 0])[0]
        elif dis < 10000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.7, 0.3, 0.00, 0])[0]
        elif dis < 15000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.9, 0.1, 0, 0])[0]
        else:
            cls_ = np.random.choice(tra_mode, 1, p=[0.9, 0, 0, 0.1])[0]  # the travel by by train needed at 0.1

    elif param == "NL_student":  # parameterise from the Ovin2014 dataset for school child and student, they cycle more and take more public transport
        if dis < 1000:
            cls_ = np.random.choice(tra_mode, 1, p=[0, 0.5, 0.5, 0])[0]
        elif dis < 2500:
            cls_ = np.random.choice(tra_mode, 1, p=[0.1, 0.7, 0.2, 0])[0]
        elif dis < 3700:
            cls_ = np.random.choice(tra_mode, 1, p=[0.3, 0.7, 0.0, 0])[0]
        elif dis < 5000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.35, 0.65, 0.0, 0])[0]  # start bus 0,1 bestuurder auto 0.1
        elif dis < 7500:
            cls_ = np.random.choice(tra_mode, 1, p=[0.45, 0.55, 0.0, 0])[0]
        elif dis < 10000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.6, 0.4, 0.0, 0])[0]  # auto include metro 0.1 , bus 0,1
        elif dis < 15000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.5, 0.4, 0, 0.1])[0]  # also include train
        elif dis < 30000:
            cls_ = np.random.choice(tra_mode, 1, p=[0.7, 0.1, 0, 0.2])[0]  # electronic or brombike
        else:
            cls_ = np.random.choice(tra_mode, 1, p=[0.3, 0, 0, 0.7])[0]  # the travel by by train needed at 0.7

def travelmean_from_distance2work_df(f_d, dis):
    """
    :param f_d: csv with probabilities for traveling
    :param dis: distance from home to work in kilometers
    :return: cls_: chosen travel mode
    """
    tra_mode = ['car', 'bicycle', 'foot', 'bus', 'train']  # But train here only for long distances
    # counts how far the dis goes down the list of distances to get the probabilities for the matching distance
    # prob = f_d.iloc[-sum(f_d.iloc[:, 0].values > dis), 1:].values  # Use instead the method below
    indx = np.count_nonzero(f_d.iloc[:, 0].values > dis)
    prob = f_d.iloc[-indx, 1:].values  # It seems that for f_d only 0.1 not 0.11 values are possible
    # one travel mode is picked depended of the probabilities
    cls_ = np.random.choice(tra_mode, 1, p=prob)[0]
    return cls_


# separating route model and schedule model
# duration and travelmean comes from the getroute_2 function.
# "travelmean" is not used for generating the schedule, only as input so that the generated schedule has "travelmean".
# "wo" means work and sports, for one person.
# input travel duration (time in seconds)
def schedule_general_wo(duration, travelmean, filedir, free_duration=1, name="work_sport", save_csv=True,
                        time_interval=0.017):  # 0.017 is the same as 1 minute as we use decimal time
    h2w_mean, w2h_mean = 8, 16  # mean time left home to go working (h2w) and mean time left work to go home (w2h)
    h2w_sd, w2h_sd = 1, 1  # standard deviation is 1 hour for both

    # to get at least a travel duration of 1 minute as time is here in decimal and rounded to 0.01
    # this was helpful as otherwise I got negative results
    mintime = 0.03  # 0.02 would be equivalent to 1 min 12 sec / 0.01 to only 36 sec

    # go to work
    home2work_start = np.random.normal(h2w_mean, h2w_sd, 1)[0]
    home2work_end = home2work_start + (duration / 3600)
    # checking that home2work_end is never before or equal to home2work_start # If travel duration is very short
    if home2work_end < home2work_start + mintime:  # Could also check only once as both routes are the same?
        home2work_end = home2work_start + mintime

    # go home
    work2home_start = np.random.normal(w2h_mean, w2h_sd, 1)[0]
    work2home_end = work2home_start + (duration / 3600)  # meaning at home again
    # checking that work2home_end is never before or equal to work2home_start  # If travel duration is very short
    if work2home_end < work2home_start + mintime:
        work2home_end = work2home_start + mintime

    if work2home_end > 23.5:
        if work2home_start >= 23.5:
            print("work2home_start later than work2home_end!")
        work2home_end = 23.5  # if going home after 23:30, we will assume him going home at 23:30

    if work2home_end > 20.5:  # at home if goes home late already.
        outdoor_evening = 23.7  # just set to the end of the day as the freetime is still at home.
        free_duration = time_interval  # just any small number as the freetime is still at home.
        freetime = 1  # at home if goes home late already.
    else:  # if goes home relatively early, the person will go outdoor
        freetime = np.random.choice([1, 4, 5, 6])
        # free time has 4 modes: 1. at home, 4. to sports, 5. 2000m buffer around home, 6. random walk around home

        # after arriving at home, people go freetime after about 1 or 1.5 hours
        outdoor_evening = work2home_end + np.random.choice([1, 1.5])

        if outdoor_evening + free_duration > 23.5:
            free_duration = 23.5 - outdoor_evening  # time should not go over 23.9

    # Add shopping to shedule
    start_time = np.round(np.array([0.0, home2work_start, home2work_end, work2home_start, work2home_end,
                                    outdoor_evening, outdoor_evening + free_duration]), 2)
    end_time = np.round(np.array([*start_time[1: len(start_time)] - 0.01, 23.9]), 2)  # =start_time - 0.01
    activity = ["home", "h2w", "work", "w2h", "home", "free_time", "home"]
    activity_code = [1, 2, 3, 2, 1, freetime, 1]
    # activity_code: 1: home, 2: h2w or w2h, 3: work,
    # 4: h2sport,
    # 5: 2000m buffer around home (the person can be anywhere around home),
    # 6: random walk around home

    # This adds travelmean only to activity h2w or w2h and not just in row 1 as before
    travel2 = [None, travelmean, None, travelmean, None, None, None]
    # Use the following if the schedule is not fixed (now inactive for computational reasons)
    # travel = []
    # for act in activity_code:
    #     travel.append(travelmean) if act == 2 else travel.append(None)

    # Get duration of each activity and not just in line 0 as before
    duration = np.round(end_time - start_time, 2)
    # input directly into dataframe instead of adding to a list, transforming and renaming
    schedule = pd.DataFrame({"start_time": start_time, "end_time": end_time, "activity": activity,
                             "activity_code": activity_code, "travel_mean": travel2, "duration": duration})

    if save_csv:
        schedule.to_csv(f'{filedir}/{name}.csv')  # name=f"ws_iter_{ite}_id_{id}"
    return schedule


def getstartend(id, df):
    xcoord_home = float(df.loc[id, "home_lon"])
    ycoord_home = float(df.loc[id, "home_lat"])
    xcoord_work = float(df.loc[id, "work_lon"])
    ycoord_work = float(df.loc[id, "work_lat"])
    start = Point(xcoord_home, ycoord_home)  # before: (ycoord_home, xcoord_home)
    end = Point(xcoord_work, ycoord_work)  # before: (ycoord_work, xcoord_work)
    return start, end


def qget(id, df):
    x1 = float(df.loc[id, "home_lon"])
    y1 = float(df.loc[id, "home_lat"])
    x2 = float(df.loc[id, "work_lon"])
    y2 = float(df.loc[id, "work_lat"])
    return y1, x1, y2, x2


def nodes_to_linestring(route, G):
    coords_list = [(G.nodes[i]['x'], G.nodes[i]['y']) for i in route]
    line = LineString(coords_list)
    return line


def roundlist(list_, n=2):
    """ Rounds every item of the list to n decimals.
    :param list_: input list
    :param n: decimals for rounding
    :return: rounded list
    """
    return list(np.round(np.array(list_), n))


def makefolder(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)


def wgs2laea(p):  # also in cal_exposure
    wgs84 = 4326
    rd = pyproj.CRS('+proj=laea +lat_0=51 +lon_0=9.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    project = pyproj.Transformer.from_crs(wgs84, rd, always_xy=True)
    p = transform(project.transform, p)
    return p


###### The following functions were added by Jan Hopfer (often with inspiration or code pieces from Meng Lu) #####


def input_bus_stops(filedir):  # Function to get bus stops
    """ Makes GeoDataFrame out of csv with bus stop locations
    :return: GeoDataFrame with bus stops
    """
    OSMstops = pd.read_csv(filedir)  # Open csv with coordinates of bus stops
    geom_bus = gpd.GeoSeries.from_wkt(OSMstops['geometry'])  # Get geometry from csv
    busstops = gpd.GeoDataFrame(OSMstops, geometry=geom_bus, crs="EPSG:4326")  # Make GeoDataFrame
    u_busstops = busstops.unary_union  # Get MultiPoint shape from GeoDataFrame
    return u_busstops


def input_shops(filedir):  # Function to get shops
    """ Makes GeoDataFrame out of csv with shop locations
    :return: GeoDataFrame with shops
    """
    OSMshops = pd.read_csv(filedir)  # Open csv with coordinates of bus stops
    geom_shop = gpd.GeoSeries.from_wkt(OSMshops['geometry'])  # Get geometry from csv
    shops = gpd.GeoDataFrame(OSMshops, geometry=geom_shop, crs="EPSG:4326")  # Make GeoDataFrame
    return shops


def get_busstops(busstops, home, destination):
    """ Gets closest bus stops to home and destination location
    :param busstops: stops as MultiPoint from .unary_union
    :param home: home Point
    :param destination: destination Point
    :return: closest busstops to home and destination
    """
    # Get coordinates of nearest bus stop to home and destination(work)
    home_bus = nearest_points(home, busstops)[1]
    des_bus = nearest_points(destination, busstops)[1]
    return home_bus, des_bus


# People who go shopping leave an hour earlier for work (8 instead of 9)
# Similar to schedule_general_wo but for going shopping from home location
def schedule_general_shop_h(duration, s_duration, travelmean, s_travelmean, filedir, free_duration=1,
                            name="work_shop_h_sport", save_csv=True, time_interval=0.017):
                            # 0.017 is the same as 1 minute as we use decimal time
    h2w_mean, w2h_mean = 8, 16  # mean time left home to go working (h2w) and mean time left work to go home (w2h)
    h2w_sd, w2h_sd = 1, 1  # standard deviation is 1 hour for both

    # to get at least a travel duration of 1 minute as time is here in decimal and rounded to 0.01
    # this was helpful as otherwise I got negative results
    mintime = 0.03  # 0.02 would be equivalent to 1 min 12 sec / 0.01 to only 36 sec

    # go to work
    home2work_start = np.random.normal(h2w_mean, h2w_sd, 1)[0]
    home2work_end = home2work_start + (duration / 3600)
    # checking that home2work_end is never before or equal to home2work_start # If travel duration is very short
    if home2work_end < home2work_start + mintime:  # Could also check only once as both routes are the same?
        home2work_end = home2work_start + mintime

    # go home
    work2home_start = np.random.normal(w2h_mean, w2h_sd, 1)[0]
    work2home_end = work2home_start + (duration / 3600)  # meaning at home again
    # checking that work2home_end is never before or equal to work2home_start  # If travel duration is very short
    if work2home_end < work2home_start + mintime:
        work2home_end = work2home_start + mintime

    # Shopping
    home2shop_start = work2home_end + np.random.choice([0.5, 0.75, 1])
    home2shop_end = home2shop_start + (s_duration / 3600)
    # checking that home2shop_end is never before or equal to home2shop_start # If travel duration is very short
    if home2shop_end < home2shop_start + mintime:  # Could also check only once as both routes are the same?
        home2shop_end = home2shop_start + mintime

    shop2home_start = home2shop_end + np.random.choice([0.25, 0.5, 0.75])
    shop2home_end = shop2home_start + (s_duration / 3600)  # meaning at home again
    # checking that shop2home_end is never before or equal to shop2home_start  # If travel duration is very short
    if shop2home_end < shop2home_start + mintime:
        shop2home_end = shop2home_start + mintime

    if shop2home_end > 22.5:  # at home if goes home late already.
        outdoor_evening = 23.7  # just set to the end of the day as the freetime is still at home.
        free_duration = time_interval  # just any small number as the freetime is still at home.
        freetime = 1  # at home if goes home late already.
    else:  # if goes home relatively early, the person will go outdoor
        freetime = np.random.choice([1, 4, 5, 6])
        # free time has 4 modes: 1. at home, 4. to sports, 5. 2000m buffer around home, 6. random walk around home

        # after arriving at home, people go freetime after about 1 or 1.5 hours
        outdoor_evening = shop2home_end + np.random.choice([0.5, 1])

        if outdoor_evening + free_duration > 23.5:
            free_duration = 23.5 - outdoor_evening  # time should not go over 23.9

    # Add shopping to shedule
    start_time = np.round(np.array([0.0, home2work_start, home2work_end, work2home_start, work2home_end,
                                    home2shop_start, home2shop_end, shop2home_start, shop2home_end,
                                    outdoor_evening, outdoor_evening + free_duration]), 2)
    end_time = np.round(np.array([*start_time[1: len(start_time)] - 0.01, 23.9]), 2)  # =start_time - 0.01
    activity = ["home", "h2w", "work", "w2h", "home", "h2s", "shop", "s2h", "home", "free_time", "home"]
    activity_code = [1, 2, 3, 2, 1, 72, 71, 72, 1, freetime, 1]
    # activity_code: 1: home, 2: h2w or w2h, 3: work,
    # 4: h2sport,
    # 5: 2000m buffer around home (the person can be anywhere around home),
    # 6: random walk around home

    # This adds travelmean only to activity h2w or w2h and not just in row 1 as before
    travel2 = [None, travelmean, None, travelmean, None, s_travelmean, None, s_travelmean, None, None, None]
    # Use the following if the schedule is not fixed (now inactive for computational reasons)
    # travel = []
    # for act in activity_code:
    #     travel.append(travelmean) if act == 2 else travel.append(None)

    # Get duration of each activity and not just in line 0 as before
    duration = np.round(end_time - start_time, 2)
    # input directly into dataframe instead of adding to a list, transforming and renaming
    schedule = pd.DataFrame({"start_time": start_time, "end_time": end_time, "activity": activity,
                             "activity_code": activity_code, "travel_mean": travel2, "duration": duration})

    if save_csv:
        schedule.to_csv(f'{filedir}/{name}.csv')  # name=f"ws_iter_{ite}_id_{id}"
    return schedule


# People who go shopping leave an hour earlier for work (8 instead of 9)
# Similar to schedule_general_wo but for going shopping on route from work to home
def schedule_general_shop_r(duration, travelmean, filedir, free_duration=1,
                            name="work_shop_r_sport", save_csv=True, time_interval=0.017):
    # 0.017 is the same as 1 minute as we use decimal time
    h2w_mean, w2h_mean = 8, 16  # mean time left home to go working (h2w) and mean time left work to go home (w2h)
    h2w_sd, w2h_sd = 1, 1  # standard deviation is 1 hour for both

    # to get at least a travel duration of 1 minute as time is here in decimal and rounded to 0.01
    # this was helpful as otherwise I got negative results
    mintime = 0.03  # 0.02 would be equivalent to 1 min 12 sec / 0.01 to only 36 sec

    # go to work
    home2work_start = np.random.normal(h2w_mean, h2w_sd, 1)[0]
    home2work_end = home2work_start + (duration / 3600)
    # checking that home2work_end is never before or equal to home2work_start # If travel duration is very short
    if home2work_end < home2work_start + mintime:  # Could also check only once as both routes are the same?
        home2work_end = home2work_start + mintime

    # go home
    work2shop_start = np.random.normal(w2h_mean, w2h_sd, 1)[0]

    duration_cut = np.random.choice([0.25, 0.5, 0.75])
    work2shop_end = work2shop_start + ((duration * duration_cut) / 3600)
    shop2home_start = work2shop_end + np.random.choice([0.25, 0.5, 0.75])
    shop2home_end = shop2home_start + ((duration * (1 - duration_cut)) / 3600)  # meaning at home again

    # checking that work2home_end is never before or equal to work2home_start  # If travel duration is very short
    if work2shop_end < work2shop_start + mintime:
        work2shop_end = work2shop_start + mintime
    if shop2home_end < shop2home_start + mintime:
        shop2home_end = shop2home_start + mintime

    if shop2home_end > 22.5:  # at home if goes home late already.
        outdoor_evening = 23.7  # just set to the end of the day as the freetime is still at home.
        free_duration = time_interval  # just any small number as the freetime is still at home.
        freetime = 1  # at home if goes home late already.
    else:  # if goes home relatively early, the person will go outdoor
        freetime = np.random.choice([1, 4, 5, 6])
        # free time has 4 modes: 1. at home, 4. to sports, 5. 2000m buffer around home, 6. random walk around home

        # after arriving at home, people go freetime after about 1 or 1.5 hours
        outdoor_evening = shop2home_end + np.random.choice([0.5, 1])
        # # before go to work in the morning, people's free time after about 1.5 hours
        # outdoor_morning = home2work_start - np.random.choice([1, 1.5])

        if outdoor_evening + free_duration > 23.5:
            free_duration = 23.5 - outdoor_evening  # time should not go over 23.9

    # Add shopping to shedule
    start_time = np.round(np.array([0.0, home2work_start, home2work_end, work2shop_start, work2shop_end,
                                    shop2home_start, shop2home_end,
                                    outdoor_evening, outdoor_evening + free_duration]), 2)
    end_time = np.round(np.array([*start_time[1: len(start_time)] - 0.01, 23.9]), 2)  # =start_time - 0.01
    activity = ["home", "h2w", "work", "w2s", "shop", "s2h", "home", "free_time", "home"]
    activity_code = [1, 2, 3, 73, 71, 73, 1, freetime, 1]
    # activity_code: 1: home, 2: h2w or w2h, 3: work,
    # 4: h2sport,
    # 5: 2000m buffer around home (the person can be anywhere around home),
    # 6: random walk around home

    # This adds travelmean only to activity h2w or w2h and not just in row 1 as before
    travel2 = [None, travelmean, None, travelmean, None, travelmean, None, None, None]
    # Use the following if the schedule is not fixed (now inactive for computational reasons)
    # travel = []
    # for act in activity_code:
    #     travel.append(travelmean) if act == 2 else travel.append(None)

    # Get duration of each activity and not just in line 0 as before
    duration = np.round(end_time - start_time, 2)
    # input directly into dataframe instead of adding to a list, transforming and renaming
    schedule = pd.DataFrame({"start_time": start_time, "end_time": end_time, "activity": activity,
                             "activity_code": activity_code, "travel_mean": travel2, "duration": duration})

    if save_csv:
        schedule.to_csv(f'{filedir}/{name}.csv')  # name=f"ws_iter_{ite}_id_{id}"
    return schedule


# Similar to schedule_general_wo but for traveling by bus
def schedule_general_bus(duration_bus, filedir, mindist_home=-999, mindist_work=-999, free_duration=1,
                         name="work_sport_bus", save_csv=True,
                         time_interval=0.017):  # 0.017 is the same as 1 minute as we use decimal time
    h2w_mean, w2h_mean = 8, 16  # mean time left home to go working (h2w) and mean time left work to go home (w2h)
    h2w_sd, w2h_sd = 1, 1  # standard deviation is 1 hour for both
    speed_walk = 5  # km/h
    wait4bus = 5  # in min

    # to get at least a travel duration of 1 minute as time is here in decimal and rounded to 0.01
    # this was helpful as otherwise I got negative results
    mintime = 0.03  # 0.02 would be equivalent to 1 min 12 sec / 0.01 to only 36 sec

    # go to work
    home2bus = np.random.normal(h2w_mean, h2w_sd, 1)[0]-0.16  # 10 min earlier
    busstop_home_arrive = home2bus + (mindist_home / 1000 / speed_walk * 3600)  # mindist_start is in km
    busstop_home_leave = busstop_home_arrive + wait4bus/60
    bus2work = busstop_home_leave + (duration_bus / 3600)  # duration_bus is in seconds
    work_start = bus2work + (mindist_work / 1000 / speed_walk * 3600)
    # checking that bus2work is never before or equal to work_start
    if work_start < bus2work + mintime:
        work_start = bus2work + mintime

    # go home
    work_end = np.random.normal(w2h_mean, w2h_sd, 1)[0]-0.16  # 10 min earlier
    busstop_work_arrive = work_end + (mindist_work / 1000 / speed_walk * 3600)  # mindist_end is in km
    busstop_work_leave = busstop_work_arrive + wait4bus / 60
    bus2home = busstop_work_leave + (duration_bus / 3600)  # duration_bus is in seconds
    home_again = bus2home + (mindist_home / 1000 / speed_walk * 3600)
    # checking that work_end is never before or equal to busstop_work_arrive
    if busstop_work_arrive < work_end + mintime:
        busstop_work_arrive = work_end + mintime

    if home_again > 23.5:
        if work_end >= 23.5:
            print("work2home_start later than work2home_end!")
        home_again = 23.5  # if going home after 23:30, we will assume him going home at 23:30

    if home_again > 20.5:  # at home if goes home late already.
        outdoor_evening = 23.7  # just set to the end of the day as the freetime is still at home.
        free_duration = time_interval  # just any small number as the freetime is still at home.
        freetime = 1  # at home if goes home late already.
    else:  # if goes home relatively early, the person will go outdoor
        freetime = np.random.choice([1, 4, 5, 6])
        # free time has 4 modes: 1. at home, 4. to sports, 5. 2000m buffer around home, 6. random walk around home

        # after arriving at home, people go freetime after about 1 or 1.5 hours
        outdoor_evening = home_again + np.random.choice([1, 1.5])

        if outdoor_evening + free_duration > 23.5:
            free_duration = 23.5 - outdoor_evening  # time should not go over 23.9

    # Add shopping to shedule
    start_time = np.round(np.array([0.0, home2bus, busstop_home_arrive, busstop_home_leave, bus2work, work_start,
                                    work_end, busstop_work_arrive, busstop_work_leave, bus2home, home_again,
                                    outdoor_evening, outdoor_evening + free_duration]), 2)
    end_time = np.round(np.array([*start_time[1: len(start_time)] - 0.01, 23.99]), 2)  # end_time = start_time - 0.01
    activity = ["home", "walk_h2b", "busstop_home", "busroute", "walk_b2w", "work",
                "walk_w2b", "busstop_work", "busroute", "walk_b2h", "home", "free_time", "home"]
    activity_code = [1, 21, 22, 25, 23, 3,
                     23, 24, 25, 21, 1, freetime, 1]
    # activity_code: 1: home, 2: h2w or w2h, 3: work,
    # 4: h2sport,
    # 5: 2000m buffer around home (the person can be anywhere around home),
    # 6: random walk around home
    # 21: walk from home to bus stop or back
    # 22: waiting at bus stop home
    # 23: walk from bus to work or back
    # 24: waiting at bus stop work

    # This adds travelmean only to activity h2w or w2h and not just in row 1 as before
    travel2 = [None, "foot", None, "bus", "foot", None,
               "foot", None, "bus", "foot", None, None, None]
    # Use the following if the schedule is not fixed (now inactive for computational reasons)
    # travel = []
    # for act in activity_code:
    #     travel.append(travelmean) if act == 2 else travel.append(None)

    # Get duration of each activity and not just in line 0 as before
    duration = np.round(end_time - start_time, 2)
    # input directly into dataframe instead of adding to a list, transforming and renaming
    schedule = pd.DataFrame({"start_time": start_time, "end_time": end_time, "activity": activity,
                             "activity_code": activity_code, "travel_mean": travel2, "duration": duration})

    if save_csv:
        schedule.to_csv(f'{filedir}/{name}.csv')  # name=f"ws_iter_{ite}_id_{id}"
    return schedule


# Similar to schedule_general_bus but for going shopping from home location
def schedule_general_bus_shop_h(duration_bus, s_travelmean, s_duration, filedir, mindist_home=-999, mindist_work=-999,
                                free_duration=1, name="work_sport", save_csv=True,
                                time_interval=0.017):  # 0.017 is the same as 1 minute as we use decimal time
    h2w_mean, w2h_mean = 8, 16  # mean time left home to go working (h2w) and mean time left work to go home (w2h)
    h2w_sd, w2h_sd = 1, 1  # standard deviation is 1 hour for both
    speed_walk = 5  # km/h
    wait4bus = 5  # in min

    # to get at least a travel duration of 1 minute as time is here in decimal and rounded to 0.01
    # this was helpful as otherwise I got negative results
    mintime = 0.03  # 0.02 would be equivalent to 1 min 12 sec / 0.01 to only 36 sec

    # go to work
    home2bus = np.random.normal(h2w_mean, h2w_sd, 1)[0]-0.16  # 10 min earlier
    busstop_home_arrive = home2bus + (mindist_home / 1000 / speed_walk * 3600)  # mindist_start is in km
    busstop_home_leave = busstop_home_arrive + wait4bus/60
    bus2work = busstop_home_leave + (duration_bus / 3600)  # duration_bus is in seconds
    work_start = bus2work + (mindist_work / 1000 / speed_walk * 3600)

    # go home
    work_end = np.random.normal(w2h_mean, w2h_sd, 1)[0]-0.16  # 10 min earlier
    busstop_work_arrive = work_end + (mindist_work / 1000 / speed_walk * 3600)  # mindist_end is in km
    busstop_work_leave = busstop_work_arrive + wait4bus / 60
    bus2home = busstop_work_leave + (duration_bus / 3600)  # duration_bus is in seconds
    home_again = bus2home + (mindist_home / 1000 / speed_walk * 3600)

    # Shopping
    home2shop_start = home_again + np.random.choice([0.5, 0.75, 1])
    home2shop_end = home2shop_start + (s_duration / 3600)
    # checking that home2shop_end is never before or equal to home2shop_start # If travel duration is very short
    if home2shop_end < home2shop_start + mintime:  # Could also check only once as both routes are the same?
        home2shop_end = home2shop_start + mintime

    shop2home_start = home2shop_end + np.random.choice([0.25, 0.5, 0.75])
    shop2home_end = shop2home_start + (s_duration / 3600)  # meaning at home again
    # checking that shop2home_end is never before or equal to shop2home_start  # If travel duration is very short
    if shop2home_end < shop2home_start + mintime:
        shop2home_end = shop2home_start + mintime

    if shop2home_end > 22.5:  # at home if goes home late already.
        outdoor_evening = 23.7  # just set to the end of the day as the freetime is still at home.
        free_duration = time_interval  # just any small number as the freetime is still at home.
        freetime = 1  # at home if goes home late already.
    else:  # if goes home relatively early, the person will go outdoor
        freetime = np.random.choice([1, 4, 5, 6])
        # free time has 4 modes: 1. at home, 4. to sports, 5. 2000m buffer around home, 6. random walk around home

        # after arriving at home, people go freetime after about 1 or 1.5 hours
        outdoor_evening = shop2home_end + np.random.choice([1, 1.5])

        if outdoor_evening + free_duration > 23.5:
            free_duration = 23.5 - outdoor_evening  # time should not go over 23.9

    # Add shopping to shedule
    start_time = np.round(np.array([0.0, home2bus, busstop_home_arrive, busstop_home_leave, bus2work, work_start,
                                    work_end, busstop_work_arrive, busstop_work_leave, bus2home, home_again,
                                    home2shop_start, home2shop_end, shop2home_start, shop2home_end,
                                    outdoor_evening, outdoor_evening + free_duration]), 2)
    end_time = np.round(np.array([*start_time[1: len(start_time)] - 0.01, 23.99]), 2)  # end_time = start_time - 0.01
    activity = ["home", "walk_h2b", "busstop_home", "busroute", "walk_b2w", "work",
                "walk_w2b", "busstop_work", "busroute", "walk_b2h",
                "home", "h2s", "shop", "s2h", "home", "free_time", "home"]
    activity_code = [1, 21, 22, 25, 23, 3,
                     23, 24, 25, 21,
                     1, 72, 71, 72, 1, freetime, 1]
    # activity_code: 1: home, 2: h2w or w2h, 3: work,
    # 4: h2sport,
    # 5: 2000m buffer around home (the person can be anywhere around home),
    # 6: random walk around home
    # 21: walk from home to bus stop or back
    # 22: waiting at bus stop home
    # 23: walk from bus to work or back
    # 24: waiting at bus stop work
    # 71: shopping inside shop
    # 72: route to the shop and back

    # This adds travelmean only to activity h2w or w2h and not just in row 1 as before
    travel2 = [None, "foot", None, "bus", "foot", None,
               "foot", None, "bus", "foot",
               None, s_travelmean, None, s_travelmean, None, None, None]
    # Use the following if the schedule is not fixed (now inactive for computational reasons)
    # travel = []
    # for act in activity_code:
    #     travel.append(travelmean) if act == 2 else travel.append(None)

    # Get duration of each activity and not just in line 0 as before
    duration = np.round(end_time - start_time, 2)
    # input directly into dataframe instead of adding to a list, transforming and renaming
    schedule = pd.DataFrame({"start_time": start_time, "end_time": end_time, "activity": activity,
                             "activity_code": activity_code, "travel_mean": travel2, "duration": duration})

    if save_csv:
        schedule.to_csv(f'{filedir}/{name}.csv')  # name=f"ws_iter_{ite}_id_{id}"
    return schedule


def travelmean(id, df, f_d, Gw, Gb, Gd, speed_walk=5, speed_bike=14):
    # Add getstartend for shopping
    start, end = getstartend(id, df=df)
    apprEucl = start.distance(end) * 110  # approximate Euclidean distance in km
    if apprEucl < 0.001:  # less than 1 m, jitter 100 m
        end = Point(end.x + 0.001, end.y)

    # Chose travel mean
    cls_ = travelmean_from_distance2work_df(f_d, apprEucl)

    if cls_ == "train":
        cls_ = "car"  # for now, train and car is the same
    # ---
    if cls_ == "foot":
        G = Gw
        speed = speed_walk
    elif cls_ == "bicycle":
        G = Gb
        speed = speed_bike
    else:
        G = Gd  # for car and train # and bus
        speed = 100  # for car and train
    return cls_, start, end, G, speed


def travelmean_bus(id, df, f_d, Gw, Gb, Gd, busstops, speed_walk=5, speed_bike=14):
    # Add getstartend for shopping
    start, end = getstartend(id, df=df)
    apprEucl = start.distance(end) * 110  # approximate Euclidean distance in km
    if apprEucl < 0.001:  # less than 1 m, jitter 100 m
        end = Point(end.x + 0.001, end.y)

    # Chose travel mean
    cls_ = travelmean_from_distance2work_df(f_d, apprEucl)

    if cls_ == "train":
        cls_ = "car"  # for now, train and car is the same
    # ---
    if cls_ == "foot":
        G = Gw
        speed = speed_walk
    elif cls_ == "bicycle":
        G = Gb
        speed = speed_bike
    else:
        G = Gd  # for car and train # and bus
        speed = 100  # for car and train

    if cls_ != "bus":
        return cls_, start, end, G, speed, Point(0, 0), Point(0, 0), -99, -99
    else:
        start_bus, end_bus = get_busstops(busstops, start, end)  # Generate nearest bus stops

        # find the distance to the nearest bus stops
        mindist_start = start.distance(start_bus) * 110
        mindist_end = end.distance(end_bus) * 110

        max_dist = 0.8  # max distance to next bus stop until car is used in km
        if mindist_start > max_dist or mindist_end > max_dist:
            cls_ = "car"
            return cls_, start, end, G, speed, Point(0, 0), Point(0, 0), -99, -99
        else:
            return cls_, start, end, G, speed, start_bus, end_bus, mindist_start, mindist_end


def getroute_2(id, cls_, start, end, G, speed):
    """
    - note for car we use routes of min travel time, but for others we use shortest distance routes.
    - for walking and biking we have a speed. this is good because we can do it differently for children and adults.
    - OSM default speed for walk and bike: https://www.targomo.com/developers/resources/concepts/assumptions_and_defaults/
    :param id:
    :param df: file includes start destination coordinates,
    :param f_d: probability distribution of travel mode as csv
    :param Gw: graph for walk (ut10kwalk.graphml)
    :param Gb: graph for bike (ut10bike.graphml)
    :param Gd: graph for drive (ut10drive.graphml)
    :param speed_walk: default is 5 km/h
    :param speed_bike: default is 14 km/h
    :return: route_, travel_time, travel_distance, cls_
    """

    Xstart, Ystart = start.x, start.y
    Xend, Yend = end.x, end.y
    start_node = ox.distance.nearest_nodes(G, X=Xstart, Y=Ystart, return_dist=False)
    end_node = ox.distance.nearest_nodes(G, X=Xend, Y=Yend, return_dist=False)

    if start_node == end_node:
        travel_distance = 0
        travel_time = 0
        route_ = start
        print(f"start = end, id: {id}, start: {start}, route: {route_} in getroute_2")

    # travel_distance is calculated using the route of min length. so not exactly the same as travel time.
    else:
        try:
            travel_distance = nx.shortest_path_length(G, start_node, end_node, weight='length')  # in meter
            if cls_ == "car":  # Also "train"
                travel_time = nx.shortest_path_length(G, start_node, end_node, weight='travel_time')
                route = ox.distance.shortest_path(G, start_node, end_node, weight='travel_time')
            elif cls_ == "bus":
                # travel_time2 = nx.shortest_path_length(G, start_node, end_node, weight='travel_time')
                # travel_time3 = travel_distance / 1000 / 25 * 3600  # in s
                travel_time = (1.95 * travel_distance/1000 + 4.60)*60  # equation from Ovin
                route = ox.distance.shortest_path(G, start_node, end_node, weight='travel_time')
            else:  # if cls_ == "bicycle" or "foot"
                travel_time = travel_distance / 1000 / speed * 3600  # in s
                route = ox.distance.shortest_path(G, start_node, end_node, weight='length')
            route_ = nodes_to_linestring(route, G)  # Method defined above
        # Maybe find out what specific Error is raised and write it behind except
        except:  # exception raised when there is no path between two points. then we just use the home location.
            travel_distance = 0
            travel_time = 0
            # route_ had to be flipped if it contains a single point as it makes otherwise problems in calc_exposure
            route_ = start  # before: Point(start[0], start[1])
            print(f"no road! {start} in getroute_2")
    return route_, travel_time, travel_distance


def distance_eucl(x1, y1, x2, y2):
    """ Calculate the Euclidean distance between two points
    :return: Euclidean distance """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def testpoint(shop: list):
    # Check if output could be valid
    if shop[1] - shop[0] > 40:  # Check if it fits with GPS coordinates in Utrecht
        # Return as Point
        p = Point(shop[0], shop[1])
        # print(f"Shop for home {home}: {p} is {shop[2]*110} km away")
        return p
    else:
        print("Error in get_shop [testpoint]")


def get_shop(shops_gdf, home, samplesize=5, outputsize=1):
    """ Returns one of the k nearest shop locations to the input points
    :param shops_gdf: GeoDataFrame with shop locations as Points in geometry column
    :param home: location as Point, list or tuple
    :param samplesize: k nearest shop locations
    :param outputsize: number of output Points (can do only one right now!)
    :return: shopping location as Point
    :type shops_gdf: GeoDataFrame
    :type home: Point or list or tuple
    """
    # Checks if input home is Point or tuple/list
    if isinstance(home, Point):
        hx = home.x
        hy = home.y
    else:
        hx = home[0]
        hy = home[1]

    # Open the input csv as an array
    shops = shops_gdf["geometry"]
    poss_shops = []
    n = len(shops)
    dist = []

    if n > 1:
        # Sample can not be more than point count
        if n <= samplesize:
            # samplesize = n
            # Sample from shops and without calculating distance
            # shop = random.choices([(x, y) for x, y in zip(shops.x, shops.y)], k=outputsize)[0]
            poss_shops = [(x, y) for x, y in zip(shops.x, shops.y)]
        else:
            # Get list with distances for shop to the home location
            for i in range(n):
                dist.append({
                    "first": distance_eucl(shops[i].x, shops[i].y, hx, hy),
                    "second": i
                })

            # Sort the list of shops
            dist = sorted(dist, key=lambda l: l["first"])

            # Chose the k (samplesize) closest shops
            for i in range(samplesize):
                pt = [shops[dist[i]["second"]].x, shops[dist[i]["second"]].y, dist[i]["first"]]
                poss_shops.append(pt)

        # Define probability that shops are not chosen randomly (could use itertools.accumulate for flexibility)
        m = len(poss_shops)
        if m == 5:
            probs = [1, 4, 7, 9, 10]
        elif m == 4:
            probs = [2, 5, 7, 10]
        elif m == 3:
            probs = [3, 7, 10]
        elif m == 2:
            probs = [4, 10]
        else:
            print(f"test nearest shops _ {m} _ {samplesize}")
            probs = None

        # Choose k (outputsize) shops out of the nearest shops
        shop = random.choices(poss_shops, cum_weights=probs, k=outputsize)[0]
        return testpoint(shop)
    elif n == 1:
        return shops[0]
    else:
        return Point(0.0, 0.0)


def getroute_shop(id, f_d_shop, Gw, Gb, Gd, shops, home, speed_walk=5, speed_bike=14):
    """
    - note for car we use routes of min travel time, but for others we use shortest distance routes.
    - for walking and biking we have a speed. this is good because we can do it differently for children and adults.
    - OSM default speed for walk and bike: https://www.targomo.com/developers/resources/concepts/assumptions_and_defaults/
    :param id: person number
    :param f_d_shop: probabilities of travel means for shopping
    :param Gw: graph for walk (ut10kwalk.graphml)
    :param Gb: graph for bike (ut10bike.graphml)
    :param Gd: graph for drive (ut10drive.graphml)
    :param shops: GeoDataFrame with shop locations as Points in geometry column
    :param home: location of home as Point
    :param speed_walk: default is 5 km/h
    :param speed_bike: default is 14 km/h
    :return: route_, travel_time, travel_distance, cls_
    """
    # Add getstartend for shopping
    start = home
    shop = get_shop(shops, home)
    apprEucl = start.distance(shop) * 110  # approximate Euclidean distance in km
    if apprEucl < 0.001:  # less than 1 m, jitter 100 m
            shop = Point(shop.x + 0.001, shop.y)

    # Chose travel mean
    # Almost nobody goes grocery shopping by bus according to the Ovin dataset # f_d_shop is without bus
    cls_ = travelmean_from_distance2work_df(f_d_shop, apprEucl)

    if cls_ == "train":
        cls_ = "car"  # for now, train and car is the same
    # ---
    if cls_ == "foot":
        G = Gw
        speed = speed_walk
    elif cls_ == "bicycle":
        G = Gb
        speed = speed_bike
    else:
        G = Gd  # for car and train # and bus
        speed = 100  # for car and train

    Xstart, Ystart = start.x, start.y
    Xend, Yend = shop.x, shop.y
    start_node = ox.distance.nearest_nodes(G, X=Xstart, Y=Ystart, return_dist=False)
    end_node = ox.distance.nearest_nodes(G, X=Xend, Y=Yend, return_dist=False)

    # Put this into own function and add option for choosing travel mean for shopping # No bus for shopping
    # input start_node end_node G and cls_ # input if shop or not
    # output travel_distance travel_time route
    if start_node == end_node:
        travel_distance = 0
        travel_time = 0
        route_ = start
        print(f"id: {id}, start: {start}, shop is too close -> no route is generated")

    # travel_distance is calculated using the route of min length. so not exactly the same as travel time.
    else:
        try:
            travel_distance = nx.shortest_path_length(G, start_node, end_node, weight='length')  # in meter
            if cls_ == "car":  # Also "train"
                travel_time = nx.shortest_path_length(G, start_node, end_node, weight='travel_time')
                route = ox.distance.shortest_path(G, start_node, end_node, weight='travel_time')
            else:  # if cls_ == "bicycle" or "foot"
                travel_time = travel_distance / 1000 / speed * 3600  # in seconds
                if travel_time > 2160:  # Nobody walks longer than 40 min to go shopping (0.6*3600 to get seconds)
                    cls_ == "bicycle"
                    G = Gb
                    travel_time = travel_distance / 1000 / speed_bike * 3600
                route = ox.distance.shortest_path(G, start_node, end_node, weight='length')
            route_ = nodes_to_linestring(route, G)  # Method defined above
        # Maybe find out what specific Error is raised and write it behind except
        except:  # exception raised when there is no path between two points. then we just use the home location.
            travel_distance = 0
            travel_time = 0
            # route_ had to be flipped if it contains a single point as it makes otherwise problems in calc_exposure
            route_ = start  # before: Point(start[0], start[1])
            print(f"no road! {start} in getroute_shop")
    return route_, travel_time, travel_distance, cls_, shop  # mindist_start, mindist_end  # cls_ -> travel mean


def get_bufferradius(travel_mean):
    """ Buffer radius for shopping on home route for different travel means
    :param travel_mean: input travel mean like "car", "bicycle" or "foot"
    :return: buffer distance in km
    :type travel_mean: str
    """
    if travel_mean == "bicycle":
        buffer = 0.4  # 400m
    elif travel_mean == "car":
        buffer = 0.8  # 800m
    elif travel_mean == "foot":  # Use else here if bus and train is excluded somewhere else
        buffer = 0.2  # 200m
    else:  # for train and bus
        buffer = 0.0
    return buffer  # in km


def checkroutebuffer(route, shops, bufferradius=0.2):
    """
    :param route: Route where the buffer will be applied
    :param shops: GeoDataFrame with shops
    :param bufferradius: radius around the route in km
    :return: number of shops inside the buffer and GeoDataframe with shops inside the buffer
    :type route: LineString
    :type shops: GeoDataFrame
    """
    # Get buffer from route
    r_buf = route.buffer(bufferradius / 110.8, cap_style=1, join_style=1)
    # a = r_buf.centroid
    # Get points (shops) inside buffer geometry
    shops_in_buf = shops[shops.within(r_buf)]
    # Count points inside buffer geometry
    count = len(shops_in_buf)
    if count > 0:
        # Change index of shops_in_buf to match requirements of get_shop
        # Just change the index to start at 0 and add plus 1 for each next row
        shops_in_buf2 = shops_in_buf.set_index(pd.Series(x for x in range(count)))
        # Get one of the closest shops to the centroid of the route buffer
        center_shop = get_shop(shops_in_buf2, r_buf.centroid, samplesize=5)
    else:
        # Return empty point as no point is forwarded if count is less then 1 (therefore no problem is generated!)
        center_shop = Point(0, 0)
    return count, center_shop


def check_in_extend(raster, df, prefix=None):
    """ Check if points are inside extend of concentration raster.
    Otherwise some points can't return concentration value at their location.
    :param raster: path for rasterfile
    :param df: dataframe with points
    :param prefix: prefix for colums "...lon" and "...lat" e.g. the default None leads to "home_lon", "home_lat", "work_lon" and "work_lat"
    """
    with rasterio.open(raster) as dataset:
        # Get the bounds of the raster
        bounds = dataset.bounds
    if prefix is None:
        prefix = ["home_", "work_"]
    allindex = []
    for pr in prefix:
        lon = f"{pr}lon"
        lat = f"{pr}lat"
        rows = []
        for index, row in df.iterrows():
            # check if the value in lon or lat are not between their corresponding lower and upper bounds
            if not (bounds.left <= row[lon] <= bounds.right) or not (bounds.bottom <= row[lat] <= bounds.top):
                # add the row to the list of rows that meet the condition
                rows.append(index)
                allindex.append(index)
        # select the rows that meet the condition
        if rows:
            result = df.loc[rows]
            print(f'prefix for column: "{pr}"')
            print(result)
    # Check if an index gets appended to allindex
    if allindex:
        # Get all rows with their index in allindex
        result2 = df.loc[sorted(set(allindex))]
        print(f"\nThese points outside {bounds} will likely cause errors:")
        print(result2)
        # return False
    else:
        print("All points are fine!\n")
        # return True
