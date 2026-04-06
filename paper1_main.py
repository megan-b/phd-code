# -*- coding: utf-8 -*-

# IMPORTS

# for analysis
import pandas as pd

import os
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import numpy as np
import datetime as dt

# for plotly plotting
import plotly.io as pio
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go

import timeit
from sklearn.mixture import GaussianMixture
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN

from scipy.spatial import distance as scidist
from scipy.spatial.distance import pdist, squareform

# CONSTANTS THAT MUST BE SET BY USER

# folder where all data is saved
dirname = "C:\\Users\\Megan\\Desktop\\PhD data\\hubs2017\\csv\\"

# For Plotly plotting
pio.renderers.default = "browser"
pyo.offline.init_notebook_mode()

# Colour palette. Change here as necessary.

mycolors_discrete = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
] # these give better contrasts for journals etc
# first six are generally colourblind friendly

mycolors_continuous = [
    "#f0f9e8",
    "#bae4bc",
    "#7bccc4",
    "#43a2ca",
    "#0868ac",
] # these give better contrasts for journals etc

mycolors_diverging = [
    "#d73027","#fc8d59","#fee090","#e0f3f8",
    "#91bfdb","#4575b4"]

# for anchoring regions
STAY_DROP_THRESHOLD = 0
MIN_POINTS_REGION = 1
MAPBUFFER = 20  # really just for mapping, buffer the anchor stops for the convex hull
STOPBUFFER = 400  # distance in m to buffer stops by for the land use
ANCHOR_REGION_SIZE = 2 * STOPBUFFER  # distance between stops for regions

# stay classes
classmap = {
    1: "W", # work
    2: "E", # school/education
    3: "S", # short
    4: "M", # medium
    5: "C", # AM transfer
    6: "C", # day transfer
    7: "C", # PM transfer
    8: "C", # overnight transfer
    9: "L", # sleep/long stay
    10: "V", # very long stay
    12: "T", # travel
    0: "drop",
}

config = {'scrollZoom': True, 'displaylogo': False}

line_colour = "rgb(217,217,217)"
bg_colour = "white"
line_width = 0.5

default_layout = dict(yaxis=dict(
        #autorange=True,
        showgrid=True,
       # zeroline=True,
        mirror=True,
        showline=True, 
        ticks="outside",
        tickcolor=line_colour,
        ticklen=5,
        linewidth=line_width, 
        linecolor=line_colour,
        gridcolor=line_colour,
        gridwidth=line_width,
        zerolinecolor=line_colour,
        zerolinewidth=line_width,
        separatethousands=True,
        tickformat=",",
    ),
        xaxis=dict(
           # zeroline=True,
            mirror=True,
            showline=True, 
            ticks="outside",
            tickcolor=line_colour,
            ticklen=5,
            linewidth=line_width, 
            linecolor=line_colour,
            zerolinecolor=line_colour,
            zerolinewidth=line_width,
            separatethousands=True,
            tickformat=",",
    ),
    paper_bgcolor=bg_colour,
    plot_bgcolor=bg_colour,
    font=dict(
        size=14,
        color="black"
    )
)

default_layout_box = dict(yaxis=dict(
        showgrid=True,
        zeroline=True,
        mirror=True,
        ticks="outside",
        tickcolor=line_colour,
        ticklen=5,
        linewidth=line_width, 
        linecolor=line_colour,
        gridcolor=line_colour,
        gridwidth=line_width,
        zerolinecolor=line_colour,
        zerolinewidth=line_width,
        separatethousands=True,
        tickformat=",",
    ),
        xaxis=dict(
            mirror=True,
            linewidth=line_width, 
            linecolor=line_colour,
            separatethousands=True,
            tickformat=",",
    ),
    paper_bgcolor=bg_colour,
    plot_bgcolor=bg_colour,
    font=dict(
        size=14,
        color="black"

    )
)

default_layout_ncaa = dict(yaxis=dict(
        showgrid=True,
        zeroline=True,
        mirror=True,
        ticks="outside",
        tickcolor=line_colour,
        ticklen=5,
        linewidth=line_width, 
        linecolor=line_colour,
        gridcolor=line_colour,
        gridwidth=line_width,
        zerolinecolor=line_colour,
        zerolinewidth=line_width,
        separatethousands=True,
        tickformat=",",
    ),
        xaxis=dict(
            mirror=True,
            linewidth=line_width, 
            linecolor=line_colour,
            separatethousands=True,
            tickformat=",",
    ),
    paper_bgcolor=bg_colour,
    plot_bgcolor=bg_colour,
    font=dict(
        size=16,
        color="black",
        family="Arial",
    )
)

#%% USEFUL FUNCTIONS

def create_point(row):
    """Returns a shapely point object based on values in x and y columns"""
    point = Point(row["X"], row["Y"])
    return point


def create_geo_stops(trainStops, busStops):
    trainStops.rename(
        columns={"StationRefNo": "StopID", "Station": "StopName"}, inplace=True
    )
    busStops.rename(
        columns={"BusStopId": "StopID", "BusStopName": "StopName"}, inplace=True
    )

    stops = pd.concat([busStops, trainStops])
    stopTable = stops[["StopID", "X", "Y"]]
    stopTable.drop_duplicates(subset="StopID", inplace=True)

    # need to split - some are in different CRSs. Where X > 130 assume 7844
    # (? something UTM) and otherwise 4326s
    # just drop for now
    stopTable = stopTable[stopTable["X"] < 130]

    # convert coords to Shapely geometry
    stopTable["geometry"] = stopTable.apply(create_point, axis=1)

    # convert to geodataframe
    geo = gpd.GeoDataFrame(stopTable, geometry="geometry")
    geo = geo.set_crs("epsg:4326", allow_override=True)
    return geo

def create_centroid(clusterID, geo, outcol):
    """Returns centroid of the cluster for each stop"""
    multi_point = MultiPoint(geo["geometry"][geo[outcol] == clusterID].to_list())
    return multi_point.centroid


def create_stop_cluster(geo, outcol, eps, minpts):
    geo = geo.to_crs(epsg=32749)
    geo["X"] = geo["geometry"].x
    geo["Y"] = geo["geometry"].y

    db = DBSCAN(eps=eps, min_samples=minpts, metric="euclidean", algorithm="ball_tree")
    geo[outcol] = db.fit_predict(geo[["Y", "X"]])

    geo.to_crs(epsg=4326, inplace=True)
    geo[outcol] = geo[outcol].astype(int)
    clusters = pd.DataFrame(
        data=np.arange(0, geo[outcol].max() + 1), columns=["clusterID"]
    )

    clusters["geometry"] = clusters.apply(
        lambda x: create_centroid(x["clusterID"], geo, outcol), axis=1
    )
    clusters = gpd.GeoDataFrame(clusters, geometry="geometry")
    clusters = clusters.set_crs(geo.crs, allow_override=True)

    return clusters, geo

#%% IMPORT JOURNEY DATA

journeys = pd.read_csv(
    dirname + "journeys.csv",
    usecols=[
        "Cardid",
        "OnDate",
        "OnLocation",
        "OnMode",
        "OffDate",
        "OnTran",
        "OffTran",
        "OffLocation",
        "Token",
    ],
)


#%% HISTOGRAM OF NUMBER OF CARD USES - BEFORE ANY TRIPS DROPPED

# Figure 1 in readme

v = journeys["Cardid"].value_counts()
x = v[v.lt(100)]  # to plot a zoomed histogram

fig = px.histogram(
    v,
    color_discrete_sequence=[mycolors_discrete[7]],
    labels={"value": "Number of trips in month"},
)
fig.update_layout(default_layout)
fig.update_layout(
    yaxis_title_text = 'Number of cards',
    showlegend=False,
    )
pyo.plot(fig, config=config)

filename = "prep_allcardshist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%%

# how many times does a card have to be used in a month for it to be kept?
# determined from this chart
MIN_USES = 12

#%% READ PROCESSED JOURNEYS DATA

# Alternative to the next cell, read it in if already processed
journeys = pd.read_pickle(dirname + "20250111-processed_journeys.pkl")

#%% JOURNEYS DATA PROCESSING

TOTAL_LEN = len(journeys)
    # total length of original journeys dataset - to work out how much is being dropped

print(TOTAL_LEN, "trips in original dataset")  # 9,643,083
v = journeys["Cardid"].value_counts()
print(len(v), " cards in data set")
print(len(v.index[v.gt(MIN_USES)]), " cards kept")
print(len(v) - len(v.index[v.gt(MIN_USES)]), " cards dropped")

journeys = journeys[journeys["Cardid"].isin(v.index[v.gt(MIN_USES)])]

DROPPED_MIN = TOTAL_LEN - len(
    journeys
)  # how many records got dropped by not being above minimum number of uses
print(
    DROPPED_MIN, "trips dropped due to card not being above minimum use threshold"
)  

print(
    DROPPED_MIN / TOTAL_LEN * 100,
    "percent of trips dropped due to card not being above minimum use threshold",
)  

# Replace unknown stops with a zero
journeys["OnLocation"] = np.where(
    journeys["OnLocation"] == "Unknown", 0, journeys["OnLocation"]
)
journeys["OffLocation"] = np.where(
    journeys["OffLocation"] == "Unknown", 0, journeys["OffLocation"]
)

# Make sure all stops are numeric (to find duplicates)
journeys["OnLocation"] = pd.to_numeric(journeys["OnLocation"])
journeys["OffLocation"] = pd.to_numeric(journeys["OffLocation"])

# Convert Date format to timestamp
journeys["OnTime"] = pd.to_datetime(journeys["OnDate"], format="%Y%m%d%H%M%S")
journeys["OffTime"] = pd.to_datetime(journeys["OffDate"], format="%Y%m%d%H%M%S")

# Separate Hour and Date
journeys["OnHour"] = pd.DatetimeIndex(journeys["OnTime"]).hour
journeys["OffHour"] = pd.DatetimeIndex(journeys["OffTime"]).hour
journeys["OnDate"] = pd.DatetimeIndex(journeys["OnTime"]).date
journeys["OffDate"] = pd.DatetimeIndex(journeys["OffTime"]).date
journeys["OnDay"] = pd.DatetimeIndex(journeys["OnTime"]).day

# Calculate duration of trip, and time since last trip
journeys["tripTime"] = journeys["OffTime"] - journeys["OnTime"]
journeys["tripTime_h"] = journeys["tripTime"] / dt.timedelta(hours=1)

journeys["timeSince"] = (journeys["OnTime"] - journeys["OffTime"].shift(1)).where(
    journeys["Cardid"] == journeys["Cardid"].shift(1)
)
# note this sometimes returns 0 for where there are synthetic activities

# used for stays
journeys["arriveHour"] = (
    journeys["OffHour"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)
journeys["fromLocation"] = (
    journeys["OffLocation"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)
journeys["originLocation"] = (
    journeys["OnLocation"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)

# NEW - where did this stay originate?
journeys["ArriveMode"] = (
    journeys["OnMode"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)

# NEW - what mode was the prior trip
journeys["OffTime_prev"] = (
    journeys["OffTime"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)
journeys["OnTime_prev"] = (
    journeys["OnTime"]
    .shift(1)
    .where(journeys["Cardid"] == journeys["Cardid"].shift(1))
)

journeys["tripTime_prev"] = journeys["OffTime_prev"] - journeys["OnTime_prev"]
journeys["tripTime_h_prev"] = journeys["tripTime_prev"] / dt.timedelta(hours=1)


# SYNTHETIC RECORDS
# synthetic records are where the system backfills a tag on or off if someone
# has forgotten to do so
# this will result in inaccurate stay times/locations, as really we don't know
# where someone has been between times they've actually used their cards

journeys["SyntheticFlag"] = np.where(
    journeys["OnTran"].str.contains("synthetic", na=False, case=False)
    | journeys["OffTran"].str.contains("synthetic", na=False, case=False),
    1,
    0,
)

print(
    "Synthetic activities comprise ",
    journeys["SyntheticFlag"].sum() / len(journeys) * 100,
    "percent, or",
    journeys["SyntheticFlag"].sum(),
    "individual activities",
)
# 2.6% of current journeys data
# 220,332 records

# ACCEPTABLE SYNTHETIC ACTIVITIES
# Warwick (stopID 2800) and Whitfords (stopID 2763)
# "Within the interchange area, you may transfer from the train to the bus,
# without tagging off the train or from the bus to the train, without tagging
# onto the train. The SmartRider will automatically transfer you to/from the
# train service. You will always need to tag on and off the bus."

# In this logic:
# OnTran
# where this is synthetic and the previous tag off is a bus within x minutes (?)
# OffTran
# where this is synthetic and the next tag on is a bus within x minutes (?)

# have used 60 minutes as this is consistent with the transfer logic

journeys["SyntheticOnOk_WW"] = np.where(
    (journeys["OnTran"].str.contains("synthetic", na=False, case=False)) &
    # synthetic tag on
    ((journeys["OnLocation"] == 2800) | (journeys["OnLocation"] == 2763)) &
    # at Warwick or Whitfords
    (journeys["OnMode"].shift(1) == "Bus") &
    # last activity was a bus
    (
        (journeys["OffTime"] - journeys["OffTime"].shift(1))
        < pd.Timedelta(60, unit="m")
    )
    &
    # time between actual tag off and last tag off is <60min
    (journeys["Cardid"] == journeys["Cardid"].shift(1)),
    # last activity was with the same card
    1,
    0,
)

journeys["SyntheticOffOk_WW"] = np.where(
    (journeys["OffTran"].str.contains("synthetic", na=False, case=False)) &
    # synthetic tag off
    ((journeys["OffLocation"] == 2800) | (journeys["OffLocation"] == 2763)) &
    # at Warwick or Whitfords
    (journeys["OnMode"].shift(-1) == "Bus") &
    # next activity is a bus
    (
        (journeys["OnTime"].shift(-1) - journeys["OnTime"])
        < pd.Timedelta(60, unit="m")
    )
    &
    # time between actual next tag on and this tag on is <60min
    (journeys["Cardid"] == journeys["Cardid"].shift(-1)),
    # next activity is with the same card
    1,
    0,
)

# Bus transfers
# The exception is when you tag onto two bus services within 60 minutes
# without tagging off the first. When this happens, we use your second tag
# on location as your tag off location for the first bus service.

# OffTran
# where this is synthetic and the next activity is a bus within 60 minutes

# but if it's not very long, transferring between a train and a bus, or from
# a train to a train, should be findable (and ok) too

journeys["SyntheticOffOk_transfer"] = np.where(
    (journeys["OffTran"].str.contains("synthetic", na=False, case=False))
    & (
        (journeys["OnTime"].shift(-1) - journeys["OnTime"])
        < pd.Timedelta(60, unit="m")
    )
    & (journeys["Cardid"] == journeys["Cardid"].shift(1)),
    1,
    0,
)

print(
    "On ok at Whitfords/Warwick:",
    journeys["SyntheticOnOk_WW"].sum() / journeys["SyntheticFlag"].sum() * 100,
    "percent of synthetic activities, or",
    journeys["SyntheticOnOk_WW"].sum(),
    "individual activities",
)

print(
    "Off ok at Whitfords/Warwick:",
    journeys["SyntheticOffOk_WW"].sum() / journeys["SyntheticFlag"].sum() * 100,
    "percent of synthetic activities, or",
    journeys["SyntheticOffOk_WW"].sum(),
    "individual activities",
)

print(
    "Off ok transfers:",
    journeys["SyntheticOffOk_transfer"].sum()
    / journeys["SyntheticFlag"].sum()
    * 100,
    "percent of synthetic activities, or",
    journeys["SyntheticOffOk_transfer"].sum(),
    "individual activities",
)

# there can be double counting here - some of the off ok bus transfers
# will be caught by the Warwick/Whitfords logic too

# overwrite synthetic flags where they're ok
journeys["SyntheticOk"] = (
    journeys["SyntheticOnOk_WW"]
    + journeys["SyntheticOffOk_WW"]
    + journeys["SyntheticOffOk_transfer"]
)
journeys["SyntheticFlag"] = np.where(
    journeys["SyntheticOk"] > 0, 0, journeys["SyntheticFlag"]
)

print(
    "Unresolved synthetic activities:",
    journeys["SyntheticFlag"].sum(),
    "individual activities",
)


#%%

# length of synthetic activities - histogram of time
# that we don't know what the person is doing

# shift(-1) gives the next row
# shift(1) gives the previous row

journeys['SyntheticType'] = 0
journeys['SyntheticType'] = np.where((journeys['SyntheticFlag']==1)&
                                     (journeys['OffTran'].str.contains("Synthetic")),"Off",journeys['SyntheticType'])
journeys['SyntheticType'] = np.where((journeys['SyntheticFlag']==1)&
                                     (journeys['OnTran'].str.contains("Synthetic")),"On",journeys['SyntheticType'])
journeys['SyntheticType'] = np.where((journeys['SyntheticFlag']==1)&
                                     (journeys['OnTran'].str.contains("Synthetic"))&
                                     (journeys['OffTran'].str.contains("Synthetic")),"Both",journeys['SyntheticType'])

journeys['SyntheticDuration'] = 0

# where tag on is synthetic, we have the tag off of the current row and the 
# tag off of the previous row - everything else in between we don't know
journeys['SyntheticDuration'] = np.where(journeys['SyntheticType']=="On",
                                         journeys['OffTime']-journeys['OffTime'].shift(1),
                                         journeys['SyntheticDuration'])

# where the tag off is synthetic, we have the tag on in the same row and the
# tag on in the next row (which is the same as the synthetic tag time)
journeys['SyntheticDuration'] = np.where(journeys['SyntheticType']=="Off",
                                         journeys['OffTime']-journeys['OnTime'],
                                         journeys['SyntheticDuration'])

# where both are synthetic, we have the previous tag off and the next tag on
# which is also the same time as the synthetic activities
journeys['SyntheticDuration'] = np.where(journeys['SyntheticType']=="Both",
                                         journeys['OffTime']-journeys['OnTime'],
                                         journeys['SyntheticDuration'])

journeys['SyntheticDuration'] = journeys['SyntheticDuration']/dt.timedelta(hours=1)

#%%
fig = px.histogram(
    journeys["SyntheticDuration"][journeys["SyntheticDuration"] >0],
    color_discrete_sequence=[mycolors_discrete[7]],
    nbins=20,
)
fig.update_layout(
    xaxis_title_text = 'Length of unknown activity (hours)', 
    yaxis_title_text = 'Number of activities',
    showlegend=False,
    )
fig.update_traces(xbins=dict( 
        start=0.0,
        end=600.0,
        size=4
    )) # force boundaries for the histogram buckets - otherwise it centres on zero/includes some negatives

fig.update_layout(default_layout)
pyo.plot(fig, config=config)

filename = "prep_syntheticduration"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%%

syn_pivot = pd.pivot_table(
    journeys, index="Cardid", columns="OnDay", values="SyntheticFlag", aggfunc="sum"
)
syn_sum = syn_pivot.melt(ignore_index=False)
syn_sum.reset_index(inplace=True)


journeys = journeys.merge(syn_sum, how="left", on=["OnDay", "Cardid"])
journeys.rename(columns={"value": "SyntheticDrop"}, inplace=True)

# so how many records would be dropped if we drop whole
# days that include synthetic records?
print(len(journeys[journeys["SyntheticDrop"] > 0]), "trips would be dropped")
print(
    "... which is",
    len(journeys[journeys["SyntheticDrop"] > 0]) / len(journeys) * 100,
    "percent of the trips",
)

# drop synthetic activities that aren't salvageable
journeys = journeys.drop(journeys[journeys["SyntheticDrop"] > 0].index).reset_index(
    drop=True)

# simplifications for density heatmaps
bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, np.inf]
names = ['5-7am', '7-9am', '9-11am', '11am-1pm', '1-3pm', '3-5pm', '5-7pm', '7-9pm', '9pm+']

journeys['OnHour_norm'] = journeys['OnHour'] - 5 # day starts at 5am
journeys['OnHour_bucket'] = pd.cut(journeys['OnHour_norm'], bins, labels=names)

token_map = {'Standard':'Standard',
            'Student 50 cent':'School',
             'Student (Up to Yr 12)':'School',
             'Student Tertiary':'Tertiary',
             'Senior':'Concession',
             'Senior Off-Peak':'Concession',
             'Health Care':'Concession',
             'Pensioner Off-Peak':'Concession',
             'Pensioner':'Concession',
             'PTA Free Pass':'Concession',
             'Freerider':'Concession',
             'PTA Concession':'Concession',
             'Veteran':'Concession'}

journeys["Token_type"] = journeys["Token"].map(token_map)

#%% PICKLE JOURNEYS DATA

journeys.to_pickle(dirname + "20250111-processed_journeys.pkl")

#%% DENSITY HEATMAP OF TRAVEL TIME

# map travel time to bins - density heatmap 
# (defining the boring commuters before any clustering)

tagon_hist = pd.pivot_table(journeys, index=['OnHour', 'OnDay','Token_type'],
                            values=['Cardid'], aggfunc="count")
tagon_hist.reset_index(inplace=True)
tagon_hist.rename(columns={"Cardid":"count"}, inplace=True)

fig = px.density_heatmap(tagon_hist, x='OnDay', 
                         y='OnHour',
                         z='count',
                         facet_col='Token_type',
                         color_continuous_scale=mycolors_continuous,
                         nbinsx=31,
                         nbinsy=24,
                             labels={
                                 "OnDay": "Day of month",
                                 "OnHour": "Hour of day",
                                # "count":"Number of activities",
                                 "Token_type":"Token"}
                             )
fig.update_layout(default_layout)
fig.update_yaxes(tickvals=[0,2,4,6,8,10,12,14,16,18,20,22])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(autorange="reversed")
pyo.plot(fig, config=config)
#%%
filename = "prep_tagon_densityhist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

# can't find an easy way to get plotly to treat the y axis as values
# and not categories - label is always in the middle of the row when
# it would be more intuitive at the top (y axis)

#%% DENSITY HEATMAP - NO DAYS


fig = px.density_heatmap(tagon_hist, x='Token_type', 
                         y='OnHour',
                         z='count',
                         nbinsx=31,
                         nbinsy=24,
                         color_continuous_scale=mycolors_continuous,
                         text_auto=",",
                             labels={
                                 "OnHour": "Hour of day",
                                 "Token_type":"Token"}
                             )

fig.update_layout(default_layout)
fig.update(layout_coloraxis_showscale=False)
fig.update_yaxes(tickvals=[0,2,4,6,8,10,12,14,16,18,20,22])
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(autorange="reversed")
pyo.plot(fig, config=config)
#%%
filename = "prep_tagon_densityhist_noday"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% READ STOP CLUSTERS

# Alternative to the next cell - read in instead of running

clusters = pd.read_pickle(dirname + "20250111-clusters.pkl")
geo = pd.read_pickle(dirname + "20250111-geo.pkl")

#%% STOP DATA

busStops = pd.read_csv(dirname + "busStops.csv")
trainStops = pd.read_csv(dirname + "trainStops.csv")

# create geodataframe of both files
geo = create_geo_stops(trainStops, busStops)

# work out which stops have been used
onstops = journeys["OnLocation"].value_counts()
onstops = pd.DataFrame(onstops)
onstops.reset_index(inplace=True)
onstops.rename(columns={"OnLocation": "StopID", "count": "Count On"}, inplace=True)

offstops = journeys["OffLocation"].value_counts()
offstops = pd.DataFrame(offstops)
offstops.reset_index(inplace=True)
offstops.rename(columns={"OffLocation": "StopID", "count": "Count Off"}, inplace=True)

geo = pd.merge(geo, onstops, left_on="StopID", right_on="StopID", how="left")

geo = pd.merge(geo, offstops, left_on="StopID", right_on="StopID", how="left")

geo["Count On"] = geo["Count On"].fillna(0)
geo["Count Off"] = geo["Count Off"].fillna(0)

geo["Count Total"] = geo["Count On"] + geo["Count Off"]


print(len(geo[geo['Count Total']==0])/len(geo)*100, "percent of all stops are unused")
# 22.75%. That seems high
print(len(geo), "total stops")
print(len(geo[geo['Count Total']>0]), "used stops")
print(len(clusters), "stop clusters")

#%% PLOT USED VS UNUSED STOPS

geo["colour"] = 0
geo["colour"] = np.where(geo["Count Total"]>0, "Used", "Unused")

fig = px.scatter_map(
    geo,
    lat=geo["geometry"].y,
    lon=geo["geometry"].x,
    color=geo["colour"],  
    hover_name="StopID",
    color_discrete_sequence=mycolors_discrete,
    zoom=12,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=10))
pyo.plot(fig, config=config)

#%%

# merge trainstops[['StopID', 'line', 'StopName']]

geo = pd.merge(geo, trainStops[['StopID','line','StopName']],left_on='StopID', right_on='StopID', how='left')

#%%
used_bayswater = geo['Count Total'][geo['StopName']=='Bayswater Stn'].sum()

geo['mode'] = np.where(~geo['line'].isna(),'Train','Bus')

used_trains = geo['Count Total'][geo['mode']=='Train'].sum()

used_bayswater/used_trains*100 # (= 0.72%)

len(geo[['line','StopName']][geo['percent_train']>0.73])
# there are 34 stations that contribute more to the total tags on/off than Bayswater
# of 69 total stations - so that's about half
# line up all of the percentages, like a cumulative one, and colour Bayswater?

geo['percent_train'] = np.where(~geo['line'].isna(), geo['Count Total']/used_trains*100, np.nan)

percent_armadale = geo['Count Total'][geo['line']=='Armadale Line'].sum()/used_trains*100

# 7.7%. But also with flow on effects of bus services etc, this is likely to have an impact

#%%

fig = px.histogram(
    geo['percent_train'],
    nbins=100,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Number of stations',
    xaxis_title_text = 'Percentage of tags on/off',
)
pyo.plot(fig, config=config)


#%%

# remove stops that aren't used
geo = geo.drop(geo[geo["Count Total"] == 0].index).reset_index(drop=True)

# STOP CLUSTERING

# DBSCAN - stop clustering
eps = 90  # metres
minpts = 1  # smallest cluster size allowed

# cluster stops
clusters, geo = create_stop_cluster(geo, "spatial_cluster", eps, minpts)

geo.to_crs(epsg=4326, inplace=True)
geo["X"] = geo["geometry"].x
geo["Y"] = geo["geometry"].y

#%% PICKLE STOP CLUSTERS

clusters.to_pickle(dirname + "20250111-clusters.pkl")
geo.to_pickle(dirname + "20250111-geo.pkl")

#%% TEST PLOT OF CLUSTERS

# Plot of all stops coloured by spatial cluster
# Demonstrates which stops have been aggregated together

# Figure 2 in readme

geopolys = geo.dissolve("spatial_cluster").convex_hull

geopolys = gpd.GeoDataFrame(geopolys)
geopolys.reset_index(inplace=True)
geopolys.rename(columns={0: "geometry"}, inplace=True)
geopolys = geopolys.set_geometry("geometry")
geopolys = geopolys.set_crs("epsg:4326", allow_override=True)

geopolys = geopolys.to_crs(epsg=32749)
geopolys["geometry"] = geopolys["geometry"].buffer(MAPBUFFER)

geo.to_crs(epsg=4326, inplace=True)
geopolys.to_crs(epsg=4326, inplace=True)

fig = px.choropleth_map(
    geopolys,
    geojson=geopolys.geometry,
    locations=geopolys.index,
    color="spatial_cluster",
    center={"lat": -31.95, "lon": 115.85},
    color_continuous_scale=mycolors_discrete,
    opacity=0.5,
    zoom=13,
)
fig.update_layout(map_style="light", coloraxis_showscale=False)

fig.add_trace(
    go.Scattermap(
        lat=geo["geometry"].y,
        lon=geo["geometry"].x,
        mode="markers",
        marker=go.scattermap.Marker(
            color=geo["spatial_cluster"],
            colorscale=mycolors_discrete,
        ),
    )
)

pyo.plot(fig, config=config)

filename = "sensitivity_stops_eps_"+str(eps)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


#%%
filename = "prep_stopclustermap"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% READ ACTIVITIES FILE

# Alternative to the next cell - read in instead of running
activities = pd.read_pickle(dirname + "20250111-activities.pkl")

#%% TURN JOURNEYS INTO STAYS

start_date = "2017-08-01 00:00:00"
start_date = pd.to_datetime(start_date, format="%Y-%m-%d %H:%M:%S")

# calculate total hours since start of the month
journeys["elapsedHours_On"] = (journeys["OnTime"] - start_date) / dt.timedelta(hours=1)

journeys["elapsedHours_Off"] = (journeys["OffTime"] - start_date) / dt.timedelta(
    hours=1
)

# get sequence of activities by card
activities = pd.melt(
    journeys,
    id_vars=["Cardid", "Token", "OnLocation", "OffLocation"],
    value_vars=["elapsedHours_On", "elapsedHours_Off"],
)
# put back in order
activities.sort_values(by=["Cardid", "value"], axis=0, ascending=True, inplace=True)
# simplify variable names
activities["location"] = np.where(
    activities["variable"] == "elapsedHours_On",
    activities["OnLocation"],
    activities["OffLocation"],
)
activities["variable"] = np.where(
    activities["variable"] == "elapsedHours_On", "On", "Off"
)

activities["Duration"] = (
    activities["value"]
    .shift(-1)
    .where(activities["Cardid"].shift(-1) == activities["Cardid"])
    - activities["value"]
)
# separate into travel vs stay
activities["activity"] = np.where(activities["variable"] == "On", "T", "stay")

# get coordinates of locations
activities = pd.merge(
    activities, geo, left_on="location", right_on="StopID", how="left"
)
activities.rename(columns={"X": "LocationLon", "Y": "LocationLat"}, inplace=True)

# use this to get the *next* spatial cluster
activities = pd.merge(
    activities, geo, left_on="OffLocation", right_on="StopID", how="left"
)

activities.drop(
    ["StopID_x", "StopID_y", "X", "Y", "geometry_x", "geometry_y"], axis=1, inplace=True
)

activities.rename(
    columns={
        "spatial_cluster_x": "spatial_cluster",
        "spatial_cluster_y": "spatial_cluster_next",
    },
    inplace=True,
)

# calculate distance travelled between start and end locations of the stay
# approximation only - ok for smaller distances
activities["Distance_stay"] = 110.3 * np.sqrt(
    np.square(activities["LocationLon"] - activities["LocationLon"].shift(-1))
    + np.square(activities["LocationLat"] - activities["LocationLat"].shift(-1))
).where(activities["Cardid"] == activities["Cardid"].shift(-1))

activities["Distance_travel"] = activities["Distance_stay"].shift(1)

activities["Start_h"] = activities["value"]
activities["Start_h_only"] = activities["value"].mod(24)

activities["End_h"] = activities["Start_h"] + activities["Duration"]
activities.rename(columns={"value": "Cumulative_h"}, inplace=True)


def learn_stay_class(
    df, col_duration, col_arrival, col_token, col_distance):
    df["class"] = 0

    # A attractor activities

    df["class"] = np.where(
        (df[col_duration] >= 5)
        & (df[col_duration] < 16)
        & (df[col_arrival] > 5)
        & (df[col_arrival] <= 12)
        & (df[col_token] != "School"),
        1,
        df["class"],
    )  # work W

    df["class"] = np.where(
        (df[col_duration] >= 5)
        & (df[col_duration] < 16)
        & (df[col_arrival] > 5)
        & (df[col_arrival] <= 12)
        & (df[col_token] == "School"),
        2,
        df["class"],
    )  # school E

    df["class"] = np.where(
        (df[col_duration] >= 1) & (df[col_duration] < 3), 3, df["class"]
    )  # short stay S

    df["class"] = np.where(
        (df[col_duration] >= 3) & (df[col_duration] < 6), 4, df["class"]
    )  # medium stay M

    df["class"] = np.where(
        (df[col_arrival] > 12)
        & (df[col_arrival] < 18)
        & (df[col_duration] >= 6)
        & (df[col_duration] < 10),
        4,
        df["class"],
    )  # medium stay M

    # C Connector activities

    df["class"] = np.where(
        (df[col_duration] >= 0) & (df[col_duration] < 1), 8, df["class"]
    )  # overnight transfer (default) before 5am or after 7pm T

    df["class"] = np.where(
        (df[col_arrival] > 5)
        & (df[col_arrival] <= 9)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        5,
        df["class"],
    )  # AM peak T

    df["class"] = np.where(
        (df[col_arrival] > 9)
        & (df[col_arrival] <= 15)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        6,
        df["class"],
    )  # day time T

    df["class"] = np.where(
        (df[col_arrival] > 15)
        & (df[col_arrival] <= 19)
        & (df[col_duration] >= 0)
        & (df[col_duration] < 1),
        7,
        df["class"],
    )  # PM peak T

    # generator activities (overnight stays)

    df["class"] = np.where(
        (df[col_arrival] >= 12) & (df[col_duration] >= 10) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep L
    df["class"] = np.where(
        (df[col_arrival] >= 18) & (df[col_duration] >= 6) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep L
    df["class"] = np.where(
        (df[col_arrival] <= 5) & (df[col_duration] >= 6) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep - early hours arrive L
    df["class"] = np.where(
        (df[col_arrival] <= 12) & (df[col_duration] >= 16) & (df[col_duration] <= 20),
        9,
        df["class"],
    )  # sleep - after AM arrive L
    df["class"] = np.where(
        (df[col_duration] > 20) & (df[col_duration] <= 3 * 24), 10, df["class"]
    )  # long stay >20h and <3 days V

    return df[col_token], df["class"]


activities["Token"], activities["class"] = learn_stay_class(
    activities, "Duration", "Start_h_only", "Token", "Distance_stay")

activities["stayclass"] = activities["class"].map(classmap)

activities["activity_alpha"] = np.where(
    activities["activity"] == "stay", activities["stayclass"], activities["activity"]
)
activities["activity_int"] = np.where(
    activities["activity"] == "stay", activities["class"], 12
)  # 12 is the class for travel in the classmap

print(len(activities),"activities before dropping unlabelled, travel or transfer stays")

print(len(activities[activities['activity_alpha']=="V"])/len(activities)*100, "percent of stays are >20h but <3 days")
# 6.92%
print(len(activities[activities['activity_alpha']=="drop"])/len(activities)*100, "percent of stays are >3 days")


activities = activities.drop(
    activities[activities["activity_alpha"] == "drop"].index
).reset_index(drop=True)

# drop all travel (T) and transfer (C) activities
activities = activities.drop(
    activities[
        (activities["activity_alpha"] == "T")
        | (activities["activity_alpha"] == "C")
    ].index
).reset_index(drop=True)

activities.drop(
    [
        "stayclass",
        "class",
        "activity",
        "variable",
        "OnLocation",
        "OffLocation",
        "Start_h_only",
    ],
    axis=1,
    inplace=True,
)

print(len(activities), "activities remaining")

#%% PICKLE ACTIVITIES FILE

activities.to_pickle(dirname + "20250111-activities.pkl")

#%% HISTOGRAM OF DISTANCES IN STAYS

# For exploration only. Distance between start and end locations of each stay
# Used to confirm assumption that the land use around the stay location can be
# used to infer activity. If the distance is large, the passenger does not remain
# in the location and thus the land use doesn't inform the activity type.


activities['Distance_bin'] = 0
activities['Distance_bin'] = np.where(activities['Distance_stay']<.8,"<800m",activities['Distance_bin'])
activities['Distance_bin'] = np.where((activities['Distance_stay']>=.8)&(activities['Distance_stay']<2),">=800m, <2km",activities['Distance_bin'])
activities['Distance_bin'] = np.where(activities['Distance_stay']>=2,">=2km",activities['Distance_bin'])

df_distance = activities['Distance_bin'][activities['Distance_bin']!='0'].value_counts()
df_distance = pd.DataFrame(df_distance)
df_distance.reset_index(inplace=True)

df_distance.rename(columns={'Distance_bin':'Distance', 'count':'Count'}, inplace=True)

fig = px.bar(df_distance, x='Distance', y='Count',
             color_discrete_sequence=[mycolors_discrete[7]],
             )
fig.update_layout(
    xaxis_title_text = 'Distance between start and end point of stay', 
    yaxis_title_text = 'Number of stays',
    showlegend=False,
    )
fig.update_layout(default_layout,
                  xaxis={'categoryorder':'array', 'categoryarray':['<800m','>=800m, <2km',">=2km"]})
pyo.plot(fig, config=config)
#%%
filename = "prep_staydistance_cat"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% HISTOGRAM OF ACTIVITY TIME
# addition for the NCAA paper - to justify the boring commuters

activities['daystart_h'] = activities['Start_h'].mod(24)

#%%
fig = px.histogram(
    activities['daystart_h'],
    nbins=48,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Number of activities',
    xaxis_title_text = 'Time of day',
)
pyo.plot(fig, config=config)


#%% READ CARD SUMMARY

cards = pd.read_pickle(dirname + "20250111-cards.pkl")

#%% GENERATE CARD SUMMARY

# summarise cards to make it easier to chart/investigate/choose what to look at
cards = activities["Cardid"].value_counts()
cards = pd.DataFrame(cards)
cards.reset_index(inplace=True)
cards.rename(columns={"Cardid": "Card"}, inplace=True)

print(cards[(cards["count"] > 100) & (cards["count"] < 150)])

#%%
cards.to_pickle(dirname + "20250111-cards.pkl")

#%% LAND USE DATA

mbpoly = gpd.read_file(dirname + "MB_2016_WA.shp")
mbpoly = mbpoly.to_crs(epsg=4326)

# create mask for land use categories
landmap = {
    "Residential": "Residential",
    "Commercial": "Industrial/Commercial",
    "Industrial": "Industrial/Commercial",
    "Transport": "Industrial/Commercial",
    "Primary Production": "Industrial/Commercial",
    "Other": "Industrial/Commercial",
    "Parkland": "Parks/Water",
    "Water": "Parks/Water",
    "Education": "Education",
    "Hospital/Medical": "Hospital/Medical",
}

mbpoly["MB_CAT16"] = mbpoly["MB_CAT16"].map(landmap)

mbdissolve = mbpoly.dissolve(by="MB_CAT16")
mbdissolve.reset_index(inplace=True)

#%% CREATE RANDOM SAMPLE OF CARDS

np.random.seed(42)

# allocate each card to a number between 1 and 5
cards["split"] = np.random.randint(1, 6, cards.shape[0])
cardrun = cards[(cards['split']==4) | (cards['split']<=2)]
cardrun.reset_index(inplace=True)

print(len(cardrun), "cards used in analysis")

#%% READ REGIONS

allhist = pd.read_pickle(dirname + "20250125-fullallhist.pkl") 
allregions = pd.read_pickle(dirname + "20250125-fullallregions.pkl") 
allregionpolys = pd.read_pickle(dirname + "20250125-fullallregionpolys.pkl") 

#%% GENERATE REGIONS

# This takes about a 2min per 1000 cards (only approximate - varies with the 
# amount of activities on each card; more activities = takes longer)

def generate_regions(
    card,
    clusters,
    activities,
    STAY_DROP_THRESHOLD,
    ANCHOR_REGION_SIZE,
    MIN_POINTS_REGION,
    MAPBUFFER,
):
    carddata = activities[activities["Cardid"] == card]

    # mean, standard deviation, count, sum
    hist = carddata.groupby(["spatial_cluster"], as_index=False).agg(
        {"Duration": ["mean", "std", "count", "sum"]}
    )

    # remove multiindex
    hist.columns = [" ".join(col).strip() for col in hist.columns.values]

    # drop where number of records is below the threshold, otherwise
    # distribution is meaningless
    hist = hist.drop(
        hist[hist["Duration count"] < STAY_DROP_THRESHOLD].index
    ).reset_index(drop=True)

    # merge hist with clusters to get geometry for cluster ID
    hist = pd.merge(
        hist, clusters, left_on="spatial_cluster", right_on="clusterID", how="left"
    )
    hist.drop(["clusterID"], axis=1, inplace=True)

    hist = gpd.GeoDataFrame(hist, geometry="geometry")

    regions, hist = create_stop_cluster(
        hist, "region_cluster", ANCHOR_REGION_SIZE, MIN_POINTS_REGION
    )

    region_summary = pd.pivot_table(
        hist,
        index="region_cluster",
        values=("Duration count", "Duration sum"),
        aggfunc=("sum"),
    )
    region_summary.reset_index(inplace=True)

    region_summary.rename(
        columns={"Duration count": "num_visits", "Duration sum": "total_stay_time"},
        inplace=True,
    )

    region_summary["avg_stay_time"] = (
        region_summary["total_stay_time"] / region_summary["num_visits"]
    )

    region_summary["fraction_time"] = (
        region_summary["total_stay_time"] / region_summary["total_stay_time"].sum()
    )
    region_summary["fraction_time_cum"] = region_summary["fraction_time"].cumsum(axis=0)

    region_summary["fraction_visits"] = (
        region_summary["num_visits"] / region_summary["num_visits"].sum()
    )
    region_summary["fraction_visits_cum"] = region_summary["fraction_visits"].cumsum(
        axis=0
    )

    region_summary = pd.merge(
        region_summary,
        regions,
        left_on="region_cluster",
        right_on="clusterID",
        how="left",
    )
    region_summary.drop(["clusterID"], axis=1, inplace=True)

    region_summary = gpd.GeoDataFrame(region_summary, geometry="geometry")

    regionpolys = hist.dissolve("region_cluster").convex_hull
    regionpolys = gpd.GeoDataFrame(regionpolys)
    regionpolys.reset_index(inplace=True)
    regionpolys.rename(columns={0: "geometry"}, inplace=True)
    regionpolys.set_geometry("geometry", inplace=True)

    # buffer the region centroid to get the circle
    regionpolys = regionpolys.to_crs(epsg=32749)
    regionpolys["geometry"] = regionpolys["geometry"].buffer(MAPBUFFER)
    regionpolys["area"] = regionpolys.area
    regionpolys["area"] = regionpolys["area"] / 1000000  # to get sq km

    return hist, region_summary, regionpolys

#%%
start = timeit.default_timer()

allhist = pd.DataFrame()
allregions = pd.DataFrame()
allregionpolys = pd.DataFrame()

for index, row in cardrun.iterrows():
    if np.remainder(index, 500) == 0:
        print(
            str(index)
            + " of "
            + str(len(cardrun))
            + " cards complete, "
            + str((timeit.default_timer() - start)/60)
            + " minutes elapsed, ("
            + str((timeit.default_timer() - start)/60/(index+1)*1000)
            + " minutes per 1000 cards)"
        )
    ret = generate_regions(
        row["Card"],
        clusters,
        activities,
        STAY_DROP_THRESHOLD,
        ANCHOR_REGION_SIZE,
        MIN_POINTS_REGION,
        MAPBUFFER,
    )

    if ret is None:
        continue
    hist, region_summary, regionpolys = ret
    hist["Card"] = row["Card"]
    region_summary["Card"] = row["Card"]
    regionpolys["Card"] = row["Card"]

    allhist = pd.concat([allhist, hist])
    allregions = pd.concat([allregions, region_summary])
    allregionpolys = pd.concat([allregionpolys, regionpolys])

print("All cards complete, " + str((timeit.default_timer() - start)/60) + " minutes elapsed")

# Export so have a full list of regions (not just the anchoring ones)

allhist.to_pickle(dirname + "20250111-fullallhist-1,2.pkl")
allregions.to_pickle(dirname + "20250111-fullallregions-1,2.pkl")
allregionpolys.to_pickle(dirname + "20250111-fullallregionpolys-1,2.pkl")

#%% HISTOGRAM OF NUMBER OF VISITS TO EACH REGION

# Used to assess for an 'elbow' - is there a number of vists at which there is
# a clear inflection?

# Figure 4 in readme

# Long tail - truncate for graphing only to make it easier to see

fig = px.histogram(
    allregions["num_visits"][allregions["num_visits"] < 20],
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    xaxis_title_text = 'Number of visits to region', 
    yaxis_title_text = 'Number of regions',
    showlegend=False,
    )

fig.update_layout(default_layout)
pyo.plot(fig, config=config)

#%%
filename = "regions_numvisitshist_lt20"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%

# From this plot, determine elbow at 5 regions
# Any region with >= 5 visits is called an 'anchor' region
# This works out to a region with a visit on average at least approximately
# once per week - so makes intuitive sense too
NUM_REGION_VISITS = 5

#%%

# make a new version of the histogram with <this value a different colour

allregions['color'] = np.where(allregions['num_visits']<NUM_REGION_VISITS,'col1','col2')

fig = px.histogram(
    allregions['num_visits'][allregions["num_visits"] < 20],
    color=allregions['color'][allregions["num_visits"] < 20],
    color_discrete_sequence=[mycolors_discrete[7], mycolors_discrete[1]],
)

fig.update_layout(default_layout_ncaa,     
                  xaxis_title_text = 'Number of visits to region', 
                  yaxis_title_text = 'Number of regions',
                  showlegend=False,
                  )
pyo.plot(fig, config=config)

filename = "fig_anchorhist"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)


#%% DETERMINE ANCHORING REGIONS

allregions["region_type"] = np.where(
    allregions["num_visits"] >= NUM_REGION_VISITS, "Anchor", "Visited"
)

allhist = pd.merge(
    allhist,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)
allregionpolys = pd.merge(
    allregionpolys,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)

#%% COVERAGE OF ANCHORING REGIONS

regionsummary = pd.pivot_table(
    allregions,
    values=["fraction_time", "fraction_visits"],
    index=["Card"],
    columns="region_type",
    fill_value=0,
    aggfunc=[np.sum, "count"],
)
regionsummary.reset_index(inplace=True)

print(len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0]), "cards have no anchor region")
print("equivalent to", len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0])/len(regionsummary)*100,"percent")
print("so",len(regionsummary)-len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0]),"cards have anchor regions")

timesummary = regionsummary.loc[:, ("sum", "fraction_time", "Anchor")]
d = {"fraction_time": timesummary}
timesummary = pd.DataFrame(data=d)

visitsummary = regionsummary["sum", "fraction_visits", "Anchor"]
d = {"fraction_visits": visitsummary}
visitsummary = pd.DataFrame(data=d)

print(
    len(timesummary[timesummary["fraction_time"] > 0.75]) / 
    len(timesummary[timesummary["fraction_time"] > 0])*100, 
    "cards that have anchoring regions spend more than 75% of their time in those regions"
)  

print(
    len(visitsummary[visitsummary["fraction_visits"] > 0.75]) / 
    len(visitsummary[visitsummary["fraction_visits"] > 0])*100,
    "cards that have anchoring regions complete more than 75% of their activities in those regions"
)  

#%% HISTOGRAM - FRACTION OF TIME COVERED BY ANCHORING REGIONS

# Figure 5a in readme

fig = px.histogram(
    timesummary,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    xaxis_title_text = 'Fraction of time covered by anchoring regions', 
    yaxis_title_text = 'Number of cards',
    showlegend=False,
    )

fig.update_layout(default_layout)
fig.update_traces(xbins=dict( 
        start=0,
        end=1,
    )) 

pyo.plot(fig, config=config)

filename = "regions_fractimeanchorhist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% HISTOGRAM - FRACTION OF VISITS COVERED BY ANCHORING REGIONS

# Figure 5b in readme

fig = px.histogram(
    visitsummary,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(default_layout)
fig.update_layout(
    xaxis_title_text = 'Fraction of visits covered by anchoring regions', 
    yaxis_title_text = 'Number of cards',
    showlegend=False,
    )
fig.update_traces(xbins=dict( 
        start=0,
        end=1,
    )) 

pyo.plot(fig, config=config)

filename = "regions_fracvisitsanchorhist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% SEPARATE OUT ANCHORING REGIONS ONLY

allhist = allhist[allhist["region_type"] == "Anchor"]
allregions = allregions[allregions["region_type"] == "Anchor"]
allregionpolys = allregionpolys[allregionpolys["region_type"] == "Anchor"]

#%% READ FILES

# instead of the next cell - import instead

allregions2 = pd.read_pickle(dirname + "20250112-final-allregions-1,2.pkl")
allregions1 = pd.read_pickle(dirname + "20250112-final-allregions.pkl")
allregions = pd.concat([allregions1,allregions2])

landusepolys2 = pd.read_pickle(dirname + "20250112-final-landusepolys-1,2.pkl")
landusepolys1 = pd.read_pickle(dirname + "20250112-final-landusepolys.pkl")
landusepolys = pd.concat([landusepolys1,landusepolys2])

regionpivot2 = pd.read_pickle(dirname + "20250112-final-regionpivot-1,2.pkl")
regionpivot1 = pd.read_pickle(dirname + "20250112-final-regionpivot.pkl")
regionpivot = pd.concat([regionpivot1,regionpivot2])

#%% OVERLAY LAND USE ON REGIONS

# This also takes ages. About 10h for 40,000 cards

start = timeit.default_timer()

# Buffer the stops by the STOPBUFFER (circles around the stops)
landusepolys = allhist.to_crs(epsg=32749)
landusepolys["geometry"] = landusepolys["geometry"].buffer(STOPBUFFER)

# Concatenate the regional polygons and the buffered stops
landusepolys = pd.concat([landusepolys, allregionpolys], ignore_index=True, sort=False)
print(
    "Geodataframes concatenated, "
    + str(timeit.default_timer() - start)
    + " seconds elapsed"
)

# Dissolve by card and region_cluster
landusepolys = landusepolys.dissolve(by=["Card", "region_cluster"])
landusepolys.reset_index(inplace=True)

landusepolys["area"] = landusepolys.area
landusepolys.sort_values(by=["area"], axis=0, ascending=False, inplace=True)

print("Regions dissolved, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Get extent of the polygons and reduce the land use dataset to those bounds
landusepolys = landusepolys.to_crs(epsg=4326)
xmin, ymin, xmax, ymax = landusepolys.total_bounds
mbdissolve_small = mbdissolve.cx[xmin:xmax, ymin:ymax]

print(
    "Land use bounds reduced, "
    + str(timeit.default_timer() - start)
    + " seconds elapsed"
)

# Make sure in same CRS
mbdissolve_small = mbdissolve_small.to_crs(epsg=32749)
landusepolys = landusepolys.to_crs(epsg=32749)

# Overlay land use - this is the slow part
ABSoverlay = mbdissolve_small.overlay(landusepolys, how="intersection")
ABSoverlay["area"] = ABSoverlay["geometry"].area

print("Land use overlayed, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Calculate area of each land use type for each region cluster
ABSpivot = pd.pivot_table(
    ABSoverlay,
    values="area",
    index=["Card", "region_cluster"],
    columns="MB_CAT16",
    fill_value=0,
    aggfunc=np.sum,
)
ABSpivot.reset_index(inplace=True)
ABSpivot["Total"] = ABSpivot.iloc[:, 2:7].sum(axis=1)

print("Areas calculated, " + str(timeit.default_timer() - start) + " seconds elapsed")

# Turn these areas into fractions
ABSpivot.iloc[:, 2:7] = ABSpivot.iloc[:, 2:7].div(ABSpivot["Total"], axis=0)
ABSpivot.drop("Total", inplace=True, axis=1)

print(
    "Fractions calculated, " + str(timeit.default_timer() - start) + " seconds elapsed"
)

activityregions = pd.merge(
    activities,
    allhist[["Card", "spatial_cluster", "region_cluster", "region_type"]],
    left_on=["Cardid", "spatial_cluster"],
    right_on=["Card", "spatial_cluster"],
    how="left",
)
activityregions = activityregions.dropna(
    axis="rows"
)  # because it hasn't been run for all cards, most will be NaN

regionpivot = pd.pivot_table(
    activityregions,
    index=["Cardid", "region_cluster", "region_type"],
    columns="activity_alpha",
    aggfunc="count",
    values="Card",
    fill_value=0,
)
regionpivot.reset_index(inplace=True)

regionpivot["Total"] = regionpivot.iloc[:, 3:9].sum(
    axis=1
)  # includes V = very long stays (>20h)
# region pivot = 159,078 records
# 22,020 of these are just V activities
# 90,969 visited regions, 68,109 anchoring regions

regionpivot["Total"] = regionpivot["Total"] - regionpivot["V"]  
# subtract off the V

# Drop regions that only have the very long stays - these will be where 
# the total now equals 0
regionpivot.drop(
    regionpivot[regionpivot["Total"] == 0].index, inplace=True
)


# This dataset includes all regions, so drop the non-anchoring ones
regionpivot.drop(
    regionpivot[regionpivot["region_type"] == "Visited"].index, inplace=True
)

regionpivot["E_frac"] = regionpivot["E"].div(regionpivot["Total"], axis=0)
regionpivot["L_frac"] = regionpivot["L"].div(regionpivot["Total"], axis=0)
regionpivot["M_frac"] = regionpivot["M"].div(regionpivot["Total"], axis=0)
regionpivot["S_frac"] = regionpivot["S"].div(regionpivot["Total"], axis=0)
regionpivot["W_frac"] = regionpivot["W"].div(regionpivot["Total"], axis=0)

regionpivot["regionid"] = (
    regionpivot["Cardid"].astype(str) + "-" 
    + regionpivot["region_cluster"].astype(str)
)
regionpivot = pd.merge(
    regionpivot,
    ABSpivot,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)

allregions.to_pickle(dirname + "20250125-final-allregions.pkl") 
landusepolys.to_pickle(dirname + "20250125-final-landusepolys.pkl") 
regionpivot.to_pickle(dirname + "20250125-final-regionpivot.pkl") 


#%% CLUSTER REGIONS - GAUSSIAN MIXTURE MODEL

X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]]

init_state = 0

# To determine the appropriate amount of clusters
# Run GMM across 1 to 20 clusters and check AIC/BIC
n_components = np.arange(1, 20)
models = [GaussianMixture(n, random_state=init_state).fit(X) for n in n_components]

d = {"BIC": [m.bic(X) for m in models], "AIC": [m.aic(X) for m in models], "num_clusters": [n for n in n_components]}
df = pd.DataFrame(data=d)

# Figure 6 in readme

fig = px.line(
    df,
    x = "num_clusters",
    y = ['BIC','AIC'],
    color_discrete_sequence=mycolors_discrete,
    labels={"num_clusters": "Number of clusters", 
            "value": "", 
            "variable": "Criteria"},
)
fig.update_layout(default_layout)
pyo.plot(fig, config=config)


#%%
filename = "regions_BICAICfull"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%
# From this plot, determine that the appropriate number of clusters
NUM_COMPONENTS = 6

#%% FIT GMM TO SELECTED NUMBER OF CLUSTERS

GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=init_state).fit(X)
colname = "GMM_cluster"

regionpivot[colname] = GMM.predict(regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]])
means0 = GMM.means_

outregion = pd.melt(
    regionpivot,
    id_vars=[colname],
    value_vars=["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"],
)
outland = pd.melt(
    regionpivot,
    id_vars=[colname],
    value_vars=[
        "Industrial/Commercial",
        "Education",
        "Hospital/Medical",
        "Parks/Water",
        "Residential",
    ],
)
outregionsum = pd.pivot_table(
    outregion, index=colname, columns="variable", aggfunc="sum"
)
outlandsum = pd.pivot_table(outland, index=colname, columns="variable", aggfunc="sum")

outland.sort_values(by=[colname, "variable"], axis=0, ascending=True, inplace=True)
outregion.sort_values(by=[colname, "variable"], axis=0, ascending=True, inplace=True)

#%% SAVING FINAL OUTPUT DATA BEFORE PLOTTING

regionpivot.to_pickle(dirname + "20250125-final-regionpivot-GMM-full.pkl")
outland.to_pickle(dirname + "20250125-outland.pkl")
outregion.to_pickle(dirname + "20250125-outregion.pkl")

#%% OPTION TO READ DATA

regionpivot = pd.read_pickle(dirname + "20250125-final-regionpivot-GMM-full.pkl")
outland = pd.read_pickle(dirname + "20250125-outland.pkl")
outregion = pd.read_pickle(dirname + "20250125-outregion.pkl")

#%% SUPPORTING OUTPUT CHARTS

numregions = pd.pivot_table(
    regionpivot, index="Cardid", columns="region_type", aggfunc="count", values=["Card"]
)
#%%
# Histogram of number of regions for each user
# Figure 7 in readme

fig = px.histogram(
    numregions.values,
    color_discrete_sequence=[mycolors_discrete[7]])
fig.update_layout(xaxis_title="Number of regions per card",
                  yaxis_title="Number of cards",
                  showlegend=False)
fig.update_layout(default_layout_ncaa)
#pyo.plot(fig, config=config)

filename = "Fig4"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)

#%%
filename = "regions_numregionshist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% BOXPLOT - ACTIVITY FRACTIONS BY REGION CLUSTER

# Figure 8 in readme, Figure 3 in paper

d = {"E_frac": "Education", "L_frac": "Long", "S_frac": "Short", 
     "M_frac": "Medium", "W_frac": "Work"}

outregion["variable"] = outregion["variable"].map(d)

# mapping - this is just to make the clusters in the same order
# as they were described in the AusDM paper
region_map = {0:2, 1:0, 2:4, 3:5, 4:1, 5:3}
outregion['new_cluster'] = outregion[colname].map(region_map)

fig = px.box(
    outregion,
    x='new_cluster',
    #x = colname, 
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    labels={colname: "Region ID", "value": "Fraction", "variable": "Activity"},
)

fig.update_layout(default_layout_box)

fig.update_layout(xaxis_title="Cluster",
                  yaxis_title="Fraction of activities",)

fig.update_yaxes(
    range=(0, 1),
    constrain='domain'
)

# fig.update_traces(boxpoints=False) # sets whiskers to min/max
fig.update_traces(
    marker=dict(opacity=0)
)  # sets whiskers to usual st dev and not 'outliers'
for i in np.arange(0,outregion[colname].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
pyo.plot(fig, config=config)

# filename = "regions_activityboxplot"
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
# fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% BOXPLOT - LAND USE FRACTIONS BY REGION CLUSTER

# Figure 9 in readme, Figure 5 in paper

# again just mapping to line up numbers with AusDM paper
outland['new_cluster'] = outland[colname].map(region_map)

fig = px.box(
    outland,
    x='new_cluster',
    #x=colname,
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    labels={'new_cluster': "Region ID", "value": "Fraction", "variable": "Land use"},
)

fig.update_layout(default_layout_box)
fig.update_layout(xaxis_title="Cluster",
                  yaxis_title="Fraction of land use",)


fig.update_yaxes(
    range=(0, 1),
    constrain='domain'
)

fig.update_traces(marker=dict(opacity=0))
for i in np.arange(0,outland['new_cluster'].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

filename = "regions_landuseboxplot"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% DEMONSTRATING THE EFFECT OF THE COMMUTERS

regiondata = regionpivot[['E_frac','L_frac','S_frac','M_frac','W_frac']].copy()

regiondata_unique = regiondata.value_counts()
regiondata_unique = pd.DataFrame(regiondata_unique)
regiondata_unique.reset_index(inplace=True)

regiondata_unique["frac"] = (
    regiondata_unique["count"] / regiondata_unique["count"].sum() * 100
)
regiondata_unique["cum_frac"] = regiondata_unique["frac"].cumsum(axis=0)


#%% NOW ALL AGAIN BUT WITH UNIQUE RECORDS ONLY

X = X.drop_duplicates()

init_state = 0

# To determine the appropriate amount of clusters
# Run GMM across 1 to 20 clusters and check AIC/BIC
n_components = np.arange(1, 20)
models = [GaussianMixture(n, random_state=init_state).fit(X) for n in n_components]

d = {"BIC": [m.bic(X) for m in models], "AIC": [m.aic(X) for m in models], "num_clusters": [n for n in n_components]}
df = pd.DataFrame(data=d)

fig = px.line(
    df,
    x = "num_clusters",
    y = ['BIC','AIC'],
    color_discrete_sequence=mycolors_discrete,
    labels={"num_clusters": "Number of clusters", 
            "value": "", 
            "variable": "Criteria"},
)
fig.update_layout(default_layout)
pyo.plot(fig, config=config)


#%%
filename = "regions_BICAICunique"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%
# From this plot, determine that the appropriate number of clusters
NUM_COMPONENTS = 5 

#%% FIT GMM TO SELECTED NUMBER OF CLUSTERS

GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=init_state).fit(X)
colname_unique = "GMM_cluster_unique"

regionpivot.drop(columns=colname, inplace=True)

regionpivot[colname_unique] = GMM.predict(regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]])
means0 = GMM.means_

outregion_unique = pd.melt(
    regionpivot,
    id_vars=[colname_unique],
    value_vars=["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"],
)
outland_unique = pd.melt(
    regionpivot,
    id_vars=[colname_unique],
    value_vars=[
        "Industrial/Commercial",
        "Education",
        "Hospital/Medical",
        "Parks/Water",
        "Residential",
    ],
)
outregionsum = pd.pivot_table(
    outregion_unique, index=colname_unique, columns="variable", aggfunc="sum"
)
outlandsum = pd.pivot_table(outland_unique, index=colname_unique, columns="variable", aggfunc="sum")

outland_unique.sort_values(by=[colname_unique, "variable"], axis=0, ascending=True, inplace=True)
outregion_unique.sort_values(by=[colname_unique, "variable"], axis=0, ascending=True, inplace=True)

#%% SAVING FINAL OUTPUT DATA BEFORE PLOTTING

regionpivot.to_pickle(dirname + "20250125-final-regionpivot-GMM-unique.pkl")
outland_unique.to_pickle(dirname + "20250125-outland_unique.pkl")
outregion_unique.to_pickle(dirname + "20250125-outregion_unique.pkl")

#%% OPTION TO READ DATA

regionpivot_unique = pd.read_pickle(dirname + "20250125-final-regionpivot-GMM-unique.pkl")
outland_unique = pd.read_pickle(dirname + "20250125-outland_unique.pkl")
outregion_unique = pd.read_pickle(dirname + "20250125-outregion_unique.pkl")

#%% BOXPLOT - ACTIVITY FRACTIONS BY REGION CLUSTER

d = {"E_frac": "Education", "L_frac": "Long", "S_frac": "Short", 
     "M_frac": "Medium", "W_frac": "Work"}

outregion_unique["variable"] = outregion_unique["variable"].map(d)

clustermap_unique = {
    0: "Education",
    1: "Residences",
    2: "Workplaces",
    3: "Residences/Leisure",
    4: "Workplaces/Leisure",
}

outregion_unique['Cluster'] = outregion_unique[colname_unique].map(clustermap_unique)

#%%

# drop education and residence clusters for clarity

outregion_unique = outregion_unique[(outregion_unique['Cluster']!="Education")&(outregion_unique['Cluster']!="Residences")]

fig = px.box(
    outregion_unique,
    x = 'Cluster', 
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    labels={colname_unique: "Region ID", "value": "Fraction", "variable": "Activity"},
)

fig.update_layout(default_layout_ncaa)
fig.update_layout(xaxis_title="Region activity cluster",
                  yaxis_title="Fraction of activities",)


fig.update_yaxes(
    range=(0, 1),
    constrain='domain'
)

# fig.update_traces(boxpoints=False) # sets whiskers to min/max
fig.update_traces(
    marker=dict(opacity=0)
)  # sets whiskers to usual st dev and not 'outliers'
for i in np.arange(0,len(outregion_unique[colname_unique].unique())):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
pyo.plot(fig, config=config)
#%%
filename = "Fig4"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=500)

#%%
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% BOXPLOT - LAND USE FRACTIONS BY REGION CLUSTER

# Figure 9 in readme, Figure 5 in paper


outland_unique['Cluster'] = outland_unique[colname_unique].map(clustermap_unique)


fig = px.box(
    outland_unique,
    x='Cluster',
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    labels={colname_unique: "Region ID", "value": "Fraction", "variable": "Land use"},
)

fig.update_layout(default_layout_ncaa)
fig.update_layout(yaxis_title="Fraction of land use", xaxis_title='Region activity cluster')

fig.update_yaxes(
    range=(0, 1),
    constrain='domain'
)

fig.update_traces(marker=dict(opacity=0))
for i in np.arange(0,outland_unique[colname_unique].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)


pyo.plot(fig, config=config)

filename = "Fig5"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=500)

#%%
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% TESTING CLUSTER STABILITY

def calc_max_instability(n):
    return (n*(n-1)/2)/(n*n)

X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]]

def calc_instability(num_clusters, num_subsamples, sample_size):
    np.random.seed(42)
    
    states = np.random.randint(1000, size=num_subsamples)

    for i in np.arange(0,num_subsamples):
        
        Y = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]].sample(n=sample_size, random_state=states[i])
        Y = Y.drop_duplicates() # NEW 11/05
        GMM = GaussianMixture(n_components=num_clusters, random_state=states[i]).fit(Y)
        sub_col = 'subsample_'+str(i)
        regionpivot[sub_col] = GMM.predict(X)    
        means = GMM.means_
        
        if i==0:
            means0 = GMM.means_
        
        if i>0:
            dist = scidist.cdist(means0, means, 'euclidean')
            min_loc = np.argmin(dist, axis=0)     
            colmap = dict(zip(np.arange(0,num_clusters),min_loc))
            regionpivot[sub_col] = regionpivot[sub_col].map(colmap)
    
    features = regionpivot.iloc[:,-num_subsamples:]
    #regionpivot.drop(columns=regionpivot.columns[-num_subsamples:], axis=1, inplace=True)
    features = features.transpose()
    distmatrix = pdist(features, metric='hamming')
    npmatrix = squareform(distmatrix) 

    inst_max = calc_max_instability(num_subsamples)

    inst = npmatrix.sum()/np.square(num_subsamples)/inst_max

    # now do by individual
    # set results to strings so we can concat them
    regionpivot.iloc[:,-num_subsamples:] = regionpivot.iloc[:,-num_subsamples:].astype(str)
        
    individuals = pd.DataFrame()
    for i in np.arange(0,num_subsamples):
        individual_ss = regionpivot.groupby(['Cardid'])['subsample_'+str(i)].apply(''.join).reset_index()
        if i==0:
            individuals = individual_ss
        if i>0:
            individuals = pd.merge(individuals, individual_ss, left_on='Cardid', right_on='Cardid', how='left')
    
    features_ind = individuals.iloc[:,-num_subsamples:]
    features_ind = features_ind.astype(int)
    # we lose things like leading zeroes by doing this but we're only interested
    # on if they match or not, so it doesn't matter
    
    regionpivot.drop(columns=regionpivot.columns[-num_subsamples:], axis=1, inplace=True)
    features_ind = features_ind.transpose()

    distmatrix_ind = pdist(features_ind, metric='hamming')
    npmatrix_ind = squareform(distmatrix_ind) 

    inst_ind = npmatrix_ind.sum()/np.square(num_subsamples)
    
    return inst, inst_ind

#%%

k_max = 10

val10 = [calc_instability(n, 5, int(round(0.1*len(regionpivot),0))) for n in np.arange(2,k_max)]
val20 = [calc_instability(n, 5, int(round(0.2*len(regionpivot),0))) for n in np.arange(2,k_max)]
val30 = [calc_instability(n, 5, int(round(0.3*len(regionpivot),0))) for n in np.arange(2,k_max)]
val40 = [calc_instability(n, 5, int(round(0.4*len(regionpivot),0))) for n in np.arange(2,k_max)]
val50 = [calc_instability(n, 5, int(round(0.5*len(regionpivot),0))) for n in np.arange(2,k_max)]
val60 = [calc_instability(n, 5, int(round(0.6*len(regionpivot),0))) for n in np.arange(2,k_max)]

instability = {"num_clusters": [n for n in np.arange(2,k_max)], 
               "10%": [val10[i-2][0] for i in np.arange(2,k_max)],
               "20%": [val20[i-2][0] for i in np.arange(2,k_max)],
               "30%": [val30[i-2][0] for i in np.arange(2,k_max)],
               "40%": [val40[i-2][0] for i in np.arange(2,k_max)],
               "50%": [val50[i-2][0] for i in np.arange(2,k_max)],
               "60%": [val60[i-2][0] for i in np.arange(2,k_max)]}

instability = pd.DataFrame(data = instability)

instability_ind = {"num_clusters": [n for n in np.arange(2,k_max)], 
               "10%": [val10[i-2][1] for i in np.arange(2,k_max)],
               "20%": [val20[i-2][1] for i in np.arange(2,k_max)],
               "30%": [val30[i-2][1] for i in np.arange(2,k_max)],
               "40%": [val40[i-2][1] for i in np.arange(2,k_max)],
               "50%": [val50[i-2][1] for i in np.arange(2,k_max)],
               "60%": [val60[i-2][1] for i in np.arange(2,k_max)]}

instability_ind = pd.DataFrame(data = instability_ind)


#%% PLOT INSTABILITY 

fig = px.bar(
    instability,
    x="num_clusters", y=["10%","20%", "30%", "40%", "50%", "60%"],
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={'variable':'Sample size'}
)
fig.update_layout(default_layout_ncaa)
fig.update_layout(xaxis_title="Number of clusters",
                  yaxis_title="Instability",)
fig.update_layout(yaxis_range=[0,1])

for i in np.arange(instability['num_clusters'].min(),instability['num_clusters'].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

#%%
filename = "Fig3"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=500)



#%%
filename = "regions_instability"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT INDIVIDUAL INSTABILITY


fig = px.bar(
    instability_ind,
    x="num_clusters", y=["10%","20%", "30%", "40%", "50%", "60%"],
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={'variable':'Sample size'}
)
fig.update_layout(default_layout_ncaa)
fig.update_layout(xaxis_title="Number of clusters",
                  yaxis_title="Instability by card",)
fig.update_layout(yaxis_range=[0,1])

for i in np.arange(instability_ind['num_clusters'].min(),instability_ind['num_clusters'].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

#%% HISTOGRAM - REGIONS PER CARD BY REGION TYPE

# Histogram of number of regions per card by type

# Figure 10 in readme
cardsum = pd.pivot_table(
    regionpivot_unique, index="Cardid", columns=colname_unique, aggfunc="count", values=["Card"]
)
cardsum.fillna(0, inplace=True)

cardsum.columns = cardsum.columns.droplevel()

# no longer used - but names from the first version with the full
# dataset (no dropping duplicates)
clustermap = {
    0: "Education",
    1: "Residences",
    2: "Workplaces/Leisure",
    3: "Education/Leisure",
    4: "Workplaces",
    5: "Leisure/Residences",
}

clustermap_unique = {
    0: "Education",
    1: "Residences",
    2: "Workplaces",
    3: "Residences/Leisure",
    4: "Workplaces/Leisure",
}

cardsum.rename(columns=clustermap_unique, inplace=True)

cardsum["Total"] = cardsum.iloc[:, 0:5].sum(axis=1)

#%% STATS

# what percentage of cards have one residence anchoring region
len(cardsum[cardsum['Residences']==1])/len(cardsum)*100

# where cards have no residence regions, how many have only one anchoring region (so one way trips)
len(cardsum[(cardsum['Residences']==0)&(cardsum['Total']==1)])/len(cardsum[cardsum['Residences']==0])*100

# how many have no workplace region
len(cardsum[cardsum['Workplaces']==0])/len(cardsum)*100

# how many have one workplace region
len(cardsum[cardsum['Workplaces']==1])/len(cardsum)*100

# of those that have any workplace regions, how many only have one
len(cardsum[cardsum['Workplaces']==1])/len(cardsum[cardsum['Workplaces']>0])*100

# how many don't have an education region
len(cardsum[cardsum['Education']==0])/len(cardsum)*100

# of those that do, how many have only one
len(cardsum[cardsum['Education']==1])/len(cardsum[cardsum['Education']>0])*100


#%%
fig = px.histogram(
    cardsum["Workplaces"], # change here for other types
    color_discrete_sequence=[mycolors_discrete[7]],
    barmode="group",
    labels={
        "value": "Number of 'Workplace' regions per card",
        "count ": "Number of cards",
        colname: "Region cluster",
    },
)
fig.update_layout(default_layout)
pyo.plot(fig, config=config)


#%% PLOT REGION CENTROIDS BY CLUSTER TYPE

# This is to intuitively sense check the region types against a map with known
# features. (e.g. Is the CBD coming up as workplaces? Are the known shopping centres
# coming up as commercial? Needs to be plotted as continuous and not discrete

# Add point geometries
regionpivotpoint = pd.merge(
    regionpivot_unique,
    allregions,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)
regionpivotpoint = gpd.GeoDataFrame(regionpivotpoint, geometry="geometry")

# Add polygon geometries
regionpivotpoly = pd.merge(
    regionpivot,
    landusepolys,
    left_on=["Cardid", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)


regionpivotpoint = gpd.GeoDataFrame(regionpivotpoint, geometry="geometry")
regionpivotpoint.to_crs(epsg=4326, inplace=True)

regionpivotpoint["cluster_name"] = regionpivotpoint[colname_unique].map(clustermap_unique)

# suggest dropping residential to make it easier to see the others
plotregion = regionpivotpoint[regionpivotpoint["cluster_name"] != "Residences"]

plotregion.rename(columns={'cluster_name':'Cluster'},inplace=True)

fig = px.scatter_map(
    plotregion,
    lat=plotregion["geometry"].y,
    lon=plotregion["geometry"].x,
    color=plotregion["Cluster"],  
    color_discrete_sequence=mycolors_discrete,
    hover_name="regionid",
    hover_data=["E", "L", "M", "S", "W"],
    zoom=12,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=10))
pyo.plot(fig, config=config)

#%%
filename = "regions_manuallandusemap"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% plots for a selected card

#testcard = 2113413
testcard = 12897166

# map coloured by region cluster
regionplot = regionpivotpoly[regionpivotpoly["Cardid"] == testcard]
regionplot[colname_unique] = regionplot[colname_unique].map(clustermap_unique)

regionplot = gpd.GeoDataFrame(regionplot, geometry="geometry")
regionplot.to_crs(epsg=4326, inplace=True)

fig1 = px.choropleth_map(
    regionplot,
    geojson=regionplot.geometry,
    locations=regionplot.index,
    color_discrete_sequence=mycolors_discrete,
    center={"lat": -32, "lon": 116},
    zoom=10,
    map_style="light",
    opacity=0.5,
    hover_name="regionid",
    hover_data=["E", "L", "M", "S", "W"],
    color=regionplot[colname_unique],
)
fig1.update_geos(fitbounds="locations", visible=False)

pyo.plot(fig1, config=config)
#%%
filename = "regions_regionsonecardmap"
fig1.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig1.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig1.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%%

# bar chart of activities by region for selected card
# note that this isn't region type - the number that is the "region_cluster"
# is just the region ID for this card

test = pd.melt(
    regionplot, id_vars=["region_cluster"], value_vars=["E", "L", "M", "S", "W"]
)

d = {"E": "Education", "L": "Long", "S": "Short", 
     "M": "Medium", "W": "Work"}

test["variable"] = test["variable"].map(d)

test.rename(columns={'variable':'Activity', 'value':'Count'}, inplace=True)

testsum = pd.pivot_table(
    test, index="region_cluster", columns="Activity", aggfunc="sum"
)  

test["region_cluster"] = test['region_cluster'].astype(int)
# 6 and 10 for AusDM conference
test = test[test['region_cluster']==10]

fig = px.bar(
    test,
    x=test["region_cluster"].astype(str),
    y="Count",
    color="Activity",
    barmode="group",
    color_discrete_sequence=mycolors_discrete,
 )
# fig.update_layout(xaxis_title="Region ID")

# fig.update_xaxes(
#     showgrid=True,
#     tickson="boundaries",
#   )

fig.update_xaxes(visible=False)
fig.update_yaxes(dtick=5)
fig.update_layout(default_layout)
pyo.plot(fig, config=config)
#%%
filename = "regions_regionsonecardbar"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% REGIONS WITH LAND USE FOR SELECTED CARD

# TODO might want to add hover text about region types for each of these
# as a standalone there's no way of knowing which cluster each
# one belongs to

# Buffer the stops by the STOPBUFFER (circles around the stops)
landusepolys_reduced = allhist[allhist['Card']==testcard].to_crs(epsg=32749)
landusepolys_reduced["geometry"] = landusepolys_reduced["geometry"].buffer(STOPBUFFER)

allregionpolys_reduced = allregionpolys[allregionpolys['Card']==testcard]

# Concatenate the regional polygons and the buffered stops
landusepolys_reduced = pd.concat([landusepolys_reduced, allregionpolys_reduced], ignore_index=True, sort=False)

# Dissolve by card and region_cluster
landusepolys_reduced = landusepolys_reduced.dissolve(by=["Card", "region_cluster"])
landusepolys_reduced.reset_index(inplace=True)

landusepolys_reduced["area"] = landusepolys_reduced.area
landusepolys_reduced.sort_values(by=["area"], axis=0, ascending=False, inplace=True)

# Get extent of the polygons and reduce the land use dataset to those bounds
landusepolys_reduced = landusepolys_reduced.to_crs(epsg=4326)
xmin, ymin, xmax, ymax = landusepolys_reduced.total_bounds
mbdissolve_small = mbdissolve.cx[xmin:xmax, ymin:ymax]

# Make sure in same CRS
mbdissolve_small = mbdissolve_small.to_crs(epsg=32749)
landusepolys_reduced = landusepolys_reduced.to_crs(epsg=32749)

# Overlay land use - this is the slow part
ABSoverlay = mbdissolve_small.overlay(landusepolys_reduced, how="intersection")
ABSoverlay["area"] = ABSoverlay["geometry"].area

# map with regions broken out by land use for selected card

ABSoverlay.to_crs(epsg=4326, inplace=True)
ABSoverlay.reset_index(inplace=True)

ABSoverlay.rename(columns={'MB_CAT16':'Land use'}, inplace=True)

fig3 = px.choropleth_map(
    ABSoverlay,
    geojson=ABSoverlay.geometry,
    locations=ABSoverlay.index,
    center={"lat": -32, "lon": 116},
    zoom=10,
    map_style="light",
    opacity=0.5,
    hover_name="region_cluster",
    color="Land use",
    color_discrete_sequence=mycolors_discrete,
)
fig3.update_layout(default_layout)

fig3.update_geos(fitbounds="locations", visible=False)
#pyo.plot(fig3, config=config)

filename = "regions_regionsonecardmap_landuse"
fig3.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig3.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig3.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% NEW AUGUST 9: SENSITIVITY ON STOP CLUSTER EPS

count_list = []
lt50 = []
lt100 = []
lt150 = []
lt200 = []
gt200 = []
eps_list = []

import shapely

for eps in np.asarray((30, 60, 90, 120, 150)):
    eps_list.append(eps)

    clusters, geo = create_stop_cluster(geo, "spatial_cluster", eps, minpts)
    
    geo.to_crs(epsg=32749, inplace=True)
    
    cluster_coords = []
    for i in np.arange(0, len(geo['spatial_cluster'].value_counts())):
        stops = geo[geo['spatial_cluster']==i]
        transformed = np.array(stops['geometry']).reshape(-1,1)
        dist = pdist(transformed, lambda u, v: shapely.distance(u,v))
        if len(stops)==1:
            dist_max = 0
        else:
            dist_max = dist.max()

        row_data = {'spatial_cluster':i, 'num_stops':len(stops), 'dist':dist_max}
        cluster_coords.append(row_data)
        
    cluster_coords = pd.DataFrame(cluster_coords)
    
    cluster_coords.to_pickle(dirname+"20250903-stopclustercoords"+str(eps)+".pkl")

    count_list.append(len(cluster_coords))
    lt50.append(len(cluster_coords[cluster_coords['dist']<50]))
    lt100.append(len(cluster_coords[(cluster_coords['dist']>=50)&(cluster_coords['dist']<100)]))
    lt150.append(len(cluster_coords[(cluster_coords['dist']>=100)&(cluster_coords['dist']<150)]))
    lt200.append(len(cluster_coords[(cluster_coords['dist']>=150)&(cluster_coords['dist']<200)]))
    gt200.append(len(cluster_coords[(cluster_coords['dist']>=200)]))

    # save plot to demonstrate the point
    geo.to_crs(epsg=4326, inplace=True)
    geopolys = geo.dissolve("spatial_cluster").convex_hull

    geopolys = gpd.GeoDataFrame(geopolys)
    geopolys.reset_index(inplace=True)
    geopolys.rename(columns={0: "geometry"}, inplace=True)
    geopolys = geopolys.set_geometry("geometry")
    geopolys = geopolys.set_crs("epsg:4326", allow_override=True)

    geopolys = geopolys.to_crs(epsg=32749)
    geopolys["geometry"] = geopolys["geometry"].buffer(MAPBUFFER)

    geo.to_crs(epsg=4326, inplace=True)
    geopolys.to_crs(epsg=4326, inplace=True)

    fig = px.choropleth_map(
        geopolys,
        geojson=geopolys.geometry,
        locations=geopolys.index,
        color="spatial_cluster",
        center={"lat": -31.95, "lon": 115.85},
        color_continuous_scale=mycolors_discrete,
        opacity=0.5,
        zoom=13,
    )
    fig.update_layout(map_style="light", coloraxis_showscale=False)

    fig.add_trace(
        go.Scattermap(
            lat=geo["geometry"].y,
            lon=geo["geometry"].x,
            mode="markers",
            marker=go.scattermap.Marker(
                color=geo["spatial_cluster"],
                colorscale=mycolors_discrete,
            ),
        )
    )

    # filename = "sensitivity_stops_eps_"+str(eps)
    # fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


data = {'eps': eps_list, 'cluster_count': count_list, '<50m': lt50, '>=50 <100m':lt100, 
        '>=100 <150m':lt150, '>=150 <200m':lt200, '>=200m':gt200} 

eps_stats = pd.DataFrame(data)

#%%

fig = go.Figure()

for eps in np.asarray((30, 60, 90, 120, 150)):
    colour = np.where(np.asarray((30, 60, 90, 120, 150)) == eps)[0][0]
    cluster_coords = pd.read_pickle(dirname+"20250903-stopclustercoords"+str(eps)+".pkl")
    cluster_coords.rename(columns={'dist':str(eps)+'m'}, inplace=True)
    fig.add_traces(px.ecdf(
        cluster_coords[str(eps)+'m'],
        color_discrete_sequence=[mycolors_discrete[colour]],
    ).data)
    
fig.update_layout(
    default_layout_ncaa,
    yaxis_title_text = 'Fraction of stop clusters',
    xaxis_title_text = 'Span of stop clusters (m)',
    xaxis_range=[0,300],
    yaxis_range=[0,1],
)

#pyo.plot(fig, config=config)

filename = "regions_stopcluster_cdf"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=450)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%

eps_stats['<50m'] = eps_stats['<50m'].div(eps_stats['cluster_count'])*100
eps_stats['>=50 <100m'] = eps_stats['>=50 <100m'].div(eps_stats['cluster_count'])*100
eps_stats['>=100 <150m'] = eps_stats['>=100 <150m'].div(eps_stats['cluster_count'])*100
eps_stats['>=150 <200m'] = eps_stats['>=150 <200m'].div(eps_stats['cluster_count'])*100
eps_stats['>=200m'] = eps_stats['>=200m'].div(eps_stats['cluster_count'])*100

#%% NEW AUGUST 9: SENSITIVITY ON NUMBER OF ACTIVITIES THAT SETS ANCHORING DEFINITION

activities = pd.read_pickle(dirname + "20250111-activities.pkl")

# parameter we're shifting is NUM_REGION_VISITS

NUM_REGION_VISITS = 5

# these get modified later to just have the anchoring regions so bring back in full file
allhist = pd.read_pickle(dirname + "20250125-fullallhist.pkl") 
allregions = pd.read_pickle(dirname + "20250125-fullallregions.pkl") 
allregionpolys = pd.read_pickle(dirname + "20250125-fullallregionpolys.pkl") 


allregions["region_type"] = np.where(
    allregions["num_visits"] >= NUM_REGION_VISITS, "Anchor", "Visited"
)

allhist = pd.merge(
    allhist,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)
allregionpolys = pd.merge(
    allregionpolys,
    allregions[["Card", "region_cluster", "region_type"]],
    left_on=["Card", "region_cluster"],
    right_on=["Card", "region_cluster"],
    how="left",
)


regionsummary = pd.pivot_table(
    allregions,
    values=["fraction_time", "fraction_visits"],
    index=["Card"],
    columns="region_type",
    fill_value=0,
    aggfunc=[np.sum, "count"],
)
regionsummary.reset_index(inplace=True)

print(len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0]), "cards have no anchor region")
print("equivalent to", len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0])/len(regionsummary)*100,"percent")
print("so",len(regionsummary)-len(regionsummary[regionsummary['count']['fraction_time']['Anchor']==0]),"cards have anchor regions")

timesummary = regionsummary.loc[:, ("sum", "fraction_time", "Anchor")]
d = {"fraction_time": timesummary}
timesummary = pd.DataFrame(data=d)

visitsummary = regionsummary["sum", "fraction_visits", "Anchor"]
d = {"fraction_visits": visitsummary}
visitsummary = pd.DataFrame(data=d)

print(
    len(timesummary[timesummary["fraction_time"] > 0.75]) / 
    len(timesummary[timesummary["fraction_time"] > 0])*100, 
    "cards that have anchoring regions spend more than 75% of their time in those regions"
)  

print(
    len(visitsummary[visitsummary["fraction_visits"] > 0.75]) / 
    len(visitsummary[visitsummary["fraction_visits"] > 0])*100,
    "cards that have anchoring regions complete more than 75% of their activities in those regions"
)  

allhist = allhist[allhist["region_type"] == "Anchor"]
allregions = allregions[allregions["region_type"] == "Anchor"]
allregionpolys = allregionpolys[allregionpolys["region_type"] == "Anchor"]

activityregions = pd.merge(
    activities,
    allhist[["Card", "spatial_cluster", "region_cluster", "region_type"]],
    left_on=["Cardid", "spatial_cluster"],
    right_on=["Card", "spatial_cluster"],
    how="left",
)
activityregions = activityregions.dropna(
    axis="rows"
)  # because it hasn't been run for all cards, most will be NaN

regionpivot = pd.pivot_table(
    activityregions,
    index=["Cardid", "region_cluster", "region_type"],
    columns="activity_alpha",
    aggfunc="count",
    values="Card",
    fill_value=0,
)
regionpivot.reset_index(inplace=True)

regionpivot["Total"] = regionpivot.iloc[:, 3:9].sum(
    axis=1
)  # includes V = very long stays (>20h)

regionpivot["Total"] = regionpivot["Total"] - regionpivot["V"]  
# subtract off the V

# Drop regions that only have the very long stays - these will be where 
# the total now equals 0
regionpivot.drop(
    regionpivot[regionpivot["Total"] == 0].index, inplace=True
)


# This dataset includes all regions, so drop the non-anchoring ones
regionpivot.drop(
    regionpivot[regionpivot["region_type"] == "Visited"].index, inplace=True
)

regionpivot["E_frac"] = regionpivot["E"].div(regionpivot["Total"], axis=0)
regionpivot["L_frac"] = regionpivot["L"].div(regionpivot["Total"], axis=0)
regionpivot["M_frac"] = regionpivot["M"].div(regionpivot["Total"], axis=0)
regionpivot["S_frac"] = regionpivot["S"].div(regionpivot["Total"], axis=0)
regionpivot["W_frac"] = regionpivot["W"].div(regionpivot["Total"], axis=0)

regionpivot["regionid"] = (
    regionpivot["Cardid"].astype(str) + "-" 
    + regionpivot["region_cluster"].astype(str)
)
#%%
X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]].drop_duplicates()

init_state = 0

# To determine the appropriate amount of clusters
# Run GMM across 1 to 20 clusters and check AIC/BIC
n_components = np.arange(1, 20)
models = [GaussianMixture(n, random_state=init_state).fit(X) for n in n_components]

d = {"BIC": [m.bic(X) for m in models], "AIC": [m.aic(X) for m in models], "num_clusters": [n for n in n_components]}
df = pd.DataFrame(data=d)

# Figure 6 in readme

fig = px.line(
    df,
    x = "num_clusters",
    y = ['BIC','AIC'],
    color_discrete_sequence=mycolors_discrete,
    labels={"num_clusters": "Number of clusters", 
            "value": "", 
            "variable": "Criteria"},
)
fig.update_layout(default_layout)
pyo.plot(fig, config=config)

# TEST INSTABILITY

X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]]


k_max = 10

val10 = [calc_instability(n, 5, int(round(0.1*len(regionpivot),0))) for n in np.arange(2,k_max)]
val20 = [calc_instability(n, 5, int(round(0.2*len(regionpivot),0))) for n in np.arange(2,k_max)]
val30 = [calc_instability(n, 5, int(round(0.3*len(regionpivot),0))) for n in np.arange(2,k_max)]
val40 = [calc_instability(n, 5, int(round(0.4*len(regionpivot),0))) for n in np.arange(2,k_max)]
val50 = [calc_instability(n, 5, int(round(0.5*len(regionpivot),0))) for n in np.arange(2,k_max)]
val60 = [calc_instability(n, 5, int(round(0.6*len(regionpivot),0))) for n in np.arange(2,k_max)]

instability = {"num_clusters": [n for n in np.arange(2,k_max)], 
               "10%": [val10[i-2][0] for i in np.arange(2,k_max)],
               "20%": [val20[i-2][0] for i in np.arange(2,k_max)],
               "30%": [val30[i-2][0] for i in np.arange(2,k_max)],
               "40%": [val40[i-2][0] for i in np.arange(2,k_max)],
               "50%": [val50[i-2][0] for i in np.arange(2,k_max)],
               "60%": [val60[i-2][0] for i in np.arange(2,k_max)]}

instability = pd.DataFrame(data = instability)

instability_ind = {"num_clusters": [n for n in np.arange(2,k_max)], 
               "10%": [val10[i-2][1] for i in np.arange(2,k_max)],
               "20%": [val20[i-2][1] for i in np.arange(2,k_max)],
               "30%": [val30[i-2][1] for i in np.arange(2,k_max)],
               "40%": [val40[i-2][1] for i in np.arange(2,k_max)],
               "50%": [val50[i-2][1] for i in np.arange(2,k_max)],
               "60%": [val60[i-2][1] for i in np.arange(2,k_max)]}

instability_ind = pd.DataFrame(data = instability_ind)

fig = px.bar(
    instability,
    x="num_clusters", y=["10%","20%", "30%", "40%", "50%", "60%"],
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={'variable':'Sample size'}
)
fig.update_layout(default_layout_ncaa)
fig.update_layout(xaxis_title="Number of clusters",
                  yaxis_title="Instability",)
fig.update_layout(yaxis_range=[0,1])

for i in np.arange(instability['num_clusters'].min(),instability['num_clusters'].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

#%%

# From this plot, determine that the appropriate number of clusters
NUM_COMPONENTS = 5
X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]].drop_duplicates()


GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=init_state).fit(X)
colname_unique = "GMM_cluster_unique"

regionpivot[colname_unique] = GMM.predict(regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]])
means0 = GMM.means_

outregion_unique = pd.melt(
    regionpivot,
    id_vars=[colname_unique],
    value_vars=["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"],
)

outregionsum = pd.pivot_table(
    outregion_unique, index=colname_unique, columns="activity_alpha", aggfunc="sum"
)

outregion_unique.sort_values(by=[colname_unique, "activity_alpha"], axis=0, ascending=True, inplace=True)


fig = px.box(
    outregion_unique,
    x = colname_unique, 
    y="value",
    color="activity_alpha",
    color_discrete_sequence=mycolors_discrete,
    labels={colname_unique: "Region ID", "value": "Fraction", "variable": "Activity"},
)

fig.update_layout(default_layout_ncaa)
fig.update_layout(xaxis_title="Region activity cluster",
                  yaxis_title="Fraction of activities",)


fig.update_yaxes(
    range=(0, 1),
    constrain='domain'
)

# fig.update_traces(boxpoints=False) # sets whiskers to min/max
fig.update_traces(
    marker=dict(opacity=0)
)  # sets whiskers to usual st dev and not 'outliers'
for i in np.arange(0,len(outregion_unique[colname_unique].unique())):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
pyo.plot(fig, config=config)

#%% SENSITIVITY: SIZE OF ANCHORING REGIONS

# ANCHOR_REGION_SIZE is the parameter we're testing
# default is 800

clusters = pd.read_pickle(dirname + "20250111-clusters.pkl")
geo = pd.read_pickle(dirname + "20250111-geo.pkl")
activities = pd.read_pickle(dirname + "20250111-activities.pkl")

np.random.seed(42)

# allocate each card to a number between 1 and 5
cards = pd.read_pickle(dirname + "20250111-cards.pkl")
cards["split"] = np.random.randint(1, 6, cards.shape[0])
cardrun = cards[(cards['split']==4) | (cards['split']<=2)]
cardrun.reset_index(inplace=True)

#%% CALCULATE REGIONS AT DIFFERENT EPSILON
# this takes several hours to run across all the different settings

ANCHOR_REGION_SIZES = [400, 600, 1000, 1200]

for i in len(ANCHOR_REGION_SIZES):

    ANCHOR_REGION_SIZE = ANCHOR_REGION_SIZES[i]
    start = timeit.default_timer()
    
    allhist = pd.DataFrame()
    allregions = pd.DataFrame()
    allregionpolys = pd.DataFrame()
    
    for index, row in cardrun.iterrows():
        if np.remainder(index, 500) == 0:
            print(
                str(index)
                + " of "
                + str(len(cardrun))
                + " cards complete, "
                + str((timeit.default_timer() - start)/60)
                + " minutes elapsed, ("
                + str((timeit.default_timer() - start)/60/(index+1)*1000)
                + " minutes per 1000 cards)"
            )
        ret = generate_regions(
            row["Card"],
            clusters,
            activities,
            STAY_DROP_THRESHOLD,
            ANCHOR_REGION_SIZE,
            MIN_POINTS_REGION,
            MAPBUFFER,
        )
    
        if ret is None:
            continue
        hist, region_summary, regionpolys = ret
        hist["Card"] = row["Card"]
        region_summary["Card"] = row["Card"]
        regionpolys["Card"] = row["Card"]
    
        allhist = pd.concat([allhist, hist])
        allregions = pd.concat([allregions, region_summary])
        allregionpolys = pd.concat([allregionpolys, regionpolys])
    
    print("All cards complete, " + str((timeit.default_timer() - start)/60) + " minutes elapsed")
    
    allhist.to_pickle(dirname + "20250809-fullallhist"+str(ANCHOR_REGION_SIZE)+".pkl")
    allregions.to_pickle(dirname + "20250809-fullallregions"+str(ANCHOR_REGION_SIZE)+".pkl")
    allregionpolys.to_pickle(dirname + "20250809-fullallregionpolys"+str(ANCHOR_REGION_SIZE)+".pkl")

#%% FUNCTIONS FOR TESTING CLUSTERING


def calc_regionpivot(activities, allhist, allregions):
    allregions["region_type"] = np.where(
        allregions["num_visits"] >= NUM_REGION_VISITS, "Anchor", "Visited"
    )
    
    allhist = pd.merge(
        allhist,
        allregions[["Card", "region_cluster", "region_type"]],
        left_on=["Card", "region_cluster"],
        right_on=["Card", "region_cluster"],
        how="left",
    )

    activityregions = pd.merge(
        activities,
        allhist[["Card", "spatial_cluster", "region_cluster", "region_type"]],
        left_on=["Cardid", "spatial_cluster"],
        right_on=["Card", "spatial_cluster"],
        how="left",
    )
    activityregions = activityregions.dropna(
        axis="rows"
    )  # because it hasn't been run for all cards, most will be NaN

    regionpivot = pd.pivot_table(
        activityregions,
        index=["Cardid", "region_cluster", "region_type"],
        columns="activity_alpha",
        aggfunc="count",
        values="Card",
        fill_value=0,
    )
    regionpivot.reset_index(inplace=True)

    regionpivot["Total"] = regionpivot.iloc[:, 3:9].sum(
        axis=1
    )  # includes V = very long stays (>20h)
 
    regionpivot["Total"] = regionpivot["Total"] - regionpivot["V"]  
    # subtract off the V

    # Drop regions that only have the very long stays - these will be where 
    # the total now equals 0
    regionpivot.drop(
        regionpivot[regionpivot["Total"] == 0].index, inplace=True
    )

    regionpivot.drop(
        regionpivot[regionpivot["region_type"] == "Visited"].index, inplace=True
    )

    regionpivot["E_frac"] = regionpivot["E"].div(regionpivot["Total"], axis=0)
    regionpivot["L_frac"] = regionpivot["L"].div(regionpivot["Total"], axis=0)
    regionpivot["M_frac"] = regionpivot["M"].div(regionpivot["Total"], axis=0)
    regionpivot["S_frac"] = regionpivot["S"].div(regionpivot["Total"], axis=0)
    regionpivot["W_frac"] = regionpivot["W"].div(regionpivot["Total"], axis=0)

    regionpivot["regionid"] = (
        regionpivot["Cardid"].astype(str) + "-" 
        + regionpivot["region_cluster"].astype(str)
    )
    return regionpivot

def getAICBICinst(regionpivot):
    X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]].drop_duplicates()

    init_state = 0

    # To determine the appropriate amount of clusters
    # Run GMM across 1 to 20 clusters and check AIC/BIC
    n_components = np.arange(1, 20)
    models = [GaussianMixture(n, random_state=init_state).fit(X) for n in n_components]

    d = {"BIC": [m.bic(X) for m in models], "AIC": [m.aic(X) for m in models], "num_clusters": [n for n in n_components]}
    df = pd.DataFrame(data=d)

    # Figure 6 in readme

    fig = px.line(
        df,
        x = "num_clusters",
        y = ['BIC','AIC'],
        color_discrete_sequence=mycolors_discrete,
        labels={"num_clusters": "Number of clusters", 
                "value": "", 
                "variable": "Criteria"},
    )
    fig.update_layout(default_layout)
    pyo.plot(fig, config=config)
      
    k_max = 10
    
    val10 = [calc_instability(n, 5, int(round(0.1*len(regionpivot),0))) for n in np.arange(2,k_max)]
    val20 = [calc_instability(n, 5, int(round(0.2*len(regionpivot),0))) for n in np.arange(2,k_max)]
    val30 = [calc_instability(n, 5, int(round(0.3*len(regionpivot),0))) for n in np.arange(2,k_max)]
    val40 = [calc_instability(n, 5, int(round(0.4*len(regionpivot),0))) for n in np.arange(2,k_max)]
    val50 = [calc_instability(n, 5, int(round(0.5*len(regionpivot),0))) for n in np.arange(2,k_max)]
    val60 = [calc_instability(n, 5, int(round(0.6*len(regionpivot),0))) for n in np.arange(2,k_max)]
    
    instability = {"num_clusters": [n for n in np.arange(2,k_max)], 
                   "10%": [val10[i-2][0] for i in np.arange(2,k_max)],
                   "20%": [val20[i-2][0] for i in np.arange(2,k_max)],
                   "30%": [val30[i-2][0] for i in np.arange(2,k_max)],
                   "40%": [val40[i-2][0] for i in np.arange(2,k_max)],
                   "50%": [val50[i-2][0] for i in np.arange(2,k_max)],
                   "60%": [val60[i-2][0] for i in np.arange(2,k_max)]}
    
    instability = pd.DataFrame(data = instability)
    
    fig = px.bar(
        instability,
        x="num_clusters", y=["10%","20%", "30%", "40%", "50%", "60%"],
        color_discrete_sequence=mycolors_discrete,
        barmode="group",
        labels={'variable':'Sample size'}
    )
    fig.update_layout(default_layout_ncaa)
    fig.update_layout(xaxis_title="Number of clusters",
                      yaxis_title="Instability",)
    fig.update_layout(yaxis_range=[0,1])
    
    for i in np.arange(instability['num_clusters'].min(),instability['num_clusters'].max()):
            fig.add_shape(
                type="line", xref='x', yref='y2 domain',
                                x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
    
    pyo.plot(fig, config=config)
    
def run_final_GMM(NUM_COMPONENTS,regionpivot,colname):
    X = regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]].drop_duplicates()

    GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=init_state).fit(X)
    colname = "GMM_cluster_unique"

    regionpivot[colname] = GMM.predict(regionpivot[["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"]])

    outregion_unique = pd.melt(
        regionpivot,
        id_vars=[colname],
        value_vars=["E_frac", "L_frac", "M_frac", "S_frac", "W_frac"],
    )
    outregion_unique.sort_values(by=[colname, "activity_alpha"], axis=0, ascending=True, inplace=True)


    fig = px.box(
        outregion_unique,
        x = colname, 
        y="value",
        color="activity_alpha",
        color_discrete_sequence=mycolors_discrete,
        labels={colname_unique: "Region ID", "value": "Fraction", "variable": "Activity"},
    )

    fig.update_layout(default_layout_ncaa)
    fig.update_layout(xaxis_title="Region activity cluster",
                      yaxis_title="Fraction of activities",)


    fig.update_yaxes(
        range=(0, 1),
        constrain='domain'
    )

    # fig.update_traces(boxpoints=False) # sets whiskers to min/max
    fig.update_traces(
        marker=dict(opacity=0)
    )  # sets whiskers to usual st dev and not 'outliers'
    for i in np.arange(0,len(outregion_unique[colname].unique())):
            fig.add_shape(
                type="line", xref='x', yref='y2 domain',
                                x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
    pyo.plot(fig, config=config)

#%%

# filenames are:
  # 800 is from the main run "20250125-fullallhist.pkl"
  # allregions = pd.read_pickle(dirname + "20250125-fullallregions.pkl") 


NUM_REGION_VISITS = 5
init_state = 0

allhist_sens = pd.read_pickle(dirname+"20250809-fullallhist400.pkl")
allregions_sens = pd.read_pickle(dirname+"20250809-fullallregions400.pkl")

regionpivot_400 = calc_regionpivot(activities, allhist_sens, allregions_sens)

getAICBICinst(regionpivot_400)

filename = "sensitivity_regions_eps_inst"+str(400)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


#%%

run_final_GMM(5, regionpivot_400, 'GMM_cluster')
# less good separation on the work ones
# all L, almost entirely E, mostly M, mostly W but also some S and M,
# mix of all except E

filename = "sensitivity_regions_eps_profiles"+str(400)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


#%%

allhist_sens = pd.read_pickle(dirname+"20250809-fullallhist600.pkl")
allregions_sens = pd.read_pickle(dirname+"20250809-fullallregions600.pkl")

regionpivot_600 = calc_regionpivot(activities, allhist_sens, allregions_sens)

getAICBICinst(regionpivot_600)

filename = "sensitivity_regions_eps_inst"+str(600)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)

#%%

# 4 gives better instability, but 5 isn't too bad and gives better AIC/BIC
# go with 5 as it's what we used in the other runs
run_final_GMM(5, regionpivot_600, 'GMM_cluster')
# all L, basically all E, mix of E and W, W with a bit of S/M
# mix of all except E

filename = "sensitivity_regions_eps_"+str(600)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


#%%

allhist_sens = pd.read_pickle(dirname+"20250809-fullallhist1000.pkl")
allregions_sens = pd.read_pickle(dirname+"20250809-fullallregions1000.pkl")

regionpivot_1000 = calc_regionpivot(activities, allhist_sens, allregions_sens)

getAICBICinst(regionpivot_1000)
filename = "sensitivity_regions_eps_inst"+str(1000)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)

#%%
# this looks really similar to the 800 one for instability

run_final_GMM(5, regionpivot_1000, 'GMM_cluster')
# all L, basically all E, mostly W with a bit of S/M, mix of
# E, L, and a tiny bit of S/M, mix of everything but E

filename = "sensitivity_regions_eps_"+str(1000)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)


#%%

allhist_sens = pd.read_pickle(dirname+"20250809-fullallhist1200.pkl")
allregions_sens = pd.read_pickle(dirname+"20250809-fullallregions1200.pkl")

regionpivot_1200 = calc_regionpivot(activities, allhist_sens, allregions_sens)

getAICBICinst(regionpivot_1200)
filename = "sensitivity_regions_eps_inst"+str(1200)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)

#%%
run_final_GMM(5, regionpivot_1200, 'GMM_cluster')
# basically all L, basically all E, mostly W with a bit of S/M, then one that 
# are a mix of everything but E and another that is a mix of everything but W

filename = "sensitivity_regions_eps_"+str(1000)
fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)

#%% SENSE CHECKING AGAINST DISTANCE BETWEEN STOPS
# using the timetable data

stop_times = pd.read_csv(dirname + 'gtfs\\stop_times.txt')
 
arrival_times = stop_times['arrival_time'].str.split(pat=":", expand=True)
stop_times['arrival_sec'] = arrival_times[0].astype(int)*60*60 + arrival_times[1].astype(int)*60 + arrival_times[2].astype(int)

stop_times.rename(columns={'stop_id':'start_stop', 'arrival_sec':'start_time'}, inplace=True)

stop_times['end_stop'] = stop_times['start_stop'].shift(-1)
stop_times['end_time'] = stop_times['start_time'].shift(-1)

stop_times['duration'] = (stop_times['end_time'] - 
                          stop_times['start_time']).div(60).where(stop_times["trip_id"] == 
                                                                  stop_times["trip_id"].shift(-1))

stop_times = stop_times.dropna(axis="rows", subset=['duration', 'end_stop'])    
stop_times.reset_index(inplace=True)                                                             
               
stop_times['end_stop'] = stop_times['end_stop'].astype(int)

stops = pd.read_csv(dirname+'gtfs\\stops.txt')
routes = pd.read_csv(dirname+'gtfs\\routes.txt')
trips = pd.read_csv(dirname+'gtfs\\trips.txt')
# where agency_id = 'TRA-TRA' is the regional stuff
trips = pd.merge(trips, routes, left_on=['route_id'], right_on=['route_id'], how='left', suffixes=(None, "_y"))
#trips = trips[trips['agency_id']!='TRA-TRA']

stop_times_all = pd.merge(stop_times, trips, 
                      left_on=['trip_id'], right_on=['trip_id'], how='left', suffixes=(None, "_y"))

# from the agency file we can drop
regional_services = ['TRA-TRA', 'COL-COL', 'CRS-CAR', 'ESP-ESP', 
                     'GER-GER', 'GWB-KAR', 'NIC-NAR', 'PHN-PTH',
                     'PTB-PTB', 'PTR-KGL', 'STR-ALB', 'STR-BBY',
                     'STR-BSN', 'TRI-KUN', 'WSB-MAN']

stop_times = stop_times_all[~stop_times_all['agency_id'].isin(regional_services)]
# to remove the regional stops - only keep rows in the stops file where they are in stop_times

stop_times['start_stop'] = stop_times['start_stop'].astype(str)
stop_times['end_stop'] = stop_times['end_stop'].astype(str)

stops[' stop_id'] = stops[' stop_id'].astype(str)

stops = stops[stops[' stop_id'].isin(stop_times['start_stop']) | stops[' stop_id'].isin(stop_times['end_stop'])]

stops['geometry'] = stops.apply(create_point, axis=1)
stops = gpd.GeoDataFrame(stops, geometry="geometry")
stops = stops.set_crs("epsg:4326", allow_override=True)

stop_times_loc = pd.merge(stop_times, stops[[' stop_id', 'geometry']], left_on='start_stop', right_on=' stop_id')
stop_times_loc.drop(columns=' stop_id', inplace=True)
stop_times_loc.rename(columns={'geometry':'end_geometry'}, inplace=True)

stop_times_loc = pd.merge(stop_times_loc, stops[[' stop_id', 'geometry']], left_on='end_stop', right_on=' stop_id')
stop_times_loc.drop(columns=' stop_id', inplace=True)
stop_times_loc.rename(columns={'geometry':'start_geometry'}, inplace=True)

# keep only unique combinations of start_stop and end_stop
stop_times_loc = stop_times_loc.drop_duplicates(subset=['start_stop','end_stop'])

geopolys = gpd.GeoDataFrame(stop_times_loc)
geopolys.reset_index(inplace=True)

geopolys = geopolys.set_geometry("start_geometry")
geopolys = geopolys.set_crs("epsg:4326", allow_override=True)
geopolys = geopolys.to_crs(epsg=32749)

# then set to the other geometry column and change crs
geopolys = geopolys.set_geometry("end_geometry")
geopolys = geopolys.set_crs("epsg:4326", allow_override=True)
geopolys = geopolys.to_crs(epsg=32749)

# calculate distance between start_geometry and end_geometry
geopolys['distance'] = geopolys['start_geometry'].distance(geopolys['end_geometry'])

#%% HISTOGRAM - STOP CLUSTER EPS DISTANCE

eps_threshold = 90
geopolys['color1'] = np.where(geopolys['distance']<=eps_threshold,'col1','col2')

fig = px.histogram(
    geopolys['distance'],
    nbins=100,
    color=geopolys['color1'],
    color_discrete_sequence=[mycolors_discrete[7], mycolors_discrete[1]],
)
fig.update_traces(xbins=dict( 
        start=0.0,
        end=1000.0,
        size=10
    )) 
fig.update_yaxes(
    range=(0, 500),
    constrain='domain'
) 
fig.update_layout(
    default_layout_ncaa,
    showlegend=False,
    yaxis_title_text = 'Number of distances',
    xaxis_title_text = 'Distance between stops (m)',
)
fig.update_layout(barmode='stack')
pyo.plot(fig, config=config)

filename = "fig_stopclustereps"
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)


#%% SENSITIVITY ON REGION EPS

# https://shapely.readthedocs.io/en/stable/reference/shapely.minimum_rotated_rectangle.html
# https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely

# this takes about half an hour per value of epsilon

count_list = []
lt50 = []
lt100 = []
lt150 = []
lt200 = []
gt200 = []
eps_list = []


for eps in np.asarray((400, 600, 800, 1000, 1200)):
    start = timeit.default_timer()
    
    eps_list.append(eps)

    if eps==800:
        allregionpolys_sens = pd.read_pickle(dirname + "20250125-fullallregionpolys.pkl")
        allhist_sens = pd.read_pickle(dirname + "20250125-fullallhist.pkl")
    else:
        allregionpolys_sens = pd.read_pickle(dirname + "20250809-fullallregionpolys"+str(eps)+".pkl")    
        allhist_sens = pd.read_pickle(dirname + "20250809-fullallhist"+str(eps)+".pkl")    
      
    allregionpolys_sens['card-region'] = allregionpolys_sens['Card'].astype(str)+"-"+allregionpolys_sens['region_cluster'].astype(str)
    allhist_sens['card-region'] = allhist_sens['Card'].astype(str)+"-"+allhist_sens['region_cluster'].astype(str)
    
    stops_per_region = pd.DataFrame(allhist_sens['card-region'].value_counts())
    stops_per_region.reset_index(inplace=True)
    
    num_agg = len(stops_per_region[stops_per_region['count']>1])/len(stops_per_region)*100
    
    print("eps:", eps, num_agg, " percent of regions have more than one stop in them")
    allregionpolys_sens.to_crs(epsg=32749, inplace=True)
    
    cluster_coords = []
    for i in np.arange(0, len(allregionpolys_sens)):
        
        if np.remainder(i, 10000) == 0:
            print(
                "eps:",
                eps,
                str(i)
                + " of "
                + str(len(allregionpolys_sens))
                + " regions complete, "
                + str((timeit.default_timer() - start)/60)
                + " minutes elapsed")
                
        poly = allregionpolys_sens['geometry'].iloc[i]
        box = poly.minimum_rotated_rectangle
        x, y = box.exterior.coords.xy
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        dist_max = max(edge_length)
        row_data = {'regionpoly':i, 'dist':dist_max}
        cluster_coords.append(row_data)
        
    cluster_coords = pd.DataFrame(cluster_coords)
    cluster_coords['dist'] = cluster_coords['dist']-2*MAPBUFFER
    # these polygons already have the buffer applied so have to subtract this from both sides from every distance

    cluster_coords['dist'] = np.maximum(cluster_coords['dist'], 0)
    # to account for some small rounding issues - a few regions have a distance of 39.9
        
    cluster_coords.to_pickle(dirname + "20250824-regionclustercoords"+str(eps)+".pkl")
    

#%%

bins = np.arange(0,7000,10)

cluster_coords['bin'] = pd.cut(cluster_coords['dist'], bins=bins)

#%%
fig = px.histogram(
    cluster_coords['dist'],
 #   nbins=100,
    cumulative=True, 
    histnorm='percent',
    color_discrete_sequence=[mycolors_discrete[7]],
)
# fig.update_traces(xbins=dict( 
#         start=0.0,
#         end=1000.0,
#         size=10
#     )) 
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Number of region clusters',
    xaxis_title_text = 'Breadth of region clusters (m)',
)
fig.update_layout(barmode='stack')
pyo.plot(fig, config=config)

#%%

fig = go.Figure()

for eps in np.asarray((400, 600, 800, 1000, 1200)):
    colour = np.where(np.asarray((400, 600, 800, 1000, 1200)) == eps)[0][0]
    cluster_coords = pd.read_pickle(dirname+"20250824-regionclustercoords"+str(eps)+".pkl")
    cluster_coords.rename(columns={'dist':str(eps)+'m'}, inplace=True)
    fig.add_traces(px.ecdf(
        cluster_coords[str(eps)+'m'],
        color_discrete_sequence=[mycolors_discrete[colour]],
    ).data)
    
fig.update_layout(
    default_layout_ncaa,
    yaxis_title_text = 'Fraction of region clusters',
    xaxis_title_text = 'Span of region clusters (m)',
    xaxis_range=[0,1600],
    yaxis_range=[0,1],
)

pyo.plot(fig, config=config)
#%%
filename = "regions_regioncluster_cdf"
#fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=450)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)
