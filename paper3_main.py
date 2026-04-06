# -*- coding: utf-8 -*-

# IMPORTS

# for analysis
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np

# for plotly plotting
import plotly.io as pio
import plotly.offline as pyo
import plotly.express as px
import datetime as dt

from shapely.geometry import box, Point, MultiPoint
from sklearn.cluster import DBSCAN
import geopandas as gpd
import networkx as nx
import plotly.graph_objects as go

import timeit
import pulp

#%% CONSTANTS THAT MUST BE SET BY USER

# folder where all data is saved
dirname = "C:\\Users\\megan\\Desktop\\PhD data\\hubs2017\\csv\\"

# OTHER CONSTANTS

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
    "#d73027",
    "#fc8d59",
    "#fee090",
    "#e0f3f8",
    "#91bfdb",
    "#4575b4"]

# intensities of green
mycolors_continuous_map = [
    "#A1D7C7",
    "#1B9E77"]

mycolors_continuous_r = mycolors_continuous.copy()
mycolors_continuous_r.reverse() # for reversed continuous gradient

mycolors_continuous_map_r = mycolors_continuous_map.copy()
mycolors_continuous_map_r.reverse() # for reversed continuous gradient

config = {'scrollZoom': True}

line_colour = "rgb(217,217,217)"
bg_colour = "white"
line_width = 0.5
map_center = {"lat": -31.952195, "lon": 115.864055}

default_layout = dict(yaxis=dict(
#        autorange=True,
        showgrid=True,
        zeroline=True,
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
            zeroline=True,
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

    trainStops['type'] = "Train"
    busStops['type'] = "Bus"

    stops = pd.concat([busStops, trainStops])
    stopTable = stops[["StopID", "X", "Y", "StopName","line","type"]]
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

#%% LOOKING AT STAYS IN THE VISITED REGIONS

activities = pd.read_pickle(dirname+"20250606-processed_activities.pkl")

visited_activities = pd.pivot_table(activities[activities['region_type']=='Visited'], index=['Cardid', 'final_regionID'], 
                             columns='activity_alpha', aggfunc='count', values='Card')
visited_activities = visited_activities.fillna(0)

len(visited_activities.value_counts())
# there are 212 different combinations of stay counts by activity

visited_activities['Total'] = visited_activities.sum(axis=1)

visited_activities['E_perc'] = visited_activities['E'].div(visited_activities['Total'])
visited_activities['L_perc'] = visited_activities['L'].div(visited_activities['Total'])
visited_activities['M_perc'] = visited_activities['M'].div(visited_activities['Total'])
visited_activities['S_perc'] = visited_activities['S'].div(visited_activities['Total'])
visited_activities['V_perc'] = visited_activities['V'].div(visited_activities['Total'])
visited_activities['W_perc'] = visited_activities['W'].div(visited_activities['Total'])

len(visited_activities[['E_perc','L_perc','M_perc','S_perc','V_perc','W_perc']].value_counts())
# there are 156 different breakdowns of activity profiles

visited_profiles = pd.DataFrame(visited_activities[['E_perc','L_perc','M_perc','S_perc','V_perc','W_perc']].value_counts())
visited_profiles.reset_index(inplace=True)

visited_profiles['cumsum'] = visited_profiles['count'].cumsum()
visited_profiles['perc'] = visited_profiles['count']/visited_profiles['count'].sum()*100
visited_profiles['cum_percent'] = visited_profiles['perc'].cumsum()

visited_profiles[['E_perc','L_perc','M_perc','S_perc','V_perc','W_perc','cum_percent']].head(15)

#%% READ JOURNEYS DATA

journeys = pd.read_pickle(dirname + "20250111-processed_journeys.pkl")
# this has already tidied up the cards infrequently used and the synthetic activities
# BUT still includes transfers. So journeys might have A-B-C but activities
# might only have A-C, so might not technically find a match


#%% CALCULATE STOP DATA

busStops = pd.read_csv(dirname + "busStops.csv")
trainStops = pd.read_csv(dirname + "trainStops_modified.csv")

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

onstops['StopID'] = onstops['StopID'].astype(str)
offstops['StopID'] = offstops['StopID'].astype(str)
geo['StopID'] = geo['StopID'].astype(str)


geo = pd.merge(geo, onstops, left_on="StopID", right_on="StopID", how="left")

geo = pd.merge(geo, offstops, left_on="StopID", right_on="StopID", how="left")

geo["Count On"] = geo["Count On"].fillna(0)
geo["Count Off"] = geo["Count Off"].fillna(0)

geo["Count Total"] = geo["Count On"] + geo["Count Off"]

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

clusters_old = pd.read_pickle(dirname + "20250111-clusters.pkl")
geo_old = pd.read_pickle(dirname + "20250111-geo.pkl")

geo_old['StopID'] = geo_old['StopID'].astype(str)
geo = pd.merge(geo, geo_old[['StopID','spatial_cluster']], on='StopID', how='left')

geo.rename(columns={'spatial_cluster_x':'spatial_cluster', 'spatial_cluster_y':'spatial_cluster_old'}, inplace=True)

geometry = [box(115.493774,-32.685620,116.180420,-31.447410)] # coast to Mt Helena, Manudurah to Yanchep
boxdf = gpd.GeoDataFrame(geometry=geometry)
boxdf = boxdf.set_crs("epsg:4326", allow_override=True)

# check it has been set up correctly
# fig = px.choropleth_map(boxdf, 
#                         geojson=boxdf.geometry, 
#                         locations=boxdf.index,     
#                         center=map_center,
#                         color_discrete_sequence=mycolors_discrete,
#                         opacity=0.6,
#                         zoom=10)
# pyo.plot(fig, config=config)

# add a column to that says whether or not each stop cluster is in the bbox
def in_gdf(row):
    return boxdf.contains(row['geometry']).any()

clusters['bbox check'] = clusters.apply(in_gdf, axis=1)

geo['newName'] = geo['StopName'].str.split(' Stand').str[0]
geo['newName'] = geo['newName'].str.split(' Bus Stn').str[0]
geo['newName'] = geo['newName'].str.split(' Stn').str[0]
geo['newName'] = geo['newName'].str.split(' Yellow Cat').str[0]
geo['newName'] = geo['newName'].str.split(' Red Cat').str[0]
geo['newName'] = geo['newName'].str.split(' Blue Cat').str[0]

geo['newName'] = geo['newName'].str.split(' After').str[0]
geo['newName'] = geo['newName'].str.split(' Before').str[0]
geo['newName'] = geo['newName'].str.split(' Opposite').str[0]

geo_agg = geo[['newName','spatial_cluster','StopName']].groupby(['newName','spatial_cluster']).count()
geo_agg.reset_index(inplace=True)
geo_agg.drop(columns=['StopName'], inplace=True)
geo_agg.drop_duplicates(inplace=True)
geo_agg = geo_agg.groupby(['spatial_cluster']).agg(list)
geo_agg['count_names'] = geo_agg['newName'].str.len()

clusters = pd.merge(clusters, geo_agg['newName'], left_on='clusterID', right_index=True, how='left')

clusters.to_pickle(dirname + "20260329-clusters.pkl")
geo.to_pickle(dirname + "20260329-geo.pkl")

#%% ALTERNATIVE: READ STOP CLUSTERS INSTEAD

geo = pd.read_pickle(dirname + "20260329-geo.pkl")
clusters = pd.read_pickle(dirname + "20260329-clusters.pkl")

#%% PROCESS JOURNEYS DATA PART 1

start_date = "2017-08-01 00:00:00"
start_date = pd.to_datetime(start_date, format="%Y-%m-%d %H:%M:%S")

# calculate total minutes since start of the month
journeys["elapsedMinutes_On"] = (journeys["OnTime"] - start_date) / dt.timedelta(minutes=1)

journeys["elapsedMinutes_Off"] = (journeys["OffTime"] - start_date) / dt.timedelta(
    minutes=1
)

journeys['Duration'] = journeys['elapsedMinutes_Off'] - journeys['elapsedMinutes_On']

journeys['Duration_round'] = journeys['Duration'].round(0)
journeys['OnLocation'] = journeys['OnLocation'].astype(str)
journeys['OffLocation'] = journeys['OffLocation'].astype(str)

allhist = pd.read_pickle(dirname + "20250125-fullallhist.pkl")

allhist = pd.merge(allhist, geo[['spatial_cluster','spatial_cluster_old']], left_on='spatial_cluster', right_on='spatial_cluster_old', how='left')

allhist.rename(columns={'spatial_cluster_y':'spatial_cluster'}, inplace=True)
allhist.drop(columns={'spatial_cluster_x'}, inplace=True)

allhist_merge = allhist[['Card','region_cluster','spatial_cluster']]
allhist_merge.drop_duplicates(inplace=True)

journeys = pd.merge(journeys, geo[['StopID','spatial_cluster']], left_on='OnLocation', right_on='StopID', how='left')

journeys.rename(columns={'spatial_cluster':'OnLocation_cluster'}, inplace=True)
journeys.drop(columns=['StopID'], inplace=True)

journeys = pd.merge(journeys, geo[['StopID','spatial_cluster']], left_on='OffLocation', right_on='StopID', how='left')
journeys.rename(columns={'spatial_cluster':'OffLocation_cluster'}, inplace=True)
journeys.drop(columns=['StopID'], inplace=True)

# not all cards are in allhist, so this will come up with a lot of NaNs
# this is ok for now - just something to remember when sense-checking -
# this gets chopped down to just the cards that are used later on
# but we want to use the full dataset for the underlying candidate graph
journeys = pd.merge(journeys,
                         allhist_merge[['Card', 'region_cluster', 'spatial_cluster']],
                         left_on=['Cardid','OnLocation_cluster'],
                         right_on=['Card','spatial_cluster'],
                         how='left')

journeys.drop(columns={'Card','spatial_cluster'},inplace=True)
journeys.rename(columns={'region_cluster':'OnLocation_regioncluster'}, inplace=True)

journeys = pd.merge(journeys,
                         allhist_merge[['Card', 'region_cluster', 'spatial_cluster']],
                         left_on=['Cardid','OffLocation_cluster'],
                         right_on=['Card','spatial_cluster'],
                         how='left')

journeys.drop(columns={'Card','spatial_cluster'},inplace=True)
journeys.rename(columns={'region_cluster':'OffLocation_regioncluster'}, inplace=True)

#%% PROCESS JOURNEYS DATA PART 2

# if the on transaction is a transfer, or the region cluster is unknown, that's a transfer
journeys['OnLocation_regioncluster'] = np.where((journeys['OnTran'].str.contains("Transfer"))|
                                                      (journeys['OnLocation_regioncluster'].isna()),
                                                      "Transfer", journeys['OnLocation_regioncluster'])

# if next tag on is a transfer, this tag off should be a transfer
journeys['OffTran'] = np.where(journeys['OnLocation_regioncluster'].shift(-1)=='Transfer','Transfer', journeys['OffTran'])


journeys['OffLocation_regioncluster'] = np.where((journeys['OffTran'].str.contains("Transfer"))|
                                                      (journeys['OffLocation_regioncluster'].isna()),
                                                      "Transfer", journeys['OffLocation_regioncluster'])

# if tag on is < 3 days from last tag off,
# AND the tag on is in a different region cluster to the last tag off
# AND the tag on is for the same card as the last tag off,
# set tag on location to be the same as the last tag off

journeys['OnLocation_newreg'] = np.where((journeys['timeSince']<dt.timedelta(days=3))
                                              &(journeys['OffLocation_regioncluster'].shift(1)
                                                !=journeys['OnLocation_regioncluster'])&
                                              (journeys['Cardid']==journeys['Cardid'].shift(1)),
                                              journeys['OffLocation_regioncluster'].shift(1),
                                              journeys['OnLocation_regioncluster'])


journeys['OnLocation_newspat'] = np.where((journeys['timeSince']<dt.timedelta(days=3))
                                              &(journeys['OffLocation_regioncluster'].shift(1)
                                                !=journeys['OnLocation_regioncluster'])&
                                              (journeys['Cardid']==journeys['Cardid'].shift(1)),
                                              journeys['OffLocation_cluster'].shift(1),
                                              journeys['OnLocation_cluster'])

journeys['OnLocation_new'] = np.where((journeys['timeSince']<dt.timedelta(days=3))
                                              &(journeys['OffLocation_regioncluster'].shift(1)
                                                !=journeys['OnLocation_regioncluster'])&
                                              (journeys['Cardid']==journeys['Cardid'].shift(1)),
                                              journeys['OffLocation'].shift(1),
                                              journeys['OnLocation'])

# fix up just the first row, as the shift function returns NaNs
journeys['OnLocation_newreg'].iloc[0] = journeys['OnLocation_regioncluster'].iloc[0]
journeys['OnLocation_newspat'].iloc[0] = journeys['OnLocation_cluster'].iloc[0]
journeys['OnLocation_new'].iloc[0] = journeys['OnLocation'].iloc[0]

#%% PROCESS JOURNEYS DATA PART 3

unique_stops = pd.concat([journeys['OnLocation_newspat'], journeys['OffLocation_cluster']]).drop_duplicates()
unique_stops = pd.DataFrame(unique_stops, columns=['spatial_cluster'])
unique_stops = unique_stops[unique_stops['spatial_cluster'].isin(geo['spatial_cluster'])]

journeys_short = journeys[journeys['OnLocation_newspat'].isin(unique_stops['spatial_cluster'])]
journeys_short = journeys_short[journeys_short['OffLocation_cluster'].isin(unique_stops['spatial_cluster'])]

# in this dataset we also can't really have a duration of zero
journeys_short = journeys_short[journeys_short['Duration_round']>0]
journeys_short = journeys_short[['OnLocation_newspat', 'OffLocation_cluster', 'Duration_round']]
journeys_short.rename(columns={'OnLocation_newspat':'OnLocation_cluster'}, inplace=True)

journeys_counts = pd.DataFrame(journeys_short.value_counts())
journeys_counts.reset_index(inplace=True)
journeys_counts.sort_values(by=['count', 'Duration_round'], inplace=True, ascending=[False,True])
# this sorting means we keep either the most frequent duration
# or if there is equally the most frequent duration, we keep the shortest one

journeys_final = journeys_counts.drop_duplicates(subset=['OnLocation_cluster', 'OffLocation_cluster'], keep='first')


#%% DROP THOSE OUTSIDE PERTH METRO

# only keep where both start and end locations are in
journeys_final = journeys_final[journeys_final['OnLocation_cluster'].isin(clusters['clusterID'][clusters['bbox check']==True])]
journeys_final = journeys_final[journeys_final['OffLocation_cluster'].isin(clusters['clusterID'][clusters['bbox check']==True])]

journeys_final.to_pickle(dirname+'20260221-journeys_final.pkl')

#%% PLOT EXPLORING WHICH DURATION TO KEEP

# stop 2776 = Perth = spatial cluster 7047
# stop 2764 = Subiaco = spatial cluster 3449

example = journeys_short[(journeys_short['OnLocation_cluster']==7047)&(journeys_short['OffLocation_cluster']==3449)]

fig = px.histogram(
    example['Duration_round'][example['Duration_round']<50],
    nbins=50,
    color_discrete_sequence=[mycolors_discrete[7]],
    histnorm='percent',
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Frequency of occurrence (%)',
    xaxis_title_text = 'Time between Perth and Subiaco (minutes)',
    showlegend=False)
pyo.plot(fig, config=config)

filename = "pmed_time_example_perthsubiaco"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT HISTOGRAM OF ALL DURATIONS

fig = px.histogram(
    journeys_final['Duration_round'][journeys_final['Duration_round']<100],
    nbins=100,
    color_discrete_sequence=[mycolors_discrete[7]],
    histnorm='percent',
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Frequency of occurrence (%)',
    xaxis_title_text = 'Edge weight (minutes)',
    showlegend=False)
pyo.plot(fig, config=config)


#%% PLOT CDF OF ALL DURATIONS

fig = px.ecdf(x=journeys_final['Duration_round'],    
              color_discrete_sequence=[mycolors_discrete[7]],
              ecdfnorm='percent')
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Cumulative frequency of occurrence (%)',
    xaxis_title_text = 'Edge weight (minutes)',
    showlegend=False)
fig.update_xaxes(range=[0, 100])
fig.update_yaxes(range=[0, 100])
pyo.plot(fig)
# 80% of edges have duration ~26min or less


#%% READ JOURNEYS FINAL PICKLE 

journeys_final = pd.read_pickle(dirname+'20260221-journeys_final.pkl')

#%% CREATE THE UNDERLYING GRAPH

G = nx.from_pandas_edgelist(journeys_final, 'OnLocation_cluster', "OffLocation_cluster", ["Duration_round"])

pos = {}
for index, row in clusters.iterrows():
    pos[row['clusterID']] = (row['geometry'].x, row['geometry'].y)

nx.set_node_attributes(G, pos, 'coord')

node_x = []
node_y = []
node_id = []
for node in G.nodes():
    x, y = G.nodes[node]['coord']
    node_x.append(x)
    node_y.append(y)
    node_id.append(node)
   
node_data = {'X': node_x, 'Y': node_y, 'clusterID':node_id}
df_nodes = pd.DataFrame(node_data)

edge_x = []
edge_y = []
weights = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['coord']
    x1, y1 = G.nodes[edge[1]]['coord']
    weight = G.edges[edge]['Duration_round']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    weights.append(weight)
    weights.append(weight) # put in twice so weight
    # appears as hover on both ends of the line
    weights.append(None)

edge_data = {'X': edge_x, 'Y': edge_y, 'weight':weights}
df_edges_all = pd.DataFrame(edge_data)

#%% MAP THE NODES AND EDGES

fig = px.scatter_map(
    df_nodes,
    lat=df_nodes["Y"],
    lon=df_nodes['X'],
    color_discrete_sequence=['gray'],
    hover_data='clusterID',
    center=map_center,
    zoom=9,
)

edge_x = []
edge_y = []
weights = []

mycolors_discrete_long = mycolors_discrete*10

n = 12
for x in np.arange(0, n):
    edge_x = []
    edge_y = []
    weights = []

    for edge in G.edges():
        if G.edges[edge]['Duration_round']==x:        
            x0, y0 = G.nodes[edge[0]]['coord']
            x1, y1 = G.nodes[edge[1]]['coord']
            weight = G.edges[edge]['Duration_round']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            weights.append(weight)
            weights.append(weight) # put in twice so weight
            # appears as hover on both ends of the line
            weights.append(None)
   
    edge_data = {'X': edge_x, 'Y': edge_y, 'weight':weights}
    df_edges = pd.DataFrame(edge_data)
   
    fig.add_trace(
        go.Scattermap(
            lat=df_edges['Y'],
            lon=df_edges['X'],
            hovertext=df_edges['weight'],
            name="weight = "+x.astype(str),
            mode="lines",
            line = dict(
                    width = 2,
                    color = mycolors_discrete_long[x],
                )
        )
    )

# this section does the final trace, for everything greater than the max
edge_x = []
edge_y = []
weights = []

for edge in G.edges():
    if G.edges[edge]['Duration_round']>=n:        
        x0, y0 = G.nodes[edge[0]]['coord']
        x1, y1 = G.nodes[edge[1]]['coord']
        weight = G.edges[edge]['Duration_round']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        weights.append(weight)
        weights.append(weight) # put in twice so weight
        # appears as hover on both ends of the line
        weights.append(None)

edge_data = {'X': edge_x, 'Y': edge_y, 'weight':weights}
df_edges = pd.DataFrame(edge_data)

fig.add_trace(
    go.Scattermap(
        lat=df_edges['Y'],
        lon=df_edges['X'],
        hovertext=df_edges['weight'],
        name="weight >= "+str(n),
        mode="lines",
        line = dict(
                width = 2,
                color = mycolors_discrete_long[n],
            )
    )
)


fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=10))
pyo.plot(fig, config=config)

# filename = 'smartrider_graph'
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)

#%% FIND JOURNEYS USED IN REGIONS ANALYSIS

# anchoring regions by card and region cluster
allregions2 = pd.read_pickle(dirname + "20250112-final-allregions-1,2.pkl")
allregions1 = pd.read_pickle(dirname + "20250112-final-allregions.pkl")
allregions = pd.concat([allregions1,allregions2])

# first - chop journeys down to just the cards we found regions for
journeys_used = journeys[journeys['Cardid'].isin(allregions['Card'])]

# calculate stay durations
journeys_used['end_stay'] = np.where(journeys_used['Cardid']==journeys_used['Cardid'].shift(-1),
                                     journeys_used['elapsedMinutes_On'].shift(-1), np.nan)
journeys_used['stayDuration'] = journeys_used['end_stay'] - journeys_used['elapsedMinutes_Off']


# if it's in allregions it's an anchoring region
# if it's got a region cluster number but not in allregions then it's a visited region
# if it's not got a region cluster number and it's not labelled 'transfer' it's a disconnected activity
# if it's not got a region cluster number and it's labelled 'transfer' it's (surprise!) a transfer
allregions['region_cluster'] = allregions['region_cluster'].astype(float)
allregions['Card-cluster'] = allregions['Card'].astype(str)+'-'+allregions['region_cluster'].astype(str)

journeys_used['Card-cluster_on'] = journeys_used['Cardid'].astype(str)+'-'+journeys_used['OnLocation_newreg'].astype(str)
journeys_used['Card-cluster_off'] = journeys_used['Cardid'].astype(str)+'-'+journeys_used['OffLocation_regioncluster'].astype(str)

# add two new columns OnLocation_regtype and OffLocation_regtype
# it is possible to have a 'transfer' that goes through an existing region
journeys_used['OnLocation_regtype'] = '0'

journeys_used['OnLocation_regtype'] = np.where((journeys_used['OnLocation_newreg']=="Transfer")|(journeys_used['OnTran']=='Transfer'),
                                               "Transfer", journeys_used['OnLocation_regtype'])

journeys_used['OnLocation_regtype'] = np.where(journeys_used['Card-cluster_on'].isin(allregions['Card-cluster']),
                                               "Anchor", journeys_used['OnLocation_regtype'])

journeys_used['OnLocation_regtype'] = np.where(journeys_used['OnLocation_newreg']=='nan',
                                               'Disconnected', journeys_used['OnLocation_regtype'])

journeys_used['OnLocation_regtype'] = np.where(journeys_used['OnLocation_regtype']=='0',
                                               "Visited", journeys_used['OnLocation_regtype'])

journeys_used['OffLocation_regtype'] = 0

journeys_used['OffLocation_regtype'] = np.where(journeys_used['Card-cluster_off'].isin(allregions['Card-cluster']),
                                               "Anchor", journeys_used['OffLocation_regtype'])

journeys_used['OffLocation_regtype'] = np.where((journeys_used['OffLocation_regioncluster']=="Transfer")|(journeys_used['OffTran']=='Transfer'),
                                               "Transfer", journeys_used['OffLocation_regtype'])

# if the stay duration is less than 60min, it's a transfer
# aligned with the regions paper 
journeys_used['OffLocation_regtype'] = np.where(journeys_used['stayDuration']<60,"Transfer", journeys_used['OffLocation_regtype'])

journeys_used['OffLocation_regtype'] = np.where(journeys_used['OffLocation_regioncluster']=='nan',
                                               'Disconnected', journeys_used['OffLocation_regtype'])

journeys_used['OffLocation_regtype'] = np.where(journeys_used['OffLocation_regtype']=='0',
                                               "Visited", journeys_used['OffLocation_regtype'])


# correcting for the day starting at 5am - we want things that are before 5am 
# to be considered part of the day before
journeys_used['OnDay'] = np.where(journeys_used['OnHour']<5,journeys_used['OnDay']-1,journeys_used['OnDay'])

journeys_used['PeriodDay'] = journeys_used['OnDay']%7
journeys_used['Period'] = journeys_used['OnDay']//7
journeys_used['Period'] = journeys_used['Period'].astype(float)

journeys_used['seqID'] = journeys_used['Cardid'].astype(str)+'-'+journeys_used['Period'].astype(str)


#%% EXAMPLE: GET INFO FOR ONE SEQUENCE

card_interest = 1373971
seqID_interest = str(card_interest)+'-1.0'

journeys_test = journeys_used[journeys_used['seqID']==seqID_interest]

journeys_test1 = journeys_test[['OnLocation_newspat','OffLocation_cluster']].value_counts()
journeys_test1 = pd.DataFrame(journeys_test1)
journeys_test1.reset_index(inplace=True)
journeys_test1.rename(columns={'OnLocation_newspat':'OnLocation_cluster'}, inplace=True)

journeys_test_stops = pd.concat([journeys_test['OnLocation_cluster'], journeys_test['OffLocation_cluster']]).drop_duplicates()
journeys_test_stops = pd.DataFrame(journeys_test_stops, columns=['spatial_cluster'])

journeys_test_stops = pd.merge(journeys_test_stops,
                         allhist_merge[['region_cluster', 'spatial_cluster']][allhist_merge['Card']==card_interest],
                         left_on=['spatial_cluster'],
                         right_on=['spatial_cluster'],
                         how='left')

journeys_test_stops = pd.merge(journeys_test_stops,
                               allregions[['region_cluster','region_type']][allregions['Card']==card_interest],
                               left_on='region_cluster',
                               right_on='region_cluster',
                               how='left')

# anything that DOESN'T have a region cluster must be a transfer node
# anything that DOES have a region cluster but isn't in allregions must be a visited region

journeys_test_stops['region_type'] = np.where(
    (journeys_test_stops['region_type'].isna())
    &(~journeys_test_stops['region_cluster'].isna()),
    "Visited",
    journeys_test_stops['region_type'])

journeys_test_stops['region_type'] = np.where(
    journeys_test_stops['region_type'].isna(),
    "Transfer",
    journeys_test_stops['region_type'])


allregionpolys = pd.read_pickle(dirname + "20250125-fullallregionpolys.pkl")

MAPBUFFER = 20

regionpolys_used = allregionpolys[allregionpolys['Card']==card_interest]
regionpolys_used = regionpolys_used[regionpolys_used['region_cluster'].isin(journeys_test_stops['region_cluster'])]

regionpolys_used = regionpolys_used.to_crs(epsg=32749)
regionpolys_used["geometry"] = regionpolys_used["geometry"].buffer(MAPBUFFER)

regionpolys_used.to_crs(epsg=4326, inplace=True)
regionpolys_used = pd.merge(regionpolys_used, journeys_test_stops[['region_cluster','region_type']].drop_duplicates(), on='region_cluster', how='left')

#%% EXAMPLE: MAKE IT A GRAPH

J = nx.from_pandas_edgelist(journeys_test1, 'OnLocation_cluster', "OffLocation_cluster", ['count'])

# coords attribute
nx.set_node_attributes(J, pos, 'coord')

# node type attribute

node_type = {}
region_cluster = {}
for index, row in journeys_test_stops.iterrows():
    node_type[row['spatial_cluster']] = (row['region_type'])
    region_cluster[row['spatial_cluster']] = (row['region_cluster'])

nx.set_node_attributes(J, node_type, 'node_type')
nx.set_node_attributes(J, region_cluster, 'region_cluster')

node_x = []
node_y = []
node_id = []
node_type = []
region_cluster = []
for node in J.nodes():
    x, y = J.nodes[node]['coord']
    z = J.nodes[node]['node_type']
    a = J.nodes[node]['region_cluster']
    node_x.append(x)
    node_y.append(y)
    node_id.append(node)
    node_type.append(z)
    region_cluster.append(a)
   
   
node_data = {'X': node_x, 'Y': node_y, 'clusterID':node_id, 'node_type':node_type, 'region_cluster':region_cluster}
df_nodes_seq = pd.DataFrame(node_data)

edge_x = []
edge_y = []
weights = []

for edge in J.edges():
    x0, y0 = J.nodes[edge[0]]['coord']
    x1, y1 = J.nodes[edge[1]]['coord']
    weight = J.edges[edge]['count']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    weights.append(weight)
    weights.append(weight) # put in twice so weight
    # appears as hover on both ends of the line
    weights.append(None)

edge_data = {'X': edge_x, 'Y': edge_y, 'weight':weights}
df_edges_seq = pd.DataFrame(edge_data)


#%% PLOT: ONE GRAPH

grey_col = "#cbcbcb"

color_anchor = mycolors_discrete[0]
color_visited = mycolors_discrete[1]

# region polygons
fig = px.choropleth_map(
    regionpolys_used,
    geojson=regionpolys_used.geometry,
    locations=regionpolys_used.index,
    color="region_type",
    center=map_center,
    color_discrete_map={"Anchor":color_anchor, "Visited":color_visited},
    opacity=0.5,
    zoom=12,
)

# candidate stops (underlying network)
fig.add_trace(
    go.Scattermap(
        lat=df_nodes['Y'],
        lon=df_nodes['X'],
        mode="markers",
        marker=dict(
            size=8,
            color = grey_col,
        ),
        name='Candidate stops',
    )
)

# edges in this sequence
fig.add_trace(
    go.Scattermap(
        lat=df_edges_seq['Y'],
        lon=df_edges_seq['X'],
        hovertext=df_edges_seq['weight'],
        mode="lines",
        line = dict(
                width = 3,
                color =mycolors_discrete[7],
            )
    )
)

# anchor region nodes in this sequence
fig.add_trace(
    go.Scattermap(
        lat=df_nodes_seq['Y'][df_nodes_seq['node_type']=='Anchor'],
        lon=df_nodes_seq['X'][df_nodes_seq['node_type']=='Anchor'],
        mode="markers",
        marker=dict(
            size=14,
            color = color_anchor,
        ),
        name='Anchor',
    )
)

# visited region nodes in this sequence
fig.add_trace(
    go.Scattermap(
        lat=df_nodes_seq['Y'][df_nodes_seq['node_type']=='Visited'],
        lon=df_nodes_seq['X'][df_nodes_seq['node_type']=='Visited'],
        mode="markers",
        marker=dict(
            size=14,
            color = color_visited,
        ),
        name='Visited',
    )
)

# transfer/long stay nodes in this sequence
fig.add_trace(
    go.Scattermap(
        lat=df_nodes_seq['Y'][df_nodes_seq['node_type']=='Transfer'],
        lon=df_nodes_seq['X'][df_nodes_seq['node_type']=='Transfer'],
        mode="markers",
        marker=dict(
            size=11,
            color =mycolors_discrete[2],
        ),
        name='Transfer or Long Stay',
    )
)

fig.update_layout(default_layout)
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)

# filename = 'example_sequence_transfers'
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)


#%% PLOT: MANY SEQUENCES

# underlying network
max_weight = 15 # do this to make sure the edges aren't too busy, just for mapping purposes

df_edges_map = df_edges_all[~(df_edges_all['weight']>max_weight)]

grey_col = "#cbcbcb"

fig = px.scatter_map(
    df_nodes,
    lat=df_nodes["Y"],
    lon=df_nodes['X'],
    color_discrete_sequence=[grey_col],
    hover_data='clusterID',
    center=map_center,
    zoom=9,
)

# edges of underlying network - can also just leave these out as it gets busy
fig.add_trace(
    go.Scattermap(
        lat=df_edges_map['Y'],
        lon=df_edges_map['X'],
        hovertext=df_edges_map['weight'],
        mode="lines",
        line = dict(
                width = 1,
                color =grey_col,
            )
    )
)
fig.update_traces(marker=dict(size=8))

seq_counts = journeys_used['seqID'].value_counts()
seq_counts = pd.DataFrame(seq_counts)
seq_counts.reset_index(inplace=True)

np.random.seed(42)
# allocate each row to a random number - seq_counts is ordered so we don't
# want to just do them in order
seq_counts["rand"] = np.random.choice(range(0, len(seq_counts)), size=len(seq_counts), replace=False)

for n in np.arange(0, 80):
    seqID_interest = seq_counts['seqID'][seq_counts['rand']==n].iloc[0]
    journeys_test = journeys_used[journeys_used['seqID']==seqID_interest]
    journeys_test1 = journeys_test[['OnLocation_cluster','OffLocation_cluster']].value_counts()
    journeys_test1 = pd.DataFrame(journeys_test1)
    journeys_test1.reset_index(inplace=True)
   
    journeys_test_stops = pd.concat([journeys_test['OnLocation_cluster'], journeys_test['OffLocation_cluster']]).drop_duplicates()
    journeys_test_stops = pd.DataFrame(journeys_test_stops, columns=['spatial_cluster'])
   
    journeys_test_stops = pd.merge(journeys_test_stops,
                             allhist_merge[['region_cluster', 'spatial_cluster']][allhist_merge['Card']==card_interest],
                             left_on=['spatial_cluster'],
                             right_on=['spatial_cluster'],
                             how='left')
   
    journeys_test_stops = pd.merge(journeys_test_stops,
                                   allregions[['region_cluster','region_type']][allregions['Card']==card_interest],
                                   left_on='region_cluster',
                                   right_on='region_cluster',
                                   how='left')
   
    journeys_test_stops['region_type'] = np.where(
        (journeys_test_stops['region_type'].isna())
        &(~journeys_test_stops['region_cluster'].isna()),
        "Visited",
        journeys_test_stops['region_type'])
   
    journeys_test_stops['region_type'] = np.where(
        journeys_test_stops['region_type'].isna(),
        "Transfer",
        journeys_test_stops['region_type'])
   
    I = nx.from_pandas_edgelist(journeys_test1, 'OnLocation_cluster', "OffLocation_cluster", ['count'])
   
    # coords attribute
    nx.set_node_attributes(I, pos, 'coord')
   
    # node type attribute
    node_type = {}
    region_cluster = {}
    for index, row in journeys_test_stops.iterrows():
        node_type[row['spatial_cluster']] = (row['region_type'])
        region_cluster[row['spatial_cluster']] = (row['region_cluster'])
   
    nx.set_node_attributes(I, node_type, 'node_type')
    nx.set_node_attributes(I, region_cluster, 'region_cluster')
   
    node_x = []
    node_y = []
    node_id = []
    node_type = []
    region_cluster = []
    for node in I.nodes():
        x, y = I.nodes[node]['coord']
        z = I.nodes[node]['node_type']
        a = I.nodes[node]['region_cluster']
        node_x.append(x)
        node_y.append(y)
        node_id.append(node)
        node_type.append(z)
        region_cluster.append(a)
   
    node_data = {'X': node_x, 'Y': node_y, 'clusterID':node_id, 'node_type':node_type, 'region_cluster':region_cluster}
    df_nodes_seq = pd.DataFrame(node_data)
   
    edge_x = []
    edge_y = []
    weights = []
   
    for edge in I.edges():
        x0, y0 = I.nodes[edge[0]]['coord']
        x1, y1 = I.nodes[edge[1]]['coord']
        weight = I.edges[edge]['count']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        weights.append(weight)
        weights.append(weight) # put in twice so weight
        # appears as hover on both ends of the line
        weights.append(None)
   
    edge_data = {'X': edge_x, 'Y': edge_y, 'weight':weights}
    df_edges_seq = pd.DataFrame(edge_data)
   
   
    fig.add_trace(
        go.Scattermap(
            lat=df_nodes_seq['Y'],
            lon=df_nodes_seq['X'],
            hovertext=df_nodes_seq['clusterID'],
            mode="markers",
            marker=dict(
                color=mycolors_discrete_long[n],  
                size=8
            )
        )
    )
   
    fig.add_trace(
        go.Scattermap(
            lat=df_edges_seq['Y'],
            lon=df_edges_seq['X'],
            hovertext=df_edges_seq['weight'],
            mode="lines",
            line = dict(
                    width = 2,
                    color = mycolors_discrete_long[n],
                )
        )
    )

fig.update_layout(default_layout)
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)

# filename = 'demand_candidate_graph'
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
  

#%% WORKING OUT THE REGIONS EITHER SIDE OF VISITED REGIONS

# drop where both the on and off region type is a transfer
journeys_output = journeys_used[~((journeys_used['OnLocation_regtype']=='Transfer')&(journeys_used['OffLocation_regtype']=='Transfer'))]
journeys_output.reset_index(inplace=True, drop=True)

# set of columns to remove
cols_drop = ['OnMode', 'OffDate', 'Token', 'OnTime', 'OffTime', 'OnHour',
       'OffHour', 'OnDay', 'tripTime', 'tripTime_h', 'timeSince', 'arriveHour',
       'fromLocation', 'originLocation', 'ArriveMode', 'OffTime_prev',
       'OnTime_prev', 'tripTime_prev', 'tripTime_h_prev', 'SyntheticFlag',
       'SyntheticOnOk_WW', 'SyntheticOffOk_WW', 'SyntheticOffOk_transfer',
       'SyntheticOk', 'SyntheticDrop', 'OnHour_norm', 'OnHour_bucket',
       'Token_type', 'elapsedMinutes_On', 'elapsedMinutes_Off', 'Duration',
       'Duration_round', 'Card-cluster_on','Card-cluster_off', 'stayDuration','end_stay',
       'OnLocation_cluster', 'OnLocation_regioncluster', 'OnLocation']

journeys_output.drop(cols_drop, axis=1, inplace=True)

# tidy up the renamed OnLocation columns
journeys_output.rename(columns={'OnLocation_new':'OnLocation', 'OnLocation_newspat':'OnLocation_cluster', 'OnLocation_newreg':'OnLocation_regioncluster'}, inplace=True)

# create an actual off location
# where off location = Transfer, pull in the offlocation of the next row
# then drop all rows where onlocation = Transfer (that next row that is no longer necessary)

journeys_output['OffLocation_newreg'] = journeys_output['OffLocation_regioncluster'].shift(-1).where(
    (journeys_output['OffLocation_regioncluster']=='Transfer')&
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1)), other=journeys_output['OffLocation_regioncluster'])

journeys_output['OffLocation'] = journeys_output['OffLocation'].shift(-1).where(
    (journeys_output['OffLocation_regioncluster']=='Transfer')&
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1)), other=journeys_output['OffLocation'])

journeys_output['OffLocation_cluster'] = journeys_output['OffLocation_cluster'].shift(-1).where(
    (journeys_output['OffLocation_regioncluster']=='Transfer')&
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1)), other=journeys_output['OffLocation_cluster'])

journeys_output['OffLocation_regtype'] = journeys_output['OffLocation_regtype'].shift(-1).where(
    (journeys_output['OffLocation_regioncluster']=='Transfer')&
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1)), other=journeys_output['OffLocation_regtype'])

journeys_output = journeys_output[journeys_output['OnLocation_regioncluster']!='Transfer']

journeys_output.drop('OffLocation_regioncluster', axis=1, inplace=True)
journeys_output.rename(columns={'OffLocation_newreg':'OffLocation_regioncluster'}, inplace=True)
journeys_output.reset_index(inplace=True,drop=True)

# then we want
# onlocation of that row (where did they come from)
# offlocation of the next row (where did they go next)
# only pull in the next location if it's less than 3 days away
# (same logic as the stays) - otherwise they probably did something else
# in that time

journeys_output['OffLocation_next'] = journeys_output['OffLocation'].shift(-1).where(
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1))&
    ((journeys_output['OnDate'].shift(-1)-journeys_output['OnDate'])<dt.timedelta(days=3)), other=np.nan)
journeys_output['OffLocation_cluster_next'] = journeys_output['OffLocation_cluster'].shift(-1).where(
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1))&
    ((journeys_output['OnDate'].shift(-1)-journeys_output['OnDate'])<dt.timedelta(days=3)), other=np.nan)
journeys_output['OffLocation_regioncluster_next'] = journeys_output['OffLocation_regioncluster'].shift(-1).where(
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1))&
    ((journeys_output['OnDate'].shift(-1)-journeys_output['OnDate'])<dt.timedelta(days=3)), other=np.nan)
journeys_output['OffLocation_regtype_next'] = journeys_output['OffLocation_regtype'].shift(-1).where(
    (journeys_output['Cardid']==journeys_output['Cardid'].shift(-1))&
    ((journeys_output['OnDate'].shift(-1)-journeys_output['OnDate'])<dt.timedelta(days=3)), other=np.nan)

# then we want to keep just where OffLocation_regtype == Visited
journeys_V = journeys_output[journeys_output['OffLocation_regtype']=='Visited']
journeys_V = journeys_V[(journeys_V['OffLocation_regtype_next']=='Visited')|(journeys_V['OffLocation_regtype_next']=='Anchor')]
journeys_V.reset_index(inplace=True, drop=True)
# this is now only 179,350

#%% GET SPATIAL CLUSTERS IN EACH REGION

allhist = pd.read_pickle(dirname + "20250125-fullallhist.pkl")

# do a groupby on allhist to get a list of spatial clusters for each card/region
allregion_list = allhist[['Card','region_cluster','spatial_cluster']].drop_duplicates().reset_index(drop=True)
allregion_list['spatial_cluster'] = allregion_list['spatial_cluster'].astype(str)
allregion_list = allregion_list.groupby(['Card','region_cluster']).agg(list)

allregion_list.reset_index(inplace=True)
# this is 478,082 long
# doing a value_counts on this takes ages because it's checking the lists
# but there are only 40,259 unique combinations of spatial clusters in regions
# note this includes ones outside Perth as well, so it's likely even less than this

# replacing the region numbers with some other kind of unique ID back to this would make sense
# without this can't consolidate each of the rows down, so keeps the optimisation problem long/difficult
allregion_list['spatial_cluster'] = allregion_list['spatial_cluster'].apply(tuple)
codes, uniques = allregion_list['spatial_cluster'].factorize()
allregion_list['ID'] = codes

uniques = pd.DataFrame(uniques, columns=['spatial_cluster'])
uniques.reset_index(inplace=True)
uniques.rename(columns={'index':'ID'}, inplace=True)

uniques.to_pickle(dirname+"20260221-uniques.pkl")
allregion_list.to_pickle(dirname+"20260221-allregionlist.pkl")

#%% PUT THE NEW REGION IDS BACK ONTO THE DATASET

allregion_list['region_cluster'] = allregion_list['region_cluster'].astype(float)
journeys_V['OnLocation_regioncluster'] = journeys_V['OnLocation_regioncluster'].astype(float)
journeys_V['OffLocation_regioncluster'] = journeys_V['OffLocation_regioncluster'].astype(float)
journeys_V['OffLocation_regioncluster_next'] = journeys_V['OffLocation_regioncluster_next'].astype(float)

journeys_V = pd.merge(journeys_V, allregion_list[['Card','region_cluster','ID']], left_on=['Cardid','OnLocation_regioncluster'], right_on=['Card','region_cluster'], how='left')
journeys_V.drop(columns={'Card','region_cluster'}, inplace=True)
journeys_V.rename(columns={'ID':'OnLocation_ID'}, inplace=True)

journeys_V = pd.merge(journeys_V, allregion_list[['Card','region_cluster','ID']], left_on=['Cardid','OffLocation_regioncluster'], right_on=['Card','region_cluster'], how='left')
journeys_V.drop(columns={'Card','region_cluster'}, inplace=True)
journeys_V.rename(columns={'ID':'OffLocation_ID'}, inplace=True)

journeys_V = pd.merge(journeys_V, allregion_list[['Card','region_cluster','ID']], left_on=['Cardid','OffLocation_regioncluster_next'], right_on=['Card','region_cluster'], how='left')
journeys_V.drop(columns={'Card','region_cluster'}, inplace=True)
journeys_V.rename(columns={'ID':'OffLocation_ID_next'}, inplace=True)

journeys_V = journeys_V[journeys_V['OnLocation_ID']!=journeys_V['OffLocation_ID']]
journeys_V = journeys_V[journeys_V['OffLocation_ID']!=journeys_V['OffLocation_ID_next']]

journeys_V.to_pickle(dirname+'20260221-journeys-V.pkl')

#%% READ JOURNEYS_V AND UNIQUES - CAN START HERE

journeys_V = pd.read_pickle(dirname+'20260221-journeys-V.pkl')
# this was 376,423 records
# is now 179,620 without transfers

uniques = pd.read_pickle(dirname+"20260221-uniques.pkl")
# this is 40,259 records


#%% SETTING UP POLYGONS TO CUT LOCATIONS DOWN

# overall box, can delete - earlier in code
geometry = [box(115.493774,-32.685620,116.180420,-31.447410)] # coast to Mt Helena, Manudurah to Yanchep
boxdf = gpd.GeoDataFrame(geometry=geometry)
boxdf = boxdf.set_crs("epsg:4326", allow_override=True)


# JOONDALUP LINE GEO

joondalup = geo[geo['line']=='Joondalup Line']

joondalup_poly = joondalup.dissolve().convex_hull.reset_index()
joondalup_poly.rename(columns={0:'geometry'}, inplace=True)

joondalup_poly = joondalup_poly.set_geometry('geometry')
joondalup_poly = joondalup_poly.set_crs("epsg:4326", allow_override=True)

joondalup_poly = joondalup_poly.to_crs(epsg=32749)
joondalup_poly["geometry"] = joondalup_poly["geometry"].buffer(1500)
# if this gets much higher the buffer around the Leederville station starts to include Perth...

joondalup_poly.to_crs(epsg=4326, inplace=True)

fig = px.choropleth_map(joondalup_poly, 
                        geojson=joondalup_poly.geometry, 
                        locations=joondalup_poly.index,     
                        center=map_center,
                        color_discrete_sequence=mycolors_discrete,
                        opacity=0.6,
                        zoom=8)
fig.update_layout(map_style="light")
#pyo.plot(fig, config=config)

filename = "pmed_map_jdlp"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

def in_jdlp(row):
    return joondalup_poly.contains(row['geometry']).any()

clusters['jdlp check'] = clusters.apply(in_jdlp, axis=1)

# TRAINS ONLY

trains = geo[geo['type']=='Train']

# BUSES ONLY
# bit trivial as there's so many of them, and many are also at train stations

buses = geo[geo['type']=='Bus']

# NORTH OF THE RIVER

geometry = [box(115.640385,-31.939116,116.226689,-31.591836)] 
nordf = gpd.GeoDataFrame(geometry=geometry)
nordf = nordf.set_crs("epsg:4326", allow_override=True)

# check it has been set up correctly
fig = px.choropleth_map(nordf, 
                        geojson=nordf.geometry, 
                        locations=nordf.index,     
                        center=map_center,
                        color_discrete_sequence=mycolors_discrete,
                        opacity=0.6,
                        zoom=8)
fig.update_layout(map_style="light")
#pyo.plot(fig, config=config)

# add a column to that says whether or not each stop cluster is in the bbox
def in_nor(row):
    return nordf.contains(row['geometry']).any()

# SOUTH OF THE RIVER

geometry = [box(115.619730,-32.719754,116.326177,-31.966032)] 
sordf = gpd.GeoDataFrame(geometry=geometry)
sordf = sordf.set_crs("epsg:4326", allow_override=True)

#check it has been set up correctly
fig = px.choropleth_map(sordf, 
                        geojson=sordf.geometry, 
                        locations=sordf.index,     
                        center=map_center,
                        color_discrete_sequence=mycolors_discrete,
                        opacity=0.6,
                        zoom=8)
fig.update_layout(map_style="light")
#pyo.plot(fig, config=config)

# add a column to that says whether or not each stop cluster is in the bbox
def in_sor(row):
    return sordf.contains(row['geometry']).any()

# CBD

geometry = [box(115.846462,-31.965589,115.884492,-31.945135)] 
cbddf = gpd.GeoDataFrame(geometry=geometry)
cbddf = cbddf.set_crs("epsg:4326", allow_override=True)

# check it has been set up correctly
# fig = px.choropleth_map(cbddf, 
#                         geojson=cbddf.geometry, 
#                         locations=cbddf.index,     
#                         center=map_center,
#                         color_discrete_sequence=mycolors_discrete,
#                         opacity=0.6,
#                         zoom=8)
# fig.update_layout(map_style="light")
# pyo.plot(fig, config=config)

# add a column to that says whether or not each stop cluster is in the bbox
def in_cbd(row):
    return ~cbddf.contains(row['geometry']).any()
# this is inverted because we want to keep things OUTSIDE the CBD, instead of
# inside the bbox like in all other examples

# ALL EXCEPT CBD MAP

res_union = boxdf.overlay(cbddf, how='difference')

fig = px.choropleth_map(res_union, 
                        geojson=res_union.geometry, 
                        locations=res_union.index,     
                        center=map_center,
                        color_discrete_sequence=mycolors_discrete,
                        opacity=0.6,
                        zoom=8)
fig.update_layout(map_style="light")
#pyo.plot(fig, config=config)

#%% FLAGGING WHERE LOCATIONS ARE OR AREN'T TO BE INCLUDED

# everything comes back to the original spatial clusters, so put it all there first
# clusters is a geodf with clusterID and geometry
clusters['jdlp check'] = clusters.apply(in_jdlp, axis=1)
clusters['train check'] = np.where(clusters['clusterID'].isin(trains['spatial_cluster']),True,False)
clusters['NOR check'] = clusters.apply(in_nor, axis=1)
clusters['SOR check'] = clusters.apply(in_sor, axis=1)
clusters['outside CBD check'] = clusters.apply(in_cbd, axis=1)

#%% PLOT MAP TO DEMONSTRATE EXCLUSION RULES

clusters_map = clusters[(clusters['jdlp check']==True)&(clusters['bbox check']==True)]

map_center = {"lat": -32.1, "lon": 115.864055}

fig = px.scatter_map(
    clusters_map,
    lat=clusters_map["geometry"].y,
    lon=clusters_map["geometry"].x,
    color_discrete_sequence=mycolors_discrete,
    center=map_center,
    zoom=8.4,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=4))
pyo.plot(fig, config=config)

filename = "pmed_map_jdlp"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% WORK OUT WHICH SPATIAL CLUSTERS HAVE ALREADY BEEN USED

used_V = journeys_V['OffLocation_cluster'][journeys_V['OffLocation_cluster'].isin(clusters['clusterID'][clusters['bbox check']==True])].value_counts()
used_V = pd.DataFrame(used_V)
used_V.reset_index(inplace=True)
# this has 5199 records in it - there are only 6861 in the whole dataset
# but many have only been visited once

used_V['percent'] = used_V['count']/used_V['count'].sum()*100
used_V['cum_percent'] = used_V['percent'].cumsum()

used_V[used_V['percent']>0.1]
# only 109 locations contribute >0.1% of the records, which gets us to 66% coverage

used_V[used_V['percent']>1]
# only 16 contribute >1%, which is 31% coverage

len(used_V[used_V['cum_percent']>95])
# 2805 locations collectively contribute 5% of the visited activities

#%% PLOT HISTOGRAM OF CANDIDATE LOCATION USES 
# demonstrating how many are only used a small number of times

fig = px.histogram(used_V[used_V['count']<500], x="count", 
                   nbins=100, histnorm='percent', color_discrete_sequence=[mycolors_discrete[7]])
fig.update_layout(default_layout,
                  yaxis_title_text = 'Percentage of dataset',
                  xaxis_title_text = 'Number of times used as visited location')
pyo.plot(fig)

#%% SET UP FILTERS ON CANDIDATE LOCATIONS

# now we have the binary indicators for the candidate locations
df_nodes = pd.merge(df_nodes, clusters, on='clusterID', how='left')
# this has a different number now because different stops are coming up
# as used with new transfer rules

# how many total candidate locations are there?
len(df_nodes) # 6861
len(df_nodes[df_nodes['NOR check']==True]) # 2869
len(df_nodes[df_nodes['SOR check']==True]) # 3562
len(df_nodes[df_nodes['outside CBD check']==True]) # 6810
len(df_nodes[df_nodes['jdlp check']==True]) # 726
len(df_nodes[df_nodes['train check']==True]) # 70

# flag for if it's already used as a visited region
df_nodes['is_open'] = np.where(df_nodes['clusterID'].isin(used_V['OffLocation_cluster']), True, False)

# option: only consider locations covering 95% of the visited activities
#df_nodes['is_open'] = np.where(df_nodes['clusterID'].isin(used_V['OffLocation_cluster'][used_V['cum_percent']<=95]), True, False)

# how many of the candidate locations are already open?
len(df_nodes[(df_nodes['is_open']==True)]) # 5199
len(df_nodes[(df_nodes['NOR check']==True)&(df_nodes['is_open']==True)]) # 2144
len(df_nodes[(df_nodes['SOR check']==True)&(df_nodes['is_open']==True)]) # 2702
len(df_nodes[(df_nodes['outside CBD check']==True)&(df_nodes['is_open']==True)]) # 5153
len(df_nodes[(df_nodes['jdlp check']==True)&(df_nodes['is_open']==True)]) # 602
len(df_nodes[(df_nodes['train check']==True)&(df_nodes['is_open']==True)]) # 69
# so basically all of them for trains - this isn't particularly useful

#journeys_V = pd.read_pickle(dirname+'20260118-journeys-V.pkl')

used_V = journeys_V['OffLocation_cluster'].value_counts()
used_V = pd.DataFrame(used_V)
used_V.reset_index(inplace=True)

df_nodes = pd.merge(df_nodes, used_V, left_on='clusterID', right_on='OffLocation_cluster',how='left')
df_nodes.drop(columns='OffLocation_cluster', inplace=True)
df_nodes['count'] = df_nodes['count'].fillna(0)
df_nodes.rename(columns={'count':'count_V'}, inplace=True)

df_nodes.to_pickle(dirname+"20260221-df-nodes.pkl")

#%% FILTERS FOR THE DEMAND LOCATIONS

# first do this onto uniques - the unique list of regions
# then need to put this onto the before/after sets
uniques = uniques['spatial_cluster'].explode()
uniques = pd.DataFrame(uniques)
uniques.reset_index(inplace=True)
uniques.rename(columns={'index':'ID'}, inplace=True)
uniques['spatial_cluster'] = uniques['spatial_cluster'].astype(float)
uniques = pd.merge(uniques, clusters, left_on='spatial_cluster', right_on='clusterID', how='left')
uniques.drop(columns=['clusterID', 'geometry'], inplace=True)

# Then can aggregate uniques back up again
# for the spatial checks (NOR, SOR, Joondalup) we want it to return False if 
# any of the clusters within the region are False
# for Train, it can return True if *any* of the clusters in the region are a train station

# the 'bbox check' column is the Perth metro - we want to keep the region
# where any of those are true too

def check_false(x):
    count = (x==False).sum()
    if count > 0:
        return False
    return True

def check_true(x):
    count = (x==True).sum()
    if count > 0:
        return True
    return False

uniques = uniques.groupby('ID').agg({'NOR check':check_false, 
                         'SOR check':check_false,
                         'jdlp check':check_false,
                         'outside CBD check':check_false,
                         'train check':check_true,
                         'bbox check':check_true,
                         'spatial_cluster':list})
uniques.reset_index(inplace=True)
# length of uniques = 40259
# length of uniques in Perth metro (bbox=true) = 38058

#%% PUT TOGETHER THE BEFORE AND AFTER LOCATIONS

journeys_agg_on = journeys_V[['Cardid','OffLocation_ID','OnLocation_ID']].value_counts()
journeys_agg_on = pd.DataFrame(journeys_agg_on)
journeys_agg_on.reset_index(inplace=True)

journeys_agg_on = journeys_agg_on.groupby(['Cardid','OffLocation_ID'], as_index=False).agg(list)
journeys_agg_on.rename(columns={'count':'On_count'}, inplace=True)

journeys_agg_off = journeys_V[['Cardid','OffLocation_ID','OffLocation_ID_next']].value_counts()
journeys_agg_off = pd.DataFrame(journeys_agg_off)
journeys_agg_off.reset_index(inplace=True)
journeys_agg_off = journeys_agg_off.groupby(['Cardid','OffLocation_ID'], as_index=False).agg(list)
journeys_agg_off.rename(columns={'count':'Off_count'}, inplace=True)

journeys_agg_all = pd.merge(journeys_agg_on, journeys_agg_off, on=['Cardid','OffLocation_ID'], how='left')
journeys_agg_all['OnLocation_ID'] = journeys_agg_all['OnLocation_ID'].apply(tuple)
journeys_agg_all['OffLocation_ID_next'] = journeys_agg_all['OffLocation_ID_next'].apply(tuple)
# 114,161


#%% CHOP DOWN TO RECORDS WE HAVE A HOME LOCATION FOR

activities = pd.read_pickle(dirname+"20250606-processed_activities.pkl")

homeregions = activities[(activities['final_regionID']=='R1')|(activities['final_regionID']=='L1')]
homeregions = homeregions[['Cardid','region_cluster','final_regionID']]
homeregions = homeregions.drop_duplicates()

allregion_list = pd.read_pickle(dirname+"20260221-allregionlist.pkl")
# this is regionIDs for cards/region clusters

homeregions = pd.merge(homeregions, allregion_list, left_on=['Cardid','region_cluster'], right_on=['Card','region_cluster'], how='left')
homeregions = homeregions[['Cardid','final_regionID','ID']]

# so now we want to chop down homeregions
# for each card, if it has a regionID R1, keep that
# otherwise, keep L1
# so basically, drop L1 if there is already an R1 for that card

homeregions['count'] = homeregions.groupby('Cardid')['Cardid'].transform('count')
homeregions = homeregions[~((homeregions['count']==2)&(homeregions['final_regionID']=='L1'))]
homeregions.drop(columns={'count','final_regionID'},inplace=True)

journeys_agg_all = pd.merge(journeys_agg_all, homeregions, on='Cardid',how='left')
journeys_agg_all.rename(columns={'ID':'homeID'}, inplace=True)
journeys_agg_all = journeys_agg_all[~journeys_agg_all['homeID'].isna()]
journeys_agg_all['count'] = journeys_agg_all['On_count'].apply(sum)

journeys_cards = journeys_agg_all[['Cardid','OnLocation_ID','OffLocation_ID_next',
                                   'Off_count','On_count','count','homeID']].value_counts()
journeys_cards = pd.DataFrame(journeys_cards)
journeys_cards.rename(columns={'count':'record_count'}, inplace=True)
journeys_cards.reset_index(inplace=True)
journeys_cards.drop(columns='count', inplace=True)
journeys_cards.rename(columns={'record_count':'count'}, inplace=True)


journeys_agg_all['On_count'] = journeys_agg_all['On_count'].apply(tuple)
journeys_agg_all['Off_count'] = journeys_agg_all['Off_count'].apply(tuple)

journeys_unique = journeys_agg_all[['OnLocation_ID','OffLocation_ID_next','On_count','Off_count']].value_counts()
journeys_unique = pd.DataFrame(journeys_unique)
journeys_unique.reset_index(inplace=True)
# this is now 54,047 with only the journeys that have home locations being considered

journeys_cards.to_pickle(dirname+'20260307-journeys-cards.pkl')
# this is now 83,613 with only those with home locations and aggregating up different visited regions together

#%% FILTERS FOR THE DEMAND LOCATIONS

journeys_on = journeys_unique['OnLocation_ID'].explode()
journeys_on = pd.DataFrame(journeys_on)

journeys_on = journeys_on.reset_index().merge(uniques[['ID', 'bbox check', 'NOR check', 'SOR check', 'jdlp check', 'outside CBD check',
       'train check']], left_on='OnLocation_ID', right_on='ID', how='left').set_index('index')
journeys_on.drop(columns='ID', inplace=True)

journeys_on_filter = journeys_on.reset_index().groupby('index').agg({'NOR check':check_false, 
                         'SOR check':check_false,
                         'jdlp check':check_false,
                         'outside CBD check':check_false,
                         'bbox check':check_false})

journeys_on_filter = journeys_on_filter.add_suffix('_on')

journeys_off = journeys_unique['OffLocation_ID_next'].explode()
journeys_off = pd.DataFrame(journeys_off)

journeys_off = journeys_off.reset_index().merge(uniques[['ID', 'bbox check', 'NOR check', 'SOR check', 'jdlp check', 'outside CBD check',
       'train check']], left_on='OffLocation_ID_next', right_on='ID', how='left').set_index('index')
journeys_off.drop(columns='ID', inplace=True)

journeys_off_filter = journeys_off.reset_index().groupby('index').agg({'NOR check':check_false, 
                         'SOR check':check_false,
                         'jdlp check':check_false,
                         'outside CBD check':check_false,
                         'bbox check':check_false})
journeys_off_filter = journeys_off_filter.add_suffix('_off')

# now need to compare
demand_filters = pd.concat([journeys_on_filter, journeys_off_filter], axis=1)

# only keep condition as true if BOTH are true
demand_filters['NOR check'] = demand_filters['NOR check_on']&demand_filters['NOR check_off']
demand_filters['SOR check'] = demand_filters['SOR check_on']&demand_filters['SOR check_off']
demand_filters['outside CBD check'] = demand_filters['outside CBD check_on']&demand_filters['outside CBD check_off']
demand_filters['jdlp check'] = demand_filters['jdlp check_on']&demand_filters['jdlp check_off']
demand_filters['bbox check'] = demand_filters['bbox check_on']&demand_filters['bbox check_off']

# now put this back onto journeys_unique
journeys_unique = journeys_unique.merge(demand_filters[['NOR check','SOR check','jdlp check', 'outside CBD check', 'bbox check']], left_index=True, right_index=True, how='left')

# and drop everything outside Perth metro
journeys_unique = journeys_unique[journeys_unique['bbox check']==True]
# this is now 54,019 - not a huge drop as we chopped the earlier analysis
# where we determined home locations down to a metro area
# note we still haven't considered where the VISITED locations are outside the Perth metro - this comes later

len(journeys_unique[journeys_unique['NOR check']==True]) # 11441
len(journeys_unique[journeys_unique['SOR check']==True]) # 19228
len(journeys_unique[journeys_unique['outside CBD check']==True]) # 40872
len(journeys_unique[journeys_unique['jdlp check']==True]) # 2472

#%%
fig = px.histogram(
    journeys_unique['count'][(journeys_unique['count']<200)&(journeys_unique['count']>1)],
    nbins=100,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    showlegend=False)
pyo.plot(fig, config=config)

#%% MAP TO EXPLORE LOCATIONS WE'RE LOOKING AT

# geo aggregated by spatial cluster to get the convex hull polygon for that cluster
geo_polygon = geo.dissolve("spatial_cluster").convex_hull.reset_index()
geo_polygon.rename(columns={0:'geometry'}, inplace=True)

geo_polygon = geo_polygon.set_geometry('geometry')
geo_polygon = geo_polygon.set_crs("epsg:4326", allow_override=True)

# reproject to crs then centroid then back again
geo_polygon = geo_polygon.to_crs(epsg=32749)
geo_polygon['geometry'] = geo_polygon['geometry'].centroid
geo_polygon = geo_polygon.to_crs(epsg=4326)

# uniques has a column ID and spatial cluster
uniques_full = uniques['spatial_cluster'].explode()
uniques_full = pd.DataFrame(uniques_full)

uniques_full.reset_index(inplace=True)
uniques_full.rename(columns={'index':'ID'}, inplace=True)
uniques_full['spatial_cluster'] = uniques_full['spatial_cluster'].astype(float)
geo_polygon['spatial_cluster'] = geo_polygon['spatial_cluster'].astype(float)

uniques_full = pd.merge(uniques_full, geo_polygon, left_on='spatial_cluster', right_on='spatial_cluster', how='left')
# explode out, put on geo, then convex hull over the new geo]
# then we'll have a polygon for each ID
uniques_full = gpd.GeoDataFrame(uniques_full, geometry="geometry")
uniques_full = uniques_full.dissolve("ID").convex_hull.reset_index()

uniques_full.rename(columns={0:'geometry'}, inplace=True)

uniques_full = uniques_full.set_geometry('geometry')
uniques_full = uniques_full.set_crs("epsg:4326", allow_override=True)

# reproject to crs then centroid then back again
uniques_full = uniques_full.to_crs(epsg=32749)
uniques_full['geometry'] = uniques_full['geometry'].centroid
uniques_full = uniques_full.to_crs(epsg=4326)

# we now have a point for each region ID

#%% PLOT THE UNIQUE REGIONS - for reference

uniques_plot = uniques_full
uniques_plot['ID'] = uniques_plot['ID'].astype(str)
fig = px.scatter_map(uniques_plot, 
                     lat=uniques_plot.geometry.y, 
                     lon=uniques_plot.geometry.x,     
                     color="ID", 
                     center=map_center,
                     color_discrete_sequence=mycolors_discrete,
                     size_max=15, zoom=10)
fig.update_layout(map_style="light", coloraxis_showscale=False)

pyo.plot(fig, config=config)


#%% PLOT REGIONS EITHER SIDE OF A VISITED REGION

on_geo = journeys_unique[['OnLocation_ID','count']].explode('OnLocation_ID').reset_index()
on_geo = pd.merge(on_geo, uniques_full, left_on='OnLocation_ID', right_on='ID', how='left')
on_geo = gpd.GeoDataFrame(on_geo, geometry="geometry")
on_geo.drop(["ID"], axis=1, inplace=True)
on_geo.rename(columns={'OnLocation_ID':'ID'}, inplace=True)

off_geo = journeys_unique[['OffLocation_ID_next','count']].explode('OffLocation_ID_next').reset_index()
off_geo = pd.merge(off_geo, uniques_full, left_on='OffLocation_ID_next', right_on='ID', how='left')
off_geo = gpd.GeoDataFrame(off_geo, geometry="geometry")
off_geo.drop(["ID"], axis=1, inplace=True)
off_geo.rename(columns={'OffLocation_ID_next':'ID'}, inplace=True)

map_all = pd.concat([on_geo[['ID','index','count','geometry']], off_geo[['ID','index','count','geometry']]])
map_all.sort_values(by='index', inplace=True)

# grab a slice rather than trying to do all ~290k points at once
# 2000 makes chrome unhappy
map_plot = map_all.head(1000)
map_plot['index'] = map_plot['index'].astype(str)

fig = px.scatter_map(map_plot, 
                     lat=map_plot.geometry.y, 
                     lon=map_plot.geometry.x,     
                     color="index", 
                     #size="count",
                     center=map_center,
                     color_discrete_sequence=mycolors_discrete,
                     size_max=15, zoom=10)
fig.update_layout(map_style="light", coloraxis_showscale=False)

pyo.plot(fig, config=config)

# index 5 is the first that doesn't involve the CBD
# do a map_all['index'].value_counts() to find the top rated one that isn't just one stop either side
# do a histogram of this for the thesis

# filename = "map_all_regioncombinations"
# fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)

#%% PLOT DISTRIBUTION OF INDICES

# want to colour those which are entirely in the CBD a different colour
# the 'outside CBD check' column says True for anything *outside* the CBD
# we want to invert for this map so we colour True for *inside* the CBD

fig = px.bar(
    journeys_unique.head(100),
    x=journeys_unique.head(100).index,
    y='count',
    color=~journeys_unique['outside CBD check'].head(100),
    color_discrete_sequence=[mycolors_discrete[1], mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Count',
    xaxis_title_text = 'Index',
)

fig.update_xaxes(
    showgrid=True,
    tickson="boundaries",
  )
pyo.plot(fig, config=config)

#%% EXPLORE INDIVIDUAL INDICES

# get a count of the number of unique visited regions between two points
# e.g. if there are 122 that start at 7, stop somewhere, then go to 185
# how many different places is that 'somewhere'?
# this is in journeys_agg_all

# index in map_all = index in journeys_unique
# which gives us the OnLocation_ID and OffLocation_ID_next to filter on in journeys_agg_all

ix_interest = 5

on_interest = journeys_unique['OnLocation_ID'].iloc[ix_interest]
off_interest = journeys_unique['OffLocation_ID_next'].iloc[ix_interest]

journeys_interest = journeys_agg_all[(journeys_agg_all['OnLocation_ID']==on_interest)&(journeys_agg_all['OffLocation_ID_next']==off_interest)]
print(len(journeys_interest))
print(len(journeys_interest['OffLocation_ID'].value_counts()))
print(len(journeys_interest['Cardid'].value_counts()))

# this gives us the number of unique places currently used as visited regions

#%% PLOT INDIVIDUAL INDICES

map_plot = map_all[map_all['index']==ix_interest]
map_plot['index'] = map_plot['index'].astype(str)

map_plot['colour'] = 0
map_plot['colour'] = np.where((map_plot['ID'].isin(on_interest))&(~map_plot['ID'].isin(off_interest)),'Before', map_plot['colour'])
map_plot['colour'] = np.where((map_plot['ID'].isin(off_interest))&(~map_plot['ID'].isin(on_interest)),'After', map_plot['colour'])
map_plot['colour'] = np.where((map_plot['ID'].isin(on_interest))&(map_plot['ID'].isin(off_interest)),'Both', map_plot['colour'])

v_regions = journeys_interest['OffLocation_ID'].value_counts()
v_regions = pd.DataFrame(v_regions)
v_regions.reset_index(inplace=True)

v_regions = pd.merge(v_regions, uniques_full, left_on='OffLocation_ID', right_on='ID', how='left')
v_regions = gpd.GeoDataFrame(v_regions, geometry="geometry")
v_regions.drop(["ID"], axis=1, inplace=True)

fig = px.scatter_map(map_plot, 
                     lat=map_plot.geometry.y, 
                     lon=map_plot.geometry.x,     
                     color="colour", 
                     hover_name="ID",
                     center={"lat": -31.95, "lon": 115.85},
                     color_discrete_sequence=mycolors_discrete,
                     zoom=10)
fig.update_layout(map_style="light", coloraxis_showscale=False)
fig.update_traces(marker=dict(size=20))

fig.add_trace(
    go.Scattermap(
        lat=v_regions["geometry"].y,
        lon=v_regions["geometry"].x,
        mode="markers",
        name='Visited regions',
        marker=go.scattermap.Marker(
            color=mycolors_discrete[7],
            size=v_regions['count']*1.5,
            sizemin=3,
        ),

    )
)

pyo.plot(fig, config=config)
# filename = "map_index_"+str(ix_interest)
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)

#%% SENSE CHECKING THAT ALL OF THE RANDOM PLACES PEOPLE GO ARE LEGIT

# it's a bit odd that there are so many different unique places between two stations

# journeys_V is the variable with all visited locations
# journeys_output is the processed one before this - this is what we need

# can get card that uses the common pair from journeys_interest

journeys_interest['OffLocation_ID'].value_counts()

ID_interest = 928
journeys_interest['Cardid'][journeys_interest['OffLocation_ID']==928].value_counts()

#%% SENSE CHECKING - INDIVIDUAL CARD

card_interest = 13103635
journeys_map = journeys_output[journeys_output['Cardid']==card_interest]

# first plot all spatial clusters used

geo['spatial_cluster'] = geo['spatial_cluster'].astype(float)
journeys_map['OnLocation_cluster'] = journeys_map['OnLocation_cluster'].astype(float)
journeys_map['OffLocation_cluster'] = journeys_map['OffLocation_cluster'].astype(float)

geo_map = geo[(geo['spatial_cluster'].isin(journeys_map['OnLocation_cluster']))
              |(geo['spatial_cluster'].isin(journeys_map['OffLocation_cluster']))]


geo_map['spatial_cluster'] = geo_map['spatial_cluster'].astype(str)

fig = px.scatter_map(geo_map, 
                     lat=geo_map.geometry.y, 
                     lon=geo_map.geometry.x,     
                     color="spatial_cluster", 
                     hover_name="spatial_cluster",
                     center=map_center,
                     color_discrete_sequence=mycolors_discrete,
                     zoom=10)
fig.update_layout(map_style="light", coloraxis_showscale=False)
fig.update_traces(marker=dict(size=10))

pyo.plot(fig, config=config)


journeys_map[['OnDate','OffLocation_cluster','OffLocation_cluster_next']]

journeys_V[journeys_V['Cardid']==card_interest]


#%% CALCULATE PAIRWISE DISTANCES FROM UNDERLYING GRAPH

# graph G is the networkx version of the underlying graph - use this to calculate distances
# so we want to do a pairwise distance for each of the candidate_stops, looking up in G

start = timeit.default_timer()
length = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='Duration_round'))
print("Complete in " + str((timeit.default_timer() - start)) + " seconds")
# this took 40 minutes

distances = pd.DataFrame.from_dict(length, orient='index')
distances.to_pickle(dirname + "20260221-distmatrix.pkl")
# this is a df where the distance is the distance FROM the
# first column TO the cluster along the columns of the first row

#%% READ PAIRWISE DISTANCES OF UNDERLYING GRAPH

distances = pd.read_pickle(dirname + "20260221-distmatrix.pkl")
# this is a df where the distance is the distance FROM the
# first column TO the cluster along the columns of the first row

#%% SETUP

# can explode then put back again
# explode
# then groupby on the index groupby(level=0).agg(list)
uniques_all = uniques['spatial_cluster'].explode()
uniques_all = pd.DataFrame(uniques_all)
uniques_all['spatial_cluster'] = uniques_all['spatial_cluster'].astype(float)

# this gives a df where the index is the new region ID
journeys_on = journeys_unique['OnLocation_ID'].explode()
journeys_on = pd.DataFrame(journeys_on)
# so this is the on location - we want the distance TO the candidate location

journeys_off = journeys_unique['OffLocation_ID_next'].explode()
journeys_off = pd.DataFrame(journeys_off)
# so this is the off location - we want the distance FROM the candidate location

# index of these is the index in journeys_unique

#%% CALCULATE DISTANCES AS REGIONS INSTEAD OF SPATIAL CLUSTERS

start = timeit.default_timer()

# these come up with some NaNs because uniques contains regions
# outside Perth metro while distances doesn't.
# but since we won't be looking at those locations later anyway, it doesn't really matter

# A - candidate - B
# this is distances FROM the region ID to the candidate location (so A-candidate above)
distances_to = pd.merge(uniques_all,distances, left_on='spatial_cluster',right_index=True,how='left')
distances_to.reset_index(inplace=True)
distances_to.rename(columns={'index':'ID'}, inplace=True)
distances_to = distances_to.groupby("ID").min()
distances_to.drop(columns='spatial_cluster',inplace=True)

distances = distances.transpose()

# this is distances FROM the candidate location to the region ID (so candidate-B above)
distances_from = pd.merge(uniques_all,distances, left_on='spatial_cluster',right_index=True,how='left')
distances_from.reset_index(inplace=True)
distances_from.rename(columns={'index':'ID'}, inplace=True)
distances_from = distances_from.groupby("ID").min()
distances_from.drop(columns='spatial_cluster',inplace=True)

distances_to.to_pickle(dirname+"20260307-distances_to.pkl")
distances_from.to_pickle(dirname+"20260307-distances_from.pkl")

print("Complete in " + str((timeit.default_timer() - start)) + " seconds")
# this took... 4 minutes

#%% READ REGION DISTANCES

distances_to = pd.read_pickle(dirname+"20260307-distances_to.pkl")
distances_from = pd.read_pickle(dirname+"20260307-distances_from.pkl")

#%% SUMMARISING THE CURRENT TRAVEL TIMES

# Explode each column separately, keeping only relevant columns
journeys_agg_left = journeys_agg_all.drop(columns=['Cardid','OffLocation_ID_next','Off_count'])
exploded_on = journeys_agg_left.explode(['OnLocation_ID','On_count'])

journeys_on_1 = exploded_on.reset_index().merge(uniques[['ID','spatial_cluster']], left_on='OnLocation_ID', right_on='ID', how='left').set_index('index')
journeys_on_1.drop(columns='ID', inplace=True)
journeys_on_1.rename(columns={'spatial_cluster':'On_cluster'}, inplace=True)

journeys_on_1 = journeys_on_1.reset_index().merge(uniques[['ID','spatial_cluster']], left_on='OffLocation_ID', right_on='ID', how='left').set_index('index')
journeys_on_1.drop(columns='ID', inplace=True)
journeys_on_1.rename(columns={'spatial_cluster':'Off_cluster'}, inplace=True)

journeys_on_exp = journeys_on_1.explode('On_cluster')
journeys_on_exp = journeys_on_exp.explode('Off_cluster')

journeys_on_exp['On_cluster'] = journeys_on_exp['On_cluster'].astype(float)
journeys_on_exp['Off_cluster'] = journeys_on_exp['Off_cluster'].astype(float)

def get_cluster_on_distance(row):
    try:
        dist = distances.loc[row['On_cluster'], row['Off_cluster']].min()
        return dist
    except KeyError:
        return np.nan # this is needed to catch where we still have some outside the Perth metro

start = timeit.default_timer()
journeys_on_exp['dist'] = journeys_on_exp.apply(get_cluster_on_distance, axis=1)
print("Complete in " + str(timeit.default_timer() - start) + " seconds")
# 9 seconds

# groupby index, OnLocation_ID, OffLocation_ID and take the minimum
journeys_on_exp1 = journeys_on_exp.reset_index().groupby(['index','OnLocation_ID','OffLocation_ID','On_count']).agg({'dist':'min'})

journeys_on_exp1.reset_index(inplace=True)
journeys_on_exp1['dist_total'] = journeys_on_exp1['dist']*journeys_on_exp1['On_count']

# then groupby index and take the sum
journeys_on_exp1 = journeys_on_exp1.groupby('index').agg({'dist':'sum', 'dist_total':'sum'})

# and can now put this back onto journeys_agg_all
journeys_agg_all = pd.merge(journeys_agg_all, journeys_on_exp1[['dist', 'dist_total']], left_index=True, right_index=True, how='left')
journeys_agg_all.rename(columns={'dist':'dist_on', 'dist_total':'dist_total_on'}, inplace=True)

# and now do the same for the other side of the visited activities
exploded_off = journeys_agg_all[['OffLocation_ID','OffLocation_ID_next','Off_count']].explode(['OffLocation_ID_next','Off_count'])

journeys_off_1 = exploded_off.reset_index().merge(uniques[['ID','spatial_cluster']], left_on='OffLocation_ID_next', right_on='ID', how='left').set_index('index')
journeys_off_1.drop(columns='ID', inplace=True)
journeys_off_1.rename(columns={'spatial_cluster':'Off_next_cluster'}, inplace=True)

journeys_off_1 = journeys_off_1.reset_index().merge(uniques[['ID','spatial_cluster']], left_on='OffLocation_ID', right_on='ID', how='left').set_index('index')
journeys_off_1.drop(columns='ID', inplace=True)
journeys_off_1.rename(columns={'spatial_cluster':'Off_cluster'}, inplace=True)

journeys_off_exp = journeys_off_1.explode('Off_cluster')
journeys_off_exp = journeys_off_exp.explode('Off_next_cluster')

journeys_off_exp['Off_cluster'] = journeys_off_exp['Off_cluster'].astype(float)
journeys_off_exp['Off_next_cluster'] = journeys_off_exp['Off_next_cluster'].astype(float)


def get_cluster_off_distance(row):
    try:
        dist = distances.loc[row['Off_cluster'], row['Off_next_cluster']].min()
        return dist
    except KeyError:
        return np.nan # this is needed to catch where we still have some outside the Perth metro

start = timeit.default_timer()
journeys_off_exp['dist'] = journeys_off_exp.apply(get_cluster_off_distance, axis=1)
print("Complete in " + str(timeit.default_timer() - start) + " seconds")
# 9 seconds

# groupby index, OnLocation_ID, OffLocation_ID and take the minimum
journeys_off_exp1 = journeys_off_exp.reset_index().groupby(['index','OffLocation_ID','OffLocation_ID_next','Off_count']).agg({'dist':'min'})

journeys_off_exp1.reset_index(inplace=True)
journeys_off_exp1['dist_total'] = journeys_off_exp1['dist']*journeys_off_exp1['Off_count']

# then groupby index and take the sum
journeys_off_exp1 = journeys_off_exp1.groupby('index').agg({'dist':'sum', 'dist_total':'sum'})

# and can now put this back onto journeys_agg_all
journeys_agg_all = pd.merge(journeys_agg_all, journeys_off_exp1[['dist', 'dist_total']], left_index=True, right_index=True, how='left')
journeys_agg_all.rename(columns={'dist':'dist_off', 'dist_total':'dist_total_off'}, inplace=True)

journeys_agg_all['dist'] = journeys_agg_all['dist_on']+journeys_agg_all['dist_off']
journeys_agg_all['dist_total'] = journeys_agg_all['dist_total_on']+journeys_agg_all['dist_total_off']

journeys_agg_all = journeys_agg_all[~journeys_agg_all['dist'].isna()]

distances_dist = journeys_agg_all.groupby(['OnLocation_ID',"OffLocation_ID_next","On_count",'Off_count']).agg({'dist':['min', 'median', 'mean','max'], 'dist_total':['min', 'median', 'mean','max']})
distances_dist.columns = [' '.join(col).strip() for col in distances_dist.columns.values]

journeys_unique = pd.merge(journeys_unique, distances_dist, on=['OnLocation_ID','OffLocation_ID_next', "On_count",'Off_count'], how='left')

# get number of unique places currently used as visited regions
unique_visited_regions = journeys_agg_all.groupby(['OnLocation_ID','OffLocation_ID_next','On_count','Off_count'])['OffLocation_ID'].nunique()
unique_visited_regions = pd.DataFrame(unique_visited_regions)
unique_visited_regions.rename(columns={'OffLocation_ID':'num_diff_visited'}, inplace=True)
unique_visited_regions.reset_index(inplace=True)

journeys_unique = pd.merge(journeys_unique, unique_visited_regions, on=['OnLocation_ID','OffLocation_ID_next','On_count','Off_count'], how='left')

journeys_agg_all.to_pickle(dirname+"20260307-journeys-agg-all.pkl")
journeys_unique.to_pickle(dirname+"20260307-journeys-unique.pkl")

#%% READ PICKLES

journeys_unique = pd.read_pickle(dirname+"20260307-journeys-unique.pkl")
journeys_agg_all = pd.read_pickle(dirname+"20260307-journeys-agg-all.pkl")


#%% GET FULL DISTANCES BY REGION COMBINATIONS (JOURNEYS_UNIQUE)

start = timeit.default_timer()

journeys_on = journeys_unique[['OnLocation_ID','On_count']].explode(['OnLocation_ID','On_count'])
journeys_on = pd.DataFrame(journeys_on)
journeys_on = pd.merge(journeys_on, distances_to, left_on='OnLocation_ID', right_index=True, how='left')

# multiply through by the edge weighting
journeys_on= journeys_on.multiply(journeys_on["On_count"], axis="index")
journeys_on.drop(columns=['OnLocation_ID','On_count'], inplace=True)

# mask all NaN with inf
# then just do a normal groupby sum
journeys_on = journeys_on.fillna(np.inf)
journeys_on = journeys_on.groupby(journeys_on.index).sum()
journeys_on = journeys_on.replace([np.inf], np.nan)

journeys_off = journeys_unique[['OffLocation_ID_next','Off_count']].explode(['OffLocation_ID_next','Off_count'])
journeys_off = pd.DataFrame(journeys_off)

journeys_off = pd.merge(journeys_off, distances_from, left_on='OffLocation_ID_next', right_index=True, how='left')

# multiply through by the edge weighting
journeys_off = journeys_off.multiply(journeys_off["Off_count"], axis="index")
journeys_off.drop(columns=['OffLocation_ID_next','Off_count'], inplace=True)

journeys_off = journeys_off.fillna(np.inf)
journeys_off = journeys_off.groupby(journeys_off.index).sum()
journeys_off = journeys_off.replace([np.inf], np.nan)

journeys_distances = journeys_on + journeys_off
journeys_distances = pd.concat([journeys_unique, journeys_distances], axis=1)

# get the first on for reference
journeys_distances['first_on'] = journeys_distances["OnLocation_ID"].str[0]

# flag the out and backs, so we can drop them later if we like
journeys_distances['not out-back check'] = journeys_distances['OnLocation_ID']!=journeys_distances['OffLocation_ID_next']

print("Complete in " + str((timeit.default_timer() - start)) + " seconds")

journeys_distances.to_pickle(dirname+"20260307-journeys-distances.pkl")


#%% DISTANCES FOR TRADITIONAL P HUB PROBLEM

# for traditional p hub, it's going to be basically just the same as the above
# but just for journeys_on using the 'first_on' column
# and for journeys_off using the first_off_next column (so the first on each side)

journeys_on = journeys_unique["OnLocation_ID"].str[0]
journeys_on = pd.DataFrame(journeys_on)
journeys_on['On_count'] = journeys_unique['On_count'].str[0]
journeys_on = pd.merge(journeys_on, distances_to, left_on='OnLocation_ID', right_index=True, how='left')

journeys_on = journeys_on.multiply(journeys_on["On_count"], axis="index")

journeys_on.drop(columns=['On_count'], inplace=True)

journeys_off = journeys_unique['OffLocation_ID_next'].str[0]
journeys_off = pd.DataFrame(journeys_off)
journeys_off['Off_count'] = journeys_unique['Off_count'].str[0]
journeys_off = pd.merge(journeys_off, distances_from, left_on='OffLocation_ID_next', right_index=True, how='left')

journeys_off = journeys_off.multiply(journeys_off["Off_count"], axis="index")
journeys_off.drop(columns=['Off_count'], inplace=True)


journeys_distances_trad = journeys_on + journeys_off
journeys_distances_trad.drop(columns=['OffLocation_ID_next','OnLocation_ID'], inplace=True)
journeys_distances_trad = pd.concat([journeys_unique, journeys_distances_trad], axis=1)

journeys_distances_trad['not out-back check'] = journeys_distances_trad['OnLocation_ID']!=journeys_distances_trad['OffLocation_ID_next']

journeys_distances_trad.to_pickle(dirname+"20260307-journeys-distances-trad.pkl")


#%% DISTANCES FOR USING 'HOME' LOCATION

journeys_cards = pd.read_pickle(dirname+'20260307-journeys-cards.pkl')
# we have 83,613 of these with a home ID

journeys_cards['Cardid'].value_counts()
# there are 46,092 cards

journeys_cards['homeID'].value_counts()
# and there are 16,062different regions used as home regions

# need to merge journeys_cards onto journeys_unique to get the bbox checks
journeys_cards = pd.merge(journeys_cards, journeys_unique[['OnLocation_ID',
                                                           'OffLocation_ID_next',
                                                           'NOR check', 'SOR check', 
                                                           'jdlp check', 
                                                           'outside CBD check', 
                                                           'bbox check']].drop_duplicates(), on=['OnLocation_ID','OffLocation_ID_next'], how='left')

journeys_distances_home = pd.merge(journeys_cards, distances_to, left_on='homeID', right_index=True, how='left')
journeys_distances_home['not out-back check'] = journeys_distances_home['OnLocation_ID']!=journeys_distances_home['OffLocation_ID_next']

journeys_distances_home = journeys_distances_home[journeys_distances_home['bbox check']==True]

# this is 83,585
journeys_distances_home.to_pickle(dirname+"20260307-journeys-distances-home.pkl")

#%% TIDY UP DF_NODES
# some distances are all NaN
# so we don't want them to be in our candidate locations

def count_na_dist(row):
    return journeys_distances_trad[row['clusterID']].isna().sum()
    
start = timeit.default_timer()
df_nodes['count_na'] = df_nodes.apply(count_na_dist, axis=1)

df_nodes = df_nodes[df_nodes['count_na']==0]
df_nodes.drop(columns=['count_na'], inplace=True)

# also drop showgrounds station given it's only a special events station
df_nodes = df_nodes[df_nodes['clusterID']!=8402]

df_nodes.to_pickle(dirname+"20260228-df-nodes.pkl")

#%% READ JOURNEYS_DISTANCES AND DF_NODES

journeys_distances = pd.read_pickle(dirname+"20260307-journeys-distances.pkl")
journeys_distances_trad = pd.read_pickle(dirname+"20260307-journeys-distances-trad.pkl")
journeys_distances_home = pd.read_pickle(dirname+"20260307-journeys-distances-home.pkl")

df_nodes = pd.read_pickle(dirname+"20260228-df-nodes.pkl")

#%% P MEDIAN PROBLEM FUNCTIONS

def solve_pmed_existing(journeys_input, demand_nodes, df_nodes, p):
    
    existing_open_facilities = df_nodes['clusterID'][df_nodes['is_open']==True]
    candidate_locations = df_nodes['clusterID'][df_nodes['is_open']==False]
    potential_facilities = pd.concat([existing_open_facilities,candidate_locations])
    
    start = timeit.default_timer()
     
    # Create the problem
    prob = pulp.LpProblem("P-Median", pulp.LpMinimize)
    solver = pulp.CPLEX_CMD()
    print("here 0")
    
    # Decision variables
    # x_ij = 1 if demand node i is assigned to facility j, 0 otherwise
    X = pulp.LpVariable.dicts("Assign", ((i, j) for i in demand_nodes for j in potential_facilities), cat='Binary')
    print("here 1")
    
    # y_j = 1 if facility j is open, 0 otherwise
    Y = pulp.LpVariable.dicts("Open", potential_facilities, cat='Binary')
    print("here 2")
     
    # Objective function: Minimize total demand-weighted cost
    prob += pulp.lpSum(journeys_input.loc[i, j] * X[(i, j)] for i in demand_nodes for j in potential_facilities), "Total Cost"
    print("here 3")
     
    # Constraints
    # 1. Each demand node must be assigned to exactly one open facility
    for i in demand_nodes:
        prob += pulp.lpSum(X[(i, j)] for j in potential_facilities) == 1, f"Assign_Demand_{i}"
    print("here 4")
     
    # 2. A demand node can only be assigned to an open facility
    for i in demand_nodes:
        for j in potential_facilities:
            prob += X[(i, j)] <= Y[j], f"Assignment_to_Open_Facility_{i}_{j}"
    print("here 5")
     
    # 3. Exactly 'p' facilities must be opened in addition to those already open
    prob += pulp.lpSum(Y[j] for j in potential_facilities) == p+len(existing_open_facilities), "Number_of_Facilities"
    print("here 6")
 
    # # 4. Constraint for existing open facilities
    for j in existing_open_facilities:
        prob += Y[j] == 1, f"Existing_Facility_{j}_Must_Be_Open"
    print("here 7")
    
    # Solve the problem
    prob.solve(solver)
     
    # Print results
    print("Status:", pulp.LpStatus[prob.status])

    assignments = []
    for i in demand_nodes:
        for j in potential_facilities:
            if X[(i, j)].varValue == 1:
                assignments.append({'Demand':i,'Assigned':j})
    assignments = pd.DataFrame(assignments)
                  
    print("Complete in " + str(timeit.default_timer() - start) + " seconds")
    print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

    return assignments

def create_inputs(df_nodes, journeys_distances_home, journeys_distances, journeys_distances_trad, location_str, save_str):
    df_nodes_input = df_nodes[(df_nodes[location_str+' check']==True)&(df_nodes['outside CBD check']==True)]
    df_nodes_input['percent'] = df_nodes_input['count_V']/df_nodes_input['count_V'].sum()*100
    df_nodes_input.sort_values(by='count_V', inplace=True, ascending=False)
    df_nodes_input['cum_percent'] = df_nodes_input['percent'].cumsum() 
    
    print("Candidate locations (df_nodes)")
    print("Total length:", len(df_nodes_input))
    print("Not open:", len(df_nodes_input[df_nodes_input['is_open']==False]))
    print("Already open:", len(df_nodes_input[df_nodes_input['is_open']==True]))
    print("Open, used twice or more:", len(df_nodes_input[(df_nodes_input['count_V']>1)]))
    print("Open, contribute >0.01%:", len(df_nodes_input[(df_nodes_input['percent']>0.01)]))
    print("Open, contribute >0.1%:", len(df_nodes_input[(df_nodes_input['percent']>0.1)]))
    print("Open, contribute >1%:", len(df_nodes_input[(df_nodes_input['percent']>1)]))
    
    df_nodes_input = df_nodes_input[(df_nodes_input['percent']>0.1)|(df_nodes_input['is_open']==False)]
    existing_open_facilities = df_nodes_input['clusterID'][df_nodes_input['is_open']==True]
    candidate_locations = df_nodes_input['clusterID'][df_nodes_input['is_open']==False]
    potential_facilities = pd.concat([existing_open_facilities,candidate_locations])

    print("Potential facilities", len(potential_facilities))
    
    max_size = 7200000
    # this is approximately what I've worked out the desktop can handle memory-wise
    max_rows = int(max_size/len(potential_facilities))
    
    # FIRST: HOME LOCATION
    journeys_input = journeys_distances_home[(journeys_distances_home[location_str+' check']==True)&
                       (journeys_distances_home['outside CBD check']==True)]
    
    journeys_input['percent'] = journeys_input['count']/journeys_input['count'].sum()*100
    journeys_input.sort_values(by='count', inplace=True, ascending=False)
    journeys_input['cum_percent'] = journeys_input['percent'].cumsum()
    
    print("Demand locations (home location)")
    print("Total length:", len(journeys_input))
    print("Used twice or more:", len(journeys_input[(journeys_input['count']>1)]))
    print("Used three times or more:", len(journeys_input[(journeys_input['count']>2)]))
    print(journeys_input['count'].sum(), "records")
    print("Max rows:", max_rows)
    
    journeys_input = journeys_input[0:max_rows]
    # journeys_input is already sorted by count
    # so if we take the max rows we'll get all the most frequent ones
    # plus as many of the infrequent ones as we can handle
    print(journeys_input['count'].sum(), "records used with max rows")
    
    # create summary of the locations and count used
    # this becomes the basis for the other inputs
    # no other filtering or chopping down required, as that's already been done
    location_inputs = journeys_input[['OnLocation_ID','OffLocation_ID_next','On_count','Off_count']]
    location_inputs['count'] = journeys_input.groupby(['OnLocation_ID','OffLocation_ID_next','On_count','Off_count'])['count'].transform('sum')
    location_inputs = location_inputs.drop_duplicates()

    print(location_inputs['count'].sum(), "records used")
    # this should be the same number as above
    
    journeys_input = journeys_input.multiply(journeys_input["count"], axis="index")
    journeys_input = journeys_input.loc[:, journeys_input.columns.isin(potential_facilities)]    
    journeys_input.to_pickle(dirname+str(max_size)+"-journeys-input-"+save_str+"-home.pkl")
    
    # SECOND: MY APPROACH 
    journeys_input = pd.merge(location_inputs, journeys_distances, on=['OnLocation_ID', 'OffLocation_ID_next','On_count','Off_count'], how='left')
    journeys_input.rename(columns={'count_x':'count'}, inplace=True)
    
    journeys_input['percent'] = journeys_input['count']/journeys_input['count'].sum()*100
    journeys_input.sort_values(by='count', inplace=True, ascending=False)
    journeys_input['cum_percent'] = journeys_input['percent'].cumsum()
    
    print("Demand locations (mine+traditional)")
    print("Total length:", len(journeys_input))
    print(journeys_input['count'].sum(), "records used")
    
    # journeys_distances has already manually had the counts put on, so don't need it here    
    journeys_input = journeys_input.loc[:, journeys_input.columns.isin(potential_facilities)]     
    journeys_input.to_pickle(dirname+str(max_size)+"-journeys-input-"+save_str+"-mine.pkl")
    
    # THIRD: TRADITIONAL P-HUB APPROACH
    journeys_input = pd.merge(location_inputs, journeys_distances_trad, on=['OnLocation_ID', 'OffLocation_ID_next', 'On_count', 'Off_count'], how='left')
    journeys_input.rename(columns={'count_x':'count'}, inplace=True)
    print(journeys_input['count'].sum(), "records used")
    
    journeys_input = journeys_input.multiply(journeys_input["count"], axis="index")
    journeys_input = journeys_input.loc[:, journeys_input.columns.isin(potential_facilities)]
    journeys_input.to_pickle(dirname+str(max_size)+"-journeys-input-"+save_str+"-trad.pkl")


def run_algorithm(filename_suffix, data_suffix, run_type, pvals, max_rows):
    journeys_input = pd.read_pickle(dirname+str(max_rows)+"-journeys-input-"+filename_suffix+"-"+run_type+".pkl")
    df_nodes_input = df_nodes[(df_nodes[data_suffix+' check']==True)&(df_nodes['outside CBD check']==True)]
    df_nodes_input['percent'] = df_nodes_input['count_V']/df_nodes_input['count_V'].sum()*100
    df_nodes_input.sort_values(by='count_V', inplace=True, ascending=False)
    df_nodes_input['cum_percent'] = df_nodes_input['percent'].cumsum()
    
    for i in np.arange(0,len(pvals)):
        assignments = solve_pmed_existing(
            journeys_input,
            journeys_input.index,
            df_nodes_input[(df_nodes_input['percent']>0.1)|(df_nodes_input['is_open']==False)], 
            p=pvals[i])
    
        assignments.to_pickle(dirname+str(max_rows)+"-assignments-"+filename_suffix+"-"+str(pvals[i])+"-"+run_type+".pkl")
        print("p value", pvals[i],"complete")
    
    del journeys_input
    
#%% CREATE THE INPUTS

create_inputs(df_nodes, journeys_distances_home, journeys_distances, 
                  journeys_distances_trad, 'jdlp', 'jdlp')

create_inputs(df_nodes, journeys_distances_home, journeys_distances, 
                  journeys_distances_trad, 'NOR', 'NOR')

create_inputs(df_nodes, journeys_distances_home, journeys_distances, 
                  journeys_distances_trad, 'SOR', 'SOR')

create_inputs(df_nodes, journeys_distances_home, journeys_distances, 
                  journeys_distances_trad, 'outside CBD', 'outCBD')

#%% RUN THE ALGORITHM FOR JDLP


max_rows = 7200000
pvals = [5, 10, 15, 20, 25, 30]

filename_suffix = "jdlp"
data_suffix = "jdlp"

run_algorithm(filename_suffix, data_suffix, "trad", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "mine", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "home", pvals, max_rows)

#%% RUN REST OF LOCATIONS

filename_suffix = "NOR"
data_suffix = "NOR"

run_algorithm(filename_suffix, data_suffix, "trad", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "mine", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "home", pvals, max_rows)


filename_suffix = "SOR"
data_suffix = "SOR"

run_algorithm(filename_suffix, data_suffix, "trad", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "mine", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "home", pvals, max_rows)

filename_suffix = "outCBD"
data_suffix = "outside CBD"

run_algorithm(filename_suffix, data_suffix, "trad", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "mine", pvals, max_rows)
run_algorithm(filename_suffix, data_suffix, "home", pvals, max_rows)


#%% FUNCTION TO GET RESULTS

def generate_results(assignments, assignments_trad, assignments_home, journeys_distances_home, 
                     journeys_distances, p, location_str, journeys_agg_all):
    
    df_nodes_output = df_nodes[(df_nodes[location_str+' check']==True)&(df_nodes['outside CBD check']==True)]
    df_nodes_output['percent'] = df_nodes_output['count_V']/df_nodes_output['count_V'].sum()*100

    df_nodes_output['pmed_open'] = 'Closed'
    df_nodes_output['pmed_open'] = np.where(df_nodes_output['percent']>0.1, 'Existing', df_nodes_output['pmed_open'])
    
    
    journeys_results = journeys_distances_home[
                        (journeys_distances_home[location_str+' check']==True)&
                        (journeys_distances_home['outside CBD check']==True)]
    
    journeys_results['percent'] = journeys_results['count']/journeys_results['count'].sum()*100
    journeys_results.sort_values(by='count', inplace=True, ascending=False)
    journeys_results['cum_percent'] = journeys_results['percent'].cumsum()
    
    max_size = 7200000
    # this is approximately what I've worked out the desktop can handle memory-wise
    num_potential = len(df_nodes_output[df_nodes_output['pmed_open']=='Existing'])+len(df_nodes_output[df_nodes_output['is_open']==False])
    max_rows = int(max_size/num_potential)
    journeys_results = journeys_results[0:max_rows]
  
    journeys_results = journeys_results[['OnLocation_ID','OffLocation_ID_next',
                                         'Cardid','homeID','count','On_count','Off_count']]
    journeys_results['assignments_home'] = assignments_home['Assigned'].values
 
    location_inputs = journeys_results[['OnLocation_ID','OffLocation_ID_next','On_count','Off_count']]
    location_inputs = location_inputs.drop_duplicates()
    location_inputs.reset_index(inplace=True)
        
    location_inputs = pd.merge(location_inputs, assignments, left_index=True, right_on='Demand', how='left')
    location_inputs.drop(columns='Demand', inplace=True)
    location_inputs.rename(columns={'Assigned':'assignments'}, inplace=True)
    location_inputs.reset_index(inplace=True)
       
    location_inputs = pd.merge(location_inputs, assignments_trad, left_index=True, right_on='Demand', how='left')
    location_inputs.drop(columns='Demand', inplace=True)
    location_inputs.rename(columns={'Assigned':'assignments_trad'}, inplace=True)
    
    journeys_results = pd.merge(journeys_results, location_inputs, 
                                on=['OnLocation_ID', 'OffLocation_ID_next', 'On_count', 'Off_count'], how='left')

    # for each of the assigned values (3 columns) in journeys_results, we want 
    # to look up the distance in journeys_distances (NOT HOME OR TRAD)
    journeys_distances_short = pd.merge(journeys_results, journeys_distances, 
                                        on=['OnLocation_ID', 'OffLocation_ID_next', 'On_count', 'Off_count'], 
                                        how='left')

    col_indexer = journeys_distances_short.columns.get_indexer(journeys_results['assignments'])
    journeys_results['dist_assigned'] = journeys_distances_short.values[range(len(journeys_distances_short)), col_indexer]

    col_indexer = journeys_distances_short.columns.get_indexer(journeys_results['assignments_trad'])
    journeys_results['dist_assigned_trad'] = journeys_distances_short.values[range(len(journeys_distances_short)), col_indexer]

    col_indexer = journeys_distances_short.columns.get_indexer(journeys_results['assignments_home'])
    journeys_results['dist_assigned_home'] = journeys_distances_short.values[range(len(journeys_distances_short)), col_indexer]

    journeys_results.drop(columns=['level_0','index'], inplace=True)

    distances_dist = journeys_agg_all.groupby(['Cardid','OnLocation_ID',"OffLocation_ID_next",'On_count','Off_count']).agg({'current_dist_total':['min', 'median', 'mean','max']})
    distances_dist.columns = [' '.join(col).strip() for col in distances_dist.columns.values]

    journeys_results = pd.merge(journeys_results, distances_dist, 
                                on=['Cardid','OnLocation_ID','OffLocation_ID_next','On_count','Off_count'], 
                                how='left')

    journeys_results['dist_diff_trad'] = journeys_results['dist_assigned_trad']-journeys_results['dist_assigned']
    journeys_results['dist_diff_home'] = journeys_results['dist_assigned_home']-journeys_results['dist_assigned']
    
    journeys_results['on_contains_homeID'] = journeys_results.apply(lambda row: row['homeID'] in row['OnLocation_ID'], axis=1)
    journeys_results['offnext_contains_homeID'] = journeys_results.apply(lambda row: row['homeID'] in row['OffLocation_ID_next'], axis=1)
    journeys_results['either_homeID'] = journeys_results['on_contains_homeID']|journeys_results['offnext_contains_homeID']
    
    df_nodes_output['pmed_open'] = np.where((df_nodes_output['clusterID'].isin(journeys_results['assignments']))&(df_nodes_output['is_open']==False),"New",df_nodes_output['pmed_open'])

    allocated_stops = journeys_results[['assignments','count']].groupby('assignments').sum()
    df_nodes_output = pd.merge(df_nodes_output, allocated_stops, left_on='clusterID', right_index=True, how='left')
    
    journeys_agg_dist = pd.merge(journeys_results[['Cardid','OnLocation_ID', 'OffLocation_ID_next', 'dist_assigned',
    'dist_assigned_trad', 'dist_assigned_home', 'assignments', 'assignments_trad', 'assignments_home','On_count','Off_count']], journeys_agg_all, on=['Cardid','OnLocation_ID', 'OffLocation_ID_next','On_count','Off_count'], how='left')
    # this will give a LOT of NAs, pending which iteration you're working with
    # journeys_agg_dist is the full dataset, but journeys_results is just the version you've run with
    journeys_agg_dist = journeys_agg_dist[~journeys_agg_dist['dist_assigned'].isna()]
    journeys_agg_dist.drop(columns=['dist_on','dist_off', 'count'], inplace=True) 
    
    # note that journeys_agg_dist is longer than journeys_results
    # this is because a card can go to multiple different places, each will appear here
    # but the card will only appear once in journeys_results
    # length of journeys_agg_dist = sum of journeys_results['count']

    journeys_agg_dist['dist_assigned'] = journeys_agg_dist['dist_assigned'].astype(float)
    journeys_agg_dist['dist_assigned_trad'] = journeys_agg_dist['dist_assigned_trad'].astype(float)
    journeys_agg_dist['dist_assigned_home'] = journeys_agg_dist['dist_assigned_home'].astype(float)

    journeys_agg_dist['time_saved'] = journeys_agg_dist['current_dist_total'] - journeys_agg_dist['dist_assigned']
    journeys_agg_dist['time_saved_trad'] = journeys_agg_dist['current_dist_total'] - journeys_agg_dist['dist_assigned_trad']
    journeys_agg_dist['time_saved_home'] = journeys_agg_dist['current_dist_total'] - journeys_agg_dist['dist_assigned_home']

    journeys_agg_dist['dist_diff_trad'] = journeys_agg_dist['dist_assigned_trad'] - journeys_agg_dist['dist_assigned']
    journeys_agg_dist['dist_diff_home'] = journeys_agg_dist['dist_assigned_home'] - journeys_agg_dist['dist_assigned']

    return df_nodes_output, journeys_results, journeys_agg_dist

#%% GET SOME RESULTS

# to get current travel times at the card level
journeys_agg_all = pd.read_pickle(dirname+"20260307-journeys-agg-all.pkl")
journeys_agg_all.rename(columns={'dist_total':'current_dist_total'}, inplace=True)

p=10
filename_suffix='outCBD'
location_str = 'outside CBD'

assignments = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-mine.pkl")
assignments_trad = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-trad.pkl")
assignments_home = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-home.pkl")

df_nodes_output_10, journeys_results_10, journeys_agg_dist_10 = generate_results(assignments, assignments_trad, assignments_home, journeys_distances_home,
                                                     journeys_distances, p=p, location_str=location_str, journeys_agg_all=journeys_agg_all)

p=20
assignments = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-mine.pkl")
assignments_trad = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-trad.pkl")
assignments_home = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-home.pkl")

df_nodes_output_20, journeys_results_20, journeys_agg_dist_20 = generate_results(assignments, assignments_trad, assignments_home, journeys_distances_home,
                                                     journeys_distances, p=p, location_str=location_str, journeys_agg_all=journeys_agg_all)

p=30
assignments = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-mine.pkl")
assignments_trad = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-trad.pkl")
assignments_home = pd.read_pickle(dirname+"pmed results\\7200000-assignments-"+filename_suffix+"-"+str(p)+"-home.pkl")

df_nodes_output_30, journeys_results_30, journeys_agg_dist_30 = generate_results(assignments, assignments_trad, assignments_home, journeys_distances_home,
                                                     journeys_distances, p=p, location_str=location_str, journeys_agg_all=journeys_agg_all)

#%% COMBINE VALUES OF P TOGETHER

journeys_agg_dist_10['p'] = 10
journeys_agg_dist_20['p'] = 20
journeys_agg_dist_30['p'] = 30

journeys_results_10['p'] = 10
journeys_results_20['p'] = 20
journeys_results_30['p'] = 30

df_nodes_output_10['p'] = 10
df_nodes_output_20['p'] = 20
df_nodes_output_30['p'] = 30

combined_agg_dist = pd.concat([journeys_agg_dist_10, journeys_agg_dist_20, journeys_agg_dist_30], axis=0)
combined_journeys_results = pd.concat([journeys_results_10, journeys_results_20, journeys_results_30], axis=0)
combined_df_nodes_output = pd.concat([df_nodes_output_10, df_nodes_output_20, df_nodes_output_30], axis=0)

#%% COMPARING ALGORITHMS (THE DATA SCIENCE RESULTS)

combined_agg_dist['num_origin'] =  combined_agg_dist['OnLocation_ID'].str.len()
combined_agg_dist['num_destination'] = combined_agg_dist['OffLocation_ID_next'].str.len()
print(location_str, "location")
print(len(combined_agg_dist[combined_agg_dist['p']==10]), "total demand in dataset")

journeys_agg_card = combined_agg_dist[['Cardid','p','num_origin','num_destination','dist_assigned','dist_assigned_home','dist_assigned_trad','dist_diff_trad','dist_diff_home']].groupby(['Cardid','p','num_origin','num_destination']).sum()
journeys_agg_card.reset_index(inplace=True)
num_cards = len(journeys_agg_card['Cardid'].unique())
print(num_cards,"cards in set")

print("vs home")
for p_explore in [10, 20, 30]:
    print("for p=", p_explore)
    journeys_agg_card_p = journeys_agg_card[journeys_agg_card['p']==p_explore]
    journeys_agg_card_p = journeys_agg_card_p.groupby('Cardid').sum()
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']>0]),"cards with reduced time,",
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']>0])/num_cards*100,"percent")
    print(journeys_agg_card_p['dist_assigned_home'][journeys_agg_card_p['dist_diff_home']>0].median(), "median travel time (home)")
    print(journeys_agg_card_p['dist_assigned'][journeys_agg_card_p['dist_diff_home']>0].median(), "median travel time (mine)")
    print(journeys_agg_card_p['dist_diff_home'][journeys_agg_card_p['dist_diff_home']>0].median(), "median time saved")    
    
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']<0]),"cards with increased time,", 
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']<0])/num_cards*100,"percent")
    print(journeys_agg_card_p['dist_assigned_home'][journeys_agg_card_p['dist_diff_home']<0].median(), "median travel time (home)")
    print(journeys_agg_card_p['dist_assigned'][journeys_agg_card_p['dist_diff_home']<0].median(), "median travel time (mine)")
    print(journeys_agg_card_p['dist_diff_home'][journeys_agg_card_p['dist_diff_home']<0].median(), "median time added")
    
    
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']==0]),"cards with no change to time,",
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_home']==0])/num_cards*100,"percent")
    
print("\nvs trad")
journeys_agg_card_trad = journeys_agg_card[(journeys_agg_card['num_origin']!=1)|(journeys_agg_card['num_destination']!=1)]
num_cards = len(journeys_agg_card_trad['Cardid'].unique())
print(num_cards,"cards in set")

for p_explore in [10, 20, 30]:
    print("for p=", p_explore)
    journeys_agg_card_p = journeys_agg_card_trad[journeys_agg_card_trad['p']==p_explore]
    journeys_agg_card_p = journeys_agg_card_p.groupby('Cardid').sum()
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']>0]),"cards with reduced time,",
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']>0])/num_cards*100,"percent")
    print(journeys_agg_card_p['dist_assigned_trad'][journeys_agg_card_p['dist_diff_trad']>0].median(), "median travel time (trad)")
    print(journeys_agg_card_p['dist_assigned'][journeys_agg_card_p['dist_diff_trad']>0].median(), "median travel time (mine)")
    print(journeys_agg_card_p['dist_diff_trad'][journeys_agg_card_p['dist_diff_trad']>0].median(), "median time saved")    
    
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']<0]),"cards with increased time,", 
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']<0])/num_cards*100,"percent")
    print(journeys_agg_card_p['dist_assigned_trad'][journeys_agg_card_p['dist_diff_trad']<0].median(), "median travel time (trad)")
    print(journeys_agg_card_p['dist_assigned'][journeys_agg_card_p['dist_diff_trad']<0].median(), "median travel time (mine)")
    print(journeys_agg_card_p['dist_diff_trad'][journeys_agg_card_p['dist_diff_trad']<0].median(), "median time added")    
    
    print(len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']==0]),"cards with no change to time,",
          len(journeys_agg_card_p[journeys_agg_card_p['dist_diff_trad']==0])/num_cards*100,"percent")

#%% SCATTER PLOT OF TRAVEL TIME VS REFERENCE ALGORITHM TRAVEL TIME

grouped_home = journeys_agg_card[['p','dist_assigned_home','dist_assigned','Cardid']].groupby(['p','dist_assigned_home','dist_assigned']).count()
grouped_home.reset_index(inplace=True)
grouped_home.rename(columns={'Cardid':'count', 'dist_assigned_home':'dist_reference'}, inplace=True)
grouped_home['algorithm'] = 'Traditional p-median'

grouped_trad = journeys_agg_card[['p','dist_assigned_trad','dist_assigned','Cardid']].groupby(['p','dist_assigned_trad','dist_assigned']).count()
grouped_trad.reset_index(inplace=True)
grouped_trad.rename(columns={'Cardid':'count', 'dist_assigned_trad':'dist_reference'}, inplace=True)
grouped_trad['algorithm'] = 'Standard p-hub median'

grouped_results = pd.concat([grouped_home, grouped_trad])


plot_summary = grouped_results[grouped_results['dist_reference']!=grouped_results['dist_assigned']]

# manually scale
original_min = plot_summary['count'].min()
original_max = plot_summary['count'].max()
original_range = original_max - original_min

new_min = 1
new_max = 25
new_range = new_max - new_min

plot_summary['scaled_count'] = (plot_summary['count'] - original_min) / original_range * new_range + new_min
plot_summary['p'] = plot_summary['p'].astype(str)
fig = px.scatter(plot_summary, x="dist_reference", y="dist_assigned", color='p',
                 color_discrete_sequence=mycolors_discrete, size='scaled_count', size_max=new_max, 
                 facet_col='algorithm', facet_row='p', render_mode='svg')
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=new_min))

min_val = 0
max_val = 100

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='y=x line',
        line=dict(color=line_colour, width=line_width),
    ), row="all", col="all"
)

fig.update_layout(yaxis_range=[-5, max_val])
fig.update_layout(xaxis_range=[0, max_val])
fig.update_layout(showlegend=False)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.for_each_xaxis(lambda x: x.update({'title': ''}))
fig.for_each_yaxis(lambda y: y.update({'title': ''}))

fig.update_yaxes(default_layout['yaxis'])
fig.update_xaxes(default_layout['xaxis'])

fig.layout.annotations[2]['x'] = 0.99
fig.layout.annotations[2]['textangle'] = 0
fig.layout.annotations[2]['text'] = 'p = 30'

fig.layout.annotations[3]['x'] = 0.99
fig.layout.annotations[3]['textangle'] = 0
fig.layout.annotations[3]['text'] = 'p = 20'

fig.layout.annotations[4]['x'] = 0.99
fig.layout.annotations[4]['textangle'] = 0
fig.layout.annotations[4]['text'] = 'p = 10'

# add annotations
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=0.5, 
    yref='paper',
    y=-.09,
    text='Travel time with reference approach (minutes)'
)
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=-0.06, # this works best at -0.04 for html export, -0.06 otherwise
    yanchor='middle',
    yref='paper',
    y=0.5,
    textangle=-90,
    text='Travel time with multi-OD p-hub median (minutes)'
)

fig.update_layout(font=dict(size=18))
#pyo.plot(fig, config=config)

filename = "pmed_scatter"+filename_suffix
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% SCATTER PLOT OF TIME SAVED  VS REFERENCE ALGORITHM TRAVEL TIME

grouped_home = journeys_agg_card[['p','dist_diff_home','dist_assigned_home','Cardid']].groupby(['p','dist_diff_home','dist_assigned_home']).count()
grouped_home.reset_index(inplace=True)
grouped_home.rename(columns={'Cardid':'count', 'dist_diff_home':'dist_reference', 'dist_assigned_home':'dist_assigned'}, inplace=True)
grouped_home['algorithm'] = 'Traditional p-median'

grouped_trad = journeys_agg_card[['p','dist_diff_trad','dist_assigned_trad','Cardid']].groupby(['p','dist_diff_trad','dist_assigned_trad']).count()
grouped_trad.reset_index(inplace=True)
grouped_trad.rename(columns={'Cardid':'count', 'dist_diff_trad':'dist_reference','dist_assigned_trad':'dist_assigned'}, inplace=True)
grouped_trad['algorithm'] = 'Standard p-hub median'

grouped_results = pd.concat([grouped_home, grouped_trad])


#plot_summary = grouped_results[grouped_results['dist_reference']!=grouped_results['dist_assigned']]
plot_summary = grouped_results[grouped_results['dist_reference']!=0]


# manually scale
original_min = plot_summary['count'].min()
original_max = plot_summary['count'].max()
original_range = original_max - original_min

new_min = 1
new_max = 25
new_range = new_max - new_min

plot_summary['scaled_count'] = (plot_summary['count'] - original_min) / original_range * new_range + new_min
plot_summary['p'] = plot_summary['p'].astype(str)
fig = px.scatter(plot_summary, x="dist_reference", y="dist_assigned", color='p',
                 color_discrete_sequence=mycolors_discrete, size='scaled_count', size_max=new_max, 
                 facet_col='algorithm', facet_row='p', render_mode='svg')
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=new_min))

fig.update_layout(yaxis_range=[-20, 60])
fig.update_layout(xaxis_range=[-20, 60])
fig.update_layout(showlegend=False)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.for_each_xaxis(lambda x: x.update({'title': ''}))
fig.for_each_yaxis(lambda y: y.update({'title': ''}))

fig.update_yaxes(default_layout['yaxis'])
fig.update_xaxes(default_layout['xaxis'])

fig.layout.annotations[2]['x'] = 0.99
fig.layout.annotations[2]['textangle'] = 0
fig.layout.annotations[2]['text'] = 'p = 30'

fig.layout.annotations[3]['x'] = 0.99
fig.layout.annotations[3]['textangle'] = 0
fig.layout.annotations[3]['text'] = 'p = 20'

fig.layout.annotations[4]['x'] = 0.99
fig.layout.annotations[4]['textangle'] = 0
fig.layout.annotations[4]['text'] = 'p = 10'

# add annotations
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=0.5, 
    yref='paper',
    y=-.09,
    text='Time saved with multi-OD p-hub median (minutes)'
)
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=-0.06, 
    yanchor='middle',
    yref='paper',
    y=0.5,
    textangle=-90,
    text='Travel time with reference algorithm (minutes)'
)

fig.update_layout(font=dict(size=18))
pyo.plot(fig, config=config)

#%% STRIP PLOT: RESULTS VS BOTH REFERENCE CASES AT DIFFERENT VALUES OF P

# note by summing all columns most are meaningless - the card, p, distances and diff ones are fine though
# use other ones with caution
plot_agg_dist = combined_agg_dist.groupby(['Cardid','p']).sum()
plot_agg_dist.rename(columns={'dist_diff_trad':'Compared to p-hub median', 'dist_diff_home':'Compared to p-median (home locations)'}, inplace=True)
plot_agg_dist.reset_index(inplace=True)

outbox = pd.melt(
    plot_agg_dist,
    id_vars=['p'],
    value_vars=['Compared to p-hub median','Compared to p-median (home locations)'],
)

fig = px.strip(outbox, color ='p', y='value',x='variable', 
             color_discrete_sequence=mycolors_discrete, 
             labels={"value": "Travel time saved (minutes)", "variable": ""})# y="total_bill")
fig.update_yaxes(
    range=(-70, 130),
    constrain='domain'
)
strip_layout = dict(yaxis=dict(
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
            showgrid=False,
            linewidth=line_width, 
            linecolor=line_colour,
            gridcolor=line_colour,
            gridwidth=line_width,
            separatethousands=True,
            tickformat=",",
    ),
    paper_bgcolor=bg_colour,
    plot_bgcolor=bg_colour,
    font=dict(
        size=18,
        color="black"
    )
)
fig.update_layout(strip_layout)
fig.add_shape(
    type="line", xref='x', yref='y',
                    y0=-10, x0=0.5, y1=80, x1=0.5, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

filename = "pmed_stripplot"+filename_suffix
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg",width=700, height=400)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% COMPARING TO CURRENT TRAVEL TIME (THE TRANSPORT RESULTS)

journeys_agg_card = combined_agg_dist[['Cardid','p','assignments','homeID','num_origin','num_destination','dist_assigned','current_dist_total','time_saved']].groupby(['Cardid','homeID','p','num_origin','num_destination', 'assignments']).sum()
journeys_agg_card.reset_index(inplace=True)

assigned_facilities_all = pd.DataFrame()
df_nodes_output_all = pd.DataFrame()
journeys_agg_card_all = pd.DataFrame()

bins = [-np.inf, -.1, 0, 20, 40, 60, 80, 100]

for p_explore in [10, 20, 30]:
    df_nodes_output = combined_df_nodes_output[combined_df_nodes_output['p']==p_explore]
    df_nodes_output = pd.merge(df_nodes_output, clusters[['clusterID','newName']], on='clusterID', how='left')
    df_nodes_output['newName'] = df_nodes_output['newName'].str.join(' | ')
    
    counts = df_nodes_output['newName'].value_counts()
    is_duplicate = df_nodes_output['newName'].map(counts) > 1

    df_nodes_output['newName'] = df_nodes_output['newName'] + " | "+ df_nodes_output['X'].round(4).astype(str)+", "+df_nodes_output['Y'].round(4).astype(str)
    
    journeys_agg_card_p = journeys_agg_card[journeys_agg_card['p']==p_explore]
    
    assigned_facilities = journeys_agg_card_p.groupby(['assignments']).agg({'time_saved':['sum','count','min', 'median', 'mean','max']})
    assigned_facilities.reset_index(inplace=True)
    assigned_facilities.columns = [' '.join(col).strip() for col in assigned_facilities.columns.values]

    # note that count_records is the number of people allocated to this location where we've saved them time
    # not necessarily just the total allocated to that location
    assigned_facilities.rename(columns={'assignments':'clusterID',
                                        'time_saved sum':'total_saved',
                                        'time_saved min':'min saved',
                                        'time_saved median':'median saved', 
                                        'time_saved mean':'mean saved', 
                                        'time_saved max':'max saved'}, inplace=True)

    assigned_facilities = pd.merge(assigned_facilities, df_nodes_output[['clusterID','newName','X','Y','pmed_open']], on='clusterID', how='left')
    assigned_facilities['p'] = p_explore
    
    
    journeys_agg_card_p = journeys_agg_card_p.groupby(['Cardid','p']).sum()
    journeys_agg_card_p.reset_index(inplace=True)
    
    # percent of current travel time saved
    journeys_agg_card_p['perc_saved'] = journeys_agg_card_p['time_saved']/journeys_agg_card_p['current_dist_total']*100
    journeys_agg_card_p['bin'] = pd.cut(journeys_agg_card_p['perc_saved'], bins=bins, right=True, include_lowest=True)

    # For the people who we do save travel time for, how much do we save?
    fig = px.histogram(journeys_agg_card_p['perc_saved'][journeys_agg_card_p['perc_saved']>0],
                       color_discrete_sequence=[mycolors_discrete[7]])

    fig.update_layout(
        default_layout,
        yaxis_title_text = 'Number of cards',
        xaxis_title_text = 'Percentage of time saved',
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 100])

    #pyo.plot(fig, config=config)
    filename = "pmed_timesaved_dist_"+filename_suffix+str(p_explore)
    fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
    fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
    fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

    assigned_facilities_all = pd.concat([assigned_facilities_all, assigned_facilities]) 
    df_nodes_output_all = pd.concat([df_nodes_output_all, df_nodes_output]) 
    journeys_agg_card_all = pd.concat([journeys_agg_card_all, journeys_agg_card_p]) 

assigned_facilities_all.to_csv(dirname+"assigned_facilities"+filename_suffix+".csv")
df_nodes_output_all[['clusterID','is_open','pmed_open','count','newName','p']].to_csv(dirname+"df_nodes_output"+filename_suffix+".csv")
journeys_agg_card_all.to_csv(dirname+"journeys_agg_card_all"+filename_suffix+".csv")

    
#%% SCATTER PLOT OF NEW TRAVEL TIME VS CURRENT TRAVEL TIME

grouped = journeys_agg_card[['p','dist_assigned','current_dist_total','Cardid']].groupby(['p','current_dist_total','dist_assigned']).count()
grouped.reset_index(inplace=True)
grouped.rename(columns={'Cardid':'count'}, inplace=True)

# manually scale
original_min = grouped['count'].min()
original_max = grouped['count'].max()
original_range = original_max - original_min

new_min = 1
new_max = 25
new_range = new_max - new_min

grouped['scaled_count'] = (grouped['count'] - original_min) / original_range * new_range + new_min
grouped['p'] = grouped['p'].astype(str)
fig = px.scatter(grouped, x="current_dist_total", y="dist_assigned", color='p',
                 color_discrete_sequence=mycolors_discrete, size='scaled_count', size_max=new_max, 
                 facet_row='p', render_mode='svg')
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=new_min))

min_val = 0
max_val = 100

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='y=x line',
        line=dict(color=line_colour, width=line_width),
    ), row="all", col="all"
)

fig.update_layout(yaxis_range=[-5, max_val])
fig.update_layout(xaxis_range=[0, max_val])
fig.update_layout(showlegend=False)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.for_each_xaxis(lambda x: x.update({'title': ''}))
fig.for_each_yaxis(lambda y: y.update({'title': ''}))

fig.update_yaxes(default_layout['yaxis'])
fig.update_xaxes(default_layout['xaxis'])

fig.layout.annotations[0]['x'] = 0.99
fig.layout.annotations[0]['textangle'] = 0
fig.layout.annotations[0]['text'] = 'p = 30'

fig.layout.annotations[1]['x'] = 0.99
fig.layout.annotations[1]['textangle'] = 0
fig.layout.annotations[1]['text'] = 'p = 20'

fig.layout.annotations[2]['x'] = 0.99
fig.layout.annotations[2]['textangle'] = 0
fig.layout.annotations[2]['text'] = 'p = 10'

# add annotations
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=0.5, 
    yref='paper',
    y=-.09,
    text='Current travel time (minutes)'
)
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=-0.04, # this works best at -0.04 for html export, -0.06 otherwise
    yanchor='middle',
    yref='paper',
    y=0.5,
    textangle=-90,
    text='Travel time with multi-OD p-hub median (minutes)'
)

fig.update_layout(font=dict(size=18))
#pyo.plot(fig, config=config)

filename = "pmed_scatter_transport"+filename_suffix
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% SCATTER PLOT OF TIME SAVED VS CURRENT TRAVEL TIME

grouped = journeys_agg_card[['p','time_saved','current_dist_total','Cardid']].groupby(['p','current_dist_total','time_saved']).count()
grouped.reset_index(inplace=True)
grouped.rename(columns={'Cardid':'count'}, inplace=True)

# manually scale
original_min = grouped['count'].min()
original_max = grouped['count'].max()
original_range = original_max - original_min

new_min = 1
new_max = 25
new_range = new_max - new_min

grouped['scaled_count'] = (grouped['count'] - original_min) / original_range * new_range + new_min
grouped['p'] = grouped['p'].astype(str)
fig = px.scatter(grouped, x="current_dist_total", y="time_saved", color='p',
                 color_discrete_sequence=mycolors_discrete, size='scaled_count', size_max=new_max, 
                 facet_row='p', render_mode='svg')
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=new_min))

min_val = 0
max_val = 100

fig.update_layout(yaxis_range=[-5, max_val])
fig.update_layout(xaxis_range=[0, max_val])
fig.update_layout(showlegend=False)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.for_each_xaxis(lambda x: x.update({'title': ''}))
fig.for_each_yaxis(lambda y: y.update({'title': ''}))

fig.update_yaxes(default_layout['yaxis'])
fig.update_xaxes(default_layout['xaxis'])

fig.layout.annotations[0]['x'] = 0.99
fig.layout.annotations[0]['textangle'] = 0
fig.layout.annotations[0]['text'] = 'p = 30'

fig.layout.annotations[1]['x'] = 0.99
fig.layout.annotations[1]['textangle'] = 0
fig.layout.annotations[1]['text'] = 'p = 20'

fig.layout.annotations[2]['x'] = 0.99
fig.layout.annotations[2]['textangle'] = 0
fig.layout.annotations[2]['text'] = 'p = 10'

# add annotations
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=0.5, 
    yref='paper',
    y=-.09,
    text='Current travel time (minutes)'
)
fig.add_annotation(
    showarrow=False,
    xanchor='center',
    xref='paper', 
    x=-0.06, 
    yanchor='middle',
    yref='paper',
    y=0.5,
    textangle=-90,
    text='Time saved with the multi-OD p-hub median (minutes)'
)

fig.update_layout(font=dict(size=18))
pyo.plot(fig, config=config)

#%% STRIP PLOT - TRANSPORT RESULTS

plot_agg_dist = combined_agg_dist.groupby(['Cardid','p']).sum()
plot_agg_dist.rename(columns={'time_saved':'Compared to current travel time'}, inplace=True)
plot_agg_dist.reset_index(inplace=True)

outbox = pd.melt(
    plot_agg_dist,
    id_vars=['p'],
    value_vars=['Compared to current travel time'],
)

fig = px.strip(outbox, color ='p', y='value',x='variable', 
             color_discrete_sequence=mycolors_discrete, 
             labels={"value": "Travel time saved (minutes)", "variable": ""})# y="total_bill")
fig.update_yaxes(
    range=(-10, 150),
    constrain='domain'
)
strip_layout = dict(yaxis=dict(
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
            showgrid=False,
            linewidth=line_width, 
            linecolor=line_colour,
            gridcolor=line_colour,
            gridwidth=line_width,
            separatethousands=True,
            tickformat=",",
    ),
    paper_bgcolor=bg_colour,
    plot_bgcolor=bg_colour,
    font=dict(
        size=18,
        color="black"
    )
)
fig.update_layout(strip_layout)

pyo.plot(fig, config=config)

filename = "pmed_stripplot_transport"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg",width=700, height=400)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT ALLOCATED FACILITIES
# sized by number of records allocated there

min_size=15

p_explore = 30
map_center = {"lat": -31.8, "lon": 115.864055}

df_nodes_output = combined_df_nodes_output[combined_df_nodes_output['p']==p_explore]

# only plot stops which have had a demand point allocated to them
df_nodes_plot = df_nodes_output[~df_nodes_output['count'].isna()]
df_nodes_plot['size'] = df_nodes_plot['count']*min_size # minimum count will always be 1, so set to minimum

fig = px.scatter_map(
    df_nodes_plot,
    lat=df_nodes_plot["Y"],
    lon=df_nodes_plot['X'],
    color=df_nodes_plot['pmed_open'],
    color_discrete_sequence=mycolors_discrete,
    hover_data=['clusterID','count'],
    size='size',
    center=map_center,
    zoom=10,
)

fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=2))
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)

filename = "pmed_jdlp_allocated_map_"+filename_suffix+str(p_explore)
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT ALL FACILITIES (OPEN, CLOSED, EXISTING)
# scatter map of all locations coloured by their status

df_nodes_output = combined_df_nodes_output[combined_df_nodes_output['p']==10]

# plot all
fig = px.scatter_map(
    df_nodes_output,
    lat=df_nodes_output["Y"],
    lon=df_nodes_output['X'],
    color=df_nodes_output['pmed_open'],
    color_discrete_sequence=mycolors_discrete,
    hover_data=['clusterID'],
    center=map_center,
    zoom=9,
)

fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=10))
pyo.plot(fig, config=config)

# filename = "pmed_jdlp_count"
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)

#%% PLOT MAP COLOURED BY TIME SAVED
# points on a map coloured by how much total time they save if used
# relative to current travel time

p_explore = 30
assigned_facilities = combined_agg_dist[(combined_agg_dist['time_saved']>0)&(combined_agg_dist['p']==p_explore)].groupby(['assignments']).agg({'time_saved':['sum','count','min', 'median', 'mean','max']})
df_nodes_output = combined_df_nodes_output[combined_df_nodes_output['p']==p_explore]

assigned_facilities.reset_index(inplace=True)
assigned_facilities.columns = [' '.join(col).strip() for col in assigned_facilities.columns.values]

# note that count_records is the number of people allocated to this location where we've saved them time
# not necessarily just the total allocated to that location
assigned_facilities.rename(columns={'assignments':'clusterID',
                                    'time_saved sum':'total_saved',
                                    'time_saved count':'count_records', 
                                    'time_saved min':'min saved',
                                    'time_saved median':'median saved', 
                                    'time_saved mean':'mean saved', 
                                    'time_saved max':'max saved'}, inplace=True)

#assigned_facilities = assigned_facilities[assigned_facilities['count_records']>0]

assigned_facilities = pd.merge(assigned_facilities, df_nodes_output[['clusterID','X', 'Y','pmed_open']], on='clusterID')

assigned_facilities['total_saved'] = assigned_facilities['total_saved'].astype(float)

fig = px.scatter_map(
    assigned_facilities,
    lat=assigned_facilities["Y"],
    lon=assigned_facilities['X'],
    color=assigned_facilities['total_saved'], 
    color_continuous_scale=mycolors_continuous_map,
    size=assigned_facilities['total_saved'], 
    size_max=15,
    hover_data=['clusterID'],
    center=map_center,
    zoom=10,
)
fig.update_traces(marker_opacity=1) 
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=3))
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)

filename = "pmed_jdlp_totalsaved_map_"+filename_suffix+str(p_explore)
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%

p_explore = 30
assigned_facilities = combined_agg_dist[(combined_agg_dist['time_saved']>0)&(combined_agg_dist['p']==p_explore)].groupby(['assignments']).agg({'time_saved':['sum','count','min', 'median', 'mean','max']})
df_nodes_output = combined_df_nodes_output[combined_df_nodes_output['p']==p_explore]

assigned_facilities.reset_index(inplace=True)
assigned_facilities.columns = [' '.join(col).strip() for col in assigned_facilities.columns.values]

# note that count_records is the number of people allocated to this location where we've saved them time
# not necessarily just the total allocated to that location
assigned_facilities.rename(columns={'assignments':'clusterID',
                                    'time_saved sum':'total_saved',
                                    'time_saved count':'count_records', 
                                    'time_saved min':'min saved',
                                    'time_saved median':'median saved', 
                                    'time_saved mean':'mean saved', 
                                    'time_saved max':'max saved'}, inplace=True)

#assigned_facilities = assigned_facilities[assigned_facilities['count_records']>0]

assigned_facilities = pd.merge(assigned_facilities, df_nodes_output[['clusterID','X', 'Y','pmed_open']], on='clusterID')

assigned_facilities['median saved'] = assigned_facilities['median saved'].astype(float)

fig = px.scatter_map(
    assigned_facilities,
    lat=assigned_facilities["Y"],
    lon=assigned_facilities['X'],
    color=assigned_facilities['median saved'], 
    color_continuous_scale=mycolors_continuous_map,
    size=assigned_facilities['median saved'], 
    size_max=15,
    hover_data=['clusterID'],
    center=map_center,
    zoom=10,
)
fig.update_traces(marker_opacity=1) 
fig.update_layout(default_layout)
fig.update_traces(marker=dict(sizemin=3))
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)

filename = "pmed_jdlp_mediansaved_map_"+filename_suffix+str(p_explore)
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)




