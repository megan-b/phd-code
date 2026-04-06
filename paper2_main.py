# -*- coding: utf-8 -*-

# IMPORTS

# for analysis
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import numpy as np
import math

# for plotly plotting
import plotly.io as pio
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shapely.geometry import box, Point

import timeit
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance as scidist

import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as iso

from collections import Counter

from sklearn.mixture import GaussianMixture
import osmnx as ox
from sklearn.neighbors import BallTree


# CONSTANTS THAT MUST BE SET BY USER

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
    "#d73027","#fc8d59","#fee090","#e0f3f8",
    "#91bfdb","#4575b4"]


mycolors_continuous_r = mycolors_continuous.copy()
mycolors_continuous_r.reverse() # for reversed continuous gradient

MAPBUFFER = 100 # for region mapping
visited_label = "V" # select character to represent visited regions

config = {'scrollZoom': True}

line_colour = "rgb(217,217,217)"
bg_colour = "white"
line_width = 0.5

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

default_layout_box_journal = dict(yaxis=dict(
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

default_layout_journal = dict(yaxis=dict(
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
        size=16,
        color="black",
        family="Arial",
    )
)

#%%

def create_point(row):
    """Returns a shapely point object based on values in x and y columns"""
    point = Point(row["X"], row["Y"])
    return point

# https://github.com/pandas-dev/pandas/issues/34836

def explode_event(start, stop, idx, step):
    """splits an event into multiple rows if crossed over a certain time threshold"""

    starts = np.arange(start, stop, step)
    starts = starts.tolist()
    stops = np.arange(start + step, stop + step, step)
    stops = stops.tolist()
    ids = [idx] * len(starts)

    return pd.DataFrame({"start": starts, "stop": stops, "event_id": ids})


def explode_events(df, start_col, stop_col, step):
    """splits each event in a df of events into multiple events if crossed over a
    certain time threshold"""

    tmp = pd.concat(
        [
            explode_event(start=row[start_col], stop=row[stop_col], idx=idx, step=step)
            for idx, row in df.iterrows()
        ]
    )

    return (
        tmp.merge(df, how="left", left_on="event_id", right_index=True)
        .drop([start_col, stop_col, "event_id"], axis=1)
        .rename(columns={"start": start_col, "stop": stop_col})
        .reset_index(drop=True)
    )

def create_events(df, start_col, end_col):
    
    df[start_col] = round(df[start_col] * 4) / 4
    df[end_col] = round(df[end_col] * 4) / 4
    
    df = df.drop(df[df[start_col] == df[end_col]].index).reset_index(drop=True)
    all_activities = explode_events(df, start_col, end_col, 1 / 4)

    all_activities["Date"] = all_activities[start_col] / 24
    all_activities["Date"] = all_activities["Date"].astype(int)
    all_activities["Date"] = all_activities["Date"] + 1

    all_activities["Cumulative_h"] = all_activities[start_col]

    all_activities[start_col] = all_activities[start_col].mod(24)
    all_activities[end_col] = all_activities[end_col].mod(24)

    # also before pivoting would be a good time to 'fill the gaps' with the unknown
    # days and timestamps
    y_range = np.arange(0, 24, .25)  # create range for hours
    x_range = np.arange(1, 31 + 1, 1)  # create range for days

    outlist = [(i, j) for i in x_range for j in y_range]
    newdf = pd.DataFrame(data=outlist, columns=["Date", "Hours"])
    
    all_hours = pd.merge(
        newdf,
        all_activities,
        left_on=["Date", "Hours"],
        right_on=["Date", start_col],
        how="left",
    )

    return all_activities, all_hours


#%% READ PICKLES

allhist = pd.read_pickle(dirname + "20250125-fullallhist.pkl")
activities = pd.read_pickle(dirname + "20250111-activities.pkl")
regionpivot = pd.read_pickle(dirname + "20250125-final-regionpivot-GMM-unique.pkl")

regionpolys = pd.read_pickle(dirname + "20250125-fullallregionpolys.pkl")
regionpolys = gpd.GeoDataFrame(regionpolys, geometry="geometry")
regionpolys = regionpolys.to_crs(epsg=4326)

#%% ADD REGION NAMES

clustermap_unique = {
    0: "Education",
    1: "Residences",
    2: "Workplaces",
    3: "Residences/Leisure",
    4: "Workplaces/Leisure",
}

regionpivot["region_name"] = regionpivot["GMM_cluster_unique"].map(clustermap_unique)

clusteralphamap = {
    0: "E",
    1: "R",
    2: "W",
    3: "L",
    4: "C",
}

regionpivot["region_alpha"] = regionpivot['GMM_cluster_unique'].map(clusteralphamap)

#%% GET REGION COUNTS

# count of region types by card
regionpivot['region_num'] = regionpivot.groupby(["Cardid","region_alpha"])['Total'].rank(method='first', ascending=False)
regionpivot['region_num'] = regionpivot['region_num'].astype(int) 

# create ID from alpha + type
# this assumes there will never be more than 9 regions of the same type
regionpivot['region_ID'] = regionpivot['region_alpha']+regionpivot['region_num'].astype(str)

# PUT REGIONS ONTO ACTIVITIES

# merge activities and on all_hist['Cardid','spatial_cluster'] to bring in 
# 'region_cluster'- note this is unique per card, not the labelled cluster type

activities = pd.merge(
    activities,
    allhist[["Card", "spatial_cluster", "region_cluster"]],
    left_on=["Cardid", "spatial_cluster"],
    right_on=["Card", "spatial_cluster"],
    how="left",
)

# merge on 'Cardid' and 'region_cluster' to bring in the 'GMM_cluster_unique'
activities = pd.merge(
    activities,
    regionpivot[["Cardid", "region_cluster", "GMM_cluster_unique", "region_name", 
                 "region_alpha", "region_ID", "region_num"]],
    left_on=["Cardid", "region_cluster"],
    right_on=["Cardid", "region_cluster"],
    how="left",
)
# activities has length 5,014,122 (same as from regions work)

activities = activities.dropna(
    axis="rows",
    subset=['Card']
) # includes all cards, not just those that have had regions generated - so 
# drop the NaNs
# this is removing the activities that weren't taken in the random slice

print(len(activities),"activities from cards we have regions for")

activities = activities[activities['Cardid'].isin(regionpivot['Cardid'])]
# the only cards in regionpivot are the ones that have anchoring regions, 
# so only keep the activities (from all regions) from those cards

# some cards only have visited regions (which would have been in the earlier 
# region counts) as we are keeping only the cards that have anchoring regions, 
# you'll end up with less

print(len(activities),"activities from anchoring regions")


#%% LABEL VISITED REGIONS

activities['region_type'] = np.where(activities['region_alpha'].isna(), 'Visited', 'Anchor')

activities['final_regionID'] = np.where(activities['region_type']=='Visited', 
                                        visited_label+activities['region_cluster'].astype(str), 
                                        activities['region_ID'])

activities['final_act-region'] =   activities['activity_alpha'] + '-' + activities['final_regionID']

# now where region_alpha is still NaN, it's a visited region
activities['region_alpha'].fillna(visited_label, inplace=True)

#%% SUMMARISE REGIONS

regionsummary = activities[['Cardid', 'region_alpha', 'final_regionID', 
                            'region_type', 'Token', 'region_cluster']].drop_duplicates(
  subset = ['Cardid', 'final_regionID', 'region_cluster'],
  keep = 'last').reset_index(drop=True)

regionsummary['region_alpha'] = regionsummary['region_alpha'].fillna(visited_label)

regioncount = pd.pivot_table(regionsummary, index=['Cardid', 'Token'], 
                             columns='region_type', aggfunc='count', 
                             values='final_regionID')
regioncount = regioncount.fillna(0)
   
regioncount['Total'] = regioncount['Anchor'] + regioncount['Visited']
regioncount['Anchor_frac'] = regioncount['Anchor'] / regioncount['Total']
regioncount['Visited_frac'] = regioncount['Visited'] / regioncount['Total']
# note that this fraction isn't weighted by number of activities, just a straight
# split of the region types

# for identifying interesting cards
regioncount.sort_values(by='Total',ascending=False, inplace=True)
regioncount.reset_index(inplace=True)


#%% PLOT DENSITY HEATMAP OF REGION TYPES

# density heatmap - % of cards in each combination of anchor/visited regions
# ~50% of all cards have 2 anchoring regions and either 0, 1 or 2 visited
# regions

fig = px.density_heatmap(
    regioncount, 
    x='Anchor',
    y='Visited',
    color_continuous_scale=mycolors_continuous,
    text_auto=".2f",
 #   facet_col='Token',
    histnorm='percent',
    labels={
        "Anchor": "Number of anchoring regions",
        "Visited": "Number of visited regions",}
)
fig.update_layout(default_layout)
#pyo.plot(fig, config=config)

filename = "regions_AVdensity"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% CALCULATE LIFT OF REGION TYPE BY TOKEN TYPE

regionsummary1 = activities[['Cardid', 'region_alpha', 'final_regionID', 
                            'region_type', 'Token', 'region_cluster', 'region_name']].drop_duplicates(
  subset = ['Cardid', 'final_regionID', 'region_name'],
  keep = 'last').reset_index(drop=True)

regionsummary1['region_alpha'] = regionsummary1['region_alpha'].fillna(visited_label)
regionsummary1['region_name'] = regionsummary1['region_name'].fillna("Visited")


regioncount1 = pd.pivot_table(regionsummary1, index=['Cardid', 'Token'], 
                             columns='region_name', aggfunc='count', 
                             values='final_regionID')
regioncount1 = regioncount1.fillna(0)
regioncount1.reset_index(inplace=True)

regionsummary2 = regioncount1[['Token','Education','Residences','Residences/Leisure',
              'Visited','Workplaces','Workplaces/Leisure']].value_counts()

regionsummary2 = pd.DataFrame(regionsummary2)
regionsummary2.reset_index(inplace=True)

regionsummary2['cumsum'] = regionsummary2['count'].cumsum()
regionsummary2['perc'] = regionsummary2['count']/regionsummary2['count'].sum()*100
regionsummary2['cum_percent'] = regionsummary2['perc'].cumsum()

tokensummary = pd.melt(
    regioncount1,
    id_vars=['Token'],
    value_vars=['Education', 'Residences', 'Residences/Leisure',
           'Visited', 'Workplaces', 'Workplaces/Leisure'],
)

tokensummary.sort_values(by=['value'], inplace=True)
tokensummary['value'] = tokensummary['value'].astype(int)
tokensummary['value'] = np.where(tokensummary['value']>5, ">5", tokensummary['value'])


tokensummary = tokensummary.value_counts()
tokensummary= pd.DataFrame(tokensummary)
tokensummary.reset_index(inplace=True)


totals_allocated = regioncount1['Token'].value_counts()
totals_allocated = pd.DataFrame(totals_allocated)
totals_allocated.reset_index(inplace=True)
totals_allocated.rename(columns={"count": "token_total"}, inplace=True)


totals = tokensummary.groupby(['region_name','value']).sum().reset_index()
totals = pd.DataFrame(totals)
totals.rename(columns={"count": "region_val total"}, inplace=True)


tokensummary = pd.merge(tokensummary, totals_allocated[['Token','token_total']], 
                        left_on='Token', right_on='Token',
                        how='left')

tokensummary = pd.merge(tokensummary, totals[['region_name','value','region_val total']], 
                        left_on=['region_name','value'], right_on=['region_name','value'],
                        how='left')


tokensummary['perc_token'] = tokensummary['count']/tokensummary['token_total']*100
tokensummary['perc_total'] = tokensummary['region_val total']/len(regioncount1)*100


# # lift = (percent of this token type in this cluster) / (percent of this token type in dataset)
tokensummary['lift'] = tokensummary['perc_token']/tokensummary['perc_total']

# percentage of standard tokens with 1 workplace / percentage of all cards with one workplace

#%% PLOT LIFT BY REGION TYPE BY TOKEN TYPE

tokensummary['value'] = tokensummary['value'].astype(str)
fig = px.density_heatmap(tokensummary, x='region_name', 
                         y='value',
                         z='lift',
                         histfunc="avg",
                         facet_col='Token',
                         color_continuous_scale=["#EFEFEF",mycolors_discrete[1]],
                         text_auto=".2f",
                         category_orders={'value':[">5", "5", "4", "3", "2", "1", "0"]},
                           labels={
                               "value": "Number of region type",
                               "variable": " ",
                               "region_name": " ",
                          },
                     )
fig.update_layout(default_layout)
fig.update(layout_coloraxis_showscale=False)
pyo.plot(fig, config=config)

filename = "regions_lift_bytoken"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT DENSITY HEATMAP OF ANCHOR REGION TYPES

tokenplot = tokensummary[(tokensummary['region_name']!='Visited')]
# could add &(tokensummary['value']!="0") to not show the zeroes

fig = px.density_heatmap(tokenplot, x='region_name', 
                         y='value',
                         z='perc_token',
                         facet_col='Token',
                         color_continuous_scale=["#EFEFEF",mycolors_discrete[1]],
                         text_auto = ".1f",
                           labels={
                               "value": "Number of region type",
                               "variable": " ",
                               "region_name": " ",
                          },
                     )
fig.update_layout(default_layout)
fig.update(layout_coloraxis_showscale=False)
pyo.plot(fig, config=config)

filename = "regions_anchortype_bytoken"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% SUMMARISE CARDS

# summarise cards to make it easier to chart/investigate/choose what to look at
cards = activities["Cardid"].value_counts()
cards = pd.DataFrame(cards)
cards.reset_index(inplace=True)
cards.rename(columns={"Cardid": "Card"}, inplace=True)

# merge cards and regioncount to get a summary of regions and number 
# of uses in one dataframe
cards = pd.merge(cards, regioncount, left_on="Card", right_on="Cardid")
cards.drop(columns=["Cardid"], inplace=True)

# use 'count' to sort by number of uses, 'Total' by regions
cards.sort_values(by='Total',ascending=True, inplace=True)

print(cards[(cards["count"] > 80) & (cards["count"] < 150)])


#%% COMPARE TWO CARDS (PLOTS)

card1 = 12935724 
card2 = 3419726 

test1 = activities[['Start_h', 'End_h', 'Cumulative_h',	
                    'activity_alpha', 'final_regionID']][activities['Cardid']==card1]
test2 = activities[['Start_h', 'End_h', 'Cumulative_h',	
                    'activity_alpha', 'final_regionID']][activities['Cardid']==card2]

outact1, outhours1 = create_events(df=test1, start_col='Start_h', end_col='End_h')

outact2, outhours2 = create_events(df=test2, start_col='Start_h', end_col='End_h')

testout = pd.merge(outhours1[['activity_alpha', 'final_regionID', 'Date', 'Hours']], 
                   outhours2[['activity_alpha', 'final_regionID', 'Date', 'Hours']], 
                      left_on=['Date', 'Hours'], right_on=['Date', 'Hours'], how="left")

testout['Cumulative_h'] = (testout['Date']-1)*24+testout['Hours']

testout.ffill(axis=0, inplace=True)

# plot y = activity alpha x = time
fig = px.line(testout, y=["activity_alpha_x","activity_alpha_y"], x="Cumulative_h")
pyo.plot(fig, config=config)
# filename = " "
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
# fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#
# plot y = region x = time
fig = px.line(testout, y=["final_regionID_x","final_regionID_y"], x="Cumulative_h")
pyo.plot(fig, config=config)

#%% PLOT 3D DIAGRAM (ACTIVITY, REGION, TIME) OF ONE CARD

card = 13658465

selectcard = activities[activities['Cardid'] == card]

selectcard['activity_alpha'].sum()
# this joins all the activity alphas into one string. neat!
# this includes the V activities - which were excluded in the regions ID 
# analysis. But should leave them in otherwise end up with disjointed motifs

days_include = 31

graphdata = selectcard[selectcard['Cumulative_h']<=days_include*24]
graphdata['Cumulative_d'] = graphdata['Cumulative_h']/24

fig = go.Figure(data = go.Scatter3d(
    x=graphdata['activity_alpha'],
    y=graphdata['final_regionID'],
    z=graphdata['Cumulative_d'],
    marker=dict(
        size=8,
        color=graphdata['activity_int'], # this needs to be numeric, so can't be activity_alpha
        colorscale=mycolors_continuous,
    ),
    line=dict(
        color=mycolors_discrete[5],
        width=0.5,
    ),
))

fig.update_scenes(
    xaxis_title='Activity',
    yaxis_title='Region',
    zaxis_title='Time (days)',
    yaxis_categoryorder="category ascending"
)

pyo.plot(fig, config=config)
# filename = " "
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
# fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT TO DETERMINE START OF DAY

# get cutover time - mod 24 of start of stay time and plotted
activities['daystart_h'] = activities['Start_h'].mod(24)
activities['daystart_round'] = round(activities['daystart_h']*2)/2

daystart_plot = pd.pivot_table(activities[['daystart_round','Start_h']], index=['daystart_round'], aggfunc='count')
daystart_plot.reset_index(inplace=True)
daystart_plot.rename(columns={'Start_h':'Count'}, inplace=True)


#%%

fig = px.bar(
    daystart_plot,
    x='daystart_round',
    y='Count',
    color_discrete_sequence=[mycolors_discrete[7]],
)

fig = px.histogram(activities['daystart_round'],
                   color_discrete_sequence=[mycolors_discrete[7]])

fig.update_layout(
    default_layout_journal,
    yaxis_title_text = 'Number of activities',
    xaxis_title_text = 'Time of day',
    showlegend=False,
)
fig.update_xaxes(range=[0, 24])

pyo.plot(fig, config=config)
filename = "routines_daystart"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% WORK OUT WHICH ACTIVITIES OCCUR IN SAME PERIOD
# and split those that go over the time between periods

TRAVELDAYSTART = 5

activities["Travel_day_start"] = (activities['Start_h']  - TRAVELDAYSTART)//24
activities["Travel_day_end"] = (activities['End_h']  - TRAVELDAYSTART)//24

PERIODLENGTH = 7 # how many days to group together in a sequence (set to 7 for a week)

activities['days'] = activities['Travel_day_end'] - activities['Travel_day_start']
activities['Travel_period_start'] = activities['Travel_day_start']//PERIODLENGTH
activities['Travel_period_end'] = activities['Travel_day_end']//PERIODLENGTH
activities['periods'] = activities['Travel_period_end'] - activities['Travel_period_start']

activities['periods']= activities['periods'].astype(int)
activities['periods_between'] = activities.apply(lambda row: [row['Travel_period_start'] + d for d in range(row['periods']+1)], axis=1)
del activities['periods']

activities = activities.explode('periods_between')

activities['Start_h'] = np.where(activities['Travel_period_start'] == activities['periods_between'], 
                                 activities['Start_h'],
                                 (activities['periods_between'])*(24*PERIODLENGTH))

activities['End_h'] = np.where(activities['Travel_period_end'] == activities['periods_between'], 
                                 activities['End_h'],
                                 (activities['periods_between']+1)*(24*PERIODLENGTH))

activities["Travel_period_start"] = activities['Start_h']//(24*PERIODLENGTH)
activities["Travel_period_end"] = activities['End_h']//(24*PERIODLENGTH)

# want to turn cumulative_h into the h of each period (i.e. 2h into period 1 comes 
# through same as 2h into period 3)
activities['Period_h'] = activities['Cumulative_h'] - activities['Travel_period_start']*(24*PERIODLENGTH)

activities['Day_h'] = activities['Period_h']%24

# where Start_h is negative, it's because the activity has been split across periods
# so rename the activity type and set the negative to zero
activities['activity_alpha'][activities['Period_h'] < 0] = "X" 
activities['Period_h'][activities['Period_h'] < 0] = 0

# BONUS EXTRA OPTION
# every sequence will now begin with this letter, but will have different lengths
# so is meainingless - can drop
# as they'll all have the same character now, just drop where activity_alpha = X
activities = activities[activities['activity_alpha'] != "X"]

#%% DROP SEQUENCES WITH ACTIVITIES OUTSIDE PERTH

geometry = [box(115.493774,-32.685620,116.180420,-31.447410)] # coast to Mt Helena, Manudurah to Yanchep
boxdf = gpd.GeoDataFrame(geometry=geometry)
boxdf = boxdf.set_crs("epsg:4326", allow_override=True)

# check it has been set up correctly
# fig = px.choropleth_map(boxdf, 
#                         geojson=boxdf.geometry, 
#                         locations=boxdf.index,     
#                         center={"lat": -32, "lon": 116},
#                         color_discrete_sequence=mycolors_discrete,
#                         opacity=0.6,
#                         zoom=10)
# pyo.plot(fig, config=config)

# add a column to regionpolys that says whether or not each region is in the bbox
def in_box(row):
    return boxdf.contains(row['geometry']).any()

regionpolys['bbox check'] = regionpolys.apply(in_box, axis=1)

# put geometries onto activities
activities = pd.merge(activities, regionpolys[['Card','region_cluster', 'geometry', 'bbox check']], 
                      left_on=['Cardid', 'region_cluster'], right_on=['Card', 'region_cluster'], how="left")
activities = gpd.GeoDataFrame(activities, geometry='geometry', crs=4326)

activities.rename(columns={'Card_x':'Card'}, inplace=True)

activities_drop = pd.pivot_table(
    activities,
    values="Day_h",
    index=["Card","Travel_period_start"],
    columns="bbox check",
    fill_value=0,
    aggfunc="count",
)

activities_drop.reset_index(inplace=True)

# where at least one activity is outside the bbox, this sequence should be dropped
# note this brings up an exception if the data has already been run through this (i.e. there are no False)
activities_drop['to drop'] = np.where(activities_drop[False]>0,'Y','N')

# merge on the card and travel period start to label each activity with if it should be dropped
activities = pd.merge(activities, activities_drop[['Card','Travel_period_start', 'to drop']], 
                      left_on=['Card', 'Travel_period_start'], right_on=['Card', 'Travel_period_start'], how="left")

activities = activities[activities["to drop"] == "N"]

activities.drop("to drop", inplace=True, axis=1)

print(len(activities), "activities after dropping sequences that contain some activities outside Perth")

#%% CHECK IF WE NEED TO DEAL WITH TOKENS WITH MULTIPLE TYPES

card_info = activities[['Cardid','Token','Start_h']]
card_tokens = card_info[['Cardid','Token']].drop_duplicates()

card_token_counts = pd.DataFrame(card_tokens['Cardid'].value_counts())
card_token_counts.reset_index(inplace=True)
multi_cards = card_token_counts[card_token_counts['count']>1]

print(len(card_token_counts), "cards in dataset")
print(len(multi_cards)/len(card_token_counts)*100)
# 1.8% of cards have more than one token type in the month
# go with assume card has the token type it began the week with

#%% PIVOT BY CARD AND PERIOD

# create list for period_h, duration, regionID, region_alpha and activity_alpha 
# for each period/card

activities['Start_h'] = pd.to_numeric(activities['Start_h'], errors='coerce')

cardsum = activities[['Day_h','Token','Period_h','Duration','final_regionID','Distance_travel',
                      'activity_alpha','region_alpha','Cardid','Travel_period_start']].groupby(
                          ['Cardid', 'Travel_period_start'], as_index=False).agg({'Token':'first','Day_h':list, 'Period_h':list,
                                                                                  'Duration':(list,'sum'),
                                                                                  'final_regionID':list,
                                                                                  'Distance_travel':(list,'sum'),
                                                                                  'region_alpha':'sum',
                                                                                  'activity_alpha':'sum'})

cardsum.columns = [' '.join(col).strip() for col in cardsum.columns.values]
cardsum.rename(columns={'Token first':'Token','Period_h list':'Start_h list','Travel_period_start':'Period',
                        'final_regionID list':'region_ID list'}, inplace=True)

cardsum['Length'] = cardsum['Start_h list'].str.len()

#%% REMOVE INCOMPLETE SEQUENCES
# there are some days at the end of the month that don't have a complete sequence, so drop those

MAXPERIOD = cardsum['Period'].max()
cardsum = cardsum[cardsum['Period'] < MAXPERIOD]

print(len(cardsum), "complete sequences")

#%% HISTOGRAM OF SEQUENCE LENGTH

fig = px.histogram(
    cardsum['Length'],
    nbins=100,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Number of sequences',
    xaxis_title_text = 'Sequence length',
    showlegend=False)
pyo.plot(fig, config=config)

filename = "routines_sequencelength"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% GET HOME GEOMETRY & PUT HUBS ONTO SEQUENCES

# get geometry for residential region. If region alpha = R.
# if no region alpha R, use L instead
# if neither, then drop
homeregions = regionsummary[(regionsummary['final_regionID']=="R1") 
                            | (regionsummary['final_regionID']=="L1")]
homecounts = homeregions["Cardid"].value_counts()
homecounts = pd.DataFrame(homecounts)
homecounts.reset_index(inplace=True)
#homecounts.rename(columns={'index':'Cardid', 'Cardid':'count'}, inplace=True)

print(len(homecounts), "cards have a home region") # 86,899 cards have a home region
# not all of the cards with home regions will necessarily have complete sequences
# but to give an indication 86899/102734 = ~85%

homeregions = pd.merge(homeregions, homecounts, 
                       left_on="Cardid", right_on="Cardid", how="left")

homeregions = homeregions[((homeregions['final_regionID']=="R1") & (homeregions['count']==2)) | 
                          ((homeregions['final_regionID']=="L1") & (homeregions['count']==1)) | 
                          ((homeregions['final_regionID']=="R1") & (homeregions['count']==1))] 

homeregions = pd.merge(homeregions, regionpolys[['Card', 'region_cluster', 'geometry']], 
                      left_on=['Cardid', "region_cluster"], right_on=['Card', "region_cluster"], how="left")

homeregions = gpd.GeoDataFrame(homeregions, geometry='geometry', crs=4326)

homeregions.drop(columns=['count','region_cluster', 'region_alpha', 'region_type'],inplace=True)

#%% GET HUB DATA

hubsinfo = pd.read_csv(dirname+'hubsinfoforRailSmart\HubsInfo_megan.csv', skiprows=3)
hubs_def = hubsinfo.copy()
hubs_def = hubs_def[hubs_def['Total Arrivals']>0] 
# only include actual hubs with usage

# explode out stops column
hubs_def['stops'] = hubs_def['stops'].str.split(' ')
hubs_def = hubs_def.explode('stops')

# pull out number between quote marks
hubs_def['stops'] = hubs_def['stops'].str.split("'").str[1]
hubs_def['stops'] = hubs_def['stops'].astype(int)

# bring in exportHubsStopTable which has cols StopID | X | Y
hub_coords = pd.read_csv(dirname+'hubsinfoforRailSmart\exportHubsStopTable.csv')

# create shapely point for each row 
hub_coords['geometry'] = hub_coords.apply(create_point, axis=1)

# merge stops column with the stop table to get coords
hubs_def = pd.merge(hubs_def, hub_coords[['StopID', 'geometry']], left_on='stops', right_on='StopID')

hubs_def = gpd.GeoDataFrame(hubs_def, geometry="geometry")
hubs_def = hubs_def.set_crs("epsg:4326", allow_override=True)

#%% PLOT STOPS COLOURED BY HUB

fig = px.scatter_map(hubs_def, 
                        lat=hubs_def.geometry.y, 
                        lon=hubs_def.geometry.x,     
                        color=hubs_def['hubID'].astype(str), 
                        hover_name=hubs_def['hubname'],
                        color_discrete_sequence=mycolors_discrete, 
                        size_max=15, zoom=10)
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)
# filename = "routines_name"
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
# fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% GET FIVE CLOSEST HUBS VIA EUCLIDEAN DISTANCE

# convert home regions to points
homeregions.to_crs(epsg=32749, inplace=True)
homeregions['geometry'] = homeregions['geometry'].centroid
homeregions.to_crs(epsg=4326, inplace=True)

candidate_radians = np.array(hubs_def['geometry'].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())

# Create tree from the candidate points
tree = BallTree(candidate_radians, leaf_size=15, metric='euclidean')

# Find closest points and distances
source_radians = np.array(homeregions['geometry'].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
distances, indices = tree.query(source_radians, k=5)

indices = pd.DataFrame(indices)
indices.rename(columns={0:"0_ix",1:"1_ix",2:"2_ix",3:"3_ix",4:"4_ix"}, inplace=True)
# these are the index of hubs_def that are the closest

distances = distances*6378 # to turn into km
distances = pd.DataFrame(distances)
distances.rename(columns={0:"0_dist",1:"1_dist",2:"2_dist",3:"3_dist",4:"4_dist"}, inplace=True)
homeregions = pd.concat([homeregions, indices, distances], axis=1)

# test distance difference between first and second closest hubs
homeregions['1-0_dist'] = homeregions['1_dist'] - homeregions['0_dist']

def get_nearest_hub(row):
    hubs_loc = row["0_ix"]
    return hubs_def[['hubID', 'StopID']].iloc[hubs_loc]

homeregions[['nearest_hub','nearest_stop']] = homeregions.apply(get_nearest_hub, axis=1)
# get hub and stop number of the one closest by euclidean distance

#%% PLOT EACH CARD'S HOME REGION, COLOURED BY NEAREST STOP (EUCLIDEAN DIST)

# sense checks the nearest hub allocation

fig = px.scatter_map(homeregions, 
                        lat=homeregions.geometry.y, 
                        lon=homeregions.geometry.x,     
                        color=homeregions['nearest_hub'].astype(str), 
                        #color=homeregions['nearest_stop'].astype(str),
                        # there are mulitple stops per hub, just depends 
                        # how you want to see this as to which is appropriate to map
                        hover_name=homeregions['Cardid'],
                        #color_discrete_sequence=mycolors_discrete,
                        center={"lat": -31.95, "lon": 115.9}, # this is the maylands peninsula-ish
                        zoom=14,
                        )
fig.add_trace(
    go.Scattermap(
        lat=hubs_def["geometry"].y,
        lon=hubs_def["geometry"].x,
        hovertext=hubs_def['hubID'],
        mode="markers",
        marker=go.scattermap.Marker(
            color='black',
            size=12,
        ),
    )
)
fig.update_layout(map_style="light", showlegend=False)
pyo.plot(fig, config=config)
#filename = "hubprofile_allocation_euclidean"
# fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
# fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%%

def find_nearest_hub_OSM(row):
    start = timeit.default_timer()
    location_point = (row['geometry'].y, row['geometry'].x)
    G = ox.graph_from_point(location_point, dist=3000, dist_type="bbox", network_type="walk")   
    home_node = ox.nearest_nodes(G, X=row['geometry'].x, Y=row['geometry'].y)
    
    candidate_nodes = []
    for i in np.arange(0,5):
        colname = str(i)+"_ix"
        x = hubs_def['geometry'].iloc[row[colname]].x
        y = hubs_def['geometry'].iloc[row[colname]].y
        node = ox.nearest_nodes(G, x, y)
        candidate_nodes = candidate_nodes+[node]
        
    dists = []
    for i in np.arange(0,5):
        route = ox.shortest_path(G, [home_node], [candidate_nodes[i]], weight="length")
        if len(route[0])<=1:
            dist = 0
        else:
            gdf = ox.routing.route_to_gdf(G, route[0], 'length')
            dist = gdf["length"].sum()
        dists = dists+[dist]
       # fig, ax = ox.plot_graph_route(G, route[0], route_color="y", route_linewidth=6, node_size=0)
        
    ix_min = dists.index(min(dists))
    hubs_loc = row[str(ix_min)+"_ix"]
    
   # print("Card:",row['Cardid'])
    #print("Previous nearest hub:", row['nearest_hub'])
    #print("New nearest hub:", hubs_def['hubID'].iloc[hubs_loc])
    print("Row", row.name, "complete in " + str(timeit.default_timer() - start) + " seconds")
    return hubs_def['hubID'].iloc[hubs_loc]

#%%

def find_nearest_hub_OSM_graph(row):
    start = timeit.default_timer()
    location_point = (row['geometry'].y, row['geometry'].x)
    G = ox.graph_from_point(location_point, dist=3000, dist_type="bbox", network_type="walk")   
    home_node = ox.nearest_nodes(G, X=row['geometry'].x, Y=row['geometry'].y)
    
    candidate_nodes = []
    for i in np.arange(0,5):
        colname = str(i)+"_ix"
        x = hubs_def['geometry'].iloc[row[colname]].x
        y = hubs_def['geometry'].iloc[row[colname]].y
        node = ox.nearest_nodes(G, x, y)
        candidate_nodes = candidate_nodes+[node]
        
    dists = []
    for i in np.arange(0,5):
        route = ox.shortest_path(G, [home_node], [candidate_nodes[i]], weight="length")
        if len(route[0])<=1:
            dist = 0
        else:
            gdf = ox.routing.route_to_gdf(G, route[0], 'length')
            dist = gdf["length"].sum()
        dists = dists+[dist]
        fig, ax = ox.plot_graph_route(G, route[0], route_color="r", route_linewidth=6, 
                                      node_size=0, filepath=dirname+"figs\\svg\\"+'hubprofile_osm_graph_'+str(i)+'.svg', save=True)
        print("Candidate",i,", network distance (km)",dist/1000)
        
    ix_min = dists.index(min(dists))
    hubs_loc = row[str(ix_min)+"_ix"]
    
    print("Card:",row['Cardid'])
    print("Previous nearest hub:", row['nearest_hub'])
    print("New nearest hub:", hubs_def['hubID'].iloc[hubs_loc])
    print("Row", row.name, "complete in " + str(timeit.default_timer() - start) + " seconds")
    return hubs_def['hubID'].iloc[hubs_loc]

#%% DEMONSTRATE WHY OSM CHECK IS REQUIRED

# from previous figure - cards at bottom of Maylands Peninsula being allocated
# to a hub over the river are
# nearest_stop = 14366
# 1463683
# 10538356
# 13636569
# 26886517

# mounts bay road being allocated to south perth
# this one needs a longer street map network (5000)
# nearest_stop = 99998
# 12901301

# east freo being allocated to north freo
# nearest_stop = 2772
# 12739894
# 26411300
# 13099852

# south guildford being allocated to success hill
# nearest_stop = 2811
# nearest is actually guildford so not a big difference
# 27036542
# 11167474
# 12903393
# 10157102

# takes about 20-30s for one card
homeregion_test = homeregions[homeregions['Cardid']==10538356]
homeregion_test['nearest_hub_OSM'] = homeregion_test.apply(find_nearest_hub_OSM_graph, axis=1)



#%% MAP HOME REGION, CURRENT ALLOCATION, NEW ALLOCATION AND CANDIDATE STOPS

candidate_stops =  hubs_def.iloc[[homeregion_test['0_ix'].iloc[0], 
                                  homeregion_test['1_ix'].iloc[0], 
                                  homeregion_test['2_ix'].iloc[0],
                                  homeregion_test['3_ix'].iloc[0],
                                  homeregion_test['4_ix'].iloc[0]]]
                                 
current_stop = hubs_def[hubs_def['stops']==homeregion_test['nearest_stop'].iloc[0]]
new_hub = homeregion_test['nearest_hub_OSM'].iloc[0]

homeregion_test['Type'] = 'Home region'
current_stop['Type'] = 'Current allocation'
candidate_stops['Type'] = 'Candidate location'
candidate_stops['Type'] = np.where(candidate_stops['hubID']==new_hub,'Updated allocation', candidate_stops['Type'])

allstops = pd.concat([candidate_stops, homeregion_test[['geometry','Type','nearest_hub','nearest_stop']], current_stop])

fig = px.scatter_map(allstops, 
                        lat=allstops.geometry.y, 
                        lon=allstops.geometry.x,     
                        color=allstops['Type'],
                        color_discrete_sequence=mycolors_discrete,
                        center={"lat": -31.95, "lon": 115.9}, # this is the maylands peninsula-ish
                        zoom=13,
                        )

fig.update_traces(marker=dict(size=20, opacity=1))
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)
#%%
filename = "hubprofile_streetmap_example"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% GET UNIQUE GEOMETRIES
# make sure it doesn't take any longer than it has to

homeregions.to_crs(epsg=4326, inplace=True)

# get unique geometries
home_unique = homeregions.drop_duplicates(subset=['geometry'],keep='first')
home_unique.reset_index(inplace=True)

#%% GETTING CLOSEST HUBS VIA OSM

# this takes many many many many many hours
# also needs to be split due to OSM API limits

split_1 = 3000
split_2 = 6000
split_3 = 9000
split_4 = 12000
split_5 = 15000
split_6 = 18000
split_7 = 21000

home_unique_1 = home_unique[0:split_1]
home_unique_2 = home_unique[split_1:split_2]
home_unique_3 = home_unique[split_2:split_3]
home_unique_4 = home_unique[split_3:split_4]
home_unique_5 = home_unique[split_4:split_5]
home_unique_6 = home_unique[split_5:split_6]
home_unique_7 = home_unique[split_6:split_7]
home_unique_8 = home_unique[split_7:]

start = timeit.default_timer()
home_unique_1['nearest_hub_osm'] = home_unique_1.apply(find_nearest_hub_OSM, axis=1)
home_unique_1.to_pickle(dirname + "20241103-homeunique1.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_2['nearest_hub_osm'] = home_unique_2.apply(find_nearest_hub_OSM, axis=1)
home_unique_2.to_pickle(dirname + "20241103-homeunique2.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_3['nearest_hub_osm'] = home_unique_3.apply(find_nearest_hub_OSM, axis=1)
home_unique_3.to_pickle(dirname + "20241103-homeunique3.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_4['nearest_hub_osm'] = home_unique_4.apply(find_nearest_hub_OSM, axis=1)
home_unique_4.to_pickle(dirname + "20241103-homeunique4.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_5['nearest_hub_osm'] = home_unique_5.apply(find_nearest_hub_OSM, axis=1)
home_unique_5.to_pickle(dirname + "20241103-homeunique5.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_6['nearest_hub_osm'] = home_unique_6.apply(find_nearest_hub_OSM, axis=1)
home_unique_6.to_pickle(dirname + "20241103-homeunique6.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_7['nearest_hub_osm'] = home_unique_7.apply(find_nearest_hub_OSM, axis=1)
home_unique_7.to_pickle(dirname + "20241103-homeunique7.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

start = timeit.default_timer()
home_unique_8['nearest_hub_osm'] = home_unique_8.apply(find_nearest_hub_OSM, axis=1)
home_unique_8.to_pickle(dirname + "20241103-homeunique8.pkl")
print("Complete in " + str((timeit.default_timer() - start)/60) + " minutes")

home_unique_all = gpd.GeoDataFrame(pd.concat([home_unique_1,home_unique_2,home_unique_3,home_unique_4,
                             home_unique_5,home_unique_6,home_unique_7,home_unique_8]))
home_unique_all.to_pickle(dirname+'20250131-home_unique_all.pkl')


#%% READING CLOSEST HUBS VIA OSM

home_unique_all = pd.read_pickle(dirname+"20250131-home_unique_all.pkl")

#%% CALCULATE NUMBER THAT HAVE NEW GEOMETRIES

len(home_unique_all[home_unique_all['nearest_hub']!=
                    home_unique_all['nearest_hub_osm']])/len(home_unique_all)*100
# 15.6% of geometries have a different nearest hub when using the street network approach


#%% MERGE BACK ONTO FULL DATASET
# merge on just the geometry, due to the duplicate geometries (geometry is the unique feature)

homeregions = pd.merge(homeregions, home_unique_all[['geometry','nearest_hub_osm']], 
                       left_on='geometry', right_on='geometry', how='left')

#%% PUT HOME REGIONS AND HUBS BACK ONTO DATASET

cardsum = pd.merge(cardsum, homeregions, 
                        left_on='Cardid', right_on='Cardid', how='left')

# only keep where a home geometry is found
cardsum = cardsum[cardsum['geometry'] != None]

#%% CREATE SPLIT POINT AND SAVE FILES

np.random.seed(42)

# allocate each sequence to a number between 1 and N
N = 100
cardsum["split"] = np.random.randint(1, N+1, cardsum.shape[0])

#%%

cardsum.to_pickle(dirname + "20250606-cardsum.pkl")
activities.to_pickle(dirname + "20250606-processed_activities.pkl")

#%% READ FILES

cardsum = pd.read_pickle(dirname + "20250606-cardsum.pkl")
activities = pd.read_pickle(dirname + "20250606-processed_activities.pkl")

#%%

test = cardsum[(cardsum["split"] <= 20)]
test.reset_index(inplace=True)

# don't forget: can also calculate for a new number (e.g. >=18 and <= 25) and append later
# saves recalculating things already calculated

print(len(cardsum), "eligible sequences")
print(len(cardsum['Cardid'].unique()), "eligible cards (sequences and home regions)")

print(len(test), "sequences in the random slice")
print(len(test['Cardid'].unique()), "cards in the random slice")
print(test['Length'].sum(), "activities in the random slice")


#%% GENERATE ALL GRAPHS

allgraphs = []

start = timeit.default_timer()
for idx, row in test.iterrows():
    l = row['region_ID list']
    if len(l)>1:
        o = [(l[i], l[i+1]) for i in range(0,len(l)-1,1)]
        edge_counts = Counter(o)
        edge_counts = dict(edge_counts)
        G = nx.from_edgelist(edge_counts, create_using=nx.DiGraph())
        nx.set_edge_attributes(G, values = edge_counts, name = 'weight')
    else:
        G = nx.DiGraph()
        G.add_nodes_from(l)
    allgraphs.append(nx.to_dict_of_dicts(G))

print("Complete in " + str(timeit.default_timer() - start) + " seconds")


#%% GENERATE UNIQUE GRAPHS

test_unique = test.drop_duplicates(subset=['region_ID list'], keep='first')
# this is a unique list of regions, but not necessarily the unique graphs
# this is a way of making the pairwise distances faster - run
# over this part of the data then map back to the full dataset

print(len(test_unique), "unique sequences within the random slice")

allgraphs_unique = []

start = timeit.default_timer()
for idx, row in test_unique.iterrows():
    l = row['region_ID list']
    if len(l)>1:
        o = [(l[i], l[i+1]) for i in range(0,len(l)-1,1)]
        edge_counts = Counter(o)
        edge_counts = dict(edge_counts)
        G = nx.from_edgelist(edge_counts, create_using=nx.DiGraph())
        nx.set_edge_attributes(G, values = edge_counts, name = 'weight')
    else:
        G = nx.DiGraph()
        G.add_nodes_from(l)
    allgraphs_unique.append(nx.to_dict_of_dicts(G))

print("Complete in " + str(timeit.default_timer() - start) + " seconds")
 
#%%

mapping_V = {'W2':'W2', 'R1':'R1', 'W1':'W1', 'V0.0':'V', 'V2.0':'V', 'V5.0':'V', 'V4.0':'V', 'V3.0':'V', 'V1.0':'V',
       'L1':'L1', 'R2':'R2', 'V7.0':'V', 'L2':'L2', 'L3':'L3', 'C1':'C1', 'V6.0':'V', 'V9.0':'V', 'V8.0':'V', 'L4':'L4',
       'L5':'L5', 'V15.0':'V', 'V14.0':'V', 'V10.0':'V', 'V16.0':'V', 'V17.0':'V', 'V12.0':'V',
       'V13.0':'V', 'V11.0':'V', 'C2':'C2', 'W3':'W3', 'E1':'E1', 'V18.0':'V', 'R3':'R3', 'V19.0':'V', 'E2':'E2',
       'C3':'C3', 'V20.0':'V', 'W4':'W4', 'W5':'W5', 'V21.0':'V', 'E3':'E3', 'V22.0':'V', 'V25.0':'V',
       'V24.0':'V', 'V23.0':'V'}

mapping_AV = {'W2':'A', 'R1':'A', 'W1':'A', 'V0.0':'V', 'V2.0':'V', 'V5.0':'V', 'V4.0':'V', 'V3.0':'V', 'V1.0':'V',
        'L1':'A', 'R2':'A', 'V7.0':'V', 'L2':'A', 'L3':'A', 'C1':'A', 'V6.0':'V', 'V9.0':'V', 'V8.0':'V', 'L4':'A',
        'L5':'A', 'V15.0':'V', 'V14.0':'V', 'V10.0':'V', 'V16.0':'V', 'V17.0':'V', 'V12.0':'V',
        'V13.0':'V', 'V11.0':'V', 'C2':'A', 'W3':'A', 'E1':'A', 'V18.0':'V', 'R3':'A', 'V19.0':'V', 'E2':'A',
        'C3':'A', 'V20.0':'V', 'W4':'A', 'W5':'A', 'V21.0':'V', 'E3':'A', 'V22.0':'V', 'V25.0':'V',
        'V24.0':'V', 'V23.0':'V'}


#%%

G = nx.from_dict_of_dicts(allgraphs[1162], create_using=nx.DiGraph())
nx.set_node_attributes(G, mapping_V, 'region_type')

labels = nx.get_node_attributes(G, 'region_type')
col_A = mycolors_discrete[1]
col_V = mycolors_discrete[2]
color_map = {'V': col_V, 'A':col_A,'W2':col_A, 'R1':col_A, 'W1':col_A, 'L1':col_A, 'R2':col_A, 
             'L2':col_A, 'L3':col_A, 'C1':col_A, 'L4':col_A,
       'L5':col_A, 'C2':col_A, 'W3':col_A, 'E1':col_A, 'R3':col_A, 'E2':col_A, 'C3':col_A, 'W4':col_A, 
       'W5':col_A, 
       'E3':col_A}
node_colors = [color_map[node[1]['region_type']] for node in G.nodes(data=True)]

nx.draw(G, labels = labels, node_color=node_colors, connectionstyle='arc3, rad=0.1', font_size=18, node_size=1000)  

 
#%% DEMONSTRATION OF ISOMORPHISM CHECKS 

G = nx.from_dict_of_dicts(allgraphs_unique[1], create_using=nx.DiGraph())
H = nx.from_dict_of_dicts(allgraphs_unique[10], create_using=nx.DiGraph()) 

plt.figure(1)
pos=nx.spring_layout(G)
nx.draw(G,pos, with_labels=True, connectionstyle='arc3, rad=0.2')
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.figure(2)
pos=nx.spring_layout(H)
nx.draw(H,pos, with_labels=True, connectionstyle='arc3, rad=0.2')
labels = nx.get_edge_attributes(H,'weight')
nx.draw_networkx_edge_labels(H,pos,edge_labels=labels)

plt.show()
# note that networkx doesn't cope with edge labels on curved edges,
# but this is just for illustration

# ignoring node labels
print("is isomorphic, ignoring node labels:",nx.is_isomorphic(G, H)) # True
print("edit distance, ignoring node labels:",nx.graph_edit_distance(G, H)) # 0

# now considering node labels
G = nx.convert_node_labels_to_integers(G, label_attribute="label")
H = nx.convert_node_labels_to_integers(H, label_attribute="label")

nm = iso.categorical_node_match("label", None)

print("is isomorphic, with node labels:", nx.is_isomorphic(G, H, node_match=nm)) # now False
print("edit distance, with node labels: ", nx.graph_edit_distance(G, H, node_match=nm)) # 2

em = iso.numerical_edge_match("weight", 1) 

print("is isomorphic, with weights:", nx.is_isomorphic(G, H, edge_match=em)) # False
print("edit distance, with weights: ", nx.graph_edit_distance(G, H, edge_match=em))  # 2

print("is isomorphic, with labels and weights:", nx.is_isomorphic(G, H, node_match=nm, edge_match=em)) # False
print("edit distance, with node labels and weights: ", nx.graph_edit_distance(G, H, node_match=nm, edge_match=em)) # 4


#%% USEFUL FUNCTIONS

def create_graph(x):
    G = nx.from_dict_of_dicts(x, create_using=nx.DiGraph()) 
    G = nx.convert_node_labels_to_integers(G, label_attribute="label")
    nx.set_edge_attributes(G, values = edge_counts, name = 'weight')  
    
    for node1, node2, data in G.edges.data():
        G[node1][node2]['start_node'] = G.nodes[node1]['label']
        G[node1][node2]['end_node'] = G.nodes[node2]['label']
        G[node1][node2]['start_node_type'] = G.nodes[node1]['label'][0]
        G[node1][node2]['end_node_type'] = G.nodes[node2]['label'][0]
        
        if G.nodes[node1]['label'][:1]=="V":
            G.nodes[node1]['region_cluster'] = G.nodes[node1]['label'][3:]
            G.nodes[node1]['label'] = G.nodes[node1]['label'][:2]
        
    return G

def create_graph_V(x):
    G = nx.from_dict_of_dicts(x, create_using=nx.DiGraph()) 
    nx.set_node_attributes(G, mapping_V, 'region_type')
    G = nx.convert_node_labels_to_integers(G, label_attribute="label")
    nx.set_edge_attributes(G, values = edge_counts, name = 'weight')  
    
    for node1, node2, data in G.edges.data():
        G[node1][node2]['start_node'] = G.nodes[node1]['label']
        G[node1][node2]['end_node'] = G.nodes[node2]['label']
        G[node1][node2]['start_node_type'] = G.nodes[node1]['label'][0]
        G[node1][node2]['end_node_type'] = G.nodes[node2]['label'][0]
        
        if G.nodes[node1]['label'][:1]=="V":
            G.nodes[node1]['region_cluster'] = G.nodes[node1]['label'][3:]
            G.nodes[node1]['label'] = G.nodes[node1]['label'][:2]

    return G
 
def create_graph_AV(x):
    G = nx.from_dict_of_dicts(x, create_using=nx.DiGraph()) 
    nx.set_node_attributes(G, mapping_AV, 'region_type')
    G = nx.convert_node_labels_to_integers(G, label_attribute="label")
    nx.set_edge_attributes(G, values = edge_counts, name = 'weight')  
    
    for node1, node2, data in G.edges.data():
        G[node1][node2]['start_node'] = G.nodes[node1]['label']
        G[node1][node2]['end_node'] = G.nodes[node2]['label']
        G[node1][node2]['start_node_type'] = G.nodes[node1]['label'][0]
        G[node1][node2]['end_node_type'] = G.nodes[node2]['label'][0]
        
        if G.nodes[node1]['label'][:1]=="V":
            G.nodes[node1]['region_cluster'] = G.nodes[node1]['label'][3:]
            G.nodes[node1]['label'] = G.nodes[node1]['label'][:2]

    return G

def test_graph_isomorphism(x, y):    
    G1 = create_graph(x)
    G2 = create_graph(y)
    if G1.number_of_nodes() == G2.number_of_nodes():
        if nx.is_isomorphic(G1, G2):
            dist = 0 # if isomorphic, then distance is 0
        else:
            dist = 1 # set distance to 1 if not isomorphic. 
            # doesn't matter what this value is as only filter on where ==0 - this one could be anything
    else:
        dist = 1 # can't be isomorphic if have different number of nodes
    return dist

def test_graph_isomorphism_labelsV(x, y):    
    G1 = create_graph_V(x)
    G2 = create_graph_V(y)
    nm = iso.categorical_node_match("region_type", None)
    if G1.number_of_nodes() == G2.number_of_nodes():
        if nx.is_isomorphic(G1, G2, node_match=nm):
            dist = 0 # if isomorphic, then distance is 0
        else:
            dist = 1 # set distance to 1 if not isomorphic. 
            # doesn't matter what this value is as only filter on where ==0 - this one could be anything
    else:
        dist = 1 # can't be isomorphic if have different number of nodes
    return dist

def test_graph_isomorphism_labelsAV(x, y):    
    G1 = create_graph_AV(x)
    G2 = create_graph_AV(y)
    nm = iso.categorical_node_match("region_type", None)
    if G1.number_of_nodes() == G2.number_of_nodes():
        if nx.is_isomorphic(G1, G2, node_match=nm):
            dist = 0 # if isomorphic, then distance is 0
        else:
            dist = 1 # set distance to 1 if not isomorphic. 
            # doesn't matter what this value is as only filter on where ==0 - this one could be anything
    else:
        dist = 1 # can't be isomorphic if have different number of nodes
    return dist

def test_graph_isomorphism_labels(x, y):    
    G1 = create_graph(x)
    G2 = create_graph(y)
    nm = iso.categorical_node_match("label", None)
    if G1.number_of_nodes() == G2.number_of_nodes():
        if nx.is_isomorphic(G1, G2, node_match=nm):
            dist = 0 # if isomorphic, then distance is 0
        else:
            dist = 1 # set distance to 1 if not isomorphic. 
            # doesn't matter what this value is as only filter on where ==0 - this one could be anything
    else:
        dist = 1 # can't be isomorphic if they have a different number of nodes
    return dist

def divergence_calc(df, graphs, colname, metric):
    start = timeit.default_timer()

    transformed_strings = np.array(graphs).reshape(-1,1)
   
    if metric == "graph_iso_label":
        dist = pdist(transformed_strings, lambda x,y: test_graph_isomorphism_labels(x[0],y[0]))
 
    if metric == "graph_iso_label_V":
        dist = pdist(transformed_strings, lambda x,y: test_graph_isomorphism_labelsV(x[0],y[0]))   
 
    if metric == "graph_iso_label_AV":
        dist = pdist(transformed_strings, lambda x,y: test_graph_isomorphism_labelsAV(x[0],y[0]))   
 
    if metric == "graph_iso":
        dist = pdist(transformed_strings, lambda x,y: test_graph_isomorphism(x[0],y[0]))

    dist = pd.DataFrame(dist, columns=["distance"])
    
    print(colname, ", metric:", metric, "complete in",str((timeit.default_timer() - start)/60),"minutes")

    return dist

def count_V_regions(row):
    x = sum(1 for s in row if 'V' in s[0])
    return x

def create_undirected_graph(x):
    G = nx.from_dict_of_dicts(x, create_using=nx.Graph()) 
    G = nx.convert_node_labels_to_integers(G, label_attribute="label")
    nx.set_edge_attributes(G, values = edge_counts, name = 'weight')  
    
    for node1, node2, data in G.edges.data():
        G[node1][node2]['start_node'] = G.nodes[node1]['label']
        G[node1][node2]['end_node'] = G.nodes[node2]['label']
        G[node1][node2]['start_node_type'] = G.nodes[node1]['label'][0]
        G[node1][node2]['end_node_type'] = G.nodes[node2]['label'][0]
        
        if G.nodes[node1]['label'][:1]=="V":
            G.nodes[node1]['region_cluster'] = G.nodes[node1]['label'][3:]
            G.nodes[node1]['label'] = G.nodes[node1]['label'][:2]
        
    return G

def get_num_edges(x):
    G1 = create_graph(allgraphs[x])
    return len(G1.edges)#, nx.diameter(G1), nx.radius(G1) # gives number of unique edges, degree, diameter and radius

def get_diameter(x):
    G1 = create_undirected_graph(allgraphs[x])
    return nx.diameter(G1)#, nx.diameter(G1), nx.radius(G1) # gives number of unique edges, degree, diameter and radius

def get_diameter_dir(x):
    G1 = create_graph(allgraphs[x])
    try: 
        diameter = nx.diameter(G1)
    except Exception:
        diameter = math.nan
    return diameter #, nx.diameter(G1), nx.radius(G1) # gives number of unique edges, degree, diameter and radius


def get_row_counts(colname):
    npmatrix = squareform(distances[colname])    
    rows, indices, inverse, counts = np.unique(npmatrix, axis=0, 
                                      return_index=True, 
                                      return_counts=True,
                                      return_inverse=True)
    inverse = pd.DataFrame(inverse, columns=['grouping'])

    return inverse

def draw_graph_nolabels(i):
    plt.figure() 
    G = nx.from_dict_of_dicts(allgraphs[i], create_using=nx.DiGraph())
    nx.draw(G, connectionstyle='arc3, rad=0.2', node_size=900) 

# Day 1 = Tuesday, so weekend = days 5 and 6 = hours 96-144
def count_weekend(row):
    x = sum((i>=96 and i<=144) for i in row)
    return x

#TRAVELDAYSTART = 5 already defined
TRAVELMORNINGSTART = 10
TRAVELAFTERNOONSTART = 15
TRAVELNIGHTSTART = 20

def count_early(row):
    x = sum(i<TRAVELMORNINGSTART for i in row)
    return x

def count_morning(row):
    x = sum((i>=TRAVELMORNINGSTART and i<TRAVELAFTERNOONSTART) for i in row)
    return x

def count_afternoon(row):
    x = sum((i>=TRAVELAFTERNOONSTART and i<TRAVELNIGHTSTART) for i in row)
    return x

def count_night(row):
    x = sum((i>=TRAVELNIGHTSTART) for i in row)
    return x


#%% OPTION 1: CALCULATE DISTANCE FOR RANDOM SAMPLE OF CARDS

x1 = allgraphs_unique[0]
y1 = allgraphs_unique[1]

test_graph_isomorphism(x1, y1)

#%%
x2 = {'V3.0': {'R1': {'weight': 2}, 'V3.0': {'weight': 1}}, 'R1': {'V3.0': {'weight': 1}}}
y2 = {'V2.0': {'R1': {'weight': 2}, 'V2.0': {'weight': 1}}, 'R1': {'V2.0': {'weight': 1}}}
test_graph_isomorphism(x2, y2)
# is isomorphic if you ignore the labels

#%%

test_graph_isomorphism_labelsV(x2, y2)
# is isomorphic when looking at the visited regions, as V2.0 and V3.0 are seen as the same

#%%
test_graph_isomorphism_labelsAV(x2, y2)
# and thus of course is also isomorphic when looking at both region types

#%%
x3 = {'L1': {'R1': {'weight': 2}, 'L1': {'weight': 1}}, 'R1': {'L1': {'weight': 1}}}
y3 = {'V2.0': {'R1': {'weight': 2}, 'V2.0': {'weight': 1}}, 'R1': {'V2.0': {'weight': 1}}}
test_graph_isomorphism_labelsAV(x3, y3)
# not isomorphic because L1 is an anchoring region and V2.0 is a visited region

#%%

#distances = pd.DataFrame()
print(len(test_unique), "sequences in dataset")
print(len(test_unique)*len(test_unique)/2-len(test_unique)/2, "distances to calculate")

#distances['regionID graph iso'] = divergence_calc(test_unique, allgraphs_unique, 'region_ID list', 'graph_iso')
#distances['regionID graph iso labels'] = divergence_calc(test_unique, allgraphs_unique, 
 #                                                        'region_ID list', 'graph_iso_label')
#distances['regionID graph iso labels_V'] = divergence_calc(test_unique, allgraphs_unique, 'region_ID list', 'graph_iso_label_V')
distances['regionID graph iso labels_AV'] = divergence_calc(test_unique, allgraphs_unique, 'region_ID list', 'graph_iso_label_AV')

distances.to_pickle(dirname + "20250920-distances.pkl")

#%% OPTION 2: READ DISTANCE FOR RANDOM SAMPLE OF CARDS

distances = pd.read_pickle(dirname + "20250920-distances.pkl") 

#%% FINDING UNIQUE GRAPH SHAPES

inverse = get_row_counts('regionID graph iso')
inverse.rename(columns={'grouping':'iso_group'}, inplace=True)
test_unique['iso_group'] = inverse['iso_group'].values

inverse_labels = get_row_counts('regionID graph iso labels')
inverse_labels.rename(columns={'grouping':'iso_group labels'}, inplace=True)
test_unique['iso_group labels'] = inverse_labels['iso_group labels'].values 


inverse_labelsV = get_row_counts('regionID graph iso labels_V')
inverse_labelsV.rename(columns={'grouping':'iso_group labels_V'}, inplace=True)
test_unique['iso_group labelsV'] = inverse_labelsV['iso_group labels_V'].values

inverse_labelsAV = get_row_counts('regionID graph iso labels_AV')
inverse_labelsAV.rename(columns={'grouping':'iso_group labels_AV'}, inplace=True)
test_unique['iso_group labelsAV'] = inverse_labelsAV['iso_group labels_AV'].values


#%%

test['region list'] = test['region_ID list'].apply(tuple) # list isn't hashable so turn into tuple
test_unique['region list'] = test_unique['region_ID list'].apply(tuple)

test = pd.merge(test, test_unique[['region list', 'iso_group', 'iso_group labels','iso_group labelsV', 'iso_group labelsAV']], 
                left_on='region list', right_on='region list', how='left')

#%% PLOT AN EXAMPLE

group = 14 # groups are numbered sequentially, the group number doesn't refer to the index
test_row = inverse[inverse['iso_group']==group].index.to_list()[0] # get the first index where this group occurs

draw_graph_nolabels(test_row)

#%% CALCULATE USEFUL STATS ON EACH SEQUENCE

# count of number of unique regions in each sequence
test['unique regions'] = test['region_ID list'].apply(np.unique)
test['region count'] = test['unique regions'].str.len()
test['seqID'] = test['Cardid'].astype(str)+'-'+test['Period'].astype(str)
test['V unique_reg'] = test['unique regions'].apply(count_V_regions)
test['act_weekend'] = test['Start_h list'].apply(count_weekend)
test['act_weekday'] = test['Length'] - test['act_weekend']
test['act_early'] = test['Day_h list'].apply(count_early)
test['act_morning'] = test['Day_h list'].apply(count_morning)
test['act_afternoon'] = test['Day_h list'].apply(count_afternoon)
test['act_night'] = test['Day_h list'].apply(count_night)
test['anchoring_reg_count'] = test['region count'] - test['V unique_reg']
test['visited_reg_count'] = test['V unique_reg']

test['num_edges'] = test.apply(lambda row: get_num_edges(row.name), axis=1)
test['diameter'] = test.apply(lambda row: get_diameter(row.name), axis=1)

test['diameter_dir'] = test.apply(lambda row: get_diameter_dir(row.name), axis=1)
# used to work out what % of the dataset would have infinite
# diameter if we used the directed version
# len(test[test['diameter_dir'].isna()])/len(test) = approx 22%


#%% PLOT OF COUNTS OF GRAPH TYPES

# reminder: columns are
#'iso_group' = ignoring labels
#'iso_group labels = include all node labels
#'iso_group labelsV' = label all visited regions as 'V' (so should have less unique)
#'iso_group labelsAV' = label all visited regions as 'V' and all anchoring regions as 'A' (so should have even less unique)

counts = test["iso_group"].value_counts()
counts = pd.DataFrame(counts)
counts.reset_index(inplace=True)
counts['cumsum'] = counts['count'].cumsum()
counts['cum_percent'] = counts['cumsum']/counts['count'].sum()*100

countsV = test["iso_group labelsV"].value_counts()
countsV = pd.DataFrame(countsV)
countsV.reset_index(inplace=True)
countsV['cumsum'] = countsV['count'].cumsum()
countsV['cum_percent'] = countsV['cumsum']/countsV['count'].sum()*100

countsAV = test["iso_group labelsAV"].value_counts()
countsAV = pd.DataFrame(countsAV)
countsAV.reset_index(inplace=True)
countsAV['cumsum'] = countsAV['count'].cumsum()
countsAV['cum_percent'] = countsAV['cumsum']/countsAV['count'].sum()*100


perc_interest = 90

print("To cover",perc_interest,"%:")
print("Ignore weights and labels:",len(counts[counts['cum_percent']<=perc_interest]), "of total", len(counts))
print("With all visited labelled V:",len(countsV[countsV['cum_percent']<=perc_interest]), "of total", len(countsV))
print("With all visited labelled V and anchoring labelled A:",len(countsAV[countsAV['cum_percent']<=perc_interest]), "of total", len(countsAV))

#%% DRAW TOP N GRAPHS WITH NO LABELS IN FULL DATASET

n=5

groups = pd.DataFrame(test['iso_group'][test['diameter']>0].value_counts())
groups.reset_index(inplace=True)
groups.rename(columns={'iso_group':'group_id'}, inplace=True)
total_graphs = groups['count'].sum()
groups['percent'] = groups['count']/total_graphs*100
groups['cum_percent'] = groups['percent'].cumsum()


print("Contribution of top 5 graphs, no labelling:")
print(groups['percent'].head(5))

print("Percent covered by top 5:",groups['cum_percent'].iloc[4])
print("Percent covered by next 5:",groups['cum_percent'].iloc[9]-groups['cum_percent'].iloc[4])
print("Number of graphs that occur only once:",len(groups[groups['count']==1]))
print("Percent of graphs that occur only once:",len(groups[groups['count']==1])/len(groups)*100)

for i in range(0,n):
    
    groupid = groups['group_id'].loc[i]
    
    index_i = test.index[(test['iso_group']==groupid)][0] 
    
    seqID_interest = test['seqID'].iloc[index_i]
    
    seqID_info = test[test['seqID']==seqID_interest]
    graph_interest = seqID_info.index.values[0]
    
    print(seqID_interest)
    
    plt.figure((n+1)*i+1) #1, 7, 13...
    G = nx.from_dict_of_dicts(allgraphs[graph_interest], create_using=nx.DiGraph())
    nx.draw(G, connectionstyle='arc3, rad=0.1', font_size=18, node_size=1200, node_color=mycolors_discrete[1]) 

    #plt.savefig(dirname+"figs\\svg\\"+'routines_nolabel_'+str(i)+".svg")


#%% DRAW TOP N GRAPHS WITH LABELS IN FULL DATASET

n=5

groups_l = pd.DataFrame(test['iso_group labelsAV'][test['diameter']>0].value_counts())
groups_l.reset_index(inplace=True)
groups_l.rename(columns={'iso_group labelsAV':'group_id'}, inplace=True)
total_graphs = groups_l['count'].sum()
groups_l['percent'] = groups_l['count']/total_graphs*100
groups_l['cum_percent'] = groups_l['percent'].cumsum()

print("Contribution of top 5 graphs, with labelling:")
print(groups_l['percent'].head(5))

print("Percent covered by top 5:",groups_l['cum_percent'].iloc[4])
print("Percent covered by next 5:",groups_l['cum_percent'].iloc[9]-groups_l['cum_percent'].iloc[4])
print("Number of graphs that occur only once:",len(groups_l[groups_l['count']==1]))
print("Percent of graphs that occur only once:",len(groups_l[groups_l['count']==1])/len(groups_l)*100)

for i in range(0,n):
    
    groupid = groups_l['group_id'].loc[i]
    
    index_i = test.index[(test['iso_group labelsAV']==groupid)][0] 
    
    seqID_interest = test['seqID'].iloc[index_i]
    
    seqID_info = test[test['seqID']==seqID_interest]
    graph_interest = seqID_info.index.values[0]
    
    print(seqID_interest)
    
    plt.figure((n+1)*i+1) #1, 7, 13...
    G = nx.from_dict_of_dicts(allgraphs[graph_interest], create_using=nx.DiGraph())
    nx.set_node_attributes(G, mapping_AV, 'region_type')
    labels = nx.get_node_attributes(G, 'region_type')
    node_colors = [color_map[node[1]['region_type']] for node in G.nodes(data=True)]
    nx.draw(G, labels = labels, node_color=node_colors, connectionstyle='arc3, rad=0.1', font_size=18, node_size=1200) 
        
    #plt.savefig(dirname+"figs\\svg\\"+'routines_label_'+str(i)+".svg")


#%%

plotdf = pd.DataFrame()
plotdf = pd.concat([groups_l['cum_percent']], axis=1, ignore_index=True) 
plotdf.ffill(inplace=True)

plotdf.rename(columns={0:'Visited labelled V and anchoring labelled A'}, inplace=True) 
plotdf['num'] = plotdf.index+1

plotdf1 = plotdf.head(n=100)
fig = px.line(plotdf1, x="num", y=['Visited labelled V and anchoring labelled A'],
              color_discrete_sequence=[mycolors_discrete[7]], 
              labels={
                       "value": "Cumulative %",
                       "num": "Number of graphs",
                       "variable":"Graph labelling",                      
                   }) 
fig.update_layout(yaxis_range=[0,100],
                  showlegend=False,)
fig.update_layout(default_layout_journal)
pyo.plot(fig, config=config)

filename = "routines_cumulativegraphcount"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% HISTOGRAM OF GRAPH COUNTS

fig = px.histogram(groups_l, x="count", histnorm='percent', color_discrete_sequence=[mycolors_discrete[7]], nbins=20000)
fig.update_layout(xaxis_title="Frequency of graph occurrence", yaxis_title="Percent of dataset")

fig.update_xaxes(
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
            range=[0, 100])

fig.update_yaxes(
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
        range=[0, 100],
    )

fig.update_layout(default_layout_journal)
pyo.plot(fig)

filename = "routines_histgraphcount"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% GET COUNTS OF CARDS IN DATASET
# idea here is to see how different weeks appear for a card
# because of the way the random sample of cards is picked, not all cards
# will appear for the whole month

# get count of times each card appears
cardcounts = pd.DataFrame(test['Cardid'].value_counts())
print(cardcounts.head(20))

#%% EXPLORING GRAPHS FOR A CARD

card_id = 1374150

testgraphs = test[test['Cardid']==card_id]

cardgraphs = []
for idx, row in testgraphs.iterrows():
    l = row['region_ID list']
    if len(l)>1:
        o = [(l[i], l[i+1]) for i in range(0,len(l)-1,1)]
        edge_counts = Counter(o)
        edge_counts = dict(edge_counts)
        G = nx.from_edgelist(edge_counts, create_using=nx.DiGraph())
        nx.set_edge_attributes(G, values = edge_counts, name = 'weight')
    else:
        G.add_nodes_from(l)
    cardgraphs.append(nx.to_dict_of_dicts(G))

for i in range(0, len(cardgraphs)):
    plt.figure()
    G = nx.from_dict_of_dicts(cardgraphs[i], create_using=nx.DiGraph())
    nx.draw(G, connectionstyle='arc3, rad=0.1', with_labels=True, font_size=18, node_size=1000) 
    
    # this works for cards that appear in every month - as the period will be 0, 1, 2, 3 where it appears 4 times
    # if it doesn't appear four times need to work out which periods it's in for
    act_card = activities[(activities['Cardid']==card_id)&(activities['Travel_period_start']==i)]

    act_card['Travel_day_start'] = act_card['Travel_day_start'].astype(int)
    act_card['hour'] = act_card['Day_h'].astype(int)
    act_card['min'] = (act_card['Day_h'] - act_card['hour'])*60
    act_card['min'] = act_card['min'].apply(lambda x: round(x, 0))
    act_card['min'] = act_card['min'].astype(int)

    print(act_card[['Travel_day_start','hour','min','final_regionID']])

#%% CLUSTERING DATA PREPARATION

all_cols = ['num_edges', 'region count', 'anchoring_reg_count',
            'visited_reg_count','diameter', 'iso_group labelsAV', 'diameter_dir']
cluster_cols = ['num_edges', 'anchoring_reg_count',
            'visited_reg_count', 'diameter']

cluster_input = test[all_cols]
cluster_input = cluster_input[cluster_input['diameter']>0]
               
print("Length of input before dropping duplicates: ",len(cluster_input))
cluster_input = cluster_input.drop_duplicates(subset=cluster_cols+['iso_group labelsAV'])
 # include iso_group here so things with identical stats but different shape 
 # are still included
print("Length of input after dropping duplicates: ",len(cluster_input))
print(cluster_input.head(10))

#%% what percentage of cluster inputs would have infinite diameter?

len(cluster_input[cluster_input['diameter_dir'].isna()])/len(cluster_input)*100
# 49%

#%% HISTOGRAMS OF CLUSTER INPUTS


# To demonstrate distributions (and how little variation there is) before clustering

hist_input = pd.melt(cluster_input[cluster_cols])
hist_input['variable'] = np.where(hist_input['variable']=='num_edges', "Number of edges", hist_input['variable'])
hist_input['variable'] = np.where(hist_input['variable']=='anchoring_reg_count', "Number of anchoring regions", hist_input['variable'])
hist_input['variable'] = np.where(hist_input['variable']=='visited_reg_count', "Number of visited regions", hist_input['variable'])
hist_input['variable'] = np.where(hist_input['variable']=='diameter', "Diameter", hist_input['variable'])

fig = make_subplots(rows=2, cols=2,
                    shared_xaxes='all',
                    shared_yaxes='all',
                    vertical_spacing=0.09,
                    horizontal_spacing=0.04,
                    x_title="Value",
                    y_title="Frequency",
                    subplot_titles=("Number of anchoring regions", "Number of edges", "Number of visited regions", "Diameter"))

fig.add_trace(go.Histogram(x=hist_input['value'][hist_input['variable']=='Number of anchoring regions'], marker_color=mycolors_discrete[7]), row=1, col=1)
fig.add_trace(go.Histogram(x=hist_input['value'][hist_input['variable']=='Number of visited regions'], marker_color=mycolors_discrete[7]), row=2, col=1)
fig.add_trace(go.Histogram(x=hist_input['value'][hist_input['variable']=='Number of edges'], marker_color=mycolors_discrete[7]), row=1, col=2)
fig.add_trace(go.Histogram(x=hist_input['value'][hist_input['variable']=='Diameter'], marker_color=mycolors_discrete[7]), row=2, col=2)
fig.update_layout(default_layout_journal, showlegend=False)
fig.update_xaxes(
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
            range=[0, 20])

fig.update_yaxes(
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
        title_standoff = 50,
        range=[0, 2500],
        #title='Frequency'
    )

fig['layout']['annotations'][5]['y'] = 0.54 
#pyo.plot(fig, config=config)

filename = "routines_clusterinputhist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1000, height=1000)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% NORMALISE Z-SCORE

def calc_z_score(df, col, suffix):
    outcol = col+suffix
    df[outcol] = (df[col]-df[col].mean())/df[col].std()
    
suffix = "_Z"

for i in range(0,len(cluster_cols)):
    calc_z_score(cluster_input, cluster_cols[i], suffix)

cluster_cols_suffix = [sub + suffix for sub in cluster_cols] 

#%% CLUSTER GRAPHS - GAUSSIAN MIXTURE MODEL

# set random state
init_state = 475

# To determine the appropriate amount of clusters
# Run GMM across 1 to 20 clusters and check AIC/BIC
n_components = np.arange(1, 20)
models = [GaussianMixture(n, random_state=init_state).fit(cluster_input[cluster_cols_suffix]) for n in n_components]

d = {"BIC": [m.bic(cluster_input[cluster_cols_suffix]) for m in models], 
      "AIC": [m.aic(cluster_input[cluster_cols_suffix]) for m in models]}
df = pd.DataFrame(data=d)

#%% PLOT TO DETERMINE NUMBER OF CLUSTERS
fig = px.line(
    df,
    color_discrete_sequence=mycolors_discrete,
    labels={"index": "Number of clusters", 
            "value": "", 
            "variable": "Criteria"},
)
fig.update_layout(default_layout_journal)
pyo.plot(fig, config=config)

filename = "routines_AICBIC"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% TEST CLUSTER STABILITY

def hamming_distance(df, col1, col2):
    return len(df[df[col1]!=df[col2]])/len(df)

def calc_max_instability(n):
    return (n*(n-1)/2)/(n*n)

def calc_instability(num_clusters, num_subsamples, sample_size):
    np.random.seed(42)
    
    states = np.random.randint(1000, size=num_subsamples)

    for i in np.arange(0,num_subsamples):
        
        Y = cluster_input[cluster_cols_suffix].sample(n=sample_size, random_state=states[i])
        GMM = GaussianMixture(n_components=num_clusters, random_state=states[i]).fit(Y)
        sub_col = 'subsample_'+str(i)
        cluster_input[sub_col] = GMM.predict(cluster_input[cluster_cols_suffix])    
        means = GMM.means_
        
        if i==0:
            means0 = GMM.means_
        
        if i>0:
            dist = scidist.cdist(means0, means, 'euclidean')
            min_loc = np.argmin(dist, axis=0)     
            colmap = dict(zip(np.arange(0,num_clusters),min_loc))
            cluster_input[sub_col] = cluster_input[sub_col].map(colmap)
    
    features = cluster_input.iloc[:,-num_subsamples:]
    cluster_input.drop(columns=cluster_input.columns[-num_subsamples:], 
                       axis=1, inplace=True)
    features = features.transpose()
    
    distmatrix = pdist(features, metric='hamming')
    npmatrix = squareform(distmatrix) 
    
    inst_max = calc_max_instability(num_subsamples)
    
    inst = npmatrix.sum()/np.square(num_subsamples)/inst_max
    
    #fig = px.imshow(npmatrix)
    #pyo.plot(fig, config=config)

    return inst

#%%

k_max = 12

val10 = [calc_instability(n, 15, int(round(0.7*len(cluster_input),0))) for n in np.arange(2,k_max)]
val20 = [calc_instability(n, 15, int(round(0.75*len(cluster_input),0))) for n in np.arange(2,k_max)]
val30 = [calc_instability(n, 15, int(round(0.8*len(cluster_input),0))) for n in np.arange(2,k_max)]
val40 = [calc_instability(n, 15, int(round(0.85*len(cluster_input),0))) for n in np.arange(2,k_max)]
val50 = [calc_instability(n, 15, int(round(0.9*len(cluster_input),0))) for n in np.arange(2,k_max)]
val60 = [calc_instability(n, 15, int(round(0.95*len(cluster_input),0))) for n in np.arange(2,k_max)]

instability = {"num_clusters": [n for n in np.arange(2,k_max)], 
               #"70%": [val10[i-2] for i in np.arange(2,k_max)],
               "75%": [val20[i-2] for i in np.arange(2,k_max)],
               "80%": [val30[i-2] for i in np.arange(2,k_max)],
               "85%": [val40[i-2] for i in np.arange(2,k_max)],
               "90%": [val50[i-2] for i in np.arange(2,k_max)],
               "95%": [val60[i-2] for i in np.arange(2,k_max)]}

instability = pd.DataFrame(data = instability)


#%% PLOT OF CLUSTER INSTABILITY

fig = px.bar(
    instability,
    x="num_clusters", y=["75%", "80%", "85%", "90%", "95%"],
    color_discrete_sequence=mycolors_discrete,
    barmode="group",
    labels={'variable':'Sample size'}
)
fig.update_layout(default_layout_journal)
fig.update_layout(xaxis_title="Number of clusters",
                  yaxis_title="Instability",)
for i in np.arange(instability['num_clusters'].min(),instability['num_clusters'].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=1, line_color=line_colour, line_width=line_width)
fig.update_layout(yaxis_range=[0,1],     xaxis=dict(
        tickmode='linear',  # Ensures ticks are placed at regular intervals
        dtick=1             # Sets the interval between ticks to 1 unit
    ))


pyo.plot(fig, config=config)

filename = "routines_stability"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% FIT GMM TO SELECTED NUMBER OF CLUSTERS
# From this plot, determine that the appropriate number of clusters

NUM_COMPONENTS = 7 

GMM = GaussianMixture(n_components=NUM_COMPONENTS, random_state=init_state).fit(cluster_input[cluster_cols_suffix])
colname_init = "Cluster_"+str(init_state)

cluster_input[colname_init] = GMM.predict(cluster_input[cluster_cols_suffix])
print(cluster_input[colname_init].value_counts())

means0 = GMM.means_
cov0 = GMM.covariances_

# BOXPLOT OF CLUSTERS

outbox = pd.melt(
    cluster_input,
    id_vars=[colname_init],
    value_vars=cluster_cols,
    #value_vars=cluster_cols_suffix,
)

map_dict = {'num_edges':'Number edges', 'anchoring_reg_count':'Anchor regions', 'visited_reg_count':'Visited regions','diameter':'Diameter'}

outbox['variable'] = outbox['variable'].map(map_dict)

fig = px.box(
    outbox,
    x=colname_init,
    y="value",
    color="variable",
    color_discrete_sequence=mycolors_discrete,
    labels={'Cluster_475': "Cluster", "value": "Count", "variable": "Parameter"},
)
fig.update_layout(default_layout_box_journal)

fig.update_yaxes(
    range=(0, 20),
    constrain='domain'
)

for i in np.arange(0,outbox[colname_init].max()):
        fig.add_shape(
            type="line", xref='x', yref='y2 domain',
                            x0=i+0.5, y0=0, x1=i+0.5, y1=20, line_color=line_colour, line_width=line_width)

pyo.plot(fig, config=config)

filename = "routines_clusterboxplot"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% PUT CLUSTERS BACK ONTO FULL DATASET

# remove the uniqueness - put the cluster back onto the full dataset

test = pd.merge(
    test,
    cluster_input[all_cols+[colname_init]], 
    left_on=all_cols,
    right_on=all_cols,
    how="left",
)

test[colname_init].value_counts()

#%%
# and pickle this so we can jump right in at this point
test.rename(columns={'Token_x':'Token'}, inplace=True)
test.drop(columns='Token_y',inplace=True)

test.to_pickle(dirname + "20250926-testwithclusters.pkl")

#%% OPTION: READ PICKLE OF FINAL DATASET WITH CLUSTERS

test = pd.read_pickle(dirname + "20250926-testwithclusters.pkl") 

#%% 

groups_nolw = pd.DataFrame(test['iso_group'].value_counts())
groups_nolw.reset_index(inplace=True)
groups_nolw.rename(columns={'iso_group':'group_id'}, inplace=True)
total_graphs = groups_nolw['count'].sum()
groups_nolw['percent'] = groups_nolw['count']/total_graphs*100

print(groups_nolw.head(10))

#%% TOP GRAPHS WITH LABELS, FOR EACH TOP GRAPH WITH NO LABELS, IN EACH CLUSTER

n=5 # draw the top this many graphs

# with n=5, this draws six graphs per run, so could put inside another for loop
# but with 8 clusters, that's 48 graphs at once
# so do this manually to not drown in graphs

cluster_interest = 6
# any number between 0 and 6
# test[colname_init].max()

groups_nolw = pd.DataFrame(test['iso_group'][test[colname_init]==cluster_interest].value_counts())
groups_nolw.reset_index(inplace=True)
groups_nolw.rename(columns={'iso_group':'group_id'}, inplace=True)
total_graphs = groups_nolw['count'].sum()
groups_nolw['percent'] = groups_nolw['count']/total_graphs*100

print(groups_nolw.head(10))

for i in range(0,n):
    
    groupid = groups_nolw['group_id'].loc[i]
    
    topn_onlyl = test['iso_group labelsAV'][(test['iso_group']==groupid)&(test[colname_init]==cluster_interest)].value_counts()
    topn_onlyl = pd.DataFrame(topn_onlyl)
    topn_onlyl.reset_index(inplace=True)
    topn_onlyl = topn_onlyl.head(n)
    topn_onlyl.rename(columns={'iso_group labelsAV':'group_id'}, inplace=True)
    
    percent = topn_onlyl['count'].sum()/groups_nolw['count'].loc[i]
    
    topn_onlyl['percent of total'] = topn_onlyl['count']/total_graphs*100
    topn_onlyl['percent of graph'] = topn_onlyl['count']/groups_nolw['count'].loc[i]*100
    
    print("\nTop graph",i+1)
    
    print(topn_onlyl)
    
    print("Top",n,"covers", percent*100,"percent")
    
    index_i = test.index[(test['iso_group']==groupid)&(test[colname_init]==cluster_interest)][0] 
    
    seqID_interest = test['seqID'].iloc[index_i]
    
    seqID_info = test[test['seqID']==seqID_interest]
    graph_interest = seqID_info.index.values[0]
    
    plt.figure((n+1)*i+1) #1, 7, 13...
    G = nx.from_dict_of_dicts(allgraphs[graph_interest], create_using=nx.DiGraph())
    nx.draw(G, connectionstyle='arc3, rad=0.1', node_size=1800) 
    
    if len(topn_onlyl)<n:
        n = len(topn_onlyl)
        
    
    for j in range (0, n):
        groupl_id = topn_onlyl['group_id'].loc[j]
        
        index_j = test.index[test['iso_group labelsAV']==groupl_id][0] 
        seqID_interest = test['seqID'].iloc[index_j]
        
        seqID_info = test[test['seqID']==seqID_interest]
        graph_interest = seqID_info.index.values[0]
        
        print(seqID_interest)
        
        plt.figure((i*n)+(2+i)+j) # [2,3,4,5,6], [8,9,10,11,12] ...
        G = nx.from_dict_of_dicts(allgraphs[graph_interest], create_using=nx.DiGraph())
        nx.set_node_attributes(G, mapping_AV, 'region_type')
        labels = nx.get_node_attributes(G, 'region_type')
        node_colors = [color_map[node[1]['region_type']] for node in G.nodes(data=True)]
        nx.draw(G, labels = labels, node_color=node_colors, connectionstyle='arc3, rad=0.1', font_size=18, node_size=1200)  
    
    plt.show()

#%% TOP N GRAPHS WITH LABELS IN EACH CLUSTER

n=5 # draw the top this many graphs

cluster_interest = 6
# any number between 0 and 6
# test[colname_init].max()

groups_l = pd.DataFrame(test['iso_group labelsAV'][test[colname_init]==cluster_interest].value_counts())
groups_l.reset_index(inplace=True)
groups_l.rename(columns={'iso_group labelsAV':'group_id'}, inplace=True)
total_graphs = groups_l['count'].sum()
groups_l['percent'] = groups_l['count']/total_graphs*100

print(total_graphs,"records in cluster",cluster_interest)
print(len(groups_l),"unique graphs with labels in cluster",cluster_interest)

print(groups_l.head(10))

for i in range(0,n):
    
    groupid = groups_l['group_id'].loc[i]
    
    index_i = test.index[(test['iso_group labelsAV']==groupid)&(test[colname_init]==cluster_interest)][0] 
    
    seqID_interest = test['seqID'].iloc[index_i]
    
    seqID_info = test[test['seqID']==seqID_interest]
    graph_interest = seqID_info.index.values[0]
    
    print(seqID_interest)
    
    plt.figure((n+1)*i+1) #1, 7, 13...
    G = nx.from_dict_of_dicts(allgraphs[graph_interest], create_using=nx.DiGraph())
    nx.set_node_attributes(G, mapping_AV, 'region_type')
    labels = nx.get_node_attributes(G, 'region_type')
    node_colors = [color_map[node[1]['region_type']] for node in G.nodes(data=True)]
    nx.draw(G, labels = labels, node_color=node_colors, 
            connectionstyle='arc3, rad=0.1', font_size=18, node_size=1000) 
        


#%%

# also has merit. subplots of different combinations of parameters
fig1 = px.density_contour(cluster_input, x="anchoring_reg_count", y="visited_reg_count",color=colname_init, facet_col=colname_init)
fig2 = px.density_contour(cluster_input, x="anchoring_reg_count", y="num_edges",color=colname_init, facet_col=colname_init)
fig3 = px.density_contour(cluster_input, x="anchoring_reg_count", y="diameter",color=colname_init, facet_col=colname_init)
fig4 = px.density_contour(cluster_input, x="visited_reg_count", y="num_edges",color=colname_init, facet_col=colname_init)
fig5 = px.density_contour(cluster_input, x="visited_reg_count", y="diameter",color=colname_init, facet_col=colname_init)
fig6 = px.density_contour(cluster_input, x="num_edges", y="diameter",color=colname_init, facet_col=colname_init)
 
import plotly.subplots as sp

figure1_traces = []
figure2_traces = []
figure3_traces = []
figure4_traces = []
figure5_traces = []
figure6_traces = []
for trace in range(len(fig1["data"])):
    figure1_traces.append(fig1["data"][trace])
for trace in range(len(fig2["data"])):
    fig2["data"][trace]['showlegend'] = False   
    figure2_traces.append(fig2["data"][trace])
for trace in range(len(fig3["data"])):
    fig3["data"][trace]['showlegend'] = False   
    figure3_traces.append(fig3["data"][trace])
for trace in range(len(fig4["data"])):
    fig4["data"][trace]['showlegend'] = False   
    figure4_traces.append(fig4["data"][trace])
for trace in range(len(fig5["data"])):
    fig5["data"][trace]['showlegend'] = False   
    figure5_traces.append(fig5["data"][trace])
for trace in range(len(fig6["data"])):
    fig6["data"][trace]['showlegend'] = False   
    figure6_traces.append(fig6["data"][trace])

this_figure = sp.make_subplots(rows=6, cols=5, vertical_spacing = 0.09)

# Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
for i in np.arange(0,len(figure1_traces)):
    this_figure.append_trace(figure1_traces[i], row=1, col=i+1)
for i in np.arange(0,len(figure2_traces)):
    this_figure.append_trace(figure2_traces[i], row=2, col=i+1)
for i in np.arange(0,len(figure3_traces)):
    this_figure.append_trace(figure3_traces[i], row=3, col=i+1)
for i in np.arange(0,len(figure4_traces)):
    this_figure.append_trace(figure4_traces[i], row=4, col=i+1)
for i in np.arange(0,len(figure5_traces)):
    this_figure.append_trace(figure5_traces[i], row=5, col=i+1)
for i in np.arange(0,len(figure6_traces)):
    this_figure.append_trace(figure6_traces[i], row=6, col=i+1)

this_figure['layout']['xaxis']['title']='anchor region count'
this_figure['layout']['yaxis']['title']='visited region count'

this_figure['layout']['xaxis6']['title']='anchor region count'
this_figure['layout']['yaxis6']['title']='unique edges'

this_figure['layout']['xaxis11']['title']='anchor region count'
this_figure['layout']['yaxis11']['title']='diameter'

this_figure['layout']['xaxis16']['title']='visited region count'
this_figure['layout']['yaxis16']['title']='unique edges'

this_figure['layout']['xaxis21']['title']='visited region count'
this_figure['layout']['yaxis21']['title']='diameter'

this_figure['layout']['xaxis26']['title']='unique edges'
this_figure['layout']['yaxis26']['title']='diameter'

pyo.plot(this_figure, config=config)
# filename = "prep_staydistance_full"
# fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+filename+".svg")

#%% TOKEN TYPES BY CLUSTER

tokensummary = pd.melt(
    test,
    id_vars=[colname_init],
    value_vars='Token',
).groupby([colname_init, 'value']).count().reset_index()

totals_allocated = tokensummary.groupby(['value']).sum().reset_index()
totals_allocated= pd.DataFrame(totals_allocated)
totals_allocated.rename(columns={'variable':'total_allocated'}, inplace=True)

totals = tokensummary.groupby([colname_init]).sum().reset_index()
totals = pd.DataFrame(totals)
totals.rename(columns={'variable':'total_cluster'}, inplace=True)

tokensummary = pd.merge(tokensummary, totals_allocated[['value','total_allocated']], 
                        left_on='value', right_on='value',
                        how='left')

tokensummary = pd.merge(tokensummary, totals[[colname_init,'total_cluster']], 
                        left_on=colname_init, right_on=colname_init,
                        how='left')

tokensummary['perc_allocated'] = tokensummary['variable']/tokensummary['total_allocated']*100
tokensummary['perc_cluster'] = tokensummary['variable']/tokensummary['total_cluster']*100
tokensummary[colname_init] = tokensummary[colname_init].astype(str)

# perc_allocated = percentage of cards of this type that are allocated to this cluster
# perc_cluster = percentage of the cluster that this card makes up

# lift = (percent of this token type in this cluster) / (percent of this token type in dataset)
tokensummary['lift'] = (tokensummary['perc_cluster']/100)/(tokensummary['total_allocated']/tokensummary['variable'].sum())

#%% ACTIVITY WEEKDAY/WEEKEND SPLIT BY CLUSTER

weekendsummary = pd.melt(
    test,
    id_vars=[colname_init],
    value_vars=['act_weekend','act_weekday'],
)

weekendsummary = weekendsummary.groupby([colname_init, 'variable']).sum().reset_index() 

totals_allocated = weekendsummary[['variable','value']].groupby(['variable']).sum().reset_index()
totals_allocated= pd.DataFrame(totals_allocated)
totals_allocated.rename(columns={'value':'total_allocated'}, inplace=True)

totals = weekendsummary.groupby([colname_init]).sum().reset_index()
totals = pd.DataFrame(totals)
totals.rename(columns={'value':'total_cluster'}, inplace=True)

weekendsummary = pd.merge(weekendsummary, totals_allocated[['variable','total_allocated']], 
                        left_on='variable', right_on='variable',
                        how='left')

weekendsummary = pd.merge(weekendsummary, totals[[colname_init,'total_cluster']], 
                        left_on=colname_init, right_on=colname_init,
                        how='left')

weekendsummary['perc_allocated'] = weekendsummary['value']/weekendsummary['total_allocated']*100
weekendsummary['perc_cluster'] = weekendsummary['value']/weekendsummary['total_cluster']*100
weekendsummary[colname_init] = weekendsummary[colname_init].astype(str)

map_dict = {'act_weekday':'Weekday activities', 'act_weekend':'Weekend activities'}
weekendsummary['variable'] = weekendsummary['variable'].map(map_dict)

weekendsummary['lift'] = (weekendsummary['perc_cluster']/100)/(weekendsummary['total_allocated']/weekendsummary['value'].sum())


#%% ACTIVITY TIME OF DAY SPLIT BY CLUSTER

timesummary = pd.melt(
    test,
    id_vars=[colname_init],
    value_vars=['act_early', 'act_morning', 'act_afternoon', 'act_night'],
)

timesummary = timesummary.groupby([colname_init, 'variable']).sum().reset_index() 

totals_allocated = timesummary[['variable','value']].groupby(['variable']).sum().reset_index()
totals_allocated= pd.DataFrame(totals_allocated)
totals_allocated.rename(columns={'value':'total_allocated'}, inplace=True)

totals = timesummary.groupby([colname_init]).sum().reset_index()
totals = pd.DataFrame(totals)
totals.rename(columns={'value':'total_cluster'}, inplace=True)

timesummary = pd.merge(timesummary, totals_allocated[['variable','total_allocated']], 
                        left_on='variable', right_on='variable',
                        how='left')

timesummary = pd.merge(timesummary, totals[[colname_init,'total_cluster']], 
                        left_on=colname_init, right_on=colname_init,
                        how='left')

timesummary['perc_allocated'] = timesummary['value']/timesummary['total_allocated']*100
timesummary['perc_cluster'] = timesummary['value']/timesummary['total_cluster']*100

timesummary[colname_init] = timesummary[colname_init].astype(str)

map_dict = {'act_afternoon':'Afternoon', 'act_early':'Morning', 'act_morning':'Day','act_night':'Night'}
timesummary['variable'] = timesummary['variable'].map(map_dict)

timesummary['lift'] = (timesummary['perc_cluster']/100)/(timesummary['total_allocated']/timesummary['value'].sum())

#%% LIFT BY HUB

hubsummary = test[['nearest_hub_osm',colname_init]].value_counts()
hubsummary = pd.DataFrame(hubsummary)
hubsummary.reset_index(inplace=True)
hubsummary.sort_values(by='nearest_hub_osm', inplace=True)

totals_allocated = hubsummary[['nearest_hub_osm','count']].groupby(['nearest_hub_osm']).sum().reset_index()
totals_allocated= pd.DataFrame(totals_allocated)
totals_allocated.rename(columns={'count':'total_allocated'}, inplace=True)

totals = hubsummary.groupby([colname_init]).sum().reset_index()
totals = pd.DataFrame(totals)
totals.rename(columns={'count':'total_cluster'}, inplace=True)

hubsummary = pd.merge(hubsummary, totals_allocated[['nearest_hub_osm','total_allocated']], 
                        left_on='nearest_hub_osm', right_on='nearest_hub_osm',
                        how='left')

hubsummary = pd.merge(hubsummary, totals[[colname_init,'total_cluster']], 
                        left_on=colname_init, right_on=colname_init,
                        how='left')

hubsummary['perc_allocated'] = hubsummary['count']/hubsummary['total_allocated']*100
hubsummary['perc_cluster'] = hubsummary['count']/hubsummary['total_cluster']*100

hubsummary[colname_init] = hubsummary[colname_init].astype(str)
hubsummary['lift'] = (hubsummary['perc_cluster']/100)/(hubsummary['total_allocated']/hubsummary['count'].sum())

hubnames= pd.read_csv('C:\\Users\\megan\\Desktop\\PhD data\\hubs2017\\csv\\hubsinfoforRailSmart\hubnames.csv')
hubsummary = pd.merge(hubsummary, hubnames, left_on='nearest_hub_osm', right_on='hubID', how='left')

hubsummary.to_csv(dirname+"hubsummary_lift.csv")



#%% CALCULATE HUB CENTRE POINTS

# calculate centroid of stops for each hub so there is only one geographical point per hub

hub_centre = hubs_def[['hubID','geometry']].dissolve(by='hubID')

hub_centre_point = hub_centre.copy()
hub_centre_point.to_crs(epsg=32749, inplace=True)
hub_centre_point['geometry'] = hub_centre_point.centroid
hub_centre_point.to_crs(epsg=4326, inplace=True)
hub_centre_point.reset_index(inplace=True)

hub_centre_point['x'] = hub_centre_point['geometry'].x
hub_centre_point['y'] = hub_centre_point['geometry'].y
hub_centre_point['coord'] = hub_centre_point['x'].round(4).astype(str)+", "+hub_centre_point['y'].round(4).astype(str)

#%% SPLITTING AREA INTO QUADRANTS

# from the overall bbox
lon_min = 115.493774
lon_max = 116.180420
lat_min = -32.685620
lat_max = -31.447410

map_center = {"lat": -31.952195, "lon": 115.864055}

def createZoneTable(zone_factor,westlimit, southlimit, eastlimit, northlimit):
    
    box_df = gpd.GeoDataFrame()
    
    longs = np.linspace(westlimit, eastlimit, zone_factor +1 )
    lats = np.linspace(southlimit, northlimit, zone_factor +1 )
    
    j_labels = ["6", "5", "4", "3", "2", "1"]
    i_labels = ["A", "B", "C", "D", "E", "F"]
    
    for i in range(1,len(longs)):
        for j in range(1,len(lats)):
            geometry = [box(longs[i-1], lats[j-1], longs[i], lats[j])]
            this_df = gpd.GeoDataFrame(geometry=geometry)
            this_df['label'] = i_labels[i-1]+j_labels[j-1]
            box_df = pd.concat([box_df, this_df], ignore_index=True)
    return box_df


grid_df = createZoneTable(6, westlimit=lon_min, southlimit=lat_min, eastlimit=lon_max, northlimit=lat_max)
grid_df= grid_df.set_crs("epsg:4326", allow_override=True)

fig = px.choropleth_map(grid_df, 
                        geojson=grid_df.geometry, 
                        locations=grid_df.index,     
                        center=map_center,
                        color='label',
                        map_style = 'carto-positron-nolabels',
                        color_discrete_sequence=mycolors_discrete,
                        opacity=0.2,
                        zoom=10)

grid_df_point = grid_df.to_crs(epsg=32749)
grid_df_point['geometry'] = grid_df_point['geometry'].centroid
grid_df_point = grid_df_point.to_crs(epsg=4326)

fig.add_trace(
    go.Scattermap(
        lat=grid_df_point["geometry"].y,
        lon=grid_df_point["geometry"].x,
        text = grid_df_point['label'],
        mode="text",
        textfont_size=32,
        textfont_color='black',
    )
)

pyo.plot(fig, config=config)

filename = "routines_gridmap"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PUT GRID LABELS ONTO SUMMARY

hub_centre_point = hub_centre_point.overlay(grid_df, how='intersection')

hubsummary = pd.merge(hubsummary, hub_centre_point[['hubID','label']], on='hubID', how='left')
hubsummary.to_csv(dirname+"hubsummary_lift.csv")


#%% PLOT HUB CENTRE POINTS

fig = px.scatter_map(hub_centre_point, 
                        lat=hub_centre_point.geometry.y, 
                        lon=hub_centre_point.geometry.x,     
                        color='label', 
                        color_discrete_sequence=mycolors_discrete, 
                        size_max=15, zoom=10)
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)
# filename = "prep_staydistance_full"
# fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+filename+".svg")


#%% EXPLORATION OF NUMBER SEQUENCES PER HUB

# get number of unique cards per hub

hubcardcount = pd.pivot_table(test, index=['nearest_hub_osm'], aggfunc=lambda x: len(x.unique()), values='Cardid')
hubcardcount.reset_index(inplace=True)
#hubcardcount.rename(columns={'Cardid':'count'}, inplace=True)

hubcardcount = pd.merge(hubcardcount, hub_centre_point, left_on='nearest_hub_osm', right_on='hubID')

hubcardcount = gpd.GeoDataFrame(hubcardcount, geometry='geometry', crs=4326)

# get number of sequences per hub

hubrecordcount = pd.pivot_table(test, index=['nearest_hub_osm'], aggfunc='count')
hubrecordcount.reset_index(inplace=True)
hubrecordcount.rename(columns={'Cardid':'count'}, inplace=True)

hubrecordcount = hubrecordcount[['nearest_hub_osm','count']]
hubrecordcount = pd.merge(hubrecordcount, hub_centre_point, left_on='nearest_hub_osm', right_on='hubID')
hubrecordcount = gpd.GeoDataFrame(hubrecordcount, geometry='geometry', crs=4326)

#%% MAP OF HUBS, COLOURED/SIZED BY NUMBER OF SEQUENCES

fig = px.scatter_map(hubrecordcount, 
                        lat=hubrecordcount.geometry.y, 
                        lon=hubrecordcount.geometry.x,     
                        color=hubrecordcount['count'],
                        size= hubrecordcount['count'],
                        color_continuous_scale=mycolors_continuous,
                        size_max=15, zoom=10)
fig.update_layout(map_style="light")
pyo.plot(fig, config=config)
# filename = "prep_staydistance_full"
# fig.write_html(dirname+filename+".html", include_plotlyjs='cdn', config=config)
# fig.write_image(dirname+filename+".svg")


#%% HISTOGRAM OF NUMBER OF SEQUENCES PER HUB

fig = px.histogram(
    hubrecordcount['count'],
    #nbins=100,
    color_discrete_sequence=[mycolors_discrete[7]],
)
fig.update_layout(
    default_layout,
    yaxis_title_text = 'Number of hubs',
    xaxis_title_text = 'Number of sequences')
#pyo.plot(fig, config=config)

filename = "hubprofile_numhubs"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%% MAP OF HUBS, COLOURED BY LIFT

hubsummary_map = pd.merge(hubsummary, hub_centre_point, left_on='nearest_hub_osm',
                          right_on='hubID', how='left')

hubsummary_map = gpd.GeoDataFrame(hubsummary_map, geometry='geometry')
hubsummary_map.set_crs(epsg=4326, inplace=True)

map_cluster = '6.0'
hubsummary_map = hubsummary_map[hubsummary_map[colname_init]==map_cluster]

fig = px.scatter_map(hubsummary_map, 
                        lat=hubsummary_map.geometry.y, 
                        lon=hubsummary_map.geometry.x,     
                        color=hubsummary_map['lift'],
                        size= hubsummary_map['lift'],
                        color_continuous_scale=mycolors_continuous,
                        size_max=15, zoom=10)
fig.update_layout(map_style="light")
#pyo.plot(fig, config=config)

filename = "hubprofile_maplift_cluster"+map_cluster
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


# and list out the top n hubs for this cluster
n = 10
hubsummary_map.sort_values(by='lift', ascending=False, inplace=True)
print(hubsummary_map[['hubname','count','lift']].head(n))


#%% READ SA1 SHAPEFILE

shape_SA1 = gpd.read_file(dirname + "2016_SA1_shape\SA1_2016_AUST.shp")

shape_SA1['SA1_7DIGIT'] = shape_SA1['SA1_7DIGIT'].astype(int)

shape_SA1.to_crs(epsg=4326,inplace=True)

# chop down to bbox
bbox_4326 = [115.669556, -32.357923, 116.140594, -31.658057]  
SA1_small = shape_SA1.cx[
    bbox_4326[0] : bbox_4326[2], bbox_4326[1] : bbox_4326[3]
]

#%% GET ALL STOPS

stop_clusters = pd.read_pickle(dirname + "20250111-clusters.pkl")

geopolys = stop_clusters.to_crs(epsg=32749)
geopolys["geometry"] = geopolys["geometry"].buffer(MAPBUFFER)

geopolys.to_crs(epsg=4326, inplace=True)

#%% GET CENSUS DATA

census_dir = '2016_GCP_SA1_for_WA_short-header\\2016 Census GCP Statistical Area 1 for WA\\'
census_file = '2016Census_G40_WA_SA1.csv'
index_col = ['SA1_7DIGITCODE_2016']
data_cols = ['P_15_yrs_over_P',
             'lfs_Emplyed_wrked_full_time_P',
             'lfs_Emplyed_wrked_part_time_P',
             'lfs_Employed_away_from_work_P',
             'lfs_Unmplyed_lookng_for_wrk_P',
             'lfs_Tot_LF_P',
             'lfs_N_the_labour_force_P',
             'Percent_Unem_loyment_P',
             'Percnt_LabForc_prticipation_P',
             'Percnt_Employment_to_populn_P',
             ]

census_data = pd.read_csv(dirname+census_dir+census_file, usecols=index_col+data_cols)

census_data['Frac_FTwork'] = census_data['lfs_Emplyed_wrked_full_time_P'].div(census_data['P_15_yrs_over_P']).replace(np.inf, 0)
census_data['Frac_Labourforce'] = census_data['Percnt_LabForc_prticipation_P'].div(100).replace(np.inf, 0)

census_data['Frac_Labourforce'] = np.where(census_data['Frac_Labourforce']>=1, 1, census_data['Frac_Labourforce'])

census_data.drop(columns=data_cols,inplace=True)

census_data = pd.merge(shape_SA1, census_data, left_on='SA1_7DIGIT', 
                       right_on='SA1_7DIGITCODE_2016')

census_data.to_crs(epsg=4326,inplace=True)
census_data_small = census_data.cx[
    bbox_4326[0] : bbox_4326[2], bbox_4326[1] : bbox_4326[3]
]


#%% HISTOGRAM OF LABOUR FORCE PARTICIPATION

fig = px.histogram(
    census_data_small["Frac_Labourforce"],
    color_discrete_sequence=[mycolors_discrete[7]],
    histnorm="percent",
)
fig.update_layout(default_layout, showlegend=False,
                  xaxis_title_text = 'Fraction of residents in the labour force', 
                  yaxis_title_text = 'Percentage of SA1 areas',
                  )
fig.update_traces(xbins=dict( 
        start=0,
        end=1,
    ))

fig.update_yaxes(
    range=(0, 100),
    constrain='domain'
) 
#pyo.plot(fig)

filename = "hubprofile_labourforce"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=700, height=350)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT MAP - SA1 BOUNDARIES + STOP CLUSTERS WITH 100M BUFFER

geopolys_small = geopolys.cx[
    bbox_4326[0] : bbox_4326[2], bbox_4326[1] : bbox_4326[3]
]
stop_clusters_small = stop_clusters.cx[
    bbox_4326[0] : bbox_4326[2], bbox_4326[1] : bbox_4326[3]
]

SA1_small.to_crs(epsg=4326,inplace=True)
fig = px.choropleth_map(
    geopolys_small,
    geojson=geopolys_small.geometry,
    locations=geopolys_small.index,
    color="clusterID",
    center={"lat": -31.95, "lon": 115.85},
    color_continuous_scale=[[0, mycolors_discrete[1]], [1, mycolors_discrete[1]]],
    opacity=0.5,
    zoom=13,
)

fig.add_trace(
    go.Scattermap(
        lat=stop_clusters_small["geometry"].y,
        lon=stop_clusters_small["geometry"].x,
        hovertext=stop_clusters['clusterID'],
        mode="markers",
        marker=go.scattermap.Marker(
            color=mycolors_discrete[1],
            size=10,
        ),

    )
)

# now add boundaries we want
fig.update_layout(
    coloraxis_showscale=False,
    map={
        "style":"light",
        "layers": [
            {
                "source": SA1_small["geometry"].__geo_interface__,
                "type": "line",
                "color": mycolors_discrete[7],
            }
        ]
    },
)

#pyo.plot(fig)

filename = "hubprofile_SA1withstops"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% PLOT MAP - SA1 LABOUR FORCE PARTICIPATION + STOP CLUSTERS

census_data_small.to_crs(epsg=4326,inplace=True)

fig = px.choropleth_map(
    census_data_small,
    geojson=census_data_small.geometry,
    locations=census_data_small.index,
    color="Frac_Labourforce",
    center={"lat": -31.95, "lon": 115.85},
    color_continuous_scale=mycolors_continuous,
    opacity=0.7,
    zoom=13,
)

fig.add_trace(
    go.Scattermap(
        lat=stop_clusters["geometry"].y,
        lon=stop_clusters["geometry"].x,
        hovertext=stop_clusters['clusterID'],
        mode="markers",
        marker=go.scattermap.Marker(
            color=mycolors_discrete[7],
            size=10,
        ),

    )
)

fig.update_layout(map_style='light')

#pyo.plot(fig)

filename = "hubprofile_labourforcemap"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)


#%% SUBURB LEVEL ASSESSMENT

# https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/1270.0.55.003July%202016?OpenDocument

shape_suburb = gpd.read_file(dirname + "1270055003_ssc_2016_aust_shape\SSC_2016_AUST.shp")
shape_suburb.to_crs(epsg=4326,inplace=True)

# chop down to bbox
suburb_small = shape_suburb.cx[
    bbox_4326[0] : bbox_4326[2], bbox_4326[1] : bbox_4326[3]
]

#%% PLOT MAP OF SUBURBS + STOP CLUSTERS WITH 100M BUFFER

suburb_small.to_crs(epsg=4326,inplace=True)
fig = px.choropleth_map(
    geopolys_small,
    geojson=geopolys_small.geometry,
    locations=geopolys_small.index,
    color="clusterID",
    center={"lat": -31.95, "lon": 115.85},
    color_continuous_scale=[[0, mycolors_discrete[1]], [1, mycolors_discrete[1]]],
    opacity=0.5,
    zoom=13,
)


fig.add_trace(
    go.Scattermap(
        lat=stop_clusters_small["geometry"].y,
        lon=stop_clusters_small["geometry"].x,
        hovertext=stop_clusters_small['clusterID'],
        mode="markers",
        marker=go.scattermap.Marker(
            color=mycolors_discrete[1],
            size=10,
        ),

    )
)

# now add boundaries we want
fig.update_layout(
    coloraxis_showscale=False,
    map={
        "style":"light",
        "layers": [
            {
                "source": suburb_small["geometry"].__geo_interface__,
                "type": "line",
                "color": mycolors_discrete[7],
            }
        ]
    },
)
# could add another choropleth layer with opacity=0 with hover info = suburb name
#pyo.plot(fig)

filename = "hubprofile_suburbmap"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

#%%

hubpivot = pd.pivot_table(
    test,
    index="nearest_hub",
    columns=colname_init,
    aggfunc="count",
    values="Cardid",
    fill_value=0,
)
hubpivot.reset_index(inplace=True)

test = gpd.GeoDataFrame(test, geometry='geometry')
test.set_crs(epsg=4326, inplace=True)

#%%
hubpivot = pd.merge(
    hubpivot,
    hub_centre_point[['hubID', 'geometry']], 
    left_on=["nearest_hub"],
    right_on=["hubID"],
    how="left",
)

hubpivot = pd.merge(hubpivot, hubs_def[['hubID','hubname']].drop_duplicates(), 
                    left_on='nearest_hub',
                    right_on='hubID', how='left')

hubpivot = gpd.GeoDataFrame(hubpivot, geometry='geometry', crs=4326)

#%%

hubpivot["Total"] = hubpivot.iloc[:, 1:9].sum(
    axis=1)

hubpivot["0_frac"] = hubpivot[0].div(hubpivot["Total"], axis=0)
hubpivot["1_frac"] = hubpivot[1].div(hubpivot["Total"], axis=0)
hubpivot["2_frac"] = hubpivot[2].div(hubpivot["Total"], axis=0)
hubpivot["3_frac"] = hubpivot[3].div(hubpivot["Total"], axis=0)
hubpivot["4_frac"] = hubpivot[4].div(hubpivot["Total"], axis=0)
hubpivot["5_frac"] = hubpivot[5].div(hubpivot["Total"], axis=0)
hubpivot["6_frac"] = hubpivot[6].div(hubpivot["Total"], axis=0)
hubpivot["7_frac"] = hubpivot[7].div(hubpivot["Total"], axis=0)

#%%

outbox = pd.melt(
    hubpivot,
    id_vars=['nearest_hub'],
    value_vars=['0_frac','1_frac','2_frac','3_frac','4_frac','5_frac','6_frac','7_frac'],
    #value_vars=cluster_cols_suffix,
)

#%% PLOT HISTOGRAM OF FRACTION OF EACH CLUSTER AT EACH HUB

fig = px.histogram(
    outbox,
    x="value",
    color="variable",
    nbins=50,
    color_discrete_sequence=mycolors_discrete,
    facet_col='variable',
    facet_col_wrap=4,
    )
fig.update_layout(default_layout, showlegend=False,)

fig.update_xaxes(
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
            range=[0, 1])

fig.update_yaxes(
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
        range=[0, 120]
    )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

pyo.plot(fig, config=config)
#%%
filename = "hubprofile_clusterdist"
fig.write_html(dirname+"figs\\html\\"+filename+".html", include_plotlyjs='cdn', config=config)
fig.write_image(dirname+"figs\\svg\\"+filename+".svg", width=1800, height=700)
fig.write_image(dirname+"figs\\square\\"+filename+".svg", width=600,height=600)

    
#%% PLOT MAP OF HUBS BY IF SELECTED CLUSTER IS PRESENT

hubpivot['plot_colour'] = np.where(hubpivot['7_frac']>0,"Yes","No")

fig = px.scatter_map(
    hubpivot,
    lat=hubpivot["geometry"].y,
    lon=hubpivot["geometry"].x,
    color=hubpivot['plot_colour'], 
    hover_name=hubpivot['hubname'],
    center={"lat": -32, "lon": 116},
    color_discrete_sequence = [mycolors_discrete[4], mycolors_discrete[1]],
    zoom=9,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=12))
pyo.plot(fig, config=config)

#%% PLOT MAP OF HUBS BY FRACTION OF SELECTED CLUSTER

# scale fraction (so we can map where each cluster is overrepresented compared
# to the rest of the other hubs)

hubpivot['0_frac_scale'] = (hubpivot['0_frac'] - hubpivot['0_frac'].min()) / (hubpivot['0_frac'].max() - hubpivot['0_frac'].min())
hubpivot['1_frac_scale'] = (hubpivot['1_frac'] - hubpivot['1_frac'].min()) / (hubpivot['1_frac'].max() - hubpivot['1_frac'].min())
hubpivot['2_frac_scale'] = (hubpivot['2_frac'] - hubpivot['2_frac'].min()) / (hubpivot['2_frac'].max() - hubpivot['2_frac'].min())
hubpivot['3_frac_scale'] = (hubpivot['3_frac'] - hubpivot['3_frac'].min()) / (hubpivot['3_frac'].max() - hubpivot['3_frac'].min())
hubpivot['4_frac_scale'] = (hubpivot['4_frac'] - hubpivot['4_frac'].min()) / (hubpivot['4_frac'].max() - hubpivot['4_frac'].min())
hubpivot['5_frac_scale'] = (hubpivot['5_frac'] - hubpivot['5_frac'].min()) / (hubpivot['5_frac'].max() - hubpivot['5_frac'].min())
hubpivot['6_frac_scale'] = (hubpivot['6_frac'] - hubpivot['6_frac'].min()) / (hubpivot['6_frac'].max() - hubpivot['6_frac'].min())
hubpivot['7_frac_scale'] = (hubpivot['7_frac'] - hubpivot['7_frac'].min()) / (hubpivot['7_frac'].max() - hubpivot['7_frac'].min())


#%%
fig = px.scatter_map(
    hubpivot,
    lat=hubpivot["geometry"].y,
    lon=hubpivot["geometry"].x,
    color=hubpivot['7_frac_scale'],
    hover_name=hubpivot['hubname'],
#    center={"lat": -32, "lon": 116},
    center={"lat":-31.952415, "lon":115.857665}, 
   color_continuous_scale = mycolors_continuous,
    range_color=(0, 1),
    zoom=9,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=12))
pyo.plot(fig, config=config)

#%% PLOT TOP N HUBS BY FRACTION OF SELECTED CLUSTER

# to answer questions like 'what is the hub with the highest fraction of commuters'

hubpivot.sort_values(by='4_frac',ascending=False, inplace=True)

hub_num = 5

fig = px.scatter_map(
    hubpivot.head(hub_num),
    lat=hubpivot.head(hub_num)["geometry"].y,
    lon=hubpivot.head(hub_num)["geometry"].x,
    hover_name=hubpivot['hubname'].head(hub_num),
    center={"lat": -32, "lon": 116},
    color_continuous_scale = mycolors_continuous,
    range_color=(0, 1),
    zoom=9,
)
fig.update_layout(default_layout)
fig.update_layout(map_style="light")
fig.update_traces(marker=dict(size=12))
pyo.plot(fig, config=config)
