#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

'''
Provides the functionality of data visualization based on the imported social matrices.
'''

import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly
from plotly.graph_objs import *

# Import networkx graph from json file
with open(sys.argv[1], 'r') as f:
    df = pd.read_json(f)

# Build the networkx graph
G = nx.DiGraph()
nodes = list(df.index)
for i in range(len(nodes)):
    for j in range(len(nodes)):
        u = nodes[i]
        v = nodes[j]
        if not np.isnan(df[u][v]):
            G.add_edge(u, v, weight=df[u][v])

# Initialize basic info for drawing
N_nodes = len(G.nodes())
pos = nx.fruchterman_reingold_layout(G)
# Coordinates on illustration of nodes
Xn = list()
Yn = list()
label_n = list()
for res, xy in pos.items():
    Xn.append(xy[0])
    Yn.append(xy[1])
    label_n.append(res)
# Coordinates on illustration of edges
Xed = list()
Yed = list()
label_e = list()
for e in G.edges(data='weight'):
    u = e[0]
    v = e[1]
    Xed += [pos[u][0], pos[v][0], None]
    Yed += [pos[u][1], pos[v][1], None]
    label_e.append(e[2])

# Settings for drawing the graph
axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )
width=1600
height=1200
layout=Layout(title="",
              font=Font(size=12),
              #showlegend=False,
              autosize=False,
              width=width,
              height=height,
              xaxis=XAxis(axis),
              yaxis=YAxis(axis),
              margin=Margin(
                  l=40,
                  r=40,
                  b=85,
                  t=100,
              ),
              hovermode='closest',
              )

traceE=Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=0.5),
               text=label_e,
               hoverinfo='text'
               )

traceN=Scatter(x=Xn,
               y=Yn,
               mode='markers',
               name='net',
               marker=Marker(symbol='dot',
                             size=15,
                             color='#6959CD',
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=label_n,
               hoverinfo='text'
               )

# Plot the graph
data = Data([traceE, traceN])
fig = Figure(data=data, layout=layout)
plotly.offline.plot(fig)

plt.show()

