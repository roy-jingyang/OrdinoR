#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

with open(sys.argv[1], 'r') as f:
    df = pd.read_json(f)

G = nx.DiGraph()
nodes = list(df.index)
for i in range(len(nodes)):
    for j in range(len(nodes)):
        u = nodes[i]
        v = nodes[j]
        if not np.isnan(df[u][v]):
            G.add_edge(u, v, weight=df[u][v])

fig = nx.draw_networkx(G)
plt.show()

