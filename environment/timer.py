from time import time
from enum import auto
from copy import deepcopy


class Timer:
    VERTEX_CHECK = 0
    EDGE_CHECK = 4
    SAMPLE = 1
    PLAN = 2
    CREATE = 3
    FORWARD = 5
    NN = 6
    EXPAND = 7
    HEAP = 8
    GPU = 9
    SHORTEST_PATH = 10
    
    def __init__(self):
        self.log = []
    
    def start(self):
        self.st = time()
        
    def finish(self, action):
        self.log.append([float(self.st), time(), action])
        
        
if __name__ == '__main__':
    import plotly.graph_objects as go
    import plotly

    plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

    values = [4.57, 6.79-4.57]
    labels = ['Collision Check:\n{0:.1f}s'.format(values[0]), 'Others:\n{0:.1f}s'.format(values[1])]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                 textinfo='percent',
                                 textfont={'family': "Courier", 'color': ["White", "White"], 'size': 20},
                                 insidetextorientation='radial'
                                 )])
    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black")))
    fig.update_traces(hole=.35, hoverinfo="label+percent+name")
    # fig.update_layout(showlegend=False)

    fig.write_image("../data/images/cc.pdf")
