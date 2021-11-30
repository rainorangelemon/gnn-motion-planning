import pickle
import numpy as np
import sys

envs = ['Maze_2D_Easy', 'Maze_2D_Normal', 'Maze_2D_Hard', 'Kuka_7D', 'Kuka_13D', 'Kuka_14D']
envs_dimension = ['2D Easy', '2D Normal', '2D Hard', '7D', '13D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost', 'Total Time']
method_names = ['GNN', 'BIT*', 'NEXT', 'RRT*', 'GNN_no_smooth']

# a_file = open("data/results/result_b.txt", "r")
# lines = a_file.readlines()
# a_file.close()
# new_file = open("data/results/result.txt", "w")
#
# for line in lines:
#     if "1000/1000" not in line:
#         new_file.write(line)
#
# new_file.close()


def path_cost(path):
    path = np.array(path)
    cost = 0
    for i in range(0, len(path)-1):
        cost += np.linalg.norm(path[i+1]-path[i])
    return cost
#
#


a = pickle.load(open("data/batch_size.p", "rb"))


#
# b = pickle.load(open("new_1.p", "rb"))
# for key in list(b.keys()):
#     if np.isnan(b[key][0]):
#         del b[key]
#
# c = pickle.load(open("new_2.p", "rb"))
# for key in list(c.keys()):
#     if np.isnan(c[key][0]):
#         del c[key]

rd = a
print(rd.keys())

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict

envs = ['Maze_2D_Hard', 'Kuka_7D']
envs_dimension = ['2D Hard', '7D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Normalized Collision Check', 'Running Time', 'Normalized Path Cost', 'Total Time']
method_names = [50, 100, 200, 300, 500, 1000]
colors = ['rgb(236, 86, 86)', 'blue']

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

for metric_id, metric_title in enumerate(zip(metrics, titles)):
    metric, title = metric_title
    data = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        for method in method_names:
            if metric == 'collision check':
                data[env].append(rd[env, method][metric_id] / rd[env, 1000][metric_id])
            elif metric == 'success rate':
                data[env].append(rd[env, method][metric_id] / 1000)
            elif metric == 'path cost':
                data[env].append(rd[env, method][metric_id] / rd[env, 1000][metric_id])
            else:
                data[env].append(rd[env, method][metric_id])
        print(data[env])

    fig = go.Figure(
        [go.Scatter(name=env_dimension, x=method_names, y=data[env], text=data[env], marker_color=color) for color, env, env_dimension in zip(colors, envs, envs_dimension)]
    )
    # Change the bar mode
    # fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.layout.margin.autoexpand = False
    fig.layout.margin.t = 30
    fig.layout.margin.b = 40
    fig.layout.margin.l = 60
    fig.layout.margin.r = 120
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=title,
        title_xanchor='center',
        font_color='black',
        title_x=0.5,
        title_y=0.98,
        titlefont_size=18,
        xaxis_title="Batch Sample Size",
        yaxis_title=title,)
    if "success" in metric:
        fig.update_layout(yaxis_range=[0.3, 1.1])
    if "collision" in metric:
        fig.update_layout(yaxis_range=[0.6, 1.2])
    if 'path cost' in metric:
        fig.update_layout(yaxis_range=[0.9, 1.12])
    fig.update_xaxes(tick0=100, dtick=100)
    fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
    fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
    fig.write_image("data/images/%s.pdf" % metric.replace(" ", "_"))
