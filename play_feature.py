import pickle
import numpy as np
import sys
from copy import deepcopy

envs = ['maze2easy', 'maze2hard', 'kuka7', 'kuka14']
envs_dimension = ['Easy2D', 'Hard2D', 'Kuka7D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time (seconds)', 'Path Cost', 'Total Time (seconds)']
metric_values = {}
method_names = ['GNN', 'GNN_no_heuristic']

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
    if path is None or len(path)==0:
        return 0
    path = np.array(path)
    cost = 0
    for i in range(0, len(path)-1):
        cost += np.linalg.norm(path[i+1]-path[i])
    return cost
#
#


d = pickle.load(open("data/no_heuristic.p", "rb"))
print(d.keys())

rd = {**d}


f = open('data/results/feature.txt', 'w')

sys.stdout = f
collision_no_smooth = []
collision_smooth = []
for env in envs:
    for seed in ['1234']:
        valid_path = [np.all([len(rd[env, method, seed][5][i]) for method in method_names]) for i in range(1000)]
        for method in method_names:
            costs = [path_cost(p) for p in rd[env, method, seed][5]]
            if 'GNN' in method:
                costs = [path_cost(p) for p in rd[env, method, seed][6]]
            rd[env, method, seed] = list(rd[env, method, seed])
            rd[env, method, seed][3] = np.mean(np.array(costs)[valid_path])
            # print(env, method, seed, np.mean(np.array(costs)[valid_path]))
            if 'GNN' in method:
                no_smoother = method+"_ns"
                rd[env, no_smoother, seed] = deepcopy(rd[env, method, seed])
                costs = [path_cost(p) for p in rd[env, no_smoother, seed][5]]
                rd[env, no_smoother, seed] = list(rd[env, no_smoother, seed])
                rd[env, no_smoother, seed][1] = rd[env, no_smoother, seed][7]
                rd[env, no_smoother, seed][4] = rd[env, no_smoother, seed][8]
                rd[env, no_smoother, seed][3] = np.mean(np.array(costs)[valid_path])
                # print(env, no_smoother, seed, np.mean(np.array(costs)[valid_path]))

    for method in ['GNN_ns', 'GNN_no_heuristic_ns']:
        rd[env, method, 'Avg'] = tuple(
            [np.mean([rd[env, method, seed][i] for seed in ['1234']]) for i in range(5)])
        print(env, method, 'Avg')
        print('success rate:', rd[env, method, 'Avg'][0])
        print('collision check: %.2f' % rd[env, method, 'Avg'][1])
        print('running time: %.2f' % rd[env, method, 'Avg'][2])
        print('path cost: %.2f' % rd[env, method, 'Avg'][3])
        print('total time: %.2f' % rd[env, method, 'Avg'][4])
        print('')

f.close()

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict

sys.stdout = sys.__stdout__

envs = ['maze2easy', 'maze2hard', 'kuka7', 'kuka14']
envs_dimension = ['Easy2D', 'Hard2D', '7D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time (seconds)', 'Path Cost', 'Total Time (seconds)']
metric_values = {}
method_names = ['Without Heuristic', 'With Heuristic']
colors = ['rgb(236, 86, 86)', '#2ca02c']#, 'rgb(100, 114, 246)']

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

with open("data/results/feature.txt", "r") as f:
    for line in f:
        result = line.strip()
        if len(set(result.split(' ')) & set(envs)):
            key = result.split()[0]
            method = result.split()[1]
            for metric in metrics:
                value = float(f.readline().split(': ')[1])
                metric_values[tuple([method] + [metric] + [key])] = value
print(metric_values)

for metric_id, metric_title in enumerate(zip(['collision check', 'path cost'], ['Collision Check', 'Path Cost'])):
    metric, title = metric_title
    data = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        data['With Heuristic'].append(metric_values['GNN_ns', metric, env])
        data['Without Heuristic'].append(metric_values['GNN_no_heuristic_ns', metric, env])

    fig = go.Figure(
        [go.Bar(name=method, x=envs_dimension, y=data[method], text=data[method], marker_color=color) for color, method in zip(colors, method_names)]
    )
    # Change the bar mode
    # fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.layout.margin.autoexpand = False
    fig.layout.margin.t = 35
    fig.layout.margin.b = 40
    fig.layout.margin.l = 60
    legend = True
    if legend:
        fig.layout.margin.r = 3500
        fig.layout.margin.t = 100
    else:
        fig.update_layout(showlegend=False)
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=title,
        title_xanchor='center',
        font_color='black',
        title_x=0.5,
        title_y=0.98,
        font_size=30,)
        # xaxis_title="Environments",)
        #yaxis_title=title,)
    fig.layout.yaxis.title.font = dict(size=19)
    fig.layout.yaxis.tickfont = dict(size=18)
    fig.layout.xaxis.tickfont = dict(size=25)
    if "success" in metric:
        fig.update_layout(yaxis_range=[0, 1.05])
    if "collision" in metric:
        fig.update_layout(yaxis_range=[0, 999])
    fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
    fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
    fig.write_image("data/images/featue_%s.pdf" % metric.replace(" ", "_"))
    # fig.show()


# print(metric_values.items())
#
# for metric in metrics:
#     for method in method_names:
#         plt.xlabel('environment')
#         plt.title(metric)
#         plt.ylabel(metric)
#         plt.plot(envs_dimension, [v for k, v in metric_values.items() if ((k[2]==method) and (k[0]==metric))], label=method)
#     plt.legend()
#     plt.savefig("data/images/%s.pdf" % metric.replace(" ", "_"))
#     plt.clf()
