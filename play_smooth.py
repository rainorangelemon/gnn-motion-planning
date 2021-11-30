import pickle
import numpy as np
import sys

envs = ['Maze_2D_Easy', 'Maze_2D_Normal', 'Maze_2D_Hard', 'Kuka_7D', 'Kuka_13D', 'Kuka_14D']
envs_dimension = ['2D Easy', '2D Normal', '2D Hard', '7D', '13D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost', 'Total Time']
metric_values = {}
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
    if path is None or len(path)==0:
        return 0
    path = np.array(path)
    cost = 0
    for i in range(0, len(path)-1):
        cost += np.linalg.norm(path[i+1]-path[i])
    return cost
#
#


a = pickle.load(open("data/only_gnn.p", "rb"))


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

a = pickle.load(open("new_1.p", "rb"))
for key in list(a.keys()):
    if np.isnan(a[key][0]):
        del a[key]

b = pickle.load(open("new_2.p", "rb"))
print(b.keys())

rd_others = {**a, **b}


f = open('data/results/smooth_self.txt', 'w')

# sys.stdout = f
collision_no_smooth = []
collision_smooth = []
for env in envs:
    valid_path = [np.all([len(rd_others[env, method, '1234'][-1][i]) for method in ['GNN', 'BIT*', 'NEXT', 'RRT*']]) for i in range(1000)]

    collision_no_smooth = np.mean([p[1] for p in rd[env, 'GNN', '1234'][-2]])
    path_no_smooth = [p[0] for p in rd[env, 'GNN', '1234'][-2]]
    cost_no_smooth = np.mean(np.array([path_cost(p) for p in path_no_smooth if len(p)]))
    cost_smooth = np.mean(np.array([path_cost(p) for p in rd[env, 'GNN', '1234'][-1] if len(p)]))

    print(env, 'Avg')
    print('success rate:', rd[env, 'GNN', '1234'][0])
    print('collision check: %.2f' % rd[env, 'GNN', '1234'][1])
    print('collision check no smooth: %.2f' % collision_no_smooth)
    print('running time: %.2f' % rd[env, 'GNN', '1234'][2])
    print('path cost: %.2f' % cost_smooth)#rd[env, 'GNN', '1234'][3])
    print('path cost no smooth: %.2f' % cost_no_smooth)
    print('total time: %.2f' % rd[env, 'GNN', '1234'][4])
    print('')

    if env == 'Maze_2D_Hard':
        cost_difference = [path_cost(p[0])-path_cost(p_s) for p, p_s in zip(rd[env, 'GNN', '1234'][-2], rd[env, 'GNN', '1234'][-1])]
        print(np.argmax(np.array(cost_difference)))
        from utils.plot import plot_edges
        path = rd[env, 'GNN', '1234'][-2][137][0]
        path_s = rd[env, 'GNN', '1234'][-1][137]
        from environment import MazeEnv, KukaEnv, Kuka2Env
        env = MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz')
        env.init_new_problem(137)
        plot_edges(path, {tuple(node): tuple(parent) for node, parent in zip(path[:-1], path[1:])},
                   env.get_problem(), title='Path without Smoother', save='u.pdf')
        plot_edges(path_s, {tuple(node): tuple(parent) for node, parent in zip(path_s[:-1], path_s[1:])},
                   env.get_problem(), title='Path without Smoother', save='v.pdf')
        print('finished')
    print('finished')

f.close()



import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict

sys.stdout = sys.__stdout__

envs = ['Maze_2D_Easy', 'Maze_2D_Normal', 'Maze_2D_Hard', 'Kuka_7D', 'Kuka_13D', 'Kuka_14D']
envs_dimension = ['2D Easy', '2D Normal', '2D Hard', '7D', '13D', '14D']
metrics = ['success rate', 'collision check', 'collision check no smooth', 'running time', 'path cost', 'path cost no smooth', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost', 'Total Time']
metric_values = {}
method_names = ['Without Smoother', 'With Smoother']
colors = ['rgb(236, 86, 86)', '#2ca02c']#, 'rgb(100, 114, 246)']

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

with open("data/results/smooth_self.txt", "r") as f:
    for line in f:
        result = line.strip()
        if len(set(result.split(' ')) & set(envs)):
            key = result.split()[0]
            for metric in metrics:
                value = float(f.readline().split(': ')[1])
                metric_values[tuple([metric] + [key])] = value
print(metric_values)

for metric in metrics:
    method = 'GNN'
    metric_values[metric, 'Maze_2D_Easy', method] = (metric_values[metric, 'Maze_2D_Easy'] + metric_values[metric, 'Maze_2D_Normal']) / 2

envs.remove("Maze_2D_Normal")
envs_dimension.remove("2D Normal")

for metric_id, metric_title in enumerate(zip(['collision check', 'path cost'], ['Collision Check', 'Path Cost'])):
    metric, title = metric_title
    data = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        data['With Smoother'].append(metric_values[metric, env])
        data['Without Smoother'].append(metric_values[metric+' no smooth', env])

    fig = go.Figure(
        [go.Bar(name=method, x=envs_dimension, y=data[method], text=data[method], marker_color=color) for color, method in zip(colors, method_names)]
    )
    # Change the bar mode
    # fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.layout.margin.autoexpand = False
    fig.layout.margin.t = 10
    fig.layout.margin.b = 40
    fig.layout.margin.l = 60
    fig.layout.margin.r = 160
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=title,
        title_xanchor='center',
        font_color='black',
        title_x=0.5,
        title_y=0.98,
        xaxis_title="Environments",
        yaxis_title=title,)
    if "success" in metric:
        fig.update_layout(yaxis_range=[0, 1.05])
    if "collision" in metric:
        fig.update_layout(yaxis_range=[0, 999])
    fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
    fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
    fig.write_image("data/images/%s.pdf" % metric.replace(" ", "_"))
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
