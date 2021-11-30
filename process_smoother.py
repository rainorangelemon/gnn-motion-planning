import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from eval_gnn import path_cost
from environment import MazeEnv
from utils.plot import plot_edges
import numpy as np

legend = False
envs = ['maze2easy', 'maze2hard', 'ur5', 'snake7', 'kuka7', 'kuka13', 'kuka14']
envs_dimension = ['2D Easy', '2D Hard', 'UR5', 'Snake', 'Kuka7D', '13D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost Improvement', 'Total Time']
metric_values = {}
method_names = {'GNN': 'GNN Smoother',
                'Oracle': 'Oracle'}
colors = ['rgb(252, 13, 27)',
          'rgb(70, 136, 241)']

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

with open('data/new_gnn_bit_rrt.p', 'rb') as f:
    rd = pickle.load(f)

with open('data/oracle_test.p', 'rb') as f:
    rd = {**pickle.load(f), **rd}

for env in envs:
    rd[env, 'GNN'] = list(rd[env, 'GNN', '1234'])
    rd[env, 'Oracle'] = list(rd[env, 'Oracle'])

for env in envs:
    for method in method_names.keys():
        rd[env, method][3] = float(sum([path_cost(p) for p in rd[env, method][5]])) / rd[env, method][0] - rd[env, method][3]
        rd[env, method][4] = rd[env, method][4] - rd[env, method][8]
        rd[env, method][1] = rd[env, method][1] - rd[env, method][7]
        if method=='GNN':
            improvement = [path_cost(p1)-path_cost(p2) for p1, p2 in zip(rd[env, method][5], rd[env, method][6])]
            print(env,
                  np.argmax(improvement),
                  np.max(improvement),
                  path_cost(rd[env, method][5][np.argmax(improvement)]),
                  path_cost(rd[env, method][6][np.argmax(improvement)]))
            if env=='maze2hard':
                orig_path = np.array(rd[env, method][5][np.argmax(improvement)])
                new_path = np.array(rd[env, method][6][np.argmax(improvement)])
                oracle_path = np.array(rd[env, 'Oracle'][6][np.argmax(improvement)])
                ENV = MazeEnv(dim=2, map_file='maze_files/mazes_hard.npz')
                ENV.init_new_problem(135)
                plot_edges(orig_path, {tuple(p):tuple(q) for p, q in zip(orig_path[1:], orig_path[:-1])},
                           ENV.get_problem(), title='Explored Path\n', save='oracle1.pdf')
                plot_edges(new_path, {tuple(p):tuple(q) for p, q in zip(new_path[1:], new_path[:-1])},
                           ENV.get_problem(), title='Smooth Path with \nGNN Smoother', save='oracle2.pdf')
                plot_edges(oracle_path, {tuple(p): tuple(q) for p, q in zip(oracle_path[1:], oracle_path[:-1])},
                           ENV.get_problem(), title='Smooth Path with Oracle Smoother', save='oracle3.pdf')

# for metric_id, metric_title in enumerate(zip(metrics, titles)):
#     metric, title = metric_title
#     data = defaultdict(list)
#     for method, method_name in method_names.items():
#         for env, env_dimension in zip(envs, envs_dimension):
#             if metric != 'success rate':
#                 data[method].append(rd[env, method][metric_id])
#             else:
#                 data[method].append(rd[env, method][metric_id] / 1000)
#
#     fig = go.Figure(
#             [go.Bar(name=method_names[method], x=envs_dimension, y=data[method], text=data[method], marker_color=color) for color, method in zip(colors, method_names)]
#     )
#     fig.layout.margin.autoexpand = False
#     fig.layout.margin.t = 10
#     fig.layout.margin.b = 30
#     fig.layout.margin.l = 90
#     fig.layout.margin.r = 10
#     if legend:
#         fig.layout.margin.r = 3500
#         fig.layout.margin.t = 100
#     else:
#         fig.update_layout(showlegend=False)
#     fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',# title=title,
#         title_xanchor='center',
#         font_color='black',
#         title_x=0.5,
#         title_y=0.98,
#         font_size=19, font_family='"Open Sans"',
#         # xaxis_title="Environments",
#         yaxis_title=title,)
#     if legend:
#         fig.update_layout(legend=dict(
#             orientation="h",
#             yanchor="top",
#             y=0.98,
#             xanchor="left",
#             x=1.0
#         ))
#     if "success" in metric:
#         fig.update_layout(yaxis_range=[0, 1.05])
#     # if "collision" in metric:
#     #     fig.update_layout(yaxis_range=[0, 3000])
#     fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
#     fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
#     fig.write_image("data/images/%s_oracle.pdf" % metric.replace(" ", "_"))
#     # fig.show()