import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

legend = False
envs = ['maze2easy', 'maze2hard', 'ur5', 'snake7', 'kuka7', 'kuka13', 'kuka14']
envs_dimension = ['2D Easy', '2D Hard', 'UR5', 'Snake', 'Kuka7D', '13D', '14D']
probing_samples = [50, 100, 200, 300, 500, 1000]
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time', 'Path Cost', 'Total Time']
metric_values = {}
method_names = {'GNN_ns': 'GNN',
                'GNN': 'GNN + Smoother',
                'GNN_pure_ns': 'GNN w/o Obstacle Encoding',
                'GNN_pure': 'GNN w/o Obstacle Encoding + Smoother',
                'BIT*': 'BIT*',
                'NEXT': 'NEXT',
                'RRT*': 'RRT*', }
colors = ['rgb(152, 37, 251)',
          'rgb(252, 13, 27)',
          'rgb(253, 153, 39)',
          'rgb(250, 187, 45)',
          'rgb(58, 167, 87)',
          'rgb(70, 136, 241)',
          'rgb(53, 32, 115)', ]

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

with open('data/probing_samples.p', 'rb') as f:
    rd = pickle.load(f)

for key, value in rd.items():
    for i, metric in enumerate(metrics):
        metric_values[tuple(list(key)+[metric])] = value[i]
print(metric_values)

for metric_id, metric_title in enumerate(zip(metrics, titles)):
    metric, title = metric_title
    data = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        for n_sample in probing_samples:
            if metric == 'collision check' and env in ['kuka7', 'kuka13']:
                data[env_dimension].append(metric_values[env, 'GNN', n_sample, metric])
            elif metric != 'success rate':
                data[env_dimension].append(metric_values[env, 'GNN', n_sample, metric])
            else:
                data[env_dimension].append(metric_values[env, 'GNN', n_sample, metric] / 1000)

    if metric in ['success rate', 'total time']:
        fig = go.Figure(
                [go.Scatter(name=env, x=probing_samples, y=data[env], text=data[env], marker_color=color) for color, env in zip(colors, envs_dimension)]
        )
    else:
        fig = go.Figure(
            [go.Bar(name=env, x=[str(s) for s in probing_samples], y=data[env], text=data[env], marker_color=color) for color, env in zip(colors, envs_dimension)]
        )
    fig.layout.margin.autoexpand = False
    fig.layout.margin.t = 42
    fig.layout.margin.b = 37
    # fig.layout.margin.l = 10
    fig.layout.margin.r = 10
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
        font_size=30, font_family='"Open Sans"',
        # xaxis_title="Environments",
        # yaxis_title=title,
                      )
    if legend:
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.0
        ))
    if "success" in metric:
        fig.update_layout(yaxis_range=[0, 1.05])
    # if "collision" in metric:
    #     fig.update_layout(yaxis_range=[0, 3000])
    fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
    fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
    fig.write_image("data/images/%s_ps.pdf" % metric.replace(" ", "_"))
    # fig.show()