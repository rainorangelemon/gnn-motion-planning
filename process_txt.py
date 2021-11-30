import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

legend = True
envs = ['maze2easy', 'maze2hard', 'ur5', 'snake7', 'kuka7', 'kuka13', 'kuka14']
envs_dimension = ['Easy2D', 'Hard2D', 'UR5', 'Snake', 'Kuka7D', '13D', '14D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Collision Check', 'Running Time (seconds)', 'Path Cost', 'Total Time (seconds)']
metric_values = {}
metric_errors = {}
method_names = {'GNN_ns': 'GNN (Explorer Only)',
                'GNN': 'GNN + Smoother',
                'GNN_pure_ns': 'GNN w/o Obstacle Encoding',
                'GNN_pure': 'GNN w/o Obstacle Encoding + Smoother',
                'BIT*': 'BIT*',
                'NEXT': 'NEXT',
                'RRT*': 'RRT*',
                'LazySP*': 'LazySP*'}
colors = ['rgb(246, 171, 171)', 'rgb(236, 86, 86)', 'rgb(192, 135, 191)', 'rgb(127, 15, 126)',
          'orange', '#2ca02c', 'rgb(100, 114, 246)', 'rgb(152, 37, 251)']


margins = {
    'success rate': 80,
    'collision check': 100,
    'running time': 100,
    'path cost': 70,
    'total time': 100
}

plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'

# fig = go.Figure(data=[
#     go.Bar(name='GNN', x=envs, y=[1000, 1000, 1000]),
#     go.Bar(name='BIT*', x=envs, y=[1000, 1000, 1000]),
#     go.Bar(name='RRT*', x=envs, y=[958.5, 609.5, 170.25]),
#     go.Bar(name='NEXT', x=envs, y=[999, 992.25, 655.5]),
# ])
# # Change the bar mode
# fig.update_layout(barmode='group', title="Success Rates",
#     xaxis_title="environments",
#     yaxis_title="success rates",)
# fig.show()

with open("data/results/result__.txt", "r") as f:
    for line in f:
        result = line.strip()
        if len(set(result.split(' ')) & set(envs)) and 'Avg' in result:
            key = result.split()[:2]
            for metric in metrics:
                value = float(f.readline().split(': ')[1])
                metric_values[tuple([metric] + key)] = value
        elif len(set(result.split(' ')) & set(envs)) and 'Std' in result:
            key = result.split()[:2]
            for metric in metrics:
                value = float(f.readline().split(': ')[1])
                metric_errors[tuple([metric] + key)] = value
print(metric_values)

for metric_id, metric_title in enumerate(zip(metrics, titles)):
    metric, title = metric_title
    data = defaultdict(list)
    error = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        for method, v in method_names.items():
            if metric == 'success rate':
                data[v].append(metric_values[metric, env, method] / 1000)
                error[v].append(metric_errors[metric, env, method] / 1000)
            # elif metric == 'total time':
            #     data[v].append(np.log(metric_values[metric, env, method]))
            #     error[v].append(np.log(metric_errors[metric, env, method]))
            else:
                data[v].append(metric_values[metric, env, method])
                error[v].append(metric_errors[metric, env, method])

    if metric not in ['success rate', 'total time']:
        fig = go.Figure(
            [go.Bar(name=method, x=envs_dimension, y=data[method], error_y=dict(type='data', color='black', array=np.array(error[method]), visible=True),
                    text=data[method], marker_color=color) for color, method in zip(colors, method_names.values())]
        )
    else:
        fig = go.Figure(
            [go.Scatter(name=method, x=envs_dimension, y=data[method], error_y=dict(type='data', color=color, array=np.array(error[method]), visible=True),
                        text=data[method], marker_color=color) for color, method in zip(colors, method_names.values())]
        )
    if 'time' in metric:
        fig.update_yaxes(type="log")
        fig.update_layout(
            yaxis=dict(
                showexponent='all',
                exponentformat='e',
                tickmode = 'linear',
                tick0 = 2,
            )
        )
    # Change the bar mode
    # fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
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
        title_x=0.5, title_y=0.98, font_size=30, font_family='"Open Sans"',
        # title = title,
        # xaxis_title="Environments",
        # yaxis_title=title,
                      )
    if legend:
        fig.update_layout(
            legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.0
        ))
    fig.layout.yaxis.title.font = dict(size=19)
    fig.layout.yaxis.tickfont = dict(size=18)
    fig.layout.xaxis.tickfont = dict(size=25)
    if "success" in metric:
        fig.update_layout(yaxis_range=[0, 1.05])
    if "collision" in metric:
        fig.update_layout(yaxis_range=[0, 6700])
    fig.update_xaxes(showline=True, gridcolor='white', linecolor='rgb(176,176,176)')
    fig.update_yaxes(showline=True, gridcolor='rgb(176,176,176)', linecolor='rgb(176,176,176)', )
    fig.write_image("data/images/%s_legend.pdf" % metric.replace(" ", "_"))
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
