import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import defaultdict

envs = ['Maze_2D_Hard', 'Kuka_7D']
envs_dimension = ['2D Hard', '7D']
metrics = ['success rate', 'collision check', 'running time', 'path cost', 'total time']
titles = ['Success Rate', 'Normalized Collision Check', 'Running Time', 'Path Cost', 'Total Time']
metric_values = {}
method_names = ['100', '300', '500' ]
colors = ['rgb(236, 86, 86)', 'blue']# 'orange', '#2ca02c']

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

with open("data/results/result_b.txt", "r") as f:
    for line in f:
        result = line.strip()
        if len(set(result.split(' ')) & set(envs)) and 'Avg' in result:
            key = result.split()[:2]
            for metric in metrics:
                value = float(f.readline().split(': ')[1])
                metric_values[tuple([metric] + key)] = value
print(metric_values)

# for metric in metrics:
#     for method in method_names:
#         metric_values[metric, 'Maze_2D_Easy', method] = (metric_values[metric, 'Maze_2D_Easy', method] + metric_values[metric, 'Maze_2D_Normal', method]) / 2

# envs.remove("Maze_2D_Normal")
# envs_dimension.remove("2D Normal")

for metric_id, metric_title in enumerate(zip(metrics, titles)):
    metric, title = metric_title
    data = defaultdict(list)
    for env, env_dimension in zip(envs, envs_dimension):
        for method in method_names:
            if metric != 'success rate':
                data[env].append(metric_values[metric, env, method] /  metric_values[metric, env, '500'])
            else:
                data[env].append(metric_values[metric, env, method] / 1000)

    fig = go.Figure(
        [go.Scatter(name=env_dimension, x=[100, 300, 500], y=data[env], text=data[env], marker_color=color) for color, env, env_dimension in zip(colors, envs, envs_dimension)]
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
        fig.update_layout(yaxis_range=[0, 1.2])
    if "collision" in metric:
        fig.update_layout(yaxis_range=[0, 1.5])
    fig.update_xaxes(tick0=100, dtick=200)
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
