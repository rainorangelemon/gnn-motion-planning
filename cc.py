import plotly
import plotly.graph_objects as go
plotly.io.orca.config.executable = '/Users/rainorangelemon/anaconda3/envs/pybullet/bin/orca'
methods = ['BIT*', 'GNN', 'RRT*', 'NEXT', 'LazySP'][::-1]

fig = go.Figure()
fig.add_trace(go.Bar(
    y=methods,
    x=[0., 0.4, 0., 17.9, 0.][::-1],
    name='Neural Network',
    # marker=dict(
    #     color='rgba(58, 71, 80)',
    # ),
    orientation='h',
))
fig.add_trace(go.Bar(
    y=methods,
    x=[5.53, 1.41, 21.21, 5.08, 3.75][::-1],
    name='Collision Checking',
    orientation='h',
    marker=dict(
        color='rgba(236, 86, 86)',
    )
))
fig.add_trace(go.Bar(
    y=methods,
    x=[1.72, 5.3, 0., 30.8, 64.][::-1],
    name='Heap / Sorting / Shortest Path',
    orientation='h',
))
fig.add_trace(go.Bar(
    y=methods,
    x=[13.3-1.72-5.53, 8.0-0.4-5.3-1.41, 62-21.21, 70.1-17.9-30.8-5.08, 5.][::-1],
    name='Others (Rewiring, Nearest Neighbor...)',
    orientation='h',
))


fig.update_layout(barmode='stack', font_size=17, font_color='black', legend=dict(
                orientation="h"))
fig.write_image("cc1.pdf")