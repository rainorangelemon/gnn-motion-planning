import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from environment import MazeEnv
from PIL import Image
from environment.graph import EdgeAttribute


def draw_node(state, color, radius_scale=1., dim=2, face=False):
    if dim == 2:
        facecolor = 'none'
        if face:
            facecolor = color
        circle = patches.Circle(tuple(state + 1.0), radius=0.02 * radius_scale, edgecolor=color, facecolor=facecolor)
        plt.gca().add_patch(circle)

    elif dim == 3:
        a, b = MazeEnv._end_points(state)
        plt.gca().add_patch(patches.ConnectionPatch(a + 1.0, b + 1.0, 'data', arrowstyle="-", linewidth=2, color=color))
        plt.gca().add_patch(patches.Circle(a + 1.0, radius=0.02 * radius_scale, edgecolor=color, facecolor=color))


def draw_edge(state0, state1, color, dim=2, style='-'):
    path = patches.ConnectionPatch(tuple(state0[:2] + 1.0), tuple(state1[:2] + 1.0), 'data', arrowstyle=style,
                                   color=color)
    plt.gca().add_patch(path)


def plot_graph(graph, problem, title='', save=None):
    plt.clf()
    plt.close('all')
    states = np.array(graph.V)
    environment_map = problem["map"]
    init_state = problem["init_state"]
    goal_state = problem["goal_state"]
    dim = init_state.size

    fig = plt.figure(figsize=(4, 4))

    rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0, linewidth=1, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    map_width = environment_map.shape
    d_x = 2.0 / map_width[0]
    d_y = 2.0 / map_width[1]
    for i in range(map_width[0]):
        for j in range(map_width[1]):
            if environment_map[i, j] > 0:
                rect = patches.Rectangle((d_x * i, d_y * j), d_x, d_y, linewidth=1, edgecolor='#253494',
                                         facecolor='#253494')
                plt.gca().add_patch(rect)

    for i in range(len(states) - 1):
        draw_node(states[i + 1], '#bbbbbb', dim=dim)

    for edge, attribute in graph.E_attr.items():

        if attribute==EdgeAttribute.Free:
            draw_edge(states[edge[0]], states[edge[1]], 'green', dim=dim)
        elif attribute==EdgeAttribute.Unknown:
            pass
            # draw_edge(states[edge[0]], states[edge[1]], 'blue', dim=dim)
        else:
            if graph.V_attr[edge[0]]==EdgeAttribute.Free and \
                    graph.V_attr[edge[1]]==EdgeAttribute.Free:
                draw_edge(states[edge[0]], states[edge[1]], 'black', dim=dim)

    draw_node(init_state, '#e6550d', dim=dim, face=True)
    draw_node(goal_state, '#a63603', dim=dim, face=True)

    plt.axis([0.0, 2.0, 0.0, 2.0])
    plt.axis('off')
    plt.axis('square')

    plt.subplots_adjust(left=-0., right=1.0, top=1.0, bottom=-0.)

    plt.title(title)

    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

    return fig


def plot_edges(states, edges, problem, index=0,
               new_list=None, fig=None, edge_classes=None, title='', save=None, title_size=25):
    
    # edges is a list
    
    plt.clf()
    states = np.array(list(states))
    environment_map = problem["map"]
    init_state = problem["init_state"]
    goal_state = problem["goal_state"]
    dim = init_state.size

    if fig is None:
        fig = plt.figure(figsize=(4, 4))

    rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0, linewidth=1, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    map_width = environment_map.shape
    d_x = 2.0 / map_width[0]
    d_y = 2.0 / map_width[1]
    for i in range(map_width[0]):
        for j in range(map_width[1]):
            if environment_map[i, j] > 0:
                rect = patches.Rectangle((d_x * i, d_y * j), d_x, d_y, linewidth=1, edgecolor='#253494',
                                         facecolor='#253494')
                plt.gca().add_patch(rect)

    for i in range(len(states)):
        draw_node(states[i], '#bbbbbb', dim=dim)

    if isinstance(edges, dict):
        iterator = enumerate(edges.items())
    else:
        iterator = enumerate(edges)
    for index, item in iterator:
        node, parent = item
        node, parent = np.array(node), np.array(parent)
        if edge_classes is None:
            draw_edge(node, parent, 'green', dim=dim)
        else:
            if edge_classes[index]:
                draw_edge(node, parent, 'blue', dim=dim)
            else:
                draw_edge(node, parent, 'green', dim=dim)

    draw_node(init_state, '#e6550d', dim=dim, face=True)
    draw_node(goal_state, '#a63603', dim=dim, face=True)
    
    plt.annotate('start', init_state+1.0, color='black', backgroundcolor=(1., 1., 1., 0.3), fontsize=20)
    plt.annotate('goal', goal_state+1.0, color='black', fontsize=20)

    plt.axis([0.0, 2.0, 0.0, 2.0])
    plt.axis('off')
    plt.axis('square')

    plt.subplots_adjust(left=-0., right=1.0, top=1.0, bottom=-0.)

    if title == '':
        plt.title('#%d Samples' % len(states) + ' #%d Edges' % len(edges), fontdict = {'fontsize':title_size})
    else:
        plt.title(title, fontdict = {'fontsize':title_size})

    if save:
        plt.savefig(save, bbox_inches='tight')
        plt.close('all')
    else:
#         plt.show()
        pass

    return fig


def merge_pic(image_paths, column, row, save_path):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = column * max(widths)
    total_height = row * max(heights)

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    index = 0
    for i in range(row):
        for j in range(column):
            new_im.paste(images[index], (j * max(widths), i * max(heights)))
            index += 1

    new_im.save(save_path)

