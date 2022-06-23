import numpy as np
from utils.plot import plot_edges
from environment import MazeEnv
from algorithm.bit_star import BITStar
from tqdm import tqdm
import matplotlib.pyplot as plt

# cost is between [0, 2.3]
# grids num is between [57, 128]

INFINITY = float('INF')


def dist(start, goal, maze):
    frontier = [start]
    explored = []
    dists = {start: 0}
    while goal not in explored:
        current = frontier.pop()
        explored.append(current)
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            neighbor = (current[0]+direction[0], current[1]+direction[1])
            if (not (14 >= neighbor[0] >= 0 and 14 >= neighbor[1] >= 0)) or (maze[neighbor[0], neighbor[1]] == 1):
                continue
            if (neighbor not in explored) and (neighbor not in frontier):
                frontier.append(neighbor)
                if neighbor not in dists:
                    dists[neighbor] = dists[current] + np.linalg.norm(np.array(direction))
            if neighbor in dists:
                dists[neighbor] = min(dists[current] + np.linalg.norm(np.array(direction)), dists[neighbor])
    return dists[goal]


mazes = np.load('maze_files/mazes_100000.npz')
train_mazes = mazes.f.arr_0
goal_mazes = mazes.f.arr_1


def find_mazes(maze_num, density, dist2goal_threshold=1):

    maps = []
    goal_states = []
    init_states = []

    
    pbar = tqdm(range(100*len(train_mazes)))
    for index in pbar: # len(train_mazes)
        pbar.set_description("len of new data: %d" % len(maps))
        env = MazeEnv(dim=2)
        env.map = 1-train_mazes[index%len(train_mazes), :]

        free_grids = np.where(env.map==0)
        # init_index = np.random.choice(len(free_grids[0]))
        # goal_index = np.random.choice(len(free_grids[0]))

        env.set_random_init_goal()

        # env.goal_state = -1 + np.array(env.goal_state) / (15. / 2) + np.random.uniform(1./30, 3./30)
        # env.init_state = -1 + np.array(env.init_state) / (15. / 2) + np.random.uniform(1./30, 3./30)
        # costs.append([len(free_grids[0])])

        if (env.init_state == env.goal_state).all():
            continue

        if density[0] <= (225-len(free_grids[0])) <= density[1] and (np.linalg.norm(env.init_state-env.goal_state)>=dist2goal_threshold):

            maps.append(env.map)
            goal_states.append(env.goal_state)
            init_states.append(env.init_state)

            if len(maps) >= maze_num:
                return maps, init_states, goal_states

    return maps, init_states, goal_states

# costs = find_mazes(0, [1, 2])
# print(np.min([cost[0] for cost in costs]))
# print(np.max([cost[0] for cost in costs]))
# plt.xlabel('Path Length')
# plt.ylabel('Density')
# plt.title('Path Length Distribution on Maze 2D')
# plt.hist(np.array([cost[0] for cost in costs]), 50, density=True, facecolor='g', alpha=0.75)
# plt.show()

# np.savez('new_maze.npz', maps=maps, goal_states=goal_states, init_states=init_states)
#
# print('success')


# maps, init_states, goal_states = find_mazes(1000, [57, 80.6])
# np.savez('maze_files/mazes_easy.npz', maps=maps, goal_states=goal_states, init_states=init_states)
# print(len(maps))
#
# maps, init_states, goal_states = find_mazes(1000, [80.6, 104.3])
# np.savez('maze_files/mazes_normal.npz', maps=maps, goal_states=goal_states, init_states=init_states)
# print(len(maps))


if __name__ == '__main__':
    maps, init_states, goal_states = find_mazes(4000, [57, INFINITY])
    np.savez('maze_files/mazes_4000.npz', maps=maps, goal_states=goal_states, init_states=init_states)
    print(len(maps))
