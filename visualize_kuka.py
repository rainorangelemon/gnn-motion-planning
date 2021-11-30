import pickle
from environment.kuka_env import KukaEnv
import numpy as np
from PIL import Image, ImageSequence
import numpy

with open('maze_files/kukas_3_3000.pkl', 'rb') as f:
    problems = pickle.load(f)

problem_index = np.argmax([np.linalg.norm(problem[2]-problem[1]) for problem in problems])
problem_index = np.argmax([problem[-1][-1] for problem in problems])
print('most sampling number: %d' % problems[problem_index][-1][-1])
print('mean sampling number: %f' % np.mean([problem[-1][-1] for problem in problems]))

problem = problems[problem_index]
obstacles, start, goal, BIT_solution = problem


def get_bit_path(edges, start, goal):
    path = [tuple(goal)]
    current_node = tuple(goal)
    while current_node != tuple(start):
        current_node = edges[current_node]
        path.append(current_node)
    path.reverse()
    return path


path = get_bit_path(BIT_solution[1], start, goal)
env = KukaEnv(GUI=True)
env.init_new_problem(obstacles, start, goal)
gifs = env.plot(path, make_gif=True)


# Setup the 4 dimensional array
a_frames = []
for im_frame in gifs:
    a_frames.append(numpy.asarray(im_frame))
a = numpy.stack(a_frames)

print("Array shape:", a.shape)  # (31, 240, 320, 3)

ims = [Image.fromarray(a_frame) for a_frame in a]
ims[0].save("out.gif", save_all=True, append_images=ims[1:], loop=0, duration=50)
print('hello')
