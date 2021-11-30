import pickle
from environment.kuka_env import KukaEnv
import numpy as np
from PIL import Image, ImageSequence
from algorithm.bit_star import BITStar
import numpy

with open('maze_files/kukas_7_3000.pkl', 'rb') as f:
    problems = pickle.load(f)

problems[0] = [[[np.array([0.17, 0.17, 0.17]), np.array([-0.5, 0, 0.5])]],
                     (0.3542580627450649, -2.022089521226548, -2.2182339069315398, 0.9115339602971932, 2.752754863673904, 0.36717590416693646, 1.448471191228442),
                     (0.5881303831253089, 0.5849310274380142, -0.05696028570145195, 1.6483631621211003, -1.8229752960204042, -0.022967923170421845, -2.503331871691698),
               []]
# problem_index = np.argmax([np.linalg.norm(problem[2]-problem[1]) for problem in problems])
# problem_index = np.argmax([problem[-1][-1] for problem in problems])
# print('most sampling number: %d' % problems[problem_index][-1][-1])
# print('mean sampling number: %f' % np.mean([problem[-1][-1] for problem in problems]))
#
# problem = problems[problem_index]
# obstacles, start, goal, BIT_solution = problem


def get_bit_path(edges, start, goal):
    path = [tuple(goal)]
    current_node = tuple(goal)
    while current_node != tuple(start):
        current_node = edges[current_node]
        path.append(current_node)
    path.reverse()
    return path


# path = get_bit_path(BIT_solution[1], start, goal)
env = KukaEnv(GUI=False)
env.problems = problems
env.init_new_problem(0)
bit = BITStar(env, batch_size=50, T=1000, sampling=None)
bit.plan(float('inf'), refine_time_budget=0, time_budget=300)
bit.get_best_path()


# # Setup the 4 dimensional array
# a_frames = []
# for im_frame in gifs:
#     a_frames.append(numpy.asarray(im_frame))
# a = numpy.stack(a_frames)
#
# print("Array shape:", a.shape)  # (31, 240, 320, 3)
#
# ims = [Image.fromarray(a_frame) for a_frame in a]
# ims[0].save("out.gif", save_all=True, append_images=ims[1:], loop=0, duration=50)
print('hello')
