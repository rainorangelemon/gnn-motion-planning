import numpy as np
import torch
from environment import KukaEnv
from torch_geometric.nn import knn_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce
from algorithm.tsa import RRTS_plan


if __name__ == "__main__":

    data = []
    env = KukaEnv()

    time0 = time()
    solutions = []

    for problem_index in tqdm(range(2000)):

        env.init_new_problem(problem_index)
        solution = RRTS_plan(env, T=1000, stop_when_success=False)
        solutions.append(solution)

    print('hello')

