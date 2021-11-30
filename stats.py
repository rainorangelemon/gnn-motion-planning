import re
import numpy as np

def mean(strs):
    values = []
    for str in strs:
        value = re.findall(r"(\d+(?:\.\d+)?)", str)[0]
        values.append(float(value))
    return np.mean(values)


results = input("input the math:\r\n")
success_rates = re.findall(r"success rate: \d+", results)
collision_checks = re.findall(r"collision check:  \d+", results)
running_time = re.findall(r"running time: \d+\.\d+", results)
path_cost = re.findall(r"path cost: \d+\.\d+", results)
total_time = re.findall(r"total time: \d+\.\d+", results)

print('success rate: %.2f' % mean(success_rates))
print('collision check: %d' % int(mean(collision_checks)))
print('running time: %.2f' % mean(running_time))
print('path cost: %.2f' % mean(path_cost))
print('total time: %.2f' % mean(total_time))

