import sys
import numpy as np
import progressbar
import os
import subprocess
import wrapper_pkg
import json
import matplotlib.pyplot as plt
import shutil

path = "perf_test"
file = open("{}/num_thread_performance.data".format(path), "r")
results = json.load(file)
file.close()

frameworks = set()
steps_set = set()
grid_set = set()
thread_set = set()
for r in results:
frameworks.add(r['framework'])
steps_set.add(r['num_steps'])
grid_set.add(str(r['grid_dims']))
thread_set.add(r['num_threads'])
steps_list = list(steps_set)
grid_list = list(grid_set)
thread_list = sorted(list(thread_set))
frameworks_list = sorted(list(frameworks))

for num_steps in steps_list:
for grid_dims in grid_list:
    fig, ax = plt.subplots(1, 2, figsize=(7,1.5))

    for c in frameworks_list:
        r_sub = [r for r in results if r['framework'] == c
                 and r['num_steps'] == num_steps
                 and str(r['grid_dims']) == grid_dims]

        r_ave = {}
        for n in {r['num_threads'] for r in r_sub}:
            time, count = 0, 0
            for t in [r['time'] for r in r_sub if r['num_threads'] == n]:
                time += t
                count += 1
            r_ave[n] = time / count

        if len(r_ave) > 0:
            n_threads, times = r_ave.keys(), r_ave.values()
            p = ax[1].plot(n_threads, times[0] / times, label=c)
            p = ax[0].semilogy(n_threads, times, label=c)

    ax[1].set(xlabel=r"Number of Threads", ylabel=r"Speed Up")
    ax[1].get_xaxis().set_ticks(thread_list)
    ax[0].set(xlabel=r"Number of Threads", ylabel=r"Execution" "\n" "Time (s)")
    ax[0].get_xaxis().set_ticks(thread_list)
    ax[1].legend(bbox_to_anchor=(1, 0.8))
    fig.tight_layout()
    filename = '{}/num_thread_perf_s{}_g{}.png'.format(path, num_steps, grid_dims).replace(",", "").replace(" ", "_")
    print(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    plt.close()
