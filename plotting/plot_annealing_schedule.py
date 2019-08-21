#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import dwave_tools

schedule = dwave_tools.anneal_sched_custom()

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 6))

npoints = len(schedule)
x = [schedule[i][0] for i in range(npoints)]
y = [schedule[i][1] for i in range(npoints)]
print(x)
print(y)
plt.scatter(x, y, color='black')
plt.plot(x, y, label='Annealing schedule')

plt.legend()
plt.ylabel("Annealing")
plt.xlabel("Time [$\mu$s]")

ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig.savefig("anneal_schedule.pdf")
