# coding=utf-8
# generate cpu/gpu util graph based on the json data

import argparse
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("logs")
parser.add_argument("output_png")

if __name__ == "__main__":
  args = parser.parse_args()

  with open(args.logs, "r") as f:
    data = json.load(f)

  # timing as the timestamp as the x axis, and others as y axis
  # timestamp to local time in seconds
  start_time = data["timing"][0]
  timings = [round(o - start_time, 1) for o in data["timing"]]

  # cpu and gpu util
  cpu_util = [round(o, 1) for o in data["cpu_utilization"]]
  gpu_util = [round(o, 1) for o in data["gpu_utilization"]]

  # gpu mem and ram, in MB
  ram_used = [round(o, 1) for o in data["ram_used"]]
  gpu_mem = [round(o, 1) for o in data["gpu_memory"]]

  # plot!
  plt.figure(figsize=(10, 6))
  # cpu util
  plt.subplot(221)
  plt.plot(timings, cpu_util, "g-")
  plt.title("cpu util %")
  plt.xlabel("seconds")
  plt.grid(True)

  plt.subplot(222)
  plt.plot(timings, ram_used, "g-")
  plt.title("ram used (MB)")
  plt.xlabel("seconds")
  plt.grid(True)

  plt.subplot(223)
  plt.plot(timings, gpu_util, "b-")
  plt.title("gpu util %")
  plt.xlabel("seconds")
  plt.grid(True)

  plt.subplot(224)
  plt.plot(timings, gpu_mem, "b-")
  plt.title("GPU mem (MB)")
  plt.xlabel("seconds")
  plt.grid(True)

  plt.subplots_adjust(hspace=0.5, wspace=0.3)

  plt.savefig(args.output_png, dpi=400)


