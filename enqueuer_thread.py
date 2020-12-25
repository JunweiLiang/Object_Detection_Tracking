# coding=utf-8
"""Given the dataset object, make a multithread enqueuer"""
import os
import queue
import threading
import contextlib
import multiprocessing
import time
import random
import sys
import utils
import traceback

# for video queuer
from nn import resizeImage
import cv2

# modified from keras
class DatasetEnqueuer(object):
  def __init__(self, dataset, prefetch=5, num_workers=1,
               start=True, # start the dataset get thread when init
               shuffle=False,
               # whether to break down each mini-batch for each gpu
               is_multi_gpu=False,
               last_full_batch=False,  # make sure the last batch is full
              ):
    self.dataset = dataset

    self.prefetch = prefetch  # how many batch to save in queue
    self.max_queue_size = int(self.prefetch * dataset.batch_size)

    self.is_multi_gpu = is_multi_gpu
    self.last_full_batch = last_full_batch

    self.workers = num_workers
    self.queue = None
    self.run_thread = None  # the thread to spawn others
    self.stop_signal = None

    self.cur_batch_count = 0

    self.shuffle = shuffle

    if start:
      self.start()

  def is_running(self):
    return self.stop_signal is not None and not self.stop_signal.is_set()

  def start(self):
    self.queue = queue.Queue(self.max_queue_size)
    self.stop_signal = threading.Event()

    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()

  def stop(self):
    #print("stop called")
    if self.is_running():
      self._stop()

  def _stop(self):
    #print("_stop called")
    self.stop_signal.set()
    with self.queue.mutex:
      self.queue.queue.clear()
      self.queue.unfinished_tasks = 0
      self.queue.not_full.notify()

    self.run_thread.join(0)

  def __del__(self):
    if self.is_running():
      self._stop()

  # thread to start getting batches into queue
  def _run(self):
    batch_idxs = list(self.dataset.valid_idxs) * self.dataset.num_epochs

    if self.shuffle:
      batch_idxs = random.sample(batch_idxs, len(batch_idxs))
      batch_idxs = random.sample(batch_idxs, len(batch_idxs))

    if self.last_full_batch:
      # make sure the batch_idxs are multiplier of batch_size
      batch_idxs += [batch_idxs[-1] for _ in range(
          self.dataset.batch_size - len(batch_idxs) % self.dataset.batch_size)]

    while True:
      with contextlib.closing(
          multiprocessing.pool.ThreadPool(self.workers)) as executor:
        for idx in batch_idxs:
          if self.stop_signal.is_set():
            return
          # block until not full
          self.queue.put(
              executor.apply_async(self.dataset.get_sample, (idx,)), block=True)

        self._wait_queue()
        if self.stop_signal.is_set():
          # We're done
          return

  # iterator to get batch from the queue
  def get(self):
    if not self.is_running():
      self.start()
    try:
      while self.is_running():
        if self.cur_batch_count == self.dataset.num_batches:
          self._stop()
          return
        samples = []
        for i in range(self.dataset.batch_size):
          # first get got the ApplyResult object,
          # then second get to get the actual thing (block till get)
          sample = self.queue.get(block=True).get()
          self.queue.task_done()
          samples.append(sample)

        # break the mini-batch into mini-batches for multi-gpu
        if self.is_multi_gpu:
          batches = []
          # a list of [frames, boxes, labels_arr, ori_boxes, box_keys]
          this_batch_idxs = range(len(samples))

          # pack these batches for each gpu
          this_batch_idxs_gpus = utils.grouper(
              this_batch_idxs, self.dataset.batch_size_per_gpu)
          for this_batch_idxs_per_gpu in this_batch_idxs_gpus:
            batches.append(self.dataset.collect_batch(
                samples, this_batch_idxs_per_gpu))

          batch = batches
        else:
          batch = self.dataset.collect_batch(samples)

        self.cur_batch_count += 1
        yield batch

    except Exception as e:  # pylint: disable=broad-except
      self._stop()
      _type, _value, _traceback = sys.exc_info()
      print("Exception in enqueuer.get: %s" % e)
      traceback.print_tb(_traceback)
      raise Exception

  def _wait_queue(self):
    """Wait for the queue to be empty."""
    while True:
      time.sleep(0.1)
      if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
        return


def count_frame_get(total_frame, frame_gap):
  count = 0
  cur_frame = 0
  while cur_frame < total_frame:
    if cur_frame % frame_gap != 0:
      cur_frame += 1
      continue
    count += 1
    cur_frame += 1
  return count

class VideoEnqueuer(object):
  def __init__(self,
               cfg,
               vcap,
               num_frame,
               frame_gap=1,
               prefetch=5,
               start=True, # start the dataset get thread when init
               is_moviepy=False,
               batch_size=4,
              ):
    self.cfg = cfg
    self.vcap = vcap
    self.num_frame = num_frame
    self.frame_gap = frame_gap

    self.is_moviepy = is_moviepy
    self.batch_size = batch_size

    self.prefetch = prefetch  # how many batch to save in queue
    self.max_queue_size = int(self.prefetch * batch_size)

    self.queue = None
    self.run_thread = None  # the thread to spawn others
    self.stop_signal = None

    # how many frames we are actually gonna get due to frame gap
    self.get_num_frame = count_frame_get(self.num_frame, self.frame_gap)
    # compute the number of batches we gonna get so we know when to stop and exit
    # last batch is not enough batch_size
    self.num_batches = self.get_num_frame // batch_size + \
        int(self.get_num_frame % batch_size != 0)
    self.cur_batch_count = 0

    if start:
      self.start()

  def is_running(self):
    return self.stop_signal is not None and not self.stop_signal.is_set()

  def start(self):
    self.queue = queue.Queue(self.max_queue_size)
    self.stop_signal = threading.Event()

    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()

  def stop(self):
    #print("stop called")
    if self.is_running():
      self._stop()

  def _stop(self):
    #print("_stop called")
    self.stop_signal.set()
    with self.queue.mutex:
      self.queue.queue.clear()
      self.queue.unfinished_tasks = 0
      self.queue.not_full.notify()

    self.run_thread.join(0)

  def __del__(self):
    if self.is_running():
      self._stop()

  # thread to start getting batches into queue
  def _run(self):
    cfg = self.cfg

    frame_count = 0
    while frame_count < self.num_frame:
      if self.stop_signal.is_set():
        return

      if self.is_moviepy:
        suc = True
        frame = next(self.vcap)
      else:
        suc, frame = self.vcap.read()
      if not suc:
        frame_count += 1
        continue

      if frame_count % self.frame_gap != 0:
        frame_count += 1
        continue

      # process the frames
      if self.is_moviepy:
        # moviepy ask ffmpeg to get rgb24
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      im = frame.astype("float32")

      resized_image = resizeImage(im, cfg.short_edge_size, cfg.max_size)

      scale = (resized_image.shape[0] * 1.0 / im.shape[0] + \
               resized_image.shape[1] * 1.0 / im.shape[1]) / 2.0

      self.queue.put((resized_image, scale, frame_count), block=True)

      frame_count += 1

    self._wait_queue()
    if self.stop_signal.is_set():
      # We're done
      return

  # iterator to get batch from the queue
  def get(self):
    if not self.is_running():
      self.start()
    try:
      while self.is_running():
        if self.cur_batch_count == self.num_batches:
          self._stop()
          return

        batch_size = self.batch_size
        # last batch
        if (self.cur_batch_count == self.num_batches - 1) and (
            self.get_num_frame % batch_size != 0):
          batch_size = self.get_num_frame % batch_size

        samples = []
        for i in range(batch_size):
          sample = self.queue.get(block=True)
          self.queue.task_done()
          samples.append(sample)

        batch = samples

        self.cur_batch_count += 1

        yield batch

    except Exception as e:  # pylint: disable=broad-except
      self._stop()
      _type, _value, _traceback = sys.exc_info()
      print("Exception in enqueuer.get: %s" % e)
      traceback.print_tb(_traceback)
      raise Exception

  def _wait_queue(self):
    """Wait for the queue to be empty."""
    while True:
      time.sleep(0.1)
      if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
        return
