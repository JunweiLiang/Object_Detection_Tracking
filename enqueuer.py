# coding=utf-8
"""Given the dataset object, make a multiprocess/thread enqueuer"""
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
import numpy as np

# TODo: checkout https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
# ------------------------------- the following is only needed for multiprocess
# multiprocess is only good for video inputs (num_workers=num_core)
# multithreading is good enough for frame inputs
# and somehow the optimal num_workers=4, for many kinds of machine with threads

# Global variables to be shared across processes
_SHARED_DATASETS = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None
# Because multiprocessing pools are inherently unsafe, starting from a clean
# state can be essential to avoiding deadlocks. In order to accomplish this, we
# need to be able to check on the status of Pools that we create.
_WORKER_ID_QUEUE = None  # Only created if needed.

# modified from keras
class DatasetEnqueuer(object):
  def __init__(self, dataset, prefetch=5, num_workers=1,
               start=True, # start the dataset get thread when init
               shuffle=False,
               # whether to break down each mini-batch for each gpu
               is_multi_gpu=False,
               last_full_batch=False,  # make sure the last batch is full
               use_process=False, # use process instead of thread
              ):
    self.dataset = dataset

    self.prefetch = prefetch  # how many batch to save in queue
    self.max_queue_size = int(self.prefetch * dataset.batch_size)

    self.workers = num_workers
    self.queue = None
    self.run_thread = None  # the thread to spawn others
    self.stop_signal = None

    self.cur_batch_count = 0

    self.shuffle = shuffle

    self.use_process = use_process

    self.is_multi_gpu = is_multi_gpu
    self.last_full_batch = last_full_batch

    # need to have a global uid for each enqueuer so we could use train/val
    # at the same time
    global _SEQUENCE_COUNTER
    if _SEQUENCE_COUNTER is None:
      try:
        _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
      except OSError:
        # In this case the OS does not allow us to use
        # multiprocessing. We resort to an int
        # for enqueuer indexing.
        _SEQUENCE_COUNTER = 0

    if isinstance(_SEQUENCE_COUNTER, int):
      self.uid = _SEQUENCE_COUNTER
      _SEQUENCE_COUNTER += 1
    else:
      # Doing Multiprocessing.Value += x is not process-safe.
      with _SEQUENCE_COUNTER.get_lock():
        self.uid = _SEQUENCE_COUNTER.value
        _SEQUENCE_COUNTER.value += 1

    if start:
      self.start()

  def is_running(self):
    return self.stop_signal is not None and not self.stop_signal.is_set()

  def start(self):
    if self.use_process:
      self.executor_fn = self._get_executor_init(self.workers)
    else:
      self.executor_fn = lambda _: multiprocessing.pool.ThreadPool(self.workers)

    self.queue = queue.Queue(self.max_queue_size)
    self.stop_signal = threading.Event()

    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()

  def _get_executor_init(self, workers):
    """Gets the Pool initializer for multiprocessing.

    Arguments:
        workers: Number of workers.

    Returns:
        Function, a Function to initialize the pool
    """
    def pool_fn(seqs):
      pool = multiprocessing.Pool(
          workers, initializer=init_pool_generator,
          initargs=(seqs, None, get_worker_id_queue()))
      return pool

    return pool_fn

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

    _SHARED_DATASETS[self.uid] = None

  def __del__(self):
    if self.is_running():
      self._stop()

  def _send_dataset(self):
    """Sends current Iterable to all workers."""
    # For new processes that may spawn
    _SHARED_DATASETS[self.uid] = self.dataset

  # preprocess the data and put them into queue
  def _run(self):
    batch_idxs = list(self.dataset.valid_idxs) * self.dataset.num_epochs

    if self.shuffle:
      batch_idxs = random.sample(batch_idxs, len(batch_idxs))
      batch_idxs = random.sample(batch_idxs, len(batch_idxs))

    if self.last_full_batch:
      # make sure the batch_idxs are multiplier of batch_size
      batch_idxs += [batch_idxs[-1] for _ in range(
          self.dataset.batch_size - len(batch_idxs) % self.dataset.batch_size)]

    self._send_dataset()  # Share the initial dataset

    while True:
      #with contextlib.closing(
      #    multiprocessing.pool.ThreadPool(self.workers)) as executor:
      with contextlib.closing(
          self.executor_fn(_SHARED_DATASETS)) as executor:
        for idx in batch_idxs:
          if self.stop_signal.is_set():
            return
          # block until not full
          #self.queue.put(
          #    executor.apply_async(self.dataset.get_sample, (idx,)), block=True)
          self.queue.put(
              executor.apply_async(get_index, (self.uid, idx)), block=True)

        self._wait_queue()
        if self.stop_signal.is_set():
          # We're done
          return

      self._send_dataset()  # Update the pool

  # get batch from the queue
  # toDo: this is single thread, put the batch collecting into multi-thread
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
          # a list of [frames, boxes, labels_arr, ori_boxes, box_keys]
          batches = []

          this_batch_idxs = range(len(samples))

          # pack these batches for each gpu
          this_batch_idxs_gpus = utils.grouper(
              this_batch_idxs, self.dataset.batch_size_per_gpu)
          batches = []
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


def get_worker_id_queue():
  """Lazily create the queue to track worker ids."""
  global _WORKER_ID_QUEUE
  if _WORKER_ID_QUEUE is None:
    _WORKER_ID_QUEUE = multiprocessing.Queue()
  return _WORKER_ID_QUEUE

def get_index(uid, i):
  """Get the value from the Ddataset `uid` at index `i`.

  To allow multiple Sequences to be used at the same time, we use `uid` to
  get a specific one. A single Sequence would cause the validation to
  overwrite the training Sequence.

  Arguments:
      uid: int, Sequence identifier
      i: index

  Returns:
      The value at index `i`.
  """
  return _SHARED_DATASETS[uid].get_sample(i)

def init_pool_generator(gens, random_seed=None, id_queue=None):
  """Initializer function for pool workers.

  Args:
    gens: State which should be made available to worker processes.
    random_seed: An optional value with which to seed child processes.
    id_queue: A multiprocessing Queue of worker ids. This is used to indicate
      that a worker process was created by Keras and can be terminated using
      the cleanup_all_keras_forkpools utility.
  """
  global _SHARED_DATASETS
  _SHARED_DATASETS = gens

  worker_proc = multiprocessing.current_process()

  # name isn't used for anything, but setting a more descriptive name is helpful
  # when diagnosing orphaned processes.
  worker_proc.name = 'Enqueuer_worker_{}'.format(worker_proc.name)

  if random_seed is not None:
    np.random.seed(random_seed + worker_proc.ident)

  if id_queue is not None:
    # If a worker dies during init, the pool will just create a replacement.
    id_queue.put(worker_proc.ident, block=True, timeout=0.1)
