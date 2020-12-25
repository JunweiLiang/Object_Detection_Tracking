# coding=utf-8
# dataset object definition
import cv2
import os
import logging
import math
import numpy as np
from nn import resizeImage

# dataset object need to implement the following function
# get_sample(self, idx)
# collect_batch(self, datalist)
class ImageDataset(object):
  def __init__(self, cfg, split, imglst, annotations=None):
    """
      imglst: a file containing a list of absolute path to all the images
    """
    self.cfg = cfg  # this should include short_edge_size, max_size, etc.
    self.split = split
    self.imglst = imglst
    self.annotations = annotations

    # machine-specific config
    self.num_gpu = cfg.gpu
    self.batch_size = cfg.im_batch_size
    self.batch_size_per_gpu = self.batch_size // cfg.gpu
    assert self.batch_size % cfg.gpu == 0, "bruh"


    if self.split == "train":
      self.num_epochs = cfg.num_epochs
    else:
      self.num_epochs = 1

    # load the img file list
    self.imgs = [line.strip() for line in open(self.imglst).readlines()]

    self.num_samples = len(self.imgs)  # one epoch length

    self.num_batches_per_epoch = int(
        math.ceil(self.num_samples / float(self.batch_size)))
    self.num_batches = int(self.num_batches_per_epoch * self.num_epochs)
    self.valid_idxs = range(self.num_samples)

    logging.info("Loaded %s imgs", len(self.imgs))

  def get_sample(self, idx):
    """
    preprocess one sample from the list
    """
    cfg = self.cfg
    img_file_path = self.imgs[idx]

    imgname = os.path.splitext(os.path.basename(img_file_path))[0]

    frame = cv2.imread(img_file_path)
    im = frame.astype("float32")

    resized_image = resizeImage(im, cfg.short_edge_size, cfg.max_size)

    scale = (resized_image.shape[0] * 1.0 / im.shape[0] + \
             resized_image.shape[1] * 1.0 / im.shape[1]) / 2.0

    return resized_image, scale, imgname, (im.shape[0], im.shape[1])

  def collect_batch(self, data, idxs=None):
    """
    collect the idxs of the data list into a dictionary
    """
    if idxs is None:
      idxs = range(len(data))
    imgs, scales, imgnames, shapes = zip(*[data[idx] for idx in idxs])

    return {
        "imgs": imgs,
        "scales": scales,
        "imgnames": imgnames,
        "ori_shapes": shapes
    }


