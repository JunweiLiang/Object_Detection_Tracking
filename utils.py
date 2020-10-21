# coding=utf-8
"""Some util functions/classes."""


import random
import itertools
import math
import sys
import os
import operator
import cv2
#import commands
if sys.version_info > (3, 0):
  import subprocess as commands
else:
  import commands

import tensorflow as tf
from operator import mul
#from itertools import izip_longest
import itertools

from collections import defaultdict
import numpy as np

import pycocotools.mask as cocomask

# these pycocotools guys will cause
# 'Unable to init server: Could not connect: Connection refused'
# for python 3
# so need this
import matplotlib
matplotlib.use("Agg")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#from nn import soft_nms, nms
from generate_anchors import generate_anchors


class Summary():
  def __init__(self):
    self.lines = []
  def add(self, string, print_it=True):
    if print_it:
      print(string)
    self.lines.append(string)

  def writeTo(self, path):
    with open(path, "w") as f:
      f.writelines("%s" % ("\n".join(self.lines)))

def grouper(l, n, fillvalue=None):
  # given a list and n(batch_size), devide list into n sized chunks
  # last one will fill None
  args = [iter(l)]*n
  if sys.version_info > (3, 0):
    out = itertools.zip_longest(*args, fillvalue=None)
  else:
    out = itertools.izip_longest(*args, fillvalue=None)
  out = list(out)
  return out

# simple FIFO class for moving average computation
class FIFO_ME:
  def __init__(self, N):
    self.N = N
    self.lst = []
    assert N > 0

  def put(self, val):
    if val is None:
      return None
    self.lst.append(val)
    if len(self.lst) > self.N:
      self.lst.pop(0)
    return 1

  def me(self):
    if not self.lst:
      return None
    return np.mean(self.lst)

# return the gpu utilization at the moment. float between 0~1.0
# tested for nvidia 384.90
# gpuid_range is a tuple of (gpu_startid, gpu_num)
def parse_nvidia_smi(gpuid_range):
  nvi_out = commands.getoutput("nvidia-smi")
  gpu_info_blocks = get_gpu_info_block(nvi_out)[
      gpuid_range[0]:(gpuid_range[0] + gpuid_range[1])]
  # num_gpu = len(gpu_info_blocks)  # the ones we care
  # all are a list of
  temps = [float(info_block.strip().strip("|").split()[1].strip("C"))
           for info_block in gpu_info_blocks]
  utilizations = [float(
      info_block.strip().strip("|").split()[-2].strip("%")) / 100.0
                  for info_block in gpu_info_blocks]
  return temps, utilizations

def get_gpu_info_block(nvi_out):
  nvi_out = nvi_out.split("\n")
  start_idx = -1
  end_idx = -1
  for i, line in enumerate(nvi_out):
    if line.startswith("|====="):
      start_idx = i+1
      break
  for i, line in enumerate(nvi_out):
    if line.startswith("     "):
      end_idx = i
      break
  assert (start_idx >= 0) and (end_idx >= 0), nvi_out
  # each gpu contains two line
  gpu_info_blocks = []
  for i in range(start_idx, end_idx, 3):
    # nvi_out[i]:"|   0  GeForce GTX TIT...  Off  | 00000000:01:00.0 Off |
    #                N/A |"
    # nvi_out[i+1]: "| 47%   81C    P2    87W / 250W |  10547MiB / 12205MiB |
    #   0%      Default |"
    gpu_info_blocks.append(nvi_out[i+1])
  return gpu_info_blocks


def nms_wrapper(final_boxes, final_probs, config):
  # in this mode,
  # final_boxes would be [num_class-1, num_prop, 4]
  # final_probs would be [num_class-1, num_prop]
  # 1. make one dets matrix
  # [num_class-1, num_prop, 5]
  dets = np.concatenate([final_boxes, np.expand_dims(final_probs, axis=-1)],
                        axis=-1)

  final_boxes, final_probs, final_labels = [], [], []
  for c in range(dets.shape[0]):  # 0- num_class-1
    this_dets = dets[c]
    # hard limit of confident score
    select_ids = this_dets[:, -1] > config.result_score_thres
    this_dets = this_dets[select_ids, :]

    classid = c + 1  # first one is BG

    # 2. nms, get [K, 5]
    #if config.use_soft_nms:
    #  keep = soft_nms(this_dets)
    #else:
    keep = nms(this_dets, config.fastrcnn_nms_iou_thres)
    this_dets = this_dets[keep, :]

    # sort the output and keep only k for each class
    boxes = this_dets[:, :4] # [K,4]
    probs = this_dets[:, 4] # [K]

    final_boxes.extend(boxes)
    final_probs.extend(probs)
    final_labels.extend([classid for i in range(len(probs))])

  # they could be empty, for empty scenes when filtered using result_score_thres
  if not final_boxes:
    return [], [], []

  final_boxes_all = np.array(final_boxes, dtype="float")
  final_probs_all = np.array(final_probs)
  final_labels_all = np.array(final_labels)

  # keep max result across all class
  ranks = np.argsort(final_probs)[::-1]
  final_boxes = final_boxes_all[ranks, :][:config.result_per_im]
  final_probs = final_probs_all[ranks][:config.result_per_im]
  final_labels = final_labels_all[ranks][:config.result_per_im]
  return final_boxes, final_labels, final_probs


class Dataset():
  # data should be
  """
  data = {"imgs":[],"ids":[],"gt":[]}

  """

  def __init__(self, data, add_gt=False, valid_idxs=None):
    self.data = data
    self.add_gt = add_gt
    self.valid_idxs = range(len(next(iter(self.data.values())))) \
                      if valid_idxs is None else valid_idxs
    self.num_examples = len(self.valid_idxs) # get one var "x" and get the len


  def get_by_idxs(self, idxs):
    out = defaultdict(list) # so the initial value is a list
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs) # extend with one whole list

    return out

  # retrun num_batchs , each batch is batch_size.
  # if cap, will make sure the total sample used <= dataset size
  def get_batches(self, batch_size, num_batches, shuffle=True, cap=False):

    num_batches_per_epoch = int(
        math.ceil(self.num_examples / float(batch_size)))
    if cap and (num_batches > num_batches_per_epoch):
      num_batches = num_batches_per_epoch

    # this may be zero
    num_epochs = int(math.ceil(num_batches/float(num_batches_per_epoch)))
    # shuflle
    if shuffle:
      # this is the list of shuffled all idxs
      random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
      # all batch idxs for one epoch
      random_grouped = lambda: list(grouper(random_idxs, batch_size))

      grouped = random_grouped
    else:

      raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
      grouped = raw_grouped

    # all batches idxs from multiple epochs
    batch_idxs_iter = itertools.chain.from_iterable(
        grouped() for _ in range(num_epochs))
    # so how all the epoch is order is fixed here

    for _ in range(num_batches):
      # so in the end batch, the None will not included
      batch_idxs = tuple(i for i in next(batch_idxs_iter) if i is not None)

      # a dict of {"x":[],"y":[],"ids":[]...}
      # batch_idxs could be str?
      #batch_data = self.get_by_idxs(batch_idxs)

      #yield batch_idxs,Dataset(batch_data) # make a new Dataset object
      # will continue next time it is called, i.e., in the next loop

      # modififiled for multi gpu setting, each image has one Dataset Object
      batch_datas = [self.get_by_idxs([idx]) for idx in batch_idxs]
      #print(batch_idxs
      #print(batch_datas
      yield batch_idxs, [Dataset(batch_data) for batch_data in batch_datas]


# helper function for eval
def gather_dt(boxes, probs, labels, eval_target, targetid2class, tococo=False,
              coco_class_names=None):
  target_dt_boxes = {one:[] for one in eval_target.keys()}
  for box, prob, label in zip(boxes, probs, labels):
    # coco box
    box[2] -= box[0]
    box[3] -= box[1]

    assert label > 0

    if tococo:
      cat_name = coco_class_names[label]
    else:
      # diva class trained from scratch
      cat_name = targetid2class[label]

    target_class = None

    if tococo:
      for t in eval_target:
        if cat_name in eval_target[t]:
          target_class = t
    else:
      if cat_name in eval_target:
        target_class = cat_name

    if target_class is None: # box from other class of mscoco/diva
      continue

    prob = float(round(prob, 4))
    #box = list(map(lambda x:float(round(x, 2)),box))
    box = [float(round(x, 2)) for x in box]

    target_dt_boxes[target_class].append((box, prob))
  return target_dt_boxes


def aggregate_eval(e, maxDet=100):
  aps = {}
  ars = {}
  for catId in e:
    e_c = e[catId]
    # put all detection scores from all image together
    dscores = np.concatenate([e_c[imageid]["dscores"][:maxDet]
                              for imageid in e_c])
    # sort
    inds = np.argsort(-dscores, kind="mergesort")
    # dscores_sorted = dscores[inds]

    # put all detection annotation together based on the score sorting
    dm = np.concatenate([e_c[imageid]["dm"][:maxDet] for imageid in e_c])[inds]
    num_gt = np.sum([e_c[imageid]["gt_num"] for imageid in e_c])

    # here the average precision should also put the unmatched ground truth
    #as detection box with lowest score
    #aps[catId] = computeAP(dm)
    aps[catId] = computeAP_v2(dm, num_gt)
    ars[catId] = computeAR_2(dm, num_gt)

  return aps, ars


def weighted_average(aps, ars, eval_target_weight=None):

  if eval_target_weight is not None:
    average_ap = sum([aps[class_]*eval_target_weight[class_] for class_ in aps])
    average_ar = sum([ars[class_]*eval_target_weight[class_] for class_ in ars])
  else:
    average_ap = sum(aps.values())/float(len(aps))
    average_ar = sum(ars.values())/float(len(ars))

  return average_ap, average_ar


def gather_gt(anno_boxes, anno_labels, eval_target, targetid2class):
  gt_boxes = {one:[] for one in eval_target.keys()}
  for box, label in zip(anno_boxes, anno_labels):
    label = targetid2class[label]
    if label in eval_target:
      #gt_box = list(map(lambda x:float(round(x,1)),box))
      gt_box = [float(round(x, 1)) for x in box]
      # gt_box is in (x1,y1,x2,y2)
      # convert to coco box
      gt_box[2] -= gt_box[0]
      gt_box[3] -= gt_box[1]

      gt_boxes[label].append(gt_box)
  return gt_boxes

# change e in place
def match_dt_gt(e, imgid, target_dt_boxes, gt_boxes, eval_target):
  for target_class in eval_target.keys():
    #if len(gt_boxes[target_class]) == 0:
    #  continue
    target_dt_boxes[target_class].sort(key=operator.itemgetter(1), reverse=True)
    d = [box for box, prob in target_dt_boxes[target_class]]
    dscores = [prob for box, prob in target_dt_boxes[target_class]]
    g = gt_boxes[target_class]

    # len(D), len(G)
    dm, gm = match_detection(d, g, cocomask.iou(
        d, g, [0 for _ in range(len(g))]), iou_thres=0.5)

    e[target_class][imgid] = {
        "dscores": dscores,
        "dm": dm,
        "gt_num": len(g)}


# for activity boxes
def gather_act_singles(actsingleboxes, actsinglelabels, topk):
  single_act_boxes = []
  single_act_labels = []
  single_act_probs = []
  # [K,num_act_class]
  # descending order
  sorted_prob_single = np.argsort(actsinglelabels, axis=-1)[:, ::-1]
  BG_ids = sorted_prob_single[:, 0] == 0 # [K] of bool

  for j in range(len(actsinglelabels)):
    if BG_ids[j]:
      continue
    labelIds = [sorted_prob_single[j, k] for k in range(topk)]
    # ignore BG class # or ignore everything after BG class?
    this_labels = [lid for lid in labelIds if lid != 0]
    this_probs = [actsinglelabels[j, lid] for lid in this_labels]
    this_boxes = [actsingleboxes[j] for _ in range(len(this_labels))]

    single_act_probs.extend(this_probs)
    single_act_labels.extend(this_labels)
    single_act_boxes.extend(this_boxes)
  return single_act_boxes, single_act_labels, single_act_probs


def match_detection(d, g, ious, iou_thres=0.5):
  D = len(d)
  G = len(g)
  # < 0 to note it is not matched, once matched will be the index of the d
  gtm = -np.ones((G)) # whether a gt box is matched
  dtm = -np.ones((D))

  # for each detection bounding box (ranked), will get the best IoU
  # matched ground truth box
  for didx, _ in enumerate(d):
    iou = iou_thres # the matched iou
    m = -1 # used to remember the matched gidx
    for gidx, _ in enumerate(g):
      # if this gt box is matched
      if gtm[gidx] >= 0:
        continue

      # the di,gi pair doesn"t have the required iou
      # or not better than before
      if ious[didx, gidx] < iou:
        continue

      # got one
      iou = ious[didx, gidx]
      m = gidx

    if m == -1:
      continue
    gtm[m] = didx
    dtm[didx] = m
  return dtm, gtm


def get_all_anchors(stride, sizes, ratios, max_size):
  """
  Get all anchors in the largest possible image, shifted, floatbox

  Returns:
    anchors: SxSxNUM_ANCHORx4, where S == MAX_SIZE//STRIDE, floatbox
    The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SCALE.

  """
  # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
  # are centered on stride / 2, have (approximate) sqrt areas of the specified
  # sizes, and aspect ratios as given.
  # got all anchor start from center (8,8) [so the base box is (0,0,15,15)]
  # -> ratios * scales
  cell_anchors = generate_anchors(
      stride, scales=np.array(sizes, dtype=np.float) / stride,
      ratios=np.array(ratios, dtype=np.float))
  # anchors are intbox here.
  # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

  # 1920/16 -> 120
  # previous tensorpack code
  #field_size = max_size // stride # how many anchor position in an image
  # at one axis
  field_size = int(np.ceil(max_size / stride))
  # 0, 120, ...., 1920
  # 120*120 (x,y)
  shifts = np.arange(0, field_size) * stride # each position"s (x,y)
  shift_x, shift_y = np.meshgrid(shifts, shifts)

  shift_x = shift_x.flatten()
  shift_y = shift_y.flatten()
  # for 1920 , will be (120x120,4) # all the anchor boxes xy
  # all anchor position xy, so should be [51x51, 4]
  shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
  # Kx4, K = field_size * field_size
  K = shifts.shape[0]  # 1920 gets 120x120

  A = cell_anchors.shape[0] # number of anchor at 1 position
  field_of_anchors = (
      cell_anchors.reshape((1, A, 4)) +
      shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
  field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
  # FSxFSxAx4
  # Many rounding happens inside the anchor code anyway
  #assert np.all(field_of_anchors == field_of_anchors.astype("int32")),
  #(field_of_anchors,field_of_anchors.astype("int32"))
  # 1920 -> (120,120,NA,4)
  field_of_anchors = field_of_anchors.astype("float32")
  # the last 4 is (x1,y1,x2,y2)
  # (x1,y1+1,x2+1,y2)??
  field_of_anchors[:, :, :, [2, 3]] += 1
  return field_of_anchors

# flatten a tensor
# [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
# keep how many dimension in the end, so final rank is keep + 1
def flatten(tensor, keep):
  # get the shape
  fixed_shape = tensor.get_shape().as_list() #[N, JQ, di] # [N, M, JX, di]
  # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
  start = len(fixed_shape) - keep
  # each num in the [] will a*b*c*d...
  # so [0] -> just N here for left
  # for [N, M, JX, di] , left is N*M
  left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i]
                      for i in range(start)])
  # [N, JQ,di]
  # [N*M, JX, di]
  out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start, len(fixed_shape))]
  # reshape
  flat = tf.reshape(tensor, out_shape)
  return flat

def evalcoco(res, annofile, add_mask=False):
  coco = COCO(annofile)
  cocoDt = coco.loadRes(res)
  cocoEval = COCOeval(coco, cocoDt, "bbox")
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()

  if add_mask:
    cocoEval = COCOeval(coco, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


#给定秒数，换成 H M S
def sec2time(secs):
  #return strftime("%H:%M:%S",time.gmtime(secs)) # doesnt support millisec  """
  m, s = divmod(secs, 60)
  #print(m,s
  h, m = divmod(m, 60)
  if s >= 10.0:
    return "%02d:%02d:%.3f"%(h, m, s)
  else:
    return "%02d:%02d:0%.3f"%(h, m, s)


def get_op_tensor_name(name):
  """
  Will automatically determine if ``name`` is a tensor name (ends with ":x")
  or a op name.
  If it is an op name, the corresponding tensor name is assumed to be
  ``op_name + ":0"``.

  Args:
    name(str): name of an op or a tensor
  Returns:
    tuple: (op_name, tensor_name)
  """
  if len(name) >= 3 and name[-2] == ":":
    return name[:-2], name
  else:
    return name, name + ":0"


# from tensorpack
def draw_boxes(im, boxes, labels=None, colors=None):
  """
  Args:
    im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
    boxes (np.ndarray or list[BoxBase]): If an ndarray,
      must be of shape Nx4 where the second dimension is [x1, y1, x2, y2].
    labels: (list[str] or None)
    color: a 3-tuple (in range [0, 255]). By default will choose automatically.
  Returns:
    np.ndarray: a new image.
  """
  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = 0.4
  if isinstance(boxes, list):
    arr = np.zeros((len(boxes), 4), dtype="int32")
    for idx, b in enumerate(boxes):
      assert isinstance(b, BoxBase), b
      arr[idx, :] = [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
    boxes = arr
  else:
    boxes = boxes.astype("int32")
  if labels is not None:
    assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
  areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
  sorted_inds = np.argsort(-areas)  # draw large ones first
  assert areas.min() > 0, areas.min()
  # allow equal, because we are not very strict about rounding error here
  assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
    and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
    "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

  im = im.copy()
  COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2), dtype="int32")
  COLOR_CANDIDATES = PALETTE_RGB[:, ::-1]
  if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
  for i in sorted_inds:
    box = boxes[i, :]

    best_color = colors[i] if colors is not None else (255, 0, 0)
    if labels is not None:
      label = labels[i]

      # find the best placement for the text
      ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
      bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
      top_left = [box[0] + 1, box[1] - 1.3 * lineh]
      if top_left[1] < 0:   # out of image
        top_left[1] = box[3] - 1.3 * lineh
        bottom_left[1] = box[3] - 0.3 * lineh
      textbox = IntBox(int(top_left[0]), int(top_left[1]),
                       int(top_left[0] + linew), int(top_left[1] + lineh))
      textbox.clip_by_shape(im.shape[:2])


      cv2.putText(im, label, (textbox.x1, textbox.y2),
                  FONT, FONT_SCALE, color=best_color)#, lineType=cv2.LINE_AA)
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                  color=best_color, thickness=1)
  return im


# a lists of floats, if < 0 means false positive, otherwise true positive
# assume lists is sorted
def computeAP(lists):

  #相关的总数
  rels = 0
  #当前排名
  rank = 0
  #AP 分数
  score = 0.0
  for one in lists:
    rank += 1
    #是相关的
    if one >= 0:
      rels += 1
      score += rels / float(rank)
  if rels != 0:
    score /= float(rels)
  return score

def computeAP_v2(lists, total_gt):

  #相关的总数
  rels = 0
  #当前排名
  rank = 0
  #AP 分数
  score = 0.0
  for one in lists:
    rank += 1
    #是相关的
    if one >= 0:
      rels += 1
      score += rels / float(rank)
  if total_gt != 0:
    score /= float(total_gt)
  return score

# given a fixed number (recall_k) of detection,
# assume d is sorted, and each d should be < 0
# if false positive, true positive d[i] == gidx
def computeAR(d, g, recall_k):
  TrueDetections = len([one for one in d[:recall_k] if one >= 0])
  num_gt = len(g)
  if len(g) > recall_k:
    num_gt = recall_k
  if not g:
    return 1.0
  else:
    return TrueDetections/float(num_gt)


def computeAR_2(d, num_gt):
  true_positives = len([one for one in d if one >= 0])
  if num_gt == 0:
    return 1.0
  else:
    return true_positives/float(num_gt)



PALETTE_HEX = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#A30059",
    "#997D87", "#FF2F80", "#D16100", "#00846F", "#001E09", "#788D66", "#886F4C",
    "#938A81", "#1E6E00", "#9B9700", "#922329", "#6A3A4C", "#222800", "#5B4E51",
    "#7ED379", "#012C58"]


def _parse_hex_color(s):
  r = int(s[1:3], 16)
  g = int(s[3:5], 16)
  b = int(s[5:7], 16)
  return (r, g, b)

PALETTE_RGB = np.asarray(
    list(map(_parse_hex_color, PALETTE_HEX)),
    dtype="int32")


# conver from COCO format (x,y,w,h) to (x1,y1,x2,y2)
def box_wh_to_x1x2(box):
  return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

class BoxBase(object):
  __slots__ = ["x1", "y1", "x2", "y2"]

  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

  def copy(self):
    new = type(self)()
    for i in self.__slots__:
      setattr(new, i, getattr(self, i))
    return new

  def __str__(self):
    return "{}(x1={}, y1={}, x2={}, y2={})".format(
      type(self).__name__, self.x1, self.y1, self.x2, self.y2)

  __repr__ = __str__

  def area(self):
    return self.w * self.h

  def is_box(self):
    return self.w > 0 and self.h > 0


class IntBox(BoxBase):
  def __init__(self, x1, y1, x2, y2):
    for k in [x1, y1, x2, y2]:
      assert isinstance(k, int)
    super(IntBox, self).__init__(x1, y1, x2, y2)

  @property
  def w(self):
    return self.x2 - self.x1 + 1

  @property
  def h(self):
    return self.y2 - self.y1 + 1

  def is_valid_box(self, shape):
    """
    Check that this rect is a valid bounding box within this shape.
    Args:
      shape: int [h, w] or None.
    Returns:
      bool
    """
    if min(self.x1, self.y1) < 0:
      return False
    if min(self.w, self.h) <= 0:
      return False
    if self.x2 >= shape[1]:
      return False
    if self.y2 >= shape[0]:
      return False
    return True

  def clip_by_shape(self, shape):
    """
    Clip xs and ys to be valid coordinates inside shape
    Args:
      shape: int [h, w] or None.
    """
    self.x1 = np.clip(self.x1, 0, shape[1] - 1)
    self.x2 = np.clip(self.x2, 0, shape[1] - 1)
    self.y1 = np.clip(self.y1, 0, shape[0] - 1)
    self.y2 = np.clip(self.y2, 0, shape[0] - 1)

  def roi(self, img):
    assert self.is_valid_box(img.shape[:2]), \
           "{} vs {}".format(self, img.shape[:2])
    return img[self.y1:self.y2 + 1, self.x1:self.x2 + 1]
