# coding=utf-8
"""
  run object detection and tracking inference
"""

import argparse
import cv2
import math
import json
import random
import sys
import time
import threading
import operator
import os
import pickle
import datasets
import enqueuer_thread
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True

import matplotlib
# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3
matplotlib.use('Agg')

from tqdm import tqdm
from PIL import Image

import numpy as np
import tensorflow as tf

# detection stuff
from models import get_model
from nn import resizeImage
from nn import fill_full_mask
from utils import get_op_tensor_name
from utils import parse_nvidia_smi
from utils import sec2time

import pycocotools.mask as cocomask

# tracking stuff
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos,linear_inter_bbox,filter_short_objs

# for mask
import pycocotools.mask as cocomask

# class ids stuff
from class_ids import targetClass2id_new_nopo
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj
from class_ids import coco_id_mapping

targetClass2id = targetClass2id_new_nopo
targetid2class = {targetClass2id[one]: one for one in targetClass2id}

def get_args():
  """Parse arguments and intialize some hyper-params."""
  global targetClass2id, targetid2class
  parser = argparse.ArgumentParser()

  parser.add_argument("--img_lst", default=None, help="path to imgs")

  parser.add_argument("--out_dir", default=None,
                      help="out_dir/imgname.json")


  parser.add_argument("--threshold_conf", default=0.0001, type=float)

  parser.add_argument("--is_load_from_pb", action="store_true",
                      help="load from a frozen graph")
  parser.add_argument("--log_time_and_gpu", action="store_true")


  parser.add_argument("--version", type=int, default=4, help="model version")
  parser.add_argument("--is_coco_model", action="store_true",
                      help="is coco model, will output coco classes instead")
  parser.add_argument("--use_gn", action="store_true",
                      help="it is group norm model")
  parser.add_argument("--use_conv_frcnn_head", action="store_true",
                      help="group norm model from tensorpack uses conv head")


  # ---- gpu params
  parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
  parser.add_argument("--gpuid_start", default=0, type=int,
                      help="start of gpu id")
  parser.add_argument("--im_batch_size", type=int, default=1)
  parser.add_argument("--use_all_mem", action="store_true")

  # ----------- model params
  parser.add_argument("--num_class", type=int, default=15,
                      help="num catagory + 1 background")

  parser.add_argument("--model_path", default="/app/object_detection_model")

  parser.add_argument("--rpn_batch_size", type=int, default=256,
                      help="num roi per image for RPN  training")
  parser.add_argument("--frcnn_batch_size", type=int, default=512,
                      help="num roi per image for fastRCNN training")

  parser.add_argument("--rpn_test_post_nms_topk", type=int, default=1000,
                      help="test post nms, input to fast rcnn")

  parser.add_argument("--max_size", type=int, default=1920,
                      help="num roi per image for RPN and fastRCNN training")
  parser.add_argument("--short_edge_size", type=int, default=1080,
                      help="num roi per image for RPN and fastRCNN training")


  parser.add_argument("--bupt_exp", action="store_true",
                      help="activity box experiemnt")

  # ---- tempory: for activity detection model
  parser.add_argument("--actasobj", action="store_true")
  parser.add_argument("--actmodel_path",
                      default="/app/activity_detection_model")

  parser.add_argument("--resnet152", action="store_true", help="")
  parser.add_argument("--resnet50", action="store_true", help="")
  parser.add_argument("--resnet34", action="store_true", help="")
  parser.add_argument("--resnet18", action="store_true", help="")
  parser.add_argument("--use_se", action="store_true",
                      help="use squeeze and excitation in backbone")
  parser.add_argument("--use_frcnn_class_agnostic", action="store_true",
                      help="use class agnostic fc head")
  parser.add_argument("--use_resnext", action="store_true", help="")
  parser.add_argument("--use_att_frcnn_head", action="store_true",
                      help="use attention to sum [K, 7, 7, C] feature into"
                           " [K, C]")

  parser.add_argument("--is_efficientdet", action="store_true",
                      help="efficientDet")

  # ---- COCO Mask-RCNN model
  parser.add_argument("--add_mask", action="store_true")

  # --------------- exp junk
  parser.add_argument("--use_dilations", action="store_true",
                      help="use dilations=2 in res5")
  parser.add_argument("--use_deformable", action="store_true",
                      help="use deformable conv")
  parser.add_argument("--add_act", action="store_true",
                      help="add activitiy model")
  parser.add_argument("--finer_resolution", action="store_true",
                      help="fpn use finer resolution conv")
  parser.add_argument("--fix_fpn_model", action="store_true",
                      help="for finetuneing a fpn model, whether to fix the"
                           " lateral and poshoc weights")
  parser.add_argument("--is_cascade_rcnn", action="store_true",
                      help="cascade rcnn on top of fpn")
  parser.add_argument("--add_relation_nn", action="store_true",
                      help="add relation network feature")

  parser.add_argument("--test_frame_extraction", action="store_true")
  parser.add_argument("--use_my_naming", action="store_true")

  # for efficient use of COCO model classes
  parser.add_argument("--use_partial_classes", action="store_true")
  parser.add_argument("--person_only", action="store_true")

  parser.add_argument("--only_classes", default=None,
                      help="only these classnames (comma seperated) to save")


  # ---- for multi-thread frame preprocessing
  parser.add_argument("--prefetch", type=int, default=10,
                      help="maximum number of batch in queue")
  parser.add_argument("--num_cpu_worker", type=int, default=4,
                      help="number for thread pool")

  args = parser.parse_args()

  if args.use_partial_classes:
    args.is_coco_model = True
    args.partial_classes = [classname for classname in coco_obj_to_actev_obj]

  #assert args.gpu == args.im_batch_size  # one gpu one image
  #assert args.gpu == 1, "Currently only support single-gpu inference"

  if args.is_load_from_pb:
    args.load_from = args.model_path

  args.controller = "/cpu:0"  # parameter server

  targetid2class = targetid2class
  targetClass2id = targetClass2id

  if args.actasobj:
    from class_ids import targetAct2id
    targetClass2id = targetAct2id
    targetid2class = {targetAct2id[one]: one for one in targetAct2id}
  if args.bupt_exp:
    from class_ids import targetAct2id_bupt
    targetClass2id = targetAct2id_bupt
    targetid2class = {targetAct2id_bupt[one]: one for one in targetAct2id_bupt}

  assert len(targetClass2id) == args.num_class, (len(targetClass2id),
                                                 args.num_class)


  assert args.version in [2, 3, 4, 5, 6], \
         "Currently we only have version 2-6 model"

  if args.version == 2:
    pass
  elif args.version == 3:
    args.use_dilations = True
  elif args.version == 4:
    args.use_frcnn_class_agnostic = True
    args.use_dilations = True
  elif args.version == 5:
    args.use_frcnn_class_agnostic = True
    args.use_dilations = True
  elif args.version == 6:
    args.use_frcnn_class_agnostic = True
    args.use_se = True

  if args.is_coco_model:
    assert args.version == 2
    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class
    args.num_class = 81

    if args.person_only:
      args.num_class = 2
      targetid2class = {0: "BG", 1: "person"}
      targetClass2id = {"BG": 0, "person": 1}

    if args.use_partial_classes:
      partial_classes = ["BG"] + args.partial_classes
      targetClass2id = {classname: i
                        for i, classname in enumerate(partial_classes)}
      targetid2class = {targetClass2id[o]: o for o in targetClass2id}

  args.classname2id = targetClass2id
  args.classid2name = targetid2class

  # ---------------more defautls
  args.is_pack_model = False
  args.diva_class3 = True
  args.diva_class = False
  args.diva_class2 = False
  args.use_small_object_head = False
  args.use_so_score_thres = False
  args.use_so_association = False
  #args.use_gn = False
  #args.use_conv_frcnn_head = False
  args.so_person_topk = 10
  args.use_cpu_nms = False
  args.use_bg_score = False
  args.freeze_rpn = True
  args.freeze_fastrcnn = True
  args.freeze = 2
  args.small_objects = ["Prop", "Push_Pulled_Object",
                        "Prop_plus_Push_Pulled_Object", "Bike"]
  args.no_obj_detect = False
  #args.add_mask = False
  args.is_fpn = True
  # args.new_tensorpack_model = True
  args.mrcnn_head_dim = 256
  args.is_train = False

  args.rpn_min_size = 0
  args.rpn_proposal_nms_thres = 0.7
  args.anchor_strides = (4, 8, 16, 32, 64)

  # [3] is 32, since we build FPN with r2,3,4,5, so 2**5
  args.fpn_resolution_requirement = float(args.anchor_strides[3])

  #if args.is_efficientdet:
  #  args.fpn_resolution_requirement = 128.0  # 2 ** max_level
    #args.short_edge_size = np.ceil(
    #    args.short_edge_size / args.fpn_resolution_requirement) * \
    #        args.fpn_resolution_requirement

  args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) * \
                  args.fpn_resolution_requirement

  args.fpn_num_channel = 256

  args.fpn_frcnn_fc_head_dim = 1024

  # ---- all the mask rcnn config

  args.resnet_num_block = [3, 4, 23, 3]  # resnet 101
  args.use_basic_block = False  # for resnet-34 and resnet-18
  if args.resnet152:
    args.resnet_num_block = [3, 8, 36, 3]
  if args.resnet50:
    args.resnet_num_block = [3, 4, 6, 3]
  if args.resnet34:
    args.resnet_num_block = [3, 4, 6, 3]
    args.use_basic_block = True
  if args.resnet18:
    args.resnet_num_block = [2, 2, 2, 2]
    args.use_basic_block = True

  args.anchor_stride = 16  # has to be 16 to match the image feature
  args.anchor_sizes = (32, 64, 128, 256, 512)

  args.anchor_ratios = (0.5, 1, 2)

  args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
  # iou thres to determine anchor label
  # args.positive_anchor_thres = 0.7
  # args.negative_anchor_thres = 0.3

  # when getting region proposal, avoid getting too large boxes
  args.bbox_decode_clip = np.log(args.max_size / 16.0)

  # fastrcnn
  args.fastrcnn_batch_per_im = args.frcnn_batch_size
  args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype="float32")

  args.fastrcnn_fg_thres = 0.5  # iou thres
  # args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

  # testing
  args.rpn_test_pre_nms_topk = 6000

  args.fastrcnn_nms_iou_thres = 0.5

  args.result_score_thres = args.threshold_conf
  args.result_per_im = 100


  if args.only_classes is not None:
    args.only_classes = args.only_classes.split(",")

  return args


def initialize(config, sess):
  """
    load the tf model weights into session
  """
  tf.global_variables_initializer().run()
  allvars = tf.global_variables()
  allvars = [var for var in allvars if "global_step" not in var.name]
  restore_vars = allvars
  opts = ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
          "Adadelta", "Momentum"]
  restore_vars = [var for var in restore_vars
                  if var.name.split(":")[0].split("/")[-1] not in opts]

  saver = tf.train.Saver(restore_vars, max_to_keep=5)

  load_from = config.model_path
  ckpt = tf.train.get_checkpoint_state(load_from)
  if ckpt and ckpt.model_checkpoint_path:
    loadpath = ckpt.model_checkpoint_path
    saver.restore(sess, loadpath)
  else:
    if os.path.exists(load_from):
      if load_from.endswith(".ckpt"):
        # load_from should be a single .ckpt file
        saver.restore(sess, load_from)
      elif load_from.endswith(".npz"):
        # load from dict
        weights = np.load(load_from)
        params = {get_op_tensor_name(n)[1]:v
                  #for n, v in dict(weights).iteritems()}
                  for n, v in dict(weights).items()}
        param_names = set(params.keys())

        variables = restore_vars

        variable_names = set([k.name for k in variables])

        intersect = variable_names & param_names

        restore_vars = [v for v in variables if v.name in intersect]

        with sess.as_default():
          for v in restore_vars:
            vname = v.name
            v.load(params[vname])

        not_used = [(o, weights[o].shape)
                    for o in weights.keys()
                    if get_op_tensor_name(o)[1] not in intersect]
        if not not_used:
          print("warning, %s/%s in npz not restored:%s" % (
              len(weights.keys()) - len(intersect), len(weights.keys()),
              not_used))

      else:
        raise Exception("Not recognized model type:%s" % load_from)
    else:
      raise Exception("Model not exists")


gpu_util_logs = []
gpu_temp_logs = []


def log_gpu_util(interval, gpuid_range):
  """
    A function to keep track of gpu usage using nvidia-smi
  """
  global gpu_util_logs
  while True:
    time.sleep(interval)
    gpu_temps, gpu_utils = parse_nvidia_smi(gpuid_range)
    gpu_util_logs.extend(gpu_utils)
    gpu_temp_logs.extend(gpu_temps)


def run_detect(resized_images, scales, imgnames, imgshapes,
               args, sess, model, targetid2class, valid_frame_num=None):

  # ignore the padded images
  if valid_frame_num is None:
    valid_frame_num = len(resized_images)

  feed_dict = model.get_feed_dict_forward_multi(resized_images)


  sess_input = [model.final_boxes, model.final_labels,
                model.final_probs, model.final_valid_indices]

  # [B, num, 4], [B, num], [B, num], [B]
  batch_boxes, batch_labels, batch_probs, valid_indices = sess.run(
      sess_input, feed_dict=feed_dict)

  # ---------------- get the json outputs for object detection from each img in
  # ------ the batch

  for b in range(valid_frame_num):
    # [k, 4]
    final_boxes = batch_boxes[b][:valid_indices[b]]
    # [k]
    final_labels = batch_labels[b][:valid_indices[b]]
    # [k]
    final_probs = batch_probs[b][:valid_indices[b]]

    # scale back the box to original image size
    final_boxes = final_boxes / scales[b]

    # save as json
    pred = []

    for j, (box, prob, label) in enumerate(zip(
        final_boxes, final_probs, final_labels)):
      box[2] -= box[0]
      box[3] -= box[1]  # produce x,y,w,h output

      cat_id = int(label)
      cat_name = targetid2class[cat_id]

      if args.only_classes and cat_name not in args.only_classes:
        continue

      res = {
          "category_id": int(cat_id),
          "cat_name": cat_name,  # [0-80]
          "score": float(round(prob, 7)),
          #"bbox": list(map(lambda x: float(round(x, 2)), box)),
          "bbox": [float(round(x, 2)) for x in box],
          "segmentation": None,
          "im_size": [imgshapes[b][0], imgshapes[b][1]],
      }

      pred.append(res)

    predfile = os.path.join(args.out_dir, "%s.json" % (imgnames[b]))

    with open(predfile, "w") as f:
      json.dump(pred, f)

if __name__ == "__main__":
  args = get_args()

  if args.log_time_and_gpu:
    gpu_log_interval = 10 # every k seconds
    start_time = time.time()
    gpu_check_thread = threading.Thread(
        target=log_gpu_util,
        args=[gpu_log_interval, (args.gpuid_start, args.gpu)])
    gpu_check_thread.daemon = True
    gpu_check_thread.start()

  img_dataset = datasets.ImageDataset(args, "test", args.img_lst)
  dataset_loader = enqueuer_thread.DatasetEnqueuer(
      img_dataset, prefetch=args.prefetch,
      shuffle=False,
      is_multi_gpu=False,
      last_full_batch=True,
      start=True, num_workers=args.num_cpu_worker)
  get_batches = dataset_loader.get()  # iterator to get batch

  if args.out_dir is not None:
    if not os.path.exists(args.out_dir):
      os.makedirs(args.out_dir)

  # 1. load the object detection model
  model = get_model(args, args.gpuid_start, controller=args.controller,
                    is_multi=True)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  if not args.use_all_mem:
    tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i
                for i in range(args.gpuid_start, args.gpuid_start + args.gpu)]))

  with tf.Session(config=tfconfig) as sess:

    if not args.is_load_from_pb:
      initialize(config=args, sess=sess)

    for frame_batch in tqdm(get_batches,
                            total=img_dataset.num_batches, ascii=True):
      # run detection and save outputs
      run_detect(
          frame_batch["imgs"], frame_batch["scales"],
          frame_batch["imgnames"], frame_batch["ori_shapes"],
          args, sess, model, targetid2class)


  if args.log_time_and_gpu:
    end_time = time.time()
    print("total run time %s (%s), log gpu utilize every %s seconds and get "
          "median %.2f%% and average %.2f%%. GPU temperature median %.2f and "
          "average %.2f (C)" % (
              sec2time(end_time - start_time),
              end_time - start_time,
              gpu_log_interval,
              np.median(gpu_util_logs)*100,
              np.mean(gpu_util_logs)*100,
              np.median(gpu_temp_logs),
              np.mean(gpu_temp_logs)))
  cv2.destroyAllWindows()
