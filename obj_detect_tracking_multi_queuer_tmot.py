# coding=utf-8
"""
  run object detection and tracking inference
  tracking uses https://github.com/Zhongdao/Towards-Realtime-MOT
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from enqueuer_thread import VideoEnqueuer

# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True

import matplotlib
# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3
matplotlib.use('Agg')

from tqdm import tqdm

import numpy as np
import tensorflow as tf

# detection stuff
from models import get_model
from models import resizeImage
from nn import fill_full_mask
from utils import get_op_tensor_name
from utils import parse_nvidia_smi
from utils import sec2time
from utils import PerformanceLogger

import pycocotools.mask as cocomask

# tracking stuff
#from deep_sort import nn_matching
#from deep_sort.detection import Detection
#from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import linear_inter_bbox
from tmot.multitracker import JDETracker

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

  parser.add_argument("--video_dir", default=None)
  parser.add_argument("--video_lst_file", default=None,
                      help="video_file_path = os.path.join(video_dir, $line)")

  parser.add_argument("--obj_out_dir", default=None,
                      help="out_dir/$basename/%%d.json, start from 0 index. "
                           "This is the object box output. Leave this blank "
                           "when use tracking to avoid saving the obj class "
                           "output to save IO time.")

  parser.add_argument("--frame_gap", default=8, type=int)

  parser.add_argument("--threshold_conf", default=0.0001, type=float)

  parser.add_argument("--is_load_from_pb", action="store_true",
                      help="load from a frozen graph")

  # logging (machine-wise) cpu and gpu usage using nvidia-smi and psutil
  # this only works if you are only running this script on your machine
  parser.add_argument("--log_time_and_gpu", action="store_true")
  parser.add_argument("--util_log_interval", type=float, default=10.)
  parser.add_argument("--save_util_log_to", default=None,
                      help="save to a json for generating figures")


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
  parser.add_argument("--fix_gpuid_range", action="store_true",
                      help="for junweil.pc")
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

  # use lijun video loader, this should deal with avi videos
  # with duplicate frames
  parser.add_argument(
      "--use_lijun_video_loader", action="store_true",
      help="use video loader from https://github.com/Lijun-Yu/diva_io")
  parser.add_argument("--use_moviepy", action="store_true")


  # ----------- tracking params
  parser.add_argument("--get_tracking", action="store_true",
                      help="this will generate tracking results for each frame")
  parser.add_argument("--tracking_dir", default="/tmp",
                      help="output will be out_dir/$videoname.txt, start from 0"
                           " index")
  parser.add_argument("--tracking_objs", default="Person,Vehicle",
                      help="Objects to be tracked, default are Person and "
                           "Vehicle")
  parser.add_argument("--min_confidence", default=0.5, type=float,
                      help="Detection confidence threshold. Disregard all "
                           "detections that have a confidence lower than this "
                           "value.")
  parser.add_argument("--nms_max_overlap", default=0.5, type=float,
                      help="Non-maxima suppression threshold: Maximum detection"
                           " overlap.")
  parser.add_argument("--track_max_second_lost", type=float, default=8.,
                      help="track is considered lost after this seconds")
  parser.add_argument("--emb_agg_method", default="max",
                      help="avg / max pooling / spatial")
  parser.add_argument("--emb_max_dist", type=float, default=0.7)
  parser.add_argument("--iou_max_dist1", type=float, default=0.9)
  parser.add_argument("--iou_max_dist2", type=float, default=0.9)
  parser.add_argument("--emb_smooth_alpha", type=float, default=0.9)

  parser.add_argument("--use_kf_box_in_tracks", action="store_true",
                      help="save with KF box instead of original detection box")
  parser.add_argument("--no_inter_box", action="store_true",
                      help="no interpolation of boxes")


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

  # ------ 04/2020, efficientdet
  parser.add_argument("--is_efficientdet", action="store_true")
  parser.add_argument("--efficientdet_modelname", default="efficientdet-d0")
  parser.add_argument("--efficientdet_max_detection_topk", type=int,
                      default=5000, help="#topk boxes before NMS")
  parser.add_argument("--efficientdet_min_level", type=int, default=3)
  parser.add_argument("--efficientdet_max_level", type=int, default=7)

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


  # for efficient use of COCO model classes
  parser.add_argument("--use_partial_classes", action="store_true")

  # ---- for multi-thread frame preprocessing
  parser.add_argument("--prefetch", type=int, default=10,
                      help="maximum number of batch in queue")

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
    if args.use_partial_classes:
      partial_classes = ["BG"] + args.partial_classes
      targetClass2id = {classname: i
                        for i, classname in enumerate(partial_classes)}
      targetid2class = {targetClass2id[o]: o for o in targetClass2id}

  # ---- 04/2020, efficientdet
  if args.is_efficientdet:
    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class
    args.num_class = 81
    args.is_coco_model = True

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
  #  args.short_edge_size = np.ceil(
  #      args.short_edge_size / args.fpn_resolution_requirement) * \
  #          args.fpn_resolution_requirement
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


def check_args(args):
  """Check the argument."""
  assert args.video_dir is not None
  assert args.video_lst_file is not None
  assert args.frame_gap >= 1
  #print("cv2 version %s" % (cv2.__version__)


def preprocess_detections(final_boxes, final_probs, final_labels,
                          box_feats, targetid2class, tracking_objs, min_confidence,
                          scale, is_coco_model=False,
                          coco_to_actev_mapping=None):

  # tracking_objs is a single item
  detections = []
  tracking_boxes = final_boxes / scale
  for j, (box, prob, label) in enumerate(zip(tracking_boxes, final_probs, final_labels)):
    cat_name = targetid2class[label]
    if is_coco_model:
      if cat_name not in coco_to_actev_mapping:
        continue
      else:
        cat_name = coco_to_actev_mapping[cat_name]

    confidence_socre = float(round(prob, 7))
    if cat_name not in tracking_objs or confidence_socre < min_confidence:
      continue

    box[2] -= box[0]
    box[3] -= box[1]  # x, y, w, h
    avg_feat = box_feats[j]
    if len(avg_feat.shape) > 2:  # [C, H, W]
      avg_feat = np.mean(box_feats[j], axis=(1, 2))

    #norm_feat = avg_feat / np.linalg.norm(avg_feat)  # will be normed later

    # xywh, conf, feature

    detections.append((box, confidence_socre, avg_feat))

  return detections

def run_detect_and_track(args, frame_stack, sess, model, targetid2class,
                         tracking_objs,
                         tracker_dict, tracking_results_dict,
                         obj_out_dir=None,
                         valid_frame_num=None):
  # ignore the padded images
  if valid_frame_num is None:
    valid_frame_num = len(frame_stack)

  resized_images, scales, frame_idxs = zip(*frame_stack)

  feed_dict = model.get_feed_dict_forward_multi(resized_images)

  sess_input = [model.final_boxes, model.final_labels,
                model.final_probs, model.final_valid_indices,
                model.fpn_box_feat]
  # [B, num, 4], [B, num], [B, num], [B], [M, 256, 7, 7]
  batch_boxes, batch_labels, batch_probs, valid_indices, batch_box_feats = \
      sess.run(sess_input, feed_dict=feed_dict)
  assert np.sum(valid_indices) == batch_box_feats.shape[0], "duh"


  if len(batch_box_feats.shape) > 2:
    # use the 256 dim as embedding
    if args.emb_agg_method == "avg":
      batch_box_feats = np.mean(batch_box_feats, axis=(2, 3))
    elif args.emb_agg_method == "max":
      batch_box_feats = np.amax(batch_box_feats, axis=(2, 3))
    elif args.emb_agg_method == "spatial":
      # use the spatial 7x7 as embedding
      batch_box_feats = np.mean(batch_box_feats, axis=1)
      # [8*100, 49]
      batch_box_feats = np.reshape(
          batch_box_feats, (batch_box_feats.shape[0], -1))
    else:
      raise Exception("Not implemented agg method: %s" % args.emb_agg_method)



  for b in range(valid_frame_num):
    cur_frame = frame_idxs[b]

    # [k, 4]
    final_boxes = batch_boxes[b][:valid_indices[b]]
    # [k]
    final_labels = batch_labels[b][:valid_indices[b]]
    # [k]
    final_probs = batch_probs[b][:valid_indices[b]]
    # [k, C]
    previous_box_num = sum(valid_indices[:b])
    box_feats = batch_box_feats[previous_box_num:previous_box_num+valid_indices[b]]

    if args.get_tracking:

      assert len(box_feats) == len(final_boxes)

      for tracking_obj in tracking_objs:

        # will consider scale here
        scale = scales[b]
        # xywh, conf, feature
        detections = preprocess_detections(
            final_boxes, final_probs, final_labels, box_feats,
            targetid2class, [tracking_obj], args.min_confidence,
            scale,
            is_coco_model=args.is_coco_model,
            coco_to_actev_mapping=coco_obj_to_actev_obj)


        # Run non-maxima suppression.
        boxes = np.array([d[0] for d in detections])
        scores = np.array([d[1] for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, args.nms_max_overlap, scores)

        detections = [detections[i] for i in indices]


        # tracking
        output_stracks = tracker_dict[tracking_obj].update(detections)

        # Store results
        for track in output_stracks:
          if args.use_kf_box_in_tracks:
            tlwh = track.tlwh
          else:
            tlwh = track.cur_det_tlwh

          track_id = track.track_id
          tracking_results_dict[tracking_obj].append([
              cur_frame, track_id, tlwh[0], tlwh[1], tlwh[2],
              tlwh[3]])

    if obj_out_dir is None:  # not saving the boxes
      continue

    # ---------------- get the json outputs for object detection

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

      res = {
          "category_id": int(cat_id),
          "cat_name": cat_name,  # [0-80]
          "score": float(round(prob, 7)),
          #"bbox": list(map(lambda x: float(round(x, 2)), box)),
          "bbox": [float(round(x, 2)) for x in box],
          "segmentation": None,
      }

      pred.append(res)

    predfile = os.path.join(obj_out_dir, "%d.json" % (cur_frame))

    with open(predfile, "w") as f:
      json.dump(pred, f)


if __name__ == "__main__":
  args = get_args()

  if args.log_time_and_gpu:

    start_time = time.time()

    gpuid_range = (args.gpuid_start, args.gpu)
    if args.fix_gpuid_range:
      gpuid_range = (0, 1)

    performance_logger = PerformanceLogger(
        gpuid_range,
        interval=args.util_log_interval)
    performance_logger.start()

  check_args(args)

  videolst = [os.path.join(args.video_dir, one.strip())
              for one in open(args.video_lst_file).readlines()]

  if args.obj_out_dir is not None:
    if not os.path.exists(args.obj_out_dir):
      os.makedirs(args.obj_out_dir)

  # 2020, deal with opencv  avi video "bug":
  # https://github.com/opencv/opencv/issues/9053
  # need pyav
  if args.use_lijun_video_loader:
    # https://github.com/Lijun-Yu/diva_io
    from diva_io.video import VideoReader

  if args.use_moviepy:
    from moviepy.editor import VideoFileClip

  # 1. load the object detection model
  model = get_model(
      args, args.gpuid_start, is_multi=True, controller=args.controller)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  if not args.use_all_mem:
    tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i
                for i in range(args.gpuid_start, args.gpuid_start + args.gpu)]))

  with tf.Session(config=tfconfig) as sess:

    if not args.is_load_from_pb:
      initialize(config=args, sess=sess)

    for videofile in tqdm(videolst, ascii=True):
      # 2. read the video file
      if args.use_lijun_video_loader:
        vcap = VideoReader(videofile)
        frame_count = int(vcap.length)
      elif args.use_moviepy:
        vcap = VideoFileClip(videofile, audio=False)
        frame_count = int(vcap.fps * vcap.duration)  # uh
        vcap = vcap.iter_frames()
      else:
        try:
          vcap = cv2.VideoCapture(videofile)
          if not vcap.isOpened():
            raise Exception("cannot open %s" % videofile)
        except Exception as e:
          # raise e
          # just move on to the next video
          print("warning, cannot open %s" % videofile)
          continue

        # opencv 2
        if cv2.__version__.split(".")[0] == "2":
          frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        else:
          # opencv 3/4
          frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

      # initialize tracking module
      if args.get_tracking:
        tracking_objs = args.tracking_objs.split(",")

        tracker_dict = {}

        tracking_results_dict = {}

        for tracking_obj in tracking_objs:
          #metric = nn_matching.NearestNeighborDistanceMetric(
          #    "cosine", args.max_cosine_distance, args.nn_budget)
          tracker_dict[tracking_obj] = JDETracker(
              args.min_confidence, args.track_max_second_lost,
              args.emb_max_dist,
              args.iou_max_dist1,
              args.iou_max_dist2,
              args.emb_smooth_alpha,
              frame_gap=args.frame_gap,
              frame_rate=30.0)
          tracking_results_dict[tracking_obj] = []

      # videoname = os.path.splitext(os.path.basename(videofile))[0]
      videoname = os.path.basename(videofile)
      video_obj_out_path = None
      if args.obj_out_dir is not None:  # not saving box json to save time
        video_obj_out_path = os.path.join(args.obj_out_dir, videoname)
        if not os.path.exists(video_obj_out_path):
          os.makedirs(video_obj_out_path)

      video_queuer = VideoEnqueuer(
          args, vcap, frame_count, frame_gap=args.frame_gap,
          prefetch=args.prefetch,
          start=True, is_moviepy=args.use_moviepy,
          batch_size=args.im_batch_size)
      get_batches = video_queuer.get()


      for batch in tqdm(get_batches, total=video_queuer.num_batches):
        # batch is a list of (resized_image, scale, frame_count)
        valid_frame_num = len(batch)
        if len(batch) < args.im_batch_size:
          batch += [batch[-1]] * (args.im_batch_size - len(batch))

        run_detect_and_track(
            args, batch, sess, model, targetid2class,
            tracking_objs, tracker_dict, tracking_results_dict,
            video_obj_out_path,
            valid_frame_num=valid_frame_num)



      if not args.use_lijun_video_loader and not args.use_moviepy:
        vcap.release()

      if args.get_tracking:
        track_num = []
        for tracking_obj in tracking_objs:
          output_dir = os.path.join(args.tracking_dir, videoname, tracking_obj)
          if not os.path.exists(output_dir):
            os.makedirs(output_dir)

          output_file = os.path.join(
              output_dir, "%s.txt" % (os.path.splitext(videoname))[0])

          tracking_results = sorted(tracking_results_dict[tracking_obj],
                                    key=lambda x: (x[0], x[1]))
          # print(len(tracking_results)
          tracking_data = np.asarray(tracking_results)
          # print(tracking_data.shape
          if not args.no_inter_box:
            tracking_data = linear_inter_bbox(tracking_data, args.frame_gap)
          tracking_results = tracking_data.tolist()
          with open(output_file, "w") as fw:
            for row in tracking_results:
              line = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (
                  row[0], row[1], row[2], row[3], row[4], row[5])
              fw.write(line + "\n")

          # reset tracker
          tracker_dict[tracking_obj].reset()
          track_num.append(
              (tracking_obj, len({c[1]:1 for c in tracking_results})))
        print("Track num %s" % (track_num))



  if args.log_time_and_gpu:
    end_time = time.time()
    performance_logger.end()
    processed_frame_num = args.im_batch_size * video_queuer.num_batches
    logs = performance_logger.logs
    print("total run time %s (%.2f FPS), log utilize every %s seconds and get "
          "GPU util median %.2f%% and average %.2f%%. GPU temperature "
          "average %.2f (C), CPU util median %.2f%%" % (
              sec2time(end_time - start_time),
              #end_time - start_time,
              processed_frame_num / (end_time - start_time),
              args.util_log_interval,
              np.median(logs["gpu_utilization"]),
              np.mean(logs["gpu_utilization"]),
              np.mean(logs["gpu_temperature"]),
              np.median(logs["cpu_utilization"]),))

    if args.save_util_log_to is not None:

      with open(args.save_util_log_to, "w") as f:
        json.dump(logs, f)
      print("saved util log to %s" % args.save_util_log_to)

  cv2.destroyAllWindows()
