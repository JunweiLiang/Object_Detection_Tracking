# coding=utf-8
"""
  main script for training and testing mask rcnn on MSCOCO/DIVA/MEVA dataset
  multi gpu version
"""

import argparse
import cv2
import math
import json
import random
import operator
import time
import os
import pickle
import sys
import threading
# so here won"t have poll allocator info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# solve the issue of a bug in while loop, when you import the graph in
# multi-gpu, prefix is not added in while loop op [tf 1.14]
# https://github.com/tensorflow/tensorflow/issues/26526
os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "1"

# remove all the annoying warnings from tf v1.10 to v1.13
import logging
logging.getLogger("tensorflow").disabled = True

import tensorflow as tf
import numpy as np
import pycocotools.mask as cocomask
from pycocotools.coco import COCO

from tqdm import tqdm
from glob import glob

from models import get_model
from models import pack
from models import  initialize
from trainer import Trainer
from tester import Tester
from nn import resizeImage
from nn import fill_full_mask
from utils import evalcoco
from utils import match_detection
from utils import computeAP
from utils import computeAR_2
from utils import grouper
from utils import gather_dt
from utils import gather_gt
from utils import match_dt_gt
from utils import gather_act_singles
from utils import aggregate_eval
from utils import weighted_average
from utils import parse_nvidia_smi
from utils import sec2time
from utils import Dataset
from utils import Summary
from utils import nms_wrapper
from utils import FIFO_ME

# for using a COCO model to finetuning with DIVA data.
from class_ids import targetClass2id
from class_ids import targetAct2id
from class_ids import targetSingleAct2id
from class_ids import targetClass2id_mergeProp
from class_ids import targetClass2id_new
from class_ids import targetClass2id_new_nopo
from class_ids import targetAct2id_bupt
from class_ids import bupt_act_mapping
from class_ids import targetAct2id_meva
from class_ids import meva_act_mapping
from class_ids import coco_obj_class_to_id
from class_ids import coco_obj_id_to_class
from class_ids import coco_obj_to_actev_obj

targetid2class = {targetClass2id[one]:one for one in targetClass2id}

targetactid2class = {targetAct2id[one]:one for one in targetAct2id}

targetsingleactid2class = {
    targetSingleAct2id[one]:one for one in targetSingleAct2id}

# coco class to DIVA class
eval_target = {
    "Vehicle": ["car", "motorcycle", "bus", "truck", "vehicle"],
    "Person": "person",
}

eval_best = "Person" # not used anymore, we use average as the best metric

def get_args():
  global targetClass2id, targetid2class
  parser = argparse.ArgumentParser()

  parser.add_argument("datajson")
  parser.add_argument("imgpath")

  parser.add_argument("--log_time_and_gpu", action="store_true")

  parser.add_argument("--outbasepath", type=str, default=None,
                      help="full path will be outbasepath/modelname/runId")

  parser.add_argument("--actoutbasepath", type=str, default=None,
                      help="for activity box forward only")

  parser.add_argument("--train_skip", type=int, default=1,
                      help="when load diva train set, skip how many.")
  parser.add_argument("--train_skip_offset", type=int, default=0,
                      help="when load diva train set, offset before skip")

  parser.add_argument("--val_skip", type=int, default=1,
                      help="when load diva val set, skip how many.")
  parser.add_argument("--val_skip_offset", type=int, default=0,
                      help="when load diva train set, offset before skip")
  parser.add_argument("--exit_after_val", action="store_true")

  parser.add_argument("--forward_skip", type=int, default=1,
                      help="forward, skip how many.")
  parser.add_argument("--use_two_level_outpath", action="store_true")
  parser.add_argument("--start_from", type=int, default=0,
                      help="forward, start from which batch")

  parser.add_argument("--modelname", type=str, default=None)
  parser.add_argument("--num_class", type=int, default=81,
                      help="num catagory + 1 background")

  # ---- for training, show losses" moving average
  parser.add_argument("--show_loss_period", type=int, default=1000)
  parser.add_argument("--loss_me_step", type=int, default=100,
                      help="moving average queue size")

  # ------ extract fpn feature of the whole image
  parser.add_argument("--extract_feat", action="store_true")
  parser.add_argument("--feat_path", default=None)
  parser.add_argument("--just_feat", action="store_true",
                      help="only extract full image feature no bounding box")

  # ------ do object detection and extract the fpn feature for each *final*boxes
  parser.add_argument("--get_box_feat", action="store_true")
  parser.add_argument("--box_feat_path", default=None)

  # ---different from above, only feat no object detection
  parser.add_argument("--videolst", default=None)
  parser.add_argument("--skip", action="store_true", help="skip existing npy")

  parser.add_argument("--tococo", action="store_true",
                      help="for training in diva using coco model, map diva"
                           " class1to1 to coco")
  parser.add_argument("--diva_class", action="store_true",
                      help="the last layer is 16 (full) class output as "
                           "the diva object classes")
  parser.add_argument("--diva_class2", action="store_true",
                      help="the last layer is new classes with person_object"
                           " boxes")
  parser.add_argument("--diva_class3", action="store_true",
                      help="the last layer is new classes without person_object"
                           " boxes")
  parser.add_argument("--is_coco_model", action="store_true")


  parser.add_argument("--merge_prop", action="store_true",
                      help="use annotation that merged prop and "
                           "Push_Pulled_Object and train")
  parser.add_argument("--use_bg_score", action="store_true")

  # ------------activity detection

  parser.add_argument("--act_as_obj", action="store_true",
                      help="activity box as obj box")

  parser.add_argument("--add_act", action="store_true",
                      help="add activitiy model")

  # 07/2019
  parser.add_argument("--bupt_exp", action="store_true",
                      help="bupt activity box exp")
  parser.add_argument("--meva_exp", action="store_true",
                      help="meva activity box exp")
  parser.add_argument("--check_img_exist", action="store_true",
                      help="check image exists when load data")

  parser.add_argument("--fix_obj_model", action="store_true",
                      help="fix the object detection part including rpn")
  # v1:
  parser.add_argument("--num_act_class", type=int, default=36,
                      help="num catagory + 1 background")
  parser.add_argument("--fastrcnn_act_fg_ratio", default=0.25, type=float)
  parser.add_argument("--act_relation_nn", action="store_true",
                      help="add relation link in activity fastrnn head")
  parser.add_argument("--act_loss_weight", default=1.0, type=float)
  # ----- activity detection version 2
  parser.add_argument("--act_v2", action="store_true")
  parser.add_argument("--act_single_topk", type=int, default=5,
                      help="each box topk classes are output")
  parser.add_argument("--num_act_single_class", default=36, type=int)
  parser.add_argument("--num_act_pair_class", default=21, type=int)


  # ---------------------------------------------

  parser.add_argument("--debug", action="store_true",
                      help="load fewer image for debug in training")
  parser.add_argument("--runId", type=int, default=1)

  # forward mode: imgpath is the list of images
  # will output result to outbasepath
  # forward still need a coco validation json to get the catgory names
  parser.add_argument("--mode", type=str, default="forward",
                      help="train | test | forward | boxfeat | givenbox")
  parser.add_argument("--avg_feat", action="store_true",
                      help="for boxfeat mode, output 7x7x2048 or just "
                           "2048 for each box")
  parser.add_argument("--boxjsonpath", default=None,
                      help="json contain a dict for all the boxes, imageId"
                           " -> boxes")
  parser.add_argument("--boxfeatpath", default=None,
                      help="where to save the box feat path, will be a npy"
                           " for each image")
  parser.add_argument("--boxclass", action="store_true",
                      help="do box classification as well")

  parser.add_argument("--resnet152", action="store_true", help="")
  parser.add_argument("--resnet50", action="store_true", help="")
  parser.add_argument("--resnet34", action="store_true", help="")
  parser.add_argument("--resnet18", action="store_true", help="")
  parser.add_argument("--use_se", action="store_true",
                      help="use squeeze and excitation in backbone")
  parser.add_argument("--use_resnext", action="store_true")

  parser.add_argument("--is_fpn", action="store_true")
  parser.add_argument("--use_gn", action="store_true",
                      help="whether to use group normalization")
  parser.add_argument("--ignore_gn_vars", action="store_true",
                      help="add gn to previous model, will ignore loading "
                           "the gn var first")
  parser.add_argument("--use_conv_frcnn_head", action="store_true",
                      help="use conv in fastrcnn head")
  parser.add_argument("--use_att_frcnn_head", action="store_true",
                      help="use attention to sum [K, 7, 7, C] feature "
                           "into [K, C]")
  parser.add_argument("--use_frcnn_class_agnostic", action="store_true",
                      help="use class agnostic fc head")
  parser.add_argument("--conv_frcnn_head_dim", default=256, type=int)

  parser.add_argument("--get_rpn_out", action="store_true")
  parser.add_argument("--rpn_out_path", default=None)

  parser.add_argument("--use_cpu_nms", action="store_true")
  parser.add_argument("--no_nms", action="store_true",
                      help="not using nms in the end, "
                           "save all pre_nms_topk boxes;")
  parser.add_argument("--save_all_box", action="store_true",
                      help="for DCR experiment, save all boxes "
                           "and scores in npz file")

  parser.add_argument("--use_small_object_head", action="store_true")
  parser.add_argument("--use_so_score_thres", action="store_true",
                      help="use score threshold before final nms")
  parser.add_argument("--oversample_so_img", action="store_true")
  parser.add_argument("--oversample_x", type=int, default=1, help="x + 1 times")
  parser.add_argument("--skip_no_so_img", action="store_true")
  parser.add_argument("--skip_no_object", default=None,
                      help="Bike, single object annotation filter")
  parser.add_argument("--so_outpath", default=None)

  parser.add_argument("--use_so_association", action="store_true")
  parser.add_argument("--so_person_topk", type=int, default=10)

  parser.add_argument("--freeze_rpn", action="store_true")
  parser.add_argument("--freeze_fastrcnn", action="store_true")

  parser.add_argument("--use_dilations", action="store_true",
                      help="use dilations=2 in res5")
  parser.add_argument("--use_deformable", action="store_true",
                      help="use dilations=2 in res5")

  parser.add_argument("--fpn_frcnn_fc_head_dim", type=int, default=1024)
  parser.add_argument("--fpn_num_channel", type=int, default=256)
  parser.add_argument("--freeze", type=int, default=0,
                      help="freeze backbone resnet until group 0|2")

  parser.add_argument("--finer_resolution", action="store_true",
                      help="fpn use finer resolution conv")

  parser.add_argument("--add_relation_nn", action="store_true",
                      help="add relation network feature")

  parser.add_argument("--focal_loss", action="store_true",
                      help="use focal loss for RPN and FasterRCNN loss, "
                           "instead of cross entropy")


  # for test mode on testing on the MSCOCO dataset, if not set this,
  # will use our evaluation script
  parser.add_argument("--use_coco_eval", action="store_true")
  parser.add_argument("--coco2014_to_2017", action="store_true",
                      help="if use the cocoval 2014 json and use val2017"
                           " filepath, need this option to get the correct"
                           " file path")

  parser.add_argument("--trainlst", type=str, default=None,
                      help="training frame name list,")
  parser.add_argument("--valframepath", type=str, default=None,
                      help="path to top frame path")
  parser.add_argument("--annopath", type=str, default=None,
                      help="path to annotation, each frame.npz")
  parser.add_argument("--valannopath", type=str, default=None,
                      help="path to annotation, each frame.npz")
  parser.add_argument("--one_level_framepath", action="store_true")
  parser.add_argument("--flip_image", action="store_true",
                      help="for training, whether to random horizontal "
                           "flipping for input image, maybe not for "
                           "surveillance video")

  parser.add_argument("--add_mask", action="store_true")

  parser.add_argument("--vallst", type=str, default=None,
                      help="validation for training")

  parser.add_argument("--load", action="store_true")
  parser.add_argument("--load_best", action="store_true")

  parser.add_argument("--skip_first_eval", action="store_true")
  parser.add_argument("--best_first", type=float, default=None)
  parser.add_argument("--force_first_eval", action="store_true")

  parser.add_argument("--no_skip_error", action="store_true")

  parser.add_argument("--show_stat", action="store_true",
                      help="show data distribution only")

  # use for pre-trained model
  parser.add_argument("--load_from", type=str, default=None)
  parser.add_argument("--ignore_vars", type=str, default=None,
                      help="variables to ignore, multiple seperate by : "
                           "like: logits/W:logits/b, this var only need to "
                           "be var name's sub string to ignore")

  parser.add_argument("--print_params", action="store_true",
                      help="print params and then exit")
  parser.add_argument("--show_restore", action="store_true",
                      help="load from existing model (npz), show the"
                           " weight that is restored")


  # -------------------- save model for deployment
  parser.add_argument("--is_pack_model", action="store_true", default=False,
                      help="with is_test, this will pack the model to a path"
                           " instead of testing")
  parser.add_argument("--pack_model_path", type=str, default=None,
                      help="path to save model, a .pb file")
  parser.add_argument("--note", type=str, default=None,
                      help="leave a note for this packed model for"
                           " future reference")
  parser.add_argument("--pack_modelconfig_path", type=str, default=None,
                      help="json file to save the config and note")

  # forward with frozen gragp
  parser.add_argument("--is_load_from_pb", action="store_true")

  # for efficientdet
  parser.add_argument("--is_efficientdet", action="store_true")
  parser.add_argument("--efficientdet_modelname", default="efficientdet-d0")
  parser.add_argument("--efficientdet_max_detection_topk", type=int,
                      default=5000, help="#topk boxes before NMS")
  parser.add_argument("--efficientdet_min_level", type=int, default=3)
  parser.add_argument("--efficientdet_max_level", type=int, default=7)

  # ------------------------------------ model specifics



  # ----------------------------------training detail
  parser.add_argument("--use_all_mem", action="store_true")
  parser.add_argument("--im_batch_size", type=int, default=1)
  parser.add_argument("--rpn_batch_size", type=int, default=256,
                      help="num roi per image for RPN  training")
  parser.add_argument("--frcnn_batch_size", type=int, default=512,
                      help="num roi per image for fastRCNN training")

  parser.add_argument("--rpn_test_post_nms_topk", type=int, default=1000,
                      help="test post nms, input to fast rcnn")
  # fastrcnn output NMS suppressing iou >= this thresZ
  parser.add_argument("--fastrcnn_nms_iou_thres", type=float, default=0.5)

  parser.add_argument("--max_size", type=int, default=1333,
                      help="num roi per image for RPN and fastRCNN training")
  parser.add_argument("--short_edge_size", type=int, default=800,
                      help="num roi per image for RPN and fastRCNN training")
  parser.add_argument("--scale_jitter", action="store_true",
                      help="if set this, will random get int from min to max"
                           " to resize image;original param will still be used"
                           " in testing")
  parser.add_argument("--short_edge_size_min", type=int, default=640,
                      help="num roi per image for RPN and fastRCNN training")
  parser.add_argument("--short_edge_size_max", type=int, default=800,
                      help="num roi per image for RPN and fastRCNN training")

  # ------------------------------mixup training
  parser.add_argument("--use_mixup", action="store_true")
  parser.add_argument("--use_constant_mixup_weight", action="store_true")
  parser.add_argument("--mixup_constant_weight", type=float, default=0.5)
  parser.add_argument("--mixup_chance", type=float, default=0.5,
                      help="the possibility of using mixup")
  parser.add_argument("--max_mixup_per_frame", type=int, default=15)

  # not used for fpn
  parser.add_argument("--small_anchor_exp", action="store_true")

  parser.add_argument("--positive_anchor_thres", default=0.7, type=float)
  parser.add_argument("--negative_anchor_thres", default=0.3, type=float)


  parser.add_argument("--fastrcnn_fg_ratio", default=0.25, type=float)

  parser.add_argument("--gpu", default=1, type=int, help="number of gpu")
  parser.add_argument("--gpuid_start", default=0, type=int,
                      help="start of gpu id")
  parser.add_argument("--model_per_gpu", default=1, type=int,
                      help="it will be set as a /task:k in device")
  parser.add_argument("--controller", default="/cpu:0",
                      help="controller for multigpu training")


  #parser.add_argument("--num_step",type=int,default=360000)
  parser.add_argument("--num_epochs", type=int, default=12)

  parser.add_argument("--save_period", type=int, default=5000,
                      help="num steps to save model and eval")

  # drop out rate
  parser.add_argument("--keep_prob", default=1.0, type=float,
                      help="1.0 - drop out rate;remember to set it to 1.0 "
                           "in eval")

  # l2 weight decay
  parser.add_argument("--wd", default=None, type=float)  # 0.0001

  parser.add_argument("--init_lr", default=0.1, type=float,
                      help=("start learning rate"))


  parser.add_argument("--use_lr_decay", action="store_true")
  parser.add_argument("--learning_rate_decay", default=0.94, type=float,
                      help=("learning rate decay"))
  parser.add_argument("--num_epoch_per_decay", default=2.0, type=float,
                      help=("how epoch after which lr decay"))
  parser.add_argument("--use_cosine_schedule", action="store_true")
  parser.add_argument("--use_exp_schedule", action="store_true")

  parser.add_argument("--warm_up_steps", default=3000, type=int,
                      help=("warm up steps not epochs"))
  parser.add_argument("--same_lr_steps", default=0, type=int,
                      help=("after warm up, keep the init_lr for k steps"))

  parser.add_argument("--optimizer", default="adam", type=str,
                      help="optimizer: adam/adadelta")
  parser.add_argument("--momentum", default=0.9, type=float)

  parser.add_argument("--result_score_thres", default=0.0001, type=float)
  parser.add_argument("--result_per_im", default=100, type=int)

  # clipping, suggest 100.0
  parser.add_argument("--clip_gradient_norm", default=None, type=float,
                      help=("norm to clip gradient to"))

  # for debug
  parser.add_argument("--vis_pre", action="store_true",
                      help="visualize preprocess images")
  parser.add_argument("--vis_path", default=None)


  # for efficient use of COCO model classes
  parser.add_argument("--use_partial_classes", action="store_true")

  args = parser.parse_args()

  if args.use_cosine_schedule:
    args.use_lr_decay = True
  if args.use_exp_schedule:
    args.use_lr_decay = True
    args.use_cosine_schedule = False

  if args.save_all_box:
    args.no_nms = True

  if args.no_nms:
    args.use_cpu_nms = True # so to avoid using TF nms in the graph
  assert args.model_per_gpu == 1, "not work yet!"
  assert args.gpu*args.model_per_gpu == args.im_batch_size # one gpu one image
  #args.controller = "/cpu:0" # parameter server


  targetid2class = targetid2class
  targetClass2id = targetClass2id

  args.small_objects = ["Prop", "Push_Pulled_Object",
                        "Prop_plus_Push_Pulled_Object", "Bike"]
  if args.use_small_object_head:
    assert args.merge_prop
    args.so_eval_target = {c:1 for c in args.small_objects}
    args.small_objects_targetClass2id = {
        c: i for i, c in enumerate(["BG"] + args.small_objects)}
    args.small_objects_targetid2class = {
        args.small_objects_targetClass2id[one]: one
        for one in args.small_objects_targetClass2id}

  if args.merge_prop:
    targetClass2id = targetClass2id_mergeProp
    targetid2class = {targetClass2id_mergeProp[one]:one
                      for one in targetClass2id_mergeProp}

  if args.diva_class2:
    targetClass2id = targetClass2id_new
    targetid2class = {targetClass2id_new[one]:one for one in targetClass2id_new}

  if args.diva_class3:
    targetClass2id = targetClass2id_new_nopo
    targetid2class = {targetClass2id_new_nopo[one]:one
                      for one in targetClass2id_new_nopo}


  args.classname2id = targetClass2id
  args.classid2name = targetid2class

  if args.act_as_obj:
    # replace the obj class with actitivy class
    targetClass2id = targetAct2id
    targetid2class = {targetAct2id[one]:one for one in targetAct2id}

  if args.bupt_exp:
    args.diva_class = True
    args.act_as_obj = True
    targetClass2id = targetAct2id_bupt
    targetid2class = {targetAct2id_bupt[one]:one for one in targetAct2id_bupt}

  if args.meva_exp:
    args.diva_class = True
    args.act_as_obj = True
    targetClass2id = targetAct2id_meva
    targetid2class = {targetAct2id_meva[one]:one for one in targetAct2id_meva}

  if args.is_coco_model:
    #assert args.mode == "forward" or args.mode == "pack"
    args.diva_class = False
    targetClass2id = coco_obj_class_to_id
    targetid2class = coco_obj_id_to_class

  if args.use_partial_classes:
    assert args.is_coco_model
    args.partial_classes = [classname for classname in coco_obj_to_actev_obj]
    args.classname2id = targetClass2id
    args.classid2name = targetid2class


  if not args.tococo:
    assert len(targetid2class) == args.num_class

  if not args.tococo and ((args.mode == "train") or (args.mode == "test")):
    assert args.num_class == len(targetid2class.keys())
  args.class_names = targetClass2id.keys()

  if args.vis_pre:
    assert args.vis_path is not None
    if not os.path.exists(args.vis_path):
      os.makedirs(args.vis_path)

  if args.add_act and (args.mode == "forward"):
    assert args.actoutbasepath is not None
    mkdir(args.actoutbasepath)

  if args.outbasepath is not None:
    mkdir(args.outbasepath)

  if args.skip_first_eval:
    assert args.best_first is not None

  if (args.outbasepath is not None) and (args.modelname is not None):
    args.outpath = os.path.join(args.outbasepath,
                                args.modelname,
                                str(args.runId).zfill(2))

    args.save_dir = os.path.join(args.outpath, "save")

    args.save_dir_best = os.path.join(args.outpath, "save-best")

    args.write_self_sum = True
    args.self_summary_path = os.path.join(args.outpath, "train_sum.txt")
    # path to save each validation step"s performance and loss
    args.stats_path = os.path.join(args.outpath, "stats.json")

  args.mrcnn_head_dim = 256

  args.no_obj_detect = False
  if args.mode == "videofeat":
    args.no_obj_detect = True

  args.anchor_stride = 16 # has to be 16 to match the image feature total stride
  args.anchor_sizes = (32, 64, 128, 256, 512)

  if args.small_anchor_exp:
    args.anchor_sizes = (16, 32, 64, 96, 128, 256) # not used for fpn

  if args.is_fpn:
    args.anchor_strides = (4, 8, 16, 32, 64)

    # we will pad H,W to be a multiplier of 32
    # [3] is 32, since there is a total pixel reduce of 2x2x2x2x2
    args.fpn_resolution_requirement = float(args.anchor_strides[3])

    if args.is_efficientdet:
      args.fpn_resolution_requirement = 128.0  # 2 ** max_level
      args.short_edge_size = np.ceil(
          args.short_edge_size / args.fpn_resolution_requirement) * \
              args.fpn_resolution_requirement
    args.max_size = np.ceil(args.max_size / args.fpn_resolution_requirement) \
        * args.fpn_resolution_requirement

    #args.fpn_num_channel = 256

    #args.fpn_frcnn_fc_head_dim = 1024


  if args.load_best:
    args.load = True
  if args.load_from is not None:
    args.load = True

  if args.mode == "train":
    assert args.outbasepath is not None
    assert args.modelname is not None
    args.is_train = True
    mkdir(args.save_dir)
    mkdir(args.save_dir_best)
  else:
    args.is_train = False
    args.num_epochs = 1

  if args.get_rpn_out:
    if not os.path.exists(args.rpn_out_path):
      os.makedirs(args.rpn_out_path)

  # ---- all the mask rcnn config

  args.resnet_num_block = [3, 4, 23, 3] # resnet 101
  args.use_basic_block = False # for resnet-34 and resnet-18
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

  #args.short_edge_size = 800
  #args.max_size = 1333

  args.anchor_ratios = (0.5, 1, 2)

  args.num_anchors = len(args.anchor_sizes) * len(args.anchor_ratios)
  # iou thres to determine anchor label
  #args.positive_anchor_thres = 0.7
  #args.negative_anchor_thres = 0.3

  # when getting region proposal, avoid getting too large boxes
  args.bbox_decode_clip = np.log(args.max_size / 16.0)

  # RPN training
  args.rpn_fg_ratio = 0.5
  args.rpn_batch_per_im = args.rpn_batch_size
  args.rpn_min_size = 0 # 8?

  args.rpn_proposal_nms_thres = 0.7

  args.rpn_train_pre_nms_topk = 12000 # not used in fpn
  args.rpn_train_post_nms_topk = 2000# this is used for fpn_nms_pre


  # fastrcnn
  args.fastrcnn_batch_per_im = args.frcnn_batch_size
  args.fastrcnn_bbox_reg_weights = np.array([10, 10, 5, 5], dtype="float32")
  #args.fastrcnn_bbox_reg_weights = np.array([20, 20, 10, 10], dtype="float32")

  args.fastrcnn_fg_thres = 0.5 # iou thres
  #args.fastrcnn_fg_ratio = 0.25 # 1:3 -> pos:neg

  # testing
  args.rpn_test_pre_nms_topk = 6000

  #args.rpn_test_post_nms_topk = 700 #1300 # 700 takes 40 hours, # OOM at 1722,28,28,1024 # 800 OOM for gpu4
  #args.fastrcnn_nms_thres = 0.5
  #args.fastrcnn_nms_iou_thres = 0.5 # 0.3 is worse

  #args.result_score_thres = 0.0001
  #args.result_per_im = 100 # 400 # 100

  if args.focal_loss and args.clip_gradient_norm is None:
    print("Focal loss needs gradient clipping or will have NaN loss")
    sys.exit()

  return args



def add_coco(config,datajson):
  coco = COCO(datajson)
  cat_ids = coco.getCatIds() #[80], each is 1-90
  cat_names = [c["name"] for c in coco.loadCats(cat_ids)] # [80]

  config.classId_to_cocoId = {(i+1): v for i, v in enumerate(cat_ids)}

  config.class_names = ["BG"] + cat_names
  # 0-80
  config.class_to_classId = {c:i for i, c in enumerate(config.class_names)}
  config.classId_to_class = {i:c for i, c in enumerate(config.class_names)}




# load all ground truth into memory
def read_data_diva(config, idlst, framepath, annopath, tococo=False,
                   randp=None, is_train=False, one_level_framepath=False):
  assert idlst is not None
  assert framepath is not None
  assert annopath is not None
  assert len(targetid2class.keys()) == config.num_class

  # load the coco class name to classId so we could convert the label name
  #to label classId
  if tococo:
    add_coco(config, config.datajson)

  imgs = [os.path.splitext(os.path.basename(line.strip()))[0]
          for line in open(idlst, "r").readlines()]

  if randp is not None:
    imgs = random.sample(imgs, int(len(imgs)*randp))

  data = {"imgs":[], "gt":[]}
  if config.use_mixup and is_train:
    data["mixup_weights"] = []

  print("loading data..")
  if config.print_params:
    imgs = imgs[:100]

  # in diva dataset, some class may be ignored
  ignored_classes = {}

  targetClass2exist = {classname:0 for classname in targetClass2id}

  num_empty_actboxes = 0
  targetAct2exist = {classname:0 for classname in targetAct2id}
  ignored_act_classes = {}

  num_empty_single_actboxes = 0
  ignored_single_act_classes = {}
  targetAct2exist_single = {classname:0 for classname in targetSingleAct2id}
  act_single_fgratio = []

  if config.debug:
    imgs = imgs[:1000]

  if (config.train_skip > 1) and is_train:
    imgs.sort()
    ori_num = len(imgs)
    imgs = imgs[config.train_skip_offset::config.train_skip]
    print("skipping [%s::%s], got %s/%s" % (
        config.train_skip_offset, config.train_skip, len(imgs), ori_num))
  if (config.val_skip > 1) and not is_train:
    imgs.sort()
    ori_num = len(imgs)
    imgs = imgs[config.val_skip_offset::config.val_skip]
    print("skipping [%s::%s], got %s/%s" % (
        config.val_skip_offset, config.val_skip, len(imgs), ori_num))


  # get starts for each img, the label distribution
  # class -> [] num_box in each image
  label_dist = {classname:[] for classname in targetClass2id}
  label_dist_all = []

  for img in tqdm(imgs, ascii=True, smoothing=0.5):
    anno = os.path.join(annopath, "%s.npz"%img)
    videoname = img.strip().split("_F_")[0]
    if not os.path.exists(anno):
      continue
    if config.check_img_exist:
      if not os.path.exists(os.path.join(framepath, videoname, "%s.jpg"%img)):
        continue
    anno = dict(np.load(anno, allow_pickle=True)) # "boxes" -> [K,4]
    # boxes are x1,y1,x2,y2

    original_box_num = len(anno["boxes"])

    # feed act box as object boxes
    if config.act_as_obj:
      anno["labels"] = anno["actlabels"]
      anno["boxes"] = anno["actboxes"]

    # labels are one word, diva classname

    labels = []
    boxes = []
    no_so_box = True
    no_object = True
    for i, classname in enumerate(list(anno["labels"])):
      if classname in targetClass2id or (
          config.bupt_exp and classname in  bupt_act_mapping) or (
              config.meva_exp and classname in meva_act_mapping):

        if config.bupt_exp and classname in bupt_act_mapping:
          classname = bupt_act_mapping[classname]

        if config.meva_exp and classname in meva_act_mapping:
          classname = meva_act_mapping[classname]

        targetClass2exist[classname] = 1
        labels.append(targetClass2id[classname])
        boxes.append(anno["boxes"][i])
      else:
        ignored_classes[classname] = 1
      if classname in config.small_objects:
        no_so_box = False
      if config.skip_no_object is not None:
        if classname == config.skip_no_object:
          no_object = False

    if config.use_mixup and is_train:
      mixup_boxes = []
      mixup_labels = []
      for i, classname in enumerate(
          list(anno["mixup_labels"])[:config.max_mixup_per_frame]):
        if classname in targetClass2id:
          # not adding now, during run time will maybe add them
          #labels.append(targetClass2id[classname])
          #boxes.append(anno["mixup_boxes"][i])

          mixup_boxes.append(anno["mixup_boxes"][i])
          mixup_labels.append(targetClass2id[classname])
      anno["mixup_boxes"] = np.array(mixup_boxes, dtype="float32")
      anno["mixup_labels"] = mixup_labels

    anno["boxes"] = np.array(boxes, dtype="float32")
    anno["labels"] = labels

    #assert len(anno["boxes"]) > 0
    if len(anno["boxes"]) == 0:
      continue

    if config.skip_no_so_img and is_train:
      if no_so_box:
        continue
    if config.skip_no_object and is_train:
      if no_object:
        continue

    assert len(anno["labels"]) == len(anno["boxes"]), (
        anno["labels"], anno["boxes"])
    assert anno["boxes"].dtype == np.float32

    if config.oversample_so_img and is_train and not no_so_box:
      for i in range(config.oversample_x):
        data["imgs"].append(os.path.join(framepath, videoname, "%s.jpg"%img))
        data["gt"].append(anno)

    # statics
    if config.show_stat:
      for classname in label_dist:
        num_box_this_img = len(
            [l for l in labels if l == targetClass2id[classname]])
        label_dist[classname].append(num_box_this_img)
      label_dist_all.append(len(labels))

    if config.add_act:
      # for activity anno, we couldn"t remove any of the boxes
      assert len(anno["boxes"]) == original_box_num
      if config.act_v2:
        # make multi class labels
        # BG class is at index 0
        K = len(anno["boxes"])
        actSingleLabels = np.zeros((K, config.num_act_single_class),
                                   dtype="uint8")

        # use this to mark BG
        hasClass = np.zeros((K), dtype="bool")
        for i, classname in enumerate(list(anno["actSingleLabels"])):
          if classname in targetSingleAct2id:
            targetAct2exist_single[classname] = 1
            act_id = targetSingleAct2id[classname]
            box_id = anno["actSingleIdxs"][i]
            assert box_id >= 0 and box_id < K
            actSingleLabels[box_id, act_id] = 1
            hasClass[box_id] = True
          else:
            ignored_single_act_classes[classname] = 1

        # mark the BG for boxes that has not activity annotation
        actSingleLabels[np.logical_not(hasClass), 0] = 1
        anno["actSingleLabels_npy"] = actSingleLabels

        # compute the BG vs FG ratio for the activity boxes
        act_single_fgratio.append(sum(hasClass)/float(K))

        if sum(hasClass) == 0:
          num_empty_single_actboxes += 1
          continue

      else:
        act_labels = []
        act_good_ids = []
        for i, classname in enumerate(list(anno["actlabels"])):
          if classname in targetAct2id:
            targetAct2exist[classname] = 1
            act_labels.append(targetAct2id[classname])
            act_good_ids.append(i)
          else:
            ignored_act_classes[classname] = 1
        #print anno["actboxes"].shape
        if anno["actboxes"].shape[0] == 0:# ignore this image
          num_empty_actboxes += 1
          continue
        anno["actboxes"] = anno["actboxes"][act_good_ids]
        # it is a npy array of python list, so no :
        anno["actboxidxs"] = anno["actboxidxs"][act_good_ids]
        anno["actlabels"] = act_labels
        assert len(anno["actboxes"]) == len(anno["actlabels"])


    if config.use_mixup and is_train:
      # the training lst and annotation is framename_M_framename.npz files
      framename1, framename2 = img.strip().split("_M_")
      videoname1 = framename1.strip().split("_F_")[0]
      videoname2 = framename2.strip().split("_F_")[0]
      data["imgs"].append(
          (os.path.join(framepath, videoname1, "%s.jpg"%framename1),
           os.path.join(framepath, videoname2, "%s.jpg"%framename2)))
      data["gt"].append(anno)
      weight = np.random.beta(1.5, 1.5)
      if config.use_constant_mixup_weight:
        weight = config.mixup_constant_weight
      data["mixup_weights"].append(weight)
    else:
      if one_level_framepath:
        data["imgs"].append(os.path.join(framepath, "%s.jpg"%img))
      else:
        data["imgs"].append(os.path.join(framepath, videoname, "%s.jpg"%img))
      data["gt"].append(anno)

  print("loaded %s/%s data" % (len(data["imgs"]), len(imgs)))

  if config.show_stat:
    for classname in label_dist:
      d = label_dist[classname]
      ratios = [a/float(b) for a, b in zip(d, label_dist_all)]
      print("%s, [%s - %s], median %s per img, ratio:[%.3f - %.3f], "
            "median %.3f, no label %s/%s [%.3f]" % (
                classname, min(d), max(d), np.median(d), min(ratios),
                max(ratios),
                np.median(ratios), len([i for i in d if i == 0]), len(d),
                len([i for i in d if i == 0])/float(len(d))))
    print("each img has boxes: [%s - %s], median %s" % (
        min(label_dist_all), max(label_dist_all), np.median(label_dist_all)))


  if ignored_classes:
    print("ignored %s " % (ignored_classes.keys()))
  noDataClasses = [classname for classname in targetClass2exist
                   if targetClass2exist[classname] == 0]
  if noDataClasses:
    print("warning: class data not exists: %s, AR will be 1.0 for these" % (
        noDataClasses))
  if config.add_act:
    if config.act_v2:
      print(" each frame positive act box percentage min %.4f, max %.4f, "
            "mean %.4f" % (
                min(act_single_fgratio), max(act_single_fgratio),
                np.mean(act_single_fgratio)))
      if ignored_single_act_classes:
        print("ignored activity %s" % (ignored_single_act_classes.keys()))
      print("%s/%s has no single activity boxes" % (
          num_empty_single_actboxes, len(data["imgs"])))
      noDataClasses = [classname for classname in targetAct2exist_single
                       if targetAct2exist_single[classname] == 0]
      if noDataClasses:
        print("warning: single activity class data not exists: %s, " % (
            noDataClasses))
    else:
      if ignored_act_classes:
        print("ignored activity %s" % (ignored_act_classes.keys()))
      print("%s/%s has no activity boxes" % (
          num_empty_actboxes, len(data["imgs"])))
      noDataClasses = [classname for classname in targetAct2exist
                       if targetAct2exist[classname] == 0]
      if noDataClasses:
        print("warning: activity class data not exists: %s, " % (noDataClasses))


  return Dataset(data, add_gt=True)


# given the gen_gt_diva
# train on diva dataset
def train_diva(config):
  global eval_target, targetid2class, targetClass2id
  eval_target_weight = None
  if config.diva_class:
    # only care certain classes
    eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object", "Bike"]
    eval_target = {one:1 for one in eval_target}
    eval_target_weight = {
        "Person":0.15,
        "Vehicle":0.15,
        "Prop":0.15,
        "Push_Pulled_Object":0.15,
        "Bike":0.15,
    }

    if config.merge_prop:
      eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object",
                     "Bike", "Prop_plus_Push_Pulled_Object"]
      eval_target = {one:1 for one in eval_target}
      eval_target_weight = {
          "Person":0.15,
          "Vehicle":0.15,
          "Prop_plus_Push_Pulled_Object":0.2,
          "Bike":0.2,
          "Prop":0.15,
          "Push_Pulled_Object":0.15,
      }
  if config.diva_class2:
    # only care certain classes
    eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object", "Bike",
                   "Construction_Vehicle", "Bike_Person", "Prop_Person",
                   "Skateboard_Person"]
    eval_target = {one:1 for one in eval_target}
    eval_target_weight = {one:1.0/len(eval_target) for one in eval_target}

  if config.diva_class3:
    # only care certain classes
    # removed construction vehicle 03/2019
    eval_target = ["Vehicle", "Person", "Prop", "Push_Pulled_Object", "Bike"]
    eval_target = {one:1 for one in eval_target}
    eval_target_weight = {one:1.0/len(eval_target) for one in eval_target}

  if config.add_act:
    # same for single box act
    # "vehicle_u_turn" is not used since not exists in val set
    act_eval_target = ["vehicle_turning_right", "vehicle_turning_left",
                       "Unloading", "Transport_HeavyCarry", "Opening",
                       "Open_Trunk", "Loading", "Exiting", "Entering",
                       "Closing_Trunk", "Closing", "Interacts", "Pull",
                       "Riding", "Talking", "activity_carrying",
                       "specialized_talking_phone", "specialized_texting_phone"]
    act_eval_target = {one:1 for one in act_eval_target}
    act_eval_target_weight = {one:1.0/len(act_eval_target)
                              for one in act_eval_target}


  if config.act_as_obj:
    # "vehicle_u_turn" is not used since not exists in val set
    eval_target = ["vehicle_turning_right", "vehicle_turning_left", "Unloading",
                   "Transport_HeavyCarry", "Opening", "Open_Trunk", "Loading",
                   "Exiting", "Entering", "Closing_Trunk", "Closing",
                   "Interacts", "Pull", "Riding", "Talking",
                   "activity_carrying", "specialized_talking_phone",
                   "specialized_texting_phone"]
    if config.bupt_exp:
      eval_target = ["Person-Vehicle", "Vehicle-Turning",
                     "Transport_HeavyCarry", "Pull", "Riding", "Talking",
                     "activity_carrying", "specialized_talking_phone",
                     "specialized_texting_phone"]
    if config.meva_exp:
      eval_target = ["Person-Vehicle", "Vehicle-Turning", "Person-Structure",
                     "Person_Heavy_Carry", "People_Talking", "Riding",
                     "Person_Sitting_Down", "Person_Sets_Down_Object"]

    eval_target = {one:1 for one in eval_target}
    eval_target_weight = {one:1.0/len(eval_target) for one in eval_target}

  if config.is_coco_model:
    # finetuning person boxes for AVA
    eval_target = ["person"]
    eval_target = {one:1 for one in eval_target}
    eval_target_weight = {one:1.0/len(eval_target) for one in eval_target}

  self_summary_strs = Summary()
  stats = [] # tuples with {"metrics":,"step":,}
  # load the frame count data first

  train_data = read_data_diva(config, config.trainlst, config.imgpath,
                              config.annopath, tococo=False, is_train=True,
                              one_level_framepath=config.one_level_framepath)
  val_data = read_data_diva(config, config.vallst, config.valframepath,
                            config.valannopath, tococo=False,
                            one_level_framepath=config.one_level_framepath)
  config.train_num_examples = train_data.num_examples

  if config.show_stat:
    sys.exit()

  # the total step (iteration) the model will run
  num_steps = int(math.ceil(
      train_data.num_examples/float(config.im_batch_size))) * config.num_epochs
  num_val_steps = int(math.ceil(
      val_data.num_examples/float(config.im_batch_size))) * 1

  #config_vars = vars(config)

  # model_per_gpu > 1 not work yet, need to set distributed computing
  #model = get_model(config) # input is image paths
  models = []
  gpuids = list(range(config.gpuid_start, config.gpuid_start+config.gpu))
  gpuids = gpuids * config.model_per_gpu
  # example, model_per_gpu=2, gpu=2, gpuid_start=0
  gpuids.sort()# [0,0,1,1]
  taskids = list(range(config.model_per_gpu)) * config.gpu # [0,1,0,1]

  for i, j in zip(gpuids, taskids):
    models.append(get_model(config, gpuid=i, task=j,
                            controller=config.controller))

  config.is_train = False
  models_eval = []
  for i, j in zip(gpuids, taskids):
    models_eval.append(get_model(config, gpuid=i, task=j,
                                 controller=config.controller))
  config.is_train = True

  trainer = Trainer(models, config)
  tester = Tester(models_eval, config, add_mask=config.add_mask)

  saver = tf.train.Saver(max_to_keep=5) # how many model to keep
  bestsaver = tf.train.Saver(max_to_keep=5) # just for saving the best model

  # start training!

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  if not config.use_all_mem:
    tfconfig.gpu_options.allow_growth = True
  # so only this gpu will be used
  tfconfig.gpu_options.visible_device_list = "%s" % (",".join(
      ["%s" % i
       for i in range(config.gpuid_start, config.gpuid_start+config.gpu)]))
  with tf.Session(config=tfconfig) as sess:
    self_summary_strs.add("total parameters: %s" % (cal_total_param()))

    initialize(load=config.load, load_best=config.load_best, config=config,
               sess=sess)

    if config.print_params:
      for var in tf.global_variables():
        not_show = False
        for c in ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
                  "Adadelta", "Momentum"]:
          if c in var.name:
            not_show = True
        if not_show:
          continue
        shape = var.get_shape()
        print("%s %s\n" % (var.name, shape))
      sys.exit()

    isStart = True

    best = (-1.0, 1)
    loss_me, wd_me, rpn_label_loss_me, rpn_box_loss_me, \
        fastrcnn_label_loss_me, fastrcnn_box_loss_me, so_label_loss_me, \
            act_loss_me, lr_me = \
                [FIFO_ME(config.loss_me_step) for i in range(9)]
    for batch in tqdm(train_data.get_batches(
        config.im_batch_size, num_batches=num_steps),
                      total=num_steps, ascii=True, smoothing=1):
      # start from 0 or the previous step
      global_step = sess.run(models[0].global_step) + 1

      validation_performance = None
      if (global_step % config.save_period == 0) or \
         (config.load and isStart and ((config.ignore_vars is None) or \
          config.force_first_eval)): # time to save model
        tqdm.write("step:%s/%s (epoch:%.3f)" % (
            global_step, num_steps,
            (config.num_epochs*global_step/float(num_steps))))
        tqdm.write("\tsaving model %s..." % global_step)
        saver.save(sess, os.path.join(config.save_dir, "model"),
                   global_step=global_step)
        tqdm.write("\tdone")
        if config.skip_first_eval and isStart:
          tqdm.write("skipped first eval...")
          validation_performance = config.best_first

        else:
          # cat_id -> imgid -> {"dm","dscores"}
          e = {one:{} for one in eval_target.keys()}
          if config.add_act:
            e_act = {one:{} for one in act_eval_target.keys()}
          if config.use_small_object_head:
            e_so = {one:{} for one in config.so_eval_target.keys()}

          for val_batch_ in tqdm(val_data.get_batches(
              config.im_batch_size, num_batches=num_val_steps, shuffle=False),
                                 total=num_val_steps, ascii=True, smoothing=1):
            batch_idx, val_batches = val_batch_
            this_batch_num = len(val_batches)
            # multiple image at a time for parallel inferencing with
            # multiple gpu
            scales = []
            imgids = []
            for val_batch in val_batches:
              # load the image here and resize
              image = cv2.imread(val_batch.data["imgs"][0], cv2.IMREAD_COLOR)
              imgid = os.path.splitext(
                  os.path.basename(val_batch.data["imgs"][0]))[0]
              imgids.append(imgid)
              assert image is not None, image
              image = image.astype("float32")
              val_batch.data["imgdata"] = [image]

              resized_image = resizeImage(image, config.short_edge_size,
                                          config.max_size)

              # rememember the scale and original image
              ori_shape = image.shape[:2]
              #print(image.shape, resized_image.shape
              # average H/h and W/w ?
              scale = (resized_image.shape[0]*1.0/image.shape[0] + \
                  resized_image.shape[1]*1.0/image.shape[1])/2.0

              val_batch.data["resized_image"] = [resized_image]
              scales.append(scale)

            outputs = tester.step(sess, val_batch_)

            # post process this batch, also remember the ground truth
            for i in range(this_batch_num): # num gpu
              imgid = imgids[i]
              scale = scales[i]
              if config.add_act:
                if config.act_v2:
                  boxes, labels, probs, actsingleboxes, actsinglelabels = \
                      outputs[i]
                  actsingleboxes = actsingleboxes / scale
                else:
                  boxes, labels, probs, actboxes, actlabels, actprobs = \
                      outputs[i]
                  actboxes = actboxes / scale
              else:
                if config.add_mask:
                  boxes, labels, probs, masks = outputs[i]
                else:
                  if config.use_small_object_head:
                    boxes, labels, probs, so_boxes, so_labels, so_probs = \
                        outputs[i]
                    so_boxes = so_boxes / scale
                  else:
                    boxes, labels, probs = outputs[i]

                if config.use_cpu_nms:
                  boxes, labels, probs = nms_wrapper(boxes, probs, config)


              val_batch = val_batches[i]

              boxes = boxes / scale

              # each class"s detection box and prob
              target_dt_boxes = gather_dt(boxes, probs, labels, eval_target,
                                          targetid2class,
                                          tococo=config.tococo,
                                          coco_class_names=config.class_names)

              # gt
              anno = val_batch.data["gt"][0] # one val_batch is single image
              gt_boxes = gather_gt(anno["boxes"], anno["labels"], eval_target,
                                   targetid2class)


              # gt_boxes and target_dt_boxes for this image

              # eval on one single image
              match_dt_gt(e, imgid, target_dt_boxes, gt_boxes, eval_target)

              if config.use_small_object_head:
                target_so_dt_boxes = gather_dt(
                    so_boxes, so_probs, so_labels, config.so_eval_target,
                    config.small_objects_targetid2class)

                anno = val_batch.data["gt"][0] # one val_batch is single image
                small_object_classids = [targetClass2id[one]
                                         for one in config.small_objects]
                idxs = [i for i in range(len(anno["labels"]))
                        if anno["labels"][i] in small_object_classids]
                gt_so_boxes = [anno["boxes"][i] for i in idxs]
                # convert the original classid to the small object class id
                gt_so_labels = [
                    small_object_classids.index(anno["labels"][i])+1
                    for i in idxs]
                gt_so_boxes = gather_gt(gt_so_boxes, gt_so_labels,
                                        config.so_eval_target,
                                        config.small_objects_targetid2class)

                match_dt_gt(e_so, imgid, target_so_dt_boxes, gt_so_boxes,
                            config.so_eval_target)


              # eval the act box as well, put stuff in e_act
              if config.add_act and config.act_v2:
                # for v2, we have the single and pair boxes
                # actsingleboxes [K,4]
                # actsinglelabels [K,num_act_class]
                # first we filter the BG boxes
                # we select topk act class for each box
                topk = config.act_single_topk

                single_act_boxes, single_act_labels, single_act_probs = \
                    gather_act_singles(actsingleboxes, actsinglelabels, topk)

                target_act_dt_boxes = gather_dt(
                    single_act_boxes, single_act_probs, single_act_labels,
                    act_eval_target, targetsingleactid2class)

                # to collect the ground truth, each label will be a stand
                # alone boxes
                anno = val_batch.data["gt"][0] # one val_batch is single image
                gt_single_act_boxes = []
                gt_single_act_labels = []
                gt_obj_boxes = anno["boxes"]
                for bid, label in zip(
                    anno["actSingleIdxs"], anno["actSingleLabels"]):
                  if label in act_eval_target:
                    gt_single_act_boxes.append(gt_obj_boxes[bid])
                    gt_single_act_labels.append(targetSingleAct2id[label])

                gt_act_boxes = gather_gt(
                    gt_single_act_boxes, gt_single_act_labels,
                    act_eval_target, targetsingleactid2class)

                match_dt_gt(e_act, imgid, target_act_dt_boxes,
                            gt_act_boxes, act_eval_target)

              if config.add_act and not config.act_v2:
                target_act_dt_boxes = gather_dt(actboxes, actprobs, actlabels,
                                                act_eval_target,
                                                targetactid2class)

                #gt

                anno = val_batch.data["gt"][0] # one val_batch is single image
                gt_act_boxes = gather_gt(
                    anno["actboxes"], anno["actlabels"],
                    act_eval_target, targetactid2class)

                # gt_boxes and target_dt_boxes for this image
                match_dt_gt(e_act, imgid, target_act_dt_boxes,
                            gt_act_boxes, act_eval_target)



          # we have the dm and g matching for each image in e & e_act
          # max detection per image per category
          aps, ars = aggregate_eval(e, maxDet=100)

          aps_str = "|".join(["%s:%.5f" % (class_, aps[class_])
                              for class_ in aps])
          ars_str = "|".join(["%s:%.5f" % (class_, ars[class_])
                              for class_ in ars])
          #validation_performance = ars[eval_best]
          # now we use average AR and average AP or weighted
          average_ap, average_ar = weighted_average(
              aps, ars, eval_target_weight)


          ap_weight = 1.0
          ar_weight = 0.0
          validation_performance = average_ap*ap_weight + average_ar*ar_weight

          if config.add_act:
            obj_validation_performance = validation_performance
            aps, ars = aggregate_eval(e_act, maxDet=100)

            act_aps_str = "|".join(["%s:%.5f"%(class_, aps[class_])
                                    for class_ in aps])
            act_ars_str = "|".join(["%s:%.5f"%(class_, ars[class_])
                                    for class_ in ars])

            average_ap, average_ar = weighted_average(
                aps, ars, act_eval_target_weight)


            ap_weight = 0.9
            ar_weight = 0.1
            act_validation_performance = average_ap*ap_weight + \
                average_ar*ar_weight

            act_perf_weight = 0.5
            obj_perf_weight = 0.5
            validation_performance = obj_perf_weight \
                * obj_validation_performance + \
                    act_perf_weight*act_validation_performance

            tqdm.write("\tval in %s at step %s, Obj AP:%s, AR:%s, obj "
                       "performance %s" % (
                num_val_steps, global_step, aps_str, ars_str,
                obj_validation_performance))
            tqdm.write("\tAct AP:%s, AR:%s, this step val:%.5f, previous"
                       " best val at %s is %.5f" % (
                act_aps_str, act_ars_str, validation_performance,
                best[1], best[0]))
          else:
            if config.use_small_object_head:
              so_aps, so_ars = aggregate_eval(e_so, maxDet=100)
              so_average_ap, so_average_ar = weighted_average(so_aps, so_ars)
              so_val = so_average_ap*0.5 + so_average_ar*0.5

              so_weight = 0.5
              validation_performance = (1 - so_weight)*validation_performance \
                  + so_weight*so_val

              so_aps_str = "|".join(["%s:%.5f"%(class_, so_aps[class_])
                                     for class_ in so_aps])
              so_ars_str = "|".join(["%s:%.5f"%(class_, so_ars[class_])
                                     for class_ in so_ars])

              tqdm.write("\tval in %s at step %s, AP:%s, AR:%s, so_AP:%s, "
                         "so_AR:%s, this step val:%.5f, previous best val "
                         "at %s is %.5f" % (
                  num_val_steps, global_step, aps_str, ars_str, so_aps_str,
                  so_ars_str, validation_performance, best[1], best[0]))

            else:
              tqdm.write("\tval in %s at step %s, AP:%s, AR:%s, this step "
                         "val:%.5f, previous best val at %s is %.5f" % (
                  num_val_steps, global_step, aps_str, ars_str,
                  validation_performance, best[1], best[0]))



        if validation_performance > best[0]:
          tqdm.write("\tsaving best model %s..." % global_step)
          bestsaver.save(sess, os.path.join(config.save_dir_best, "model"),
                         global_step=global_step)
          tqdm.write("\tdone")
          best = (validation_performance, global_step)

        isStart = False
        if config.exit_after_val:
          print("exit after eval.")
          break

      # skip if the batch is not complete, usually the last few ones
      if len(batch[1]) != config.gpu:
        continue

      try:
        #loss, rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op,act_losses = trainer.step(sess,batch)
        loss, wds, rpn_label_losses, rpn_box_losses, fastrcnn_label_losses, \
            fastrcnn_box_losses, so_label_losses, act_losses, lr = \
                trainer.step(sess, batch)
      except Exception as e:
        print(e)
        bs = batch[1]
        print("trainer error, batch files:%s"%([b.data["imgs"] for b in bs]))
        sys.exit()

      if math.isnan(loss):
        tqdm.write("warning, nan loss: loss:%s,rpn_label_loss:%s, "
                   "rpn_box_loss:%s, fastrcnn_label_loss:%s, "
                   "fastrcnn_box_loss:%s" % (
            loss, rpn_label_losses, rpn_box_losses, fastrcnn_label_losses,
            fastrcnn_box_losses))
        if config.add_act:
          tqdm.write("\tact_losses:%s" % (act_losses))
        print("batch:%s" % (batch[1][0].data["imgs"]))
        sys.exit()

      # use moving average to compute loss

      loss_me.put(loss)
      lr_me.put(lr)
      for wd, rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, \
          fastrcnn_box_loss, so_label_loss, act_loss in zip(
              wds, rpn_label_losses, rpn_box_losses, fastrcnn_label_losses,
              fastrcnn_box_losses, so_label_losses, act_losses):
        wd_me.put(wd)
        rpn_label_loss_me.put(rpn_label_loss)
        rpn_box_loss_me.put(rpn_box_loss)
        fastrcnn_label_loss_me.put(fastrcnn_label_loss)
        fastrcnn_box_loss_me.put(fastrcnn_box_loss)
        so_label_loss_me.put(so_label_loss)
        act_loss_me.put(act_loss)

      if global_step % config.show_loss_period == 0:
        tqdm.write("step %s, moving average: learning_rate %.6f, loss %.6f,"
                   " weight decay loss %.6f, rpn_label_loss %.6f, rpn_box_loss"
                   " %.6f, fastrcnn_label_loss %.6f, fastrcnn_box_loss %.6f, "
                   "so_label_loss %.6f, act_loss %.6f" % (
            global_step, lr_me.me(), loss_me.me(), wd_me.me(),
            rpn_label_loss_me.me(), rpn_box_loss_me.me(),
            fastrcnn_label_loss_me.me(), fastrcnn_box_loss_me.me(),
            so_label_loss_me.me(), act_loss_me.me()))

      # save these for ploting later
      stats.append({
          "s":float(global_step),
          "l":float(loss),
          "val":validation_performance
      })
      isStart = False

    # save the last model
    if global_step % config.save_period != 0: # time to save model
      print("saved last model without evaluation.")
      saver.save(sess, os.path.join(config.save_dir, "model"),
                 global_step=global_step)

    if config.write_self_sum:
      self_summary_strs.writeTo(config.self_summary_path)

      with open(config.stats_path, "w") as f:
        json.dump(stats, f)


# given a list of images, do the forward, save each image result separately
def forward(config):
  imagelist = config.imgpath

  if config.extract_feat:
    assert config.feat_path is not None
    assert config.is_fpn
    if not os.path.exists(config.feat_path):
      os.makedirs(config.feat_path)
    print("also extracting fpn features")

  all_images = [line.strip() for line in open(config.imgpath, "r").readlines()]

  if config.forward_skip > 1:
    all_images.sort()
    ori_num = len(all_images)
    all_images = all_images[::config.forward_skip]
    print("skiiping %s, got %s/%s" % (
        config.forward_skip, len(all_images), ori_num))

  if config.check_img_exist:
    exist_imgs = []
    for image in all_images:
      if os.path.exists(image):
        exist_imgs.append(image)
    print("%s/%s image exists" % (len(exist_imgs), len(all_images)))
    all_images = exist_imgs

  print("total images to test:%s"%len(all_images))

  if config.use_small_object_head:
    if not os.path.exists(config.so_outpath):
      os.makedirs(config.so_outpath)

  models = []
  for i in range(config.gpuid_start, config.gpuid_start+config.gpu):
    models.append(get_model(config, i, controller=config.controller))

  model_final_boxes = [model.final_boxes for model in models]
  # [R]
  model_final_labels = [model.final_labels for model in models]
  model_final_probs = [model.final_probs for model in models]

  if config.extract_feat:
    model_feats = [model.fpn_feature for model in models]

  if config.add_mask:
    # [R,14,14]
    model_final_masks = [model.final_masks for model in models]

  if config.add_act:
    if config.act_v2:
      model_act_single_boxes = [model.act_single_boxes for model in models]
      model_act_single_label_logits = [model.act_single_label_logits
                                       for model in models]
    else:
      model_act_final_boxes = [model.act_final_boxes for model in models]
      # [R]
      model_act_final_labels = [model.act_final_labels for model in models]
      model_act_final_probs = [model.act_final_probs for model in models]

  tfconfig = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
  if not config.use_all_mem:
    tfconfig.gpu_options.allow_growth = True

  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s"%i
                for i in range(
                    config.gpuid_start, config.gpuid_start+config.gpu)]))

  with tf.Session(config=tfconfig) as sess:

    # for packing model, the weights are already loaded
    if not config.is_load_from_pb:
      initialize(load=True, load_best=config.load_best, config=config,
                 sess=sess)

    # num_epoch should be 1
    assert config.num_epochs == 1

    count = 0
    for images in tqdm(grouper(all_images, config.im_batch_size), ascii=True):
      count += 1
      if config.start_from > 0:
        if count <= config.start_from:
          continue
      images = [im for im in images if im is not None]
      # multigpu will need full image inpu
      this_batch_len = len(images)
      if this_batch_len != config.im_batch_size:
        need = config.im_batch_size - this_batch_len
        images.extend(all_images[:need]) # redo some images
      scales = []
      resized_images = []
      ori_shapes = []
      imagenames = []
      # the folder the image is in, for when we want a two-level output
      pathnames = []
      feed_dict = {}
      for i, image in enumerate(images):
        im = cv2.imread(image, cv2.IMREAD_COLOR)
        imagename = os.path.splitext(os.path.basename(image))[0]
        pathnames.append(image.split("/")[-2])
        imagenames.append(imagename)

        ori_shape = im.shape[:2]

        # need to resize here, otherwise
        # InvalidArgumentError (see above for traceback):
        #Expected size[1] in [0, 83], but got 120
        #[[Node: anchors/fm_anchors = Slice[Index=DT_INT32, T=DT_FLOAT,
        #_device="/job:localhost/replica:0/task:0/device:GPU:0"]
        #(anchors/all_anchors, anchors/fm_anchors/begin, anchors/stack)]]

        resized_image = resizeImage(im, config.short_edge_size, config.max_size)

        scale = (resized_image.shape[0]*1.0/im.shape[0] + \
            resized_image.shape[1]*1.0/im.shape[1])/2.0

        resized_images.append(resized_image)
        scales.append(scale)
        ori_shapes.append(ori_shape)

        feed_dict.update(models[i].get_feed_dict_forward(resized_image))

      sess_input = []

      if config.just_feat:
        outputs = sess.run(model_feats, feed_dict=feed_dict)
        for i, feat in enumerate(outputs):
          imagename = imagenames[i]

          featfile = os.path.join(config.feat_path, "%s.npy"%imagename)
          np.save(featfile, feat)

        continue # no bounding boxes

      if config.add_mask:
        for _, boxes, labels, probs, masks in zip(
            range(len(images)), model_final_boxes, model_final_labels,
            model_final_probs, model_final_masks):
          sess_input += [boxes, labels, probs, masks]
      else:
        if config.add_act:
          if config.act_v2:
            for _, boxes, labels, probs, actboxes, actlabels in zip(
                range(len(images)), model_final_boxes, model_final_labels,
                model_final_probs, model_act_single_boxes,
                model_act_single_label_logits):
              sess_input += [boxes, labels, probs, actboxes, actlabels]
          else:
            for _, boxes, labels, probs, actboxes, actlabels, actprobs in zip(
                range(len(images)), model_final_boxes, model_final_labels,
                model_final_probs, model_act_final_boxes,
                model_act_final_labels, model_act_final_probs):
              sess_input += [boxes, labels, probs, actboxes, actlabels,
                             actprobs]
        else:
          if config.extract_feat:
            for _, boxes, labels, probs, feats in zip(
                range(len(images)), model_final_boxes, model_final_labels,
                model_final_probs, model_feats):
              sess_input += [boxes, labels, probs, feats]
          else:
            if config.get_rpn_out:
              model_proposal_boxes = [model.proposal_boxes for model in models]
              model_proposal_scores = [model.proposal_scores
                                       for model in models]
              for _, boxes, labels, probs, prop_boxes, prop_scores in zip(
                  range(len(images)), model_final_boxes, model_final_labels,
                  model_final_probs, model_proposal_boxes,
                  model_proposal_scores):
                sess_input += [boxes, labels, probs, prop_boxes, prop_scores]
            else:
              if config.use_small_object_head:
                model_so_boxes = [model.so_final_boxes for model in models]
                model_so_probs = [model.so_final_probs for model in models]
                model_so_labels = [model.so_final_labels for model in models]
                for _, boxes, labels, probs, so_boxes, so_labels, \
                    so_probs in zip(
                        range(len(images)), model_final_boxes,
                        model_final_labels, model_final_probs, model_so_boxes,
                        model_so_labels, model_so_probs):
                  sess_input += [boxes, labels, probs, so_boxes, so_labels,
                                 so_probs]
              else:
                for _, boxes, labels, probs in zip(
                    range(len(images)), model_final_boxes, model_final_labels,
                    model_final_probs):
                  sess_input += [boxes, labels, probs]
      outputs = sess.run(sess_input, feed_dict=feed_dict)
      if config.add_mask:
        pn = 4
      else:
        pn = 3
        if config.add_act:
          pn = 6
          if config.act_v2:
            pn = 5
        else:
          if config.extract_feat:
            pn = 4
          elif config.get_rpn_out:
            pn = 5
          elif config.use_small_object_head:
            pn = 6
      outputs = [outputs[i*pn:(i*pn+pn)] for i in range(len(images))]

      for i, output in enumerate(outputs):
        scale = scales[i]
        ori_shape = ori_shapes[i]
        imagename = imagenames[i]
        if config.add_mask:
          final_boxes, final_labels, final_probs, final_masks = output
          final_boxes = final_boxes / scale
          final_masks = [fill_full_mask(box, mask, ori_shape)
                         for box, mask in zip(final_boxes, final_masks)]
        else:
          if config.add_act:
            if config.act_v2:
              final_boxes, final_labels, final_probs, actsingleboxes, \
                  actsinglelabels = output
              actsingleboxes = actsingleboxes / scale
            else:
              final_boxes, final_labels, final_probs, actboxes,\
                  actlabels, actprobs = output
              actboxes = actboxes / scale
          else:
            if config.extract_feat:
              final_boxes, final_labels, final_probs, final_feat = output
              #print(final_feats.shape# [1,7,7,256]
              # save the features

              featfile = os.path.join(config.feat_path, "%s.npy"%imagename)
              np.save(featfile, final_feat)
            else:
              if config.get_rpn_out:
                final_boxes, final_labels, final_probs, prop_boxes, \
                    prop_scores = output
                prop_boxes = prop_boxes / scale
                props = np.concatenate(
                    [prop_boxes, np.expand_dims(prop_scores, axis=-1)],
                    axis=-1) # [K, 5]
                # save the proposal boxes,
                prop_file = os.path.join(config.rpn_out_path,
                                         "%s.npy" % imagename)
                np.save(prop_file, props)
              else:
                if config.use_small_object_head:
                  final_boxes, final_labels, final_probs, final_so_boxes, \
                      final_so_labels, final_so_probs = output
                else:
                  final_boxes, final_labels, final_probs = output

            if config.use_cpu_nms:
              if not config.no_nms:
                final_boxes, final_labels, final_probs = nms_wrapper(
                    final_boxes, final_probs, config)

          final_boxes = final_boxes / scale
          final_masks = [None for one in final_boxes]

          if config.no_nms:
            # will leave all K boxes, each box class is the max prob class
            # final_boxes would be [num_class-1, K, 4]
            # final_probs would be [num_class-1, K]
            # final_labels is actually rcnn_boxes, [K, 4]
            if config.save_all_box: # save all output as npz file instead
              rcnn_boxes = final_labels
              rcnn_boxes = rcnn_boxes / scale
              # boxes are [x1, y1, x2, y2]
              if config.use_frcnn_class_agnostic:
                if final_boxes:
                  assert final_boxes[0, 1, 2] == final_boxes[1, 1, 2]

                  final_boxes = final_boxes[0, :, :] # [K, 4]
              data = {
                  "rcnn_boxes": rcnn_boxes, # [K, 4]
                  "frcnn_boxes": final_boxes, # [C, K, 4] / [K, 4]
                  "frcnn_probs": final_probs, # [C, K] # C is num_class -1
              }
              target_file = os.path.join(config.outbasepath, "%s.npz"%imagename)
              np.savez(target_file, **data)
              continue # next image
            else:
              num_cat, num_box = final_boxes.shape[:2]
              # [K]
              best_cat = np.argmax(final_probs, axis=0)
              # get the final labels first
              final_labels = best_cat + 1

              # use the final boxes, select the best cat for each box
              final_boxes2 = np.zeros([num_box, 4], dtype="float")
              for i in range(num_box):
                final_boxes2[i, :] = final_boxes[best_cat[i], i, :]
              final_boxes = final_boxes2
              final_probs = np.amax(final_probs, axis=0) # [K]
              final_masks = [None for one in final_boxes]

        pred = []

        for j, (box, prob, label, mask) in enumerate(
            zip(final_boxes, final_probs, final_labels, final_masks)):
          box[2] -= box[0]
          box[3] -= box[1] # produce x,y,w,h output

          cat_id = int(label)
          cat_name = targetid2class[cat_id]

          # encode mask
          rle = None
          if config.add_mask:
            rle = cocomask.encode(np.array(mask[:, :, None], order="F"))[0]
            rle["counts"] = rle["counts"].decode("ascii")

          res = {
              "category_id":cat_id,
              "cat_name":cat_name, # [0-80]
              "score":float(round(prob, 4)),
              #"bbox": list(map(lambda x:float(round(x,1)),box)),
              "bbox": [float(round(x, 1)) for x in box],
              "segmentation":rle,
          }


          pred.append(res)

        # save the data
        outbasepath = config.outbasepath
        if config.use_two_level_outpath:
          pathname = pathnames[i]
          outbasepath = os.path.join(config.outbasepath, pathname)
          if not os.path.exists(outbasepath):
            os.makedirs(outbasepath)
        resultfile = os.path.join(outbasepath, "%s.json"%imagename)
        with open(resultfile, "w") as f:
          json.dump(pred, f)

        if config.use_small_object_head:
          so_pred = []

          for j, (so_box, so_prob, so_label) in enumerate(
              zip(final_so_boxes, final_so_probs, final_so_labels)):
            so_box[2] -= so_box[0]
            so_box[3] -= so_box[1] # produce x,y,w,h output

            # so_label is the class id in the small objects,
            # here the cat_id should follow the original class
            cat_name = config.small_objects_targetid2class[so_label]
            cat_id = targetClass2id[cat_name]

            res = {
                "category_id": cat_id,
                "cat_name": cat_name,
                "score": float(round(so_prob, 4)),
                #"bbox": list(map(lambda x:float(round(x,1)), so_box)),
                "bbox": [float(round(x, 1)) for x in so_box],
                "segmentation": None,
            }

            so_pred.append(res)

          resultfile = os.path.join(config.so_outpath, "%s.json" % imagename)
          with open(resultfile, "w") as f:
            json.dump(so_pred, f)


        if config.add_act:
          act_pred = []

          if config.act_v2:
            # assemble the single boxes and pair boxes?
            topk = config.act_single_topk
            single_act_boxes, single_act_labels, single_act_probs = \
                gather_act_singles(actsingleboxes, actsinglelabels, topk)

            for j, (act_box, act_prob, act_label) in enumerate(
                zip(single_act_boxes, single_act_probs, single_act_labels)):
              act_box[2] -= act_box[0]
              act_box[3] -= act_box[1]
              act_name = targetsingleactid2class[act_label]
              res = {
                  "category_id":act_label,
                  "cat_name":act_name,
                  "score":float(round(act_prob, 4)),
                  #"bbox": list(map(lambda x:float(round(x,1)),act_box)),
                  "bbox": [float(round(x, 1)) for x in act_box],
                  "segmentation":None,
                  "v2":1,
                  "single":1,
              }
              act_pred.append(res)

          else:
            for j, (act_box, act_prob, act_label) in enumerate(
                zip(actboxes, actprobs, actlabels)):
              act_box[2] -= act_box[0]
              act_box[3] -= act_box[1]
              act_name = targetactid2class[act_label]
              res = {
                  "category_id":act_label,
                  "cat_name":act_name,
                  "score":float(round(act_prob, 4)),
                  #"bbox": list(map(lambda x:float(round(x,1)),act_box)),
                  "bbox": [float(round(x, 1)) for x in act_box],
                  "segmentation":None,
                  "v2":0,
              }
              act_pred.append(res)


          # save the act data
          resultfile = os.path.join(config.actoutbasepath, "%s.json"%imagename)
          with open(resultfile, "w") as f:
            json.dump(act_pred, f)


def read_data_coco(datajson, config, add_gt=False, load_coco_class=False):

  with open(datajson, "r") as f:
    dj = json.load(f)

  if load_coco_class:
    add_coco(config, datajson)


  data = {"imgs":[], "ids":[]}
  if add_gt:
    data = {"imgs":[], "ids":[], "gt":[]}

  # read coco annotation file
  for one in dj["images"]:
    imgid = int(one["id"])
    imgfile = os.path.join(config.imgpath, one["file_name"])
    if config.coco2014_to_2017:
      imgfile = os.path.join(config.imgpath, one["file_name"].split("_")[-1])
    data["imgs"].append(imgfile)
    data["ids"].append(imgid)
    if add_gt:
      # load the bounding box and so on
      pass


  return Dataset(data, add_gt=add_gt)


# for testing, dataset -> {"imgs":[],"ids":[]}, imgs is the image file path,
def forward_coco(dataset, num_batches, config, sess, tester, resize=True):
  assert not config.diva_class # not working for this yet
  # "id" -> (boxes, probs, labels, masks)
  #pred = {}
  # each is (image_id,cat_id,bbox,score,segmentation)
  pred = []
  for evalbatch in tqdm(
      dataset.get_batches(config.im_batch_size, num_batches=num_batches,
                          shuffle=False,cap=True), total=num_batches):

    _, batches = evalbatch

    scales = []
    ori_shapes = []
    image_ids = []
    for batch in batches:
      # load the image here and resize
      image = cv2.imread(batch.data["imgs"][0], cv2.IMREAD_COLOR)
      assert image is not None, batch.data["imgs"][0]
      image = image.astype("float32")
      imageId = batch.data["ids"][0]
      image_ids.append(imageId)
      batch.data["imgdata"] = [image]
      #if imageId != 139:
      #  continue

      # resize image
      # ppwwyyxx"s code do resizing in eval
      if resize:
        resized_image = resizeImage(image, config.short_edge_size,
                                    config.max_size)
      else:
        resized_image = image

      # rememember the scale and original image
      ori_shape = image.shape[:2]
      #print(image.shape, resized_image.shape
      # average H/h and W/w ?
      scale = (resized_image.shape[0]*1.0/image.shape[0] + \
          resized_image.shape[1]*1.0/image.shape[1])/2.0

      batch.data["resized_image"] = [resized_image]
      scales.append(scale)
      ori_shapes.append(ori_shape)

    outputs = tester.step(sess, evalbatch)

    for i, output in enumerate(outputs):
      scale = scales[i]
      ori_shape = ori_shapes[i]
      imgid = image_ids[i]
      if config.add_mask:
        final_boxes, final_labels, final_probs, final_masks = output
        final_boxes = final_boxes / scale
        final_masks = [fill_full_mask(box, mask, ori_shape)
                       for box, mask in zip(final_boxes, final_masks)]
      else:
        final_boxes, final_labels, final_probs = output
        final_boxes = final_boxes / scale
        final_masks = [None for one in final_boxes]

      for box, prob, label, mask in zip(final_boxes, final_probs,
                                        final_labels, final_masks):
        box[2] -= box[0]
        box[3] -= box[1]

        cat_id = config.classId_to_cocoId[label]

        # encode mask
        rle = None
        if config.add_mask:
          rle = cocomask.encode(np.array(mask[:, :, None], order="F"))[0]
          rle["counts"] = rle["counts"].decode("ascii")

        res = {
            "image_id":imgid,#int
            "category_id":cat_id,
            "cat_name":config.class_names[label], #[0-80]
            "score":float(round(prob, 4)),
            #"bbox": list(map(lambda x:float(round(x,1)),box)),
            "bbox": [float(round(x, 1)) for x in box],
            "segmentation":rle
        }
        pred.append(res)

    #print([(one["category_id"],one["score"],one["bbox"]) for one in pred]
    #print(imageId
    #sys.exit()

  return pred


# test on coco dataset
def test(config):
  test_data = read_data_coco(
      config.datajson, config=config, add_gt=False, load_coco_class=True)

  print("total testing samples:%s" % test_data.num_examples)

  models = []
  for i in range(config.gpuid_start, config.gpuid_start+config.gpu):
    models.append(get_model(config, i, controller=config.controller))
  tester = Tester(models, config, add_mask=config.add_mask)


  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  if not config.use_all_mem:
    tfconfig.gpu_options.allow_growth = True
  with tf.Session(config=tfconfig) as sess:

    initialize(load=True, load_best=config.load_best, config=config, sess=sess)
    # num_epoch should be 1
    assert config.num_epochs == 1
    num_steps = int(math.ceil(
        test_data.num_examples/float(config.im_batch_size)))*config.num_epochs

    # a list of imageids
    pred = forward_coco(test_data, num_steps, config, sess, tester, resize=True)

    #with open("coco.json","w") as f:
    #  json.dump(pred,f)
    if config.use_coco_eval:

      evalcoco(pred, config.datajson, add_mask=config.add_mask)

    else:

      # check our AP implementation, use our map implementation
      # load the annotation first
      all_cat_ids = {}
      with open(config.datajson, "r") as f:
        data = json.load(f)
      gt = {}  # imageid -> boxes:[], catids
      for one in data["annotations"]:
        cat_id = one["category_id"]
        all_cat_ids[cat_id] = 1
        imageid = int(one["image_id"])
        if imageid not in gt:
          gt[imageid] = {} # cat_ids -> boxes[]
        #gt[imageid]["boxes"].append(one["bbox"]) # (x,y,w,h), float
        #gt[imageid]["cat_ids"].append(one["category_id"])
        if cat_id not in gt[imageid]:
          gt[imageid][cat_id] = []
        gt[imageid][cat_id].append(one["bbox"])


      print("total category:%s" % len(all_cat_ids))

      # get the aps/ars for each frame
      dt = {} # imageid -> cat_id -> {boxes,scores}
      for one in pred:
        imageid = one["image_id"]
        dt_bbox = one["bbox"]
        score = one["score"]
        cat_id = one["category_id"]
        if imageid not in dt:
          dt[imageid] = {}
        if cat_id not in dt[imageid]:
          dt[imageid][cat_id] = []
        dt[imageid][cat_id].append((dt_bbox, score))

      # accumulate all detection and compute AP once
      e = {} # imageid -> catid
      start = time.time()
      for imageid in gt:
        e[imageid] = {}
        for cat_id in gt[imageid]:
          g = gt[imageid][cat_id]

          e[imageid][cat_id] = {
              "dscores":[],
              "dm":[],
              "gt_num":len(g),
          }

          d = []
          dscores = []
          if imageid in dt and cat_id in dt[imageid]:
            # sort the boxes based on the score first
            dt[imageid][cat_id].sort(key=operator.itemgetter(1), reverse=True)
            for boxes, score in dt[imageid][cat_id]:
              d.append(boxes)
              dscores.append(score)


          dm, gm = match_detection(
              d, g, cocomask.iou(d, g, [0 for _ in range(len(g))]),
              iou_thres=0.5)

          e[imageid][cat_id]["dscores"] = dscores
          e[imageid][cat_id]["dm"] = dm

      # accumulate results
      maxDet = 100 # max detection per image per category
      aps = {}
      ars = {}
      for catId in all_cat_ids:
        # put all detection scores from all image together
        dscores = np.concatenate(
            [e[imageid][catId]["dscores"][:maxDet]
             for imageid in e if catId in e[imageid]])
        # sort
        inds = np.argsort(-dscores, kind="mergesort")
        dscores_sorted = dscores[inds]

        # put all detection annotation together based on the score sorting
        dm = np.concatenate(
            [e[imageid][catId]["dm"][:maxDet]
             for imageid in e if catId in e[imageid]])[inds]
        num_gt = np.sum([e[imageid][catId]["gt_num"]
                         for imageid in e if catId in e[imageid]])

        aps[catId] = computeAP(dm)
        ars[catId] = computeAR_2(dm, num_gt)

      mean_ap = np.mean([aps[catId] for catId in aps])
      mean_ar = np.mean([ars[catId] for catId in ars])
      took = time.time() - start
      print("total dt image:%s, gt image:%s" % (len(dt), len(gt)))

      print("mean AP with IoU 0.5:%s, mean AR with max detection %s:%s, "
            "took %s seconds" % (mean_ap, maxDet, mean_ar, took))


def cal_total_param():
  total = 0
  for var in tf.trainable_variables():
    shape = var.get_shape()
    var_num = 1
    for dim in shape:
      var_num *= dim.value
    total += var_num
  return total


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


gpu_util_logs = []
gpu_temp_logs = []

# use nvidia-smi to
def log_gpu_util(interval, gpuid_range):
  global gpu_util_logs
  while True:
    time.sleep(interval)
    gpu_temps, gpu_utils = parse_nvidia_smi(gpuid_range)
    gpu_util_logs.extend(gpu_utils)
    gpu_temp_logs.extend(gpu_temps)


if __name__ == "__main__":
  config = get_args()

  if config.mode == "pack":
    config.is_pack_model = True
  if config.is_pack_model:
    pack(config)
  else:
    if config.log_time_and_gpu:
      gpu_log_interval = 10 # every k seconds
      start_time = time.time()
      gpu_check_thread = threading.Thread(
          target=log_gpu_util,
          args=[gpu_log_interval, (config.gpuid_start, config.gpu)])
      gpu_check_thread.daemon = True
      gpu_check_thread.start()

    if config.mode == "train":
      train_diva(config)
    elif config.mode == "test":
      test(config)
    elif config.mode == "forward":
      forward(config)
    else:
      raise Exception("mode %s not supported"%(config.mode))

    if config.log_time_and_gpu:
      end_time = time.time()
      print("total run time %s (%s), log gpu utilize every %s seconds and "
            "get median %.2f%% and average %.2f%%. GPU temperature median "
            "%.2f and average %.2f (C)" % (
          sec2time(end_time - start_time),
          end_time - start_time,
          gpu_log_interval,
          np.median(gpu_util_logs)*100,
          np.mean(gpu_util_logs)*100,
          np.median(gpu_temp_logs),
          np.mean(gpu_temp_logs),
      ))
