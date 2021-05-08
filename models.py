# coding=utf-8
"""model graph.
"""
import cv2
import json
import math
import itertools
import random
import sys
import os
import tensorflow as tf
import numpy as np

from PIL import Image

from utils import Dataset
from utils import get_all_anchors
from utils import draw_boxes
from utils import box_wh_to_x1x2
from utils import get_op_tensor_name

#import tensorflow.contrib.slim as slim
from nn import pretrained_resnet_conv4
from nn import conv2d
from nn import deconv2d
from nn import resnet_conv5
from nn import dense
from nn import pairwise_iou
from nn import get_iou_callable
from nn import resizeImage
from nn import resnet_fpn_backbone
from nn import fpn_model
from nn import decode_bbox_target
from nn import generate_rpn_proposals
from nn import sample_fast_rcnn_targets
from nn import roi_align
from nn import encode_bbox_target
from nn import focal_loss
from nn import wd_cost
from nn import clip_boxes
from nn import person_object_relation
from nn import np_iou
# for multi image batch
from nn import decode_bbox_target_multi
from nn import generate_rpn_proposals_multibatch
from nn import roi_align_multi
# this is for ugly batch norm
from nn import is_training
from nn import add_wd
#from nn import get_so_labels
from nn import group_norm

from efficientdet_wrapper import EfficientDet
from efficientdet_wrapper import EfficientDet_frozen

# need this otherwise No TRTEngineOp when load a trt graph # no use,
#TensorRT doesn"t support FPN ops yet
#import tensorflow.contrib.tensorrt as trt

# ------------------------------ multi gpu stuff
PS_OPS = [
    "Variable", "VariableV2", "AutoReloadVariable", "MutableHashTable",
    "MutableHashTableOfTensors", "MutableDenseHashTable"
]

# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(compute_device, controller_device): # ps: paramter server
  """Returns a function to place variables on the ps_device.

  Args:
    device: Device for everything but variables
    ps_device: Device to put the variables on. Example values are /GPU:0
    and /CPU:0.

  If ps_device is not set then the variables will be placed on the default
  device.
  The best device for shared varibles depends on the platform as well as the
  model. Start with CPU:0 and then test GPU:0 to see if there is an
  improvement.
  """
  def _assign(op):
    node_def = op if isinstance(op, tf.NodeDef) else op.node_def
    if node_def.op in PS_OPS:
      return controller_device
    else:
      return compute_device
  return _assign


#----------------------------------


# 05/2019, the code will still use other gpu even if we have set visible list;
# seems a v1.13 bug
# yes it is a v1.13 bug, something to do with XLA:
# https://github.com/horovod/horovod/issues/876
def get_model(config, gpuid=0, task=0, controller="/cpu:0", is_multi=False):


  with tf.device(assign_to_device("/gpu:%s"%(gpuid), controller)):
    # load from frozen model
    if config.is_load_from_pb:
      if config.is_efficientdet:
        model = EfficientDet_frozen(config, config.load_from, gpuid)
      else:
        model = Mask_RCNN_FPN_frozen(config.load_from, gpuid,
                                     add_mask=config.add_mask,
                                     is_multi=is_multi)
    else:
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        #tf.get_variable_scope().reuse_variables()
        if config.is_efficientdet:
          model = EfficientDet(config)
        elif is_multi:
          model = Mask_RCNN_FPN_multi(config, gpuid=gpuid)
        else:
          model = Mask_RCNN_FPN(config, gpuid=gpuid)

  return model

def get_model_feat(config, gpuid=0, task=0, controller="/cpu:0"):
  # task is not used
  #with tf.device("/gpu:%s"%gpuid):
  with tf.device(assign_to_device("/gpu:%s"%(gpuid), controller)):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      #tf.get_variable_scope().reuse_variables()
      model = RCNN_FPN_givenbox(config, gpuid=gpuid)

  return model


# updated 05/29, pack model
# simple tf frozen graph or TensorRT optimized model
def pack(config):

  # the graph var names to be saved

  vars_ = [
      "final_boxes",
      "final_labels",
      "final_probs",
      "fpn_box_feat"]
  if config.add_mask:
    vars_.append("final_masks")
  if config.is_multi:
    vars_.append("final_valid_indices")

  model = get_model(config, is_multi=config.is_multi)
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  with tf.Session(config=tfconfig) as sess:

    initialize(load=True, load_best=config.load_best, config=config, sess=sess)

    # also save all the model config and note into the model
    #assert config.note != "", "please add some note for the model"
    if config.note is not None:
      # remove some param?
      config_json = vars(config)
      for k in config_json:
        if type(config_json[k]) == type(np.array([1])):
          config_json[k] = config_json[k].tolist()
        if type(config_json[k]) == type(np.array([1])[0]):
          config_json[k] = int(config_json[k])
        if type(config_json[k]) == type(np.array([1.0])[0]):
          config_json[k] = float(config_json[k])
        if type(config_json[k]) == type({}.keys()):  # python3 dict_keys
          config_json[k] = list(config_json[k])
      with open(config.pack_modelconfig_path, "w") as f:
        json.dump(config_json, f)

    print("saving packed model...")
    # put into one big file to save
    input_graph_def = tf.get_default_graph().as_graph_def()
    #print [n.name for n in input_graph_def.node]
    # We use a built-in TF helper to export variables to constants
    # output node names

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes
        vars_,
    )
    output_graph = config.pack_model_path
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    print("model saved in %s, config record is in %s" % (
        config.pack_model_path, config.pack_modelconfig_path))




# load the weights at init time
# this class has the same interface as Mask_RCNN_FPN
class Mask_RCNN_FPN_frozen():
  def __init__(self, modelpath, gpuid, add_mask=False, is_multi=False):
    self.graph = tf.get_default_graph()
    self.is_multi = is_multi

    # save path is one.pb file

    with tf.gfile.GFile(modelpath, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    #print [n.name for n in graph_def.node]
    # need this to load different stuff for different gpu
    self.var_prefix = "model_%s" % gpuid
    tf.import_graph_def(
        graph_def,
        name=self.var_prefix,
        return_elements=None
    )


    # input place holders
    self.image = self.graph.get_tensor_by_name("%s/image:0" % self.var_prefix)

    self.final_boxes = self.graph.get_tensor_by_name(
        "%s/final_boxes:0" % self.var_prefix)
    self.final_labels = self.graph.get_tensor_by_name(
        "%s/final_labels:0" % self.var_prefix)
    self.final_probs = self.graph.get_tensor_by_name(
        "%s/final_probs:0" % self.var_prefix)

    if is_multi:
      self.final_valid_indices = self.graph.get_tensor_by_name(
        "%s/final_valid_indices:0" % self.var_prefix)

    if add_mask:
      self.final_masks = self.graph.get_tensor_by_name(
          "%s/final_masks:0" % self.var_prefix)

    self.fpn_box_feat = self.graph.get_tensor_by_name(
        "%s/fpn_box_feat:0" % self.var_prefix)


  def get_feed_dict_forward(self, imgdata):
    feed_dict = {}

    if self.is_multi:
      # imgdata: a list of [H, W, 3]

      # [B, H, W, 3]
      feed_dict[self.image] = np.stack(imgdata, axis=0)

    else:

      feed_dict[self.image] = imgdata

    return feed_dict

  def get_feed_dict_forward_multi(self, imgs):
    # imgs: a list of [H, W, 3]
    feed_dict = {}

    # [B, H, W, 3]
    feed_dict[self.image] = np.stack(imgs, axis=0)

    return feed_dict


class Mask_RCNN_FPN():
  def __init__(self, config, gpuid=0):
    self.gpuid = gpuid
    # for batch_norm
    global is_training
    is_training = config.is_train # change this before building model

    self.config = config

    self.num_class = config.num_class

    self.global_step = tf.get_variable(
        "global_step", shape=[], dtype="int32",
        initializer=tf.constant_initializer(0), trainable=False)

    # current model get one image at a time
    self.image = tf.placeholder(tf.float32, [None, None, 3], name="image")

    if not config.is_pack_model:
      self.is_train = tf.placeholder("bool", [], name="is_train")

    # for training
    self.anchor_labels = []
    self.anchor_boxes = []
    num_anchors = len(config.anchor_ratios)
    for k in range(len(config.anchor_strides)):
      self.anchor_labels.append(
          tf.placeholder(tf.int32, [None, None, num_anchors],
                         name="anchor_labels_lvl%s" % (k+2)))
      self.anchor_boxes.append(
          tf.placeholder(tf.float32, [None, None, num_anchors, 4],
                         name="anchor_boxes_lvl%s" % (k+2)))

    self.gt_boxes = tf.placeholder(tf.float32, [None, 4], name="gt_boxes")
    self.gt_labels = tf.placeholder(tf.int64, [None, ], name="gt_labels")

    self.so_gt_boxes = []
    self.so_gt_labels = []
    for i in range(len(config.small_objects)):
      self.so_gt_boxes.append(
          tf.placeholder(tf.float32, [None, 4], name="so_gt_boxes_c%s" % (i+1)))
      self.so_gt_labels.append(
          tf.placeholder(tf.int64, [None,], name="so_gt_labels_c%s" % (i+1)))

    # H,W,v -> {0,1}
    self.gt_mask = tf.placeholder(tf.uint8, [None, None, None], name="gt_masks")

    # the following will be added in the build_forward and loss
    self.logits = None
    self.yp = None
    self.loss = None

    self.build_preprocess()
    self.build_forward()

  # get feature map anchor and preprocess image
  def build_preprocess(self):
    config = self.config
    image = self.image

    # get feature map anchors first
    # slower if put on cpu # 1.5it/s vs 1.2it/s
    self.multilevel_anchors = []
    with tf.name_scope("fpn_anchors"):#,tf.device("/cpu:0"):
      #fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1]
      #// config.anchor_stride
      # all posible anchor box coordinates for a given max_size image,
      # so for 1920 x 1920 image, 1920/16 = 120, so (120,120,NA,4) box, NA is
      #scale*ratio boxes
      self.multilevel_anchors = self.get_all_anchors_fpn()

    bgr = True  # cv2 load image is bgr
    p_image = tf.expand_dims(image, 0)  # [1,H,W,C]

    with tf.name_scope("image_preprocess"):  # tf.device("/cpu:0"):
      if p_image.dtype.base_dtype != tf.float32:
        p_image = tf.cast(p_image, tf.float32)

      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]

      p_image = p_image * (1.0/255)

      if bgr:
        mean = mean[::-1]
        std = std[::-1]
      image_mean = tf.constant(mean, dtype=tf.float32)
      image_std = tf.constant(std, dtype=tf.float32)
      p_image = (p_image - image_mean) / image_std
      p_image = tf.transpose(p_image, [0, 3, 1, 2])

    self.p_image = p_image

  def get_all_anchors_fpn(self):
    config = self.config
    anchors = []
    assert len(config.anchor_strides) == len(config.anchor_sizes)
    for stride, size in zip(config.anchor_strides, config.anchor_sizes):
      anchors_np = get_all_anchors(
          stride=stride, sizes=[size], ratios=config.anchor_ratios,
          max_size=config.max_size)

      anchors.append(anchors_np)
    return anchors

  # make the numpy anchor match to the feature shape
  def slice_feature_and_anchors(self, image_shape2d, p23456, anchors):
    # anchors is the numpy anchors for different levels
    config = self.config
    # the anchor labels and boxes are grouped into
    gt_anchor_labels = self.anchor_labels
    gt_anchor_boxes = self.anchor_boxes
    self.sliced_anchor_labels = []
    self.sliced_anchor_boxes = []
    for i, stride in enumerate(config.anchor_strides):
      with tf.name_scope("FPN_slice_lvl%s" % (i)):
        if i < 3:
          # Images are padded for p5, which are too large for p2-p4.
          pi = p23456[i]
          target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * \
              (1.0 / stride)))

          p23456[i] = tf.slice(
              pi, [0, 0, 0, 0], tf.concat([[-1, -1], target_shape], axis=0))
          p23456[i].set_shape([1, pi.shape[1], None, None])

        shape2d = tf.shape(p23456[i])[2:] # h,W
        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)

        anchors[i] = tf.slice(anchors[i], [0, 0, 0, 0], slice4d)
        self.sliced_anchor_labels.append(
            tf.slice(gt_anchor_labels[i], [0, 0, 0], slice3d))
        self.sliced_anchor_boxes.append(tf.slice(
            gt_anchor_boxes[i], [0, 0, 0, 0], slice4d))

  def generate_fpn_proposals(self, multilevel_anchors, multilevel_label_logits,
                             multilevel_box_logits, image_shape2d):
    config = self.config
    num_lvl = len(config.anchor_strides)
    assert num_lvl == len(multilevel_anchors)
    assert num_lvl == len(multilevel_box_logits)
    assert num_lvl == len(multilevel_label_logits)
    all_boxes = []
    all_scores = []
    fpn_nms_topk = config.rpn_train_post_nms_topk \
        if config.is_train else config.rpn_test_post_nms_topk
    for lvl in range(num_lvl):
      with tf.name_scope("Lvl%s"%(lvl+2)):
        anchors = multilevel_anchors[lvl]
        pred_boxes_decoded = decode_bbox_target(
            multilevel_box_logits[lvl], anchors,
            decode_clip=config.bbox_decode_clip)

        this_fpn_nms_topk = fpn_nms_topk
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(multilevel_label_logits[lvl], [-1]), image_shape2d,
            config, pre_nms_topk=this_fpn_nms_topk)
        all_boxes.append(proposal_boxes)
        all_scores.append(proposal_scores)


    proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
    proposal_scores = tf.concat(all_scores, axis=0)  # n
    proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
    proposal_scores, topk_indices = tf.nn.top_k(proposal_scores,
                                                k=proposal_topk, sorted=False)
    proposal_boxes = tf.gather(proposal_boxes, topk_indices)
    return tf.stop_gradient(proposal_boxes, name="boxes"), \
           tf.stop_gradient(proposal_scores, name="scores")

  # based on box sizes
  def fpn_map_rois_to_levels(self, boxes):

    def tf_area(boxes):
      x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
      return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * \
        (1.0 / np.log(2))))
    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),# problems with ==?
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]

    level_ids = [tf.reshape(x, [-1], name="roi_level%s_id" % (i + 2))
                 for i, x in enumerate(level_ids)]
    #num_in_levels = [tf.size(x, name="num_roi_level%s" % (i + 2))
    #                 for i, x in enumerate(level_ids)]

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


  # output_shape is the output feature HxW
  def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
    config = self.config
    assert len(features) == 4
    # Reassign rcnn_boxes to levels # based on box area size
    level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
      with tf.name_scope("roi_level%s" % (i + 2)):
        boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])
        all_rois.append(
            roi_align(featuremap, boxes_on_featuremap, output_shape))

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois


  def build_forward(self):
    config = self.config
    image = self.p_image # [1, C, H, W]
    image_shape2d = tf.shape(image)[2:]
    # a list of numpy anchors, not sliced
    multilevel_anchors = self.multilevel_anchors

    # the feature map shared by RPN and fast RCNN
    # TODO: fix the batch norm mess
    # TODO: fix global param like data_format and
    # [1,C,FS,FS]

    c2345 = resnet_fpn_backbone(
        image, config.resnet_num_block, use_gn=config.use_gn,
        resolution_requirement=config.fpn_resolution_requirement,
        use_dilations=config.use_dilations,
        use_deformable=config.use_deformable, tf_pad_reverse=True,
        freeze=config.freeze, use_basic_block=config.use_basic_block,
        use_se=config.use_se, use_resnext=config.use_resnext)

    # include lateral 1x1 conv and final 3x3 conv
    # -> [7, 7, 256]
    p23456 = fpn_model(c2345, num_channel=config.fpn_num_channel,
                       use_gn=config.use_gn, scope="fpn")

    if config.freeze_rpn or config.freeze_fastrcnn:
      p23456 = [tf.stop_gradient(p) for p in p23456]

    # [1, H, W, channel]
    self.fpn_feature = tf.image.resize_images(tf.transpose(
        p23456[3], perm=[0, 2, 3, 1]), (7, 7)) # p5 # default bilinear

    if config.no_obj_detect: # pair with extract_feat, so only extract feature
      print("no object detect branch..")
      return True
    # given the numpy anchor for each stride,
    # slice the anchor box and label against the feature map size on each
    #level. Again?

    self.slice_feature_and_anchors(image_shape2d, p23456, multilevel_anchors)
    # now multilevel_anchors are sliced and tf type
    # added sliced gt anchor labels and boxes
    # so we have each fpn level"s anchor boxes, and the ground truth anchor
    # boxes & labels if training

    # given [1,256,FS,FS] feature, each level got len(anchor_ratios) anchor
    # outputs
    rpn_outputs = [
        self.rpn_head(pi, config.fpn_num_channel, len(config.anchor_ratios),
                      data_format="NCHW", scope="rpn") for pi in p23456]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]

    if config.freeze_rpn:
      multilevel_label_logits = [tf.stop_gradient(o)
                                 for o in multilevel_label_logits]
      multilevel_box_logits = [tf.stop_gradient(o)
                               for o in multilevel_box_logits]

    # each H,W location has a box regression and classification score,
    # here combine all positive boxes using NMS
    # [N,4]/[N] , N is the number of proposal boxes
    proposal_boxes, proposal_scores = self.generate_fpn_proposals(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits,
        image_shape2d)

    # for getting RPN performance
    # K depend on rpn_test_post_nms_topk during testing
    # K = 1000
    self.proposal_boxes = proposal_boxes  # [K, 4]
    self.proposal_scores = proposal_scores  # [K]

    if config.is_train:
      gt_boxes = self.gt_boxes
      gt_labels = self.gt_labels
      # for training, use gt_box and some proposal box as pos and neg
      # rcnn_sampled_boxes [N_FG+N_NEG,4]
      # fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
      rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
          proposal_boxes, gt_boxes, gt_labels, config=config)
    else:
      rcnn_boxes = proposal_boxes

    # NxCx7x7 # (?, 256, 7, 7)
    roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4], rcnn_boxes, 7)

    if config.use_frcnn_class_agnostic:
      # (N,num_class), (N, 1, 4)
      fastrcnn_label_logits, fastrcnn_box_logits = \
          self.fastrcnn_2fc_head_class_agnostic(
              roi_feature_fastrcnn, config.num_class,
              boxes=rcnn_boxes, scope="fastrcnn")
    else:
      # (N,num_class), (N, num_class - 1, 4)
      fastrcnn_label_logits, fastrcnn_box_logits = self.fastrcnn_2fc_head(
          roi_feature_fastrcnn, config.num_class,
          boxes=rcnn_boxes, scope="fastrcnn")

    if config.freeze_fastrcnn:
      fastrcnn_label_logits, fastrcnn_box_logits = tf.stop_gradient(
          fastrcnn_label_logits), tf.stop_gradient(fastrcnn_box_logits)

    if config.use_small_object_head:

      # 1. get all the actual boxes coordinates
      anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1),
                        [1, config.num_class-1, 1])
      boxes = decode_bbox_target(fastrcnn_box_logits / \
          tf.constant(config.fastrcnn_bbox_reg_weights, dtype=tf.float32),
                                 anchors)

      probs = tf.nn.softmax(fastrcnn_label_logits)

      boxes = tf.transpose(boxes, [1, 0, 2])  # [num_class-1, N, 4]
      probs = tf.transpose(probs[:, 1:], [1, 0])  # [num_class-1, N]

      small_object_class_ids = [config.classname2id[name] - 1
                                for name in config.small_objects]

      # C is the number of small object class
      # [C, N, 4], [C, N]
      so_boxes, so_scores = tf.gather(boxes, small_object_class_ids), \
          tf.gather(probs, small_object_class_ids)

      # 1. we do NMS for each class to get topk

      # for each catagory get the top K
      # [C, K, 4] / [C, K]
      so_boxes, so_scores = tf.map_fn(
          self.nms_return_boxes, (so_scores, so_boxes),
          dtype=(tf.float32, tf.float32), parallel_iterations=10)

      self.so_boxes = so_boxes

      so_boxes = tf.reshape(so_boxes, [-1, 4])  # [C*K, 4]
      so_scores = tf.reshape(so_scores, [-1])  # [C*K]

      # [C*K, 256, 7, 7]
      so_feature = self.multilevel_roi_align(p23456[:4], so_boxes, 7)

      # share the fc part with fast rcnn head
      with tf.variable_scope("fastrcnn", reuse=tf.AUTO_REUSE):
        dim = config.fpn_frcnn_fc_head_dim # 1024
        initializer = tf.variance_scaling_initializer()

        # sharing features
        # [C*K, dim]
        hidden = dense(so_feature, dim, W_init=initializer,
                       activation=tf.nn.relu, scope="fc6")
        hidden = dense(hidden, dim, W_init=initializer,
                       activation=tf.nn.relu, scope="fc7")
        # [C, K, dim]
        hidden = tf.reshape(hidden, [len(config.small_objects), -1, dim])

        if config.freeze_fastrcnn:
          hidden = tf.stop_gradient(hidden)

        if config.use_so_association:
          ref_class_id = config.classname2id["Person"] - 1
          # [N, 4], [N]
          ref_boxes, ref_scores = boxes[ref_class_id], probs[ref_class_id]

          # NMS to get a few peron boxes
          ref_topk = config.so_person_topk # 10

          ref_selection = tf.image.non_max_suppression(
              ref_boxes, ref_scores, max_output_size=ref_topk,
              iou_threshold=config.fastrcnn_nms_iou_thres)
          # [Rr, 4]
          ref_boxes = tf.gather(ref_boxes, ref_selection)
          ref_scores = tf.gather(ref_scores, ref_selection)
          ref_feat = self.multilevel_roi_align(p23456[:4], ref_boxes, 7)

          # share the same fc
          ref_feat = dense(ref_feat, dim, W_init=initializer,
                           activation=tf.nn.relu, scope="fc6")
          ref_feat = dense(ref_feat, dim, W_init=initializer,
                           activation=tf.nn.relu, scope="fc7")

          if config.freeze_fastrcnn:
            ref_feat = tf.stop_gradient(ref_feat)

        # new variable for small object
        with tf.variable_scope("small_objects"):
          so_label_logits = [] # each class a head

          for i in range(len(config.small_objects)):
            if config.use_so_association:
              asso_hidden = hidden[i] + person_object_relation(
                  hidden[i], self.so_boxes[i], ref_boxes, ref_feat,
                  group=16, geo_feat_dim=64, scope="person_object_relation")
              so_label_logits.append(dense(
                  asso_hidden, 2,
                  W_init=tf.random_normal_initializer(stddev=0.01),
                  scope="small_object_classification_c%s" % (i+1)))
            else:
              so_label_logits.append(dense(
                  hidden[i], 2,
                  W_init=tf.random_normal_initializer(stddev=0.01),
                  scope="small_object_classification_c%s"%(i+1)))
          add_wd(0.0001)

        # [C, K, 2]
        so_label_logits = tf.stack(so_label_logits, axis=0)


    if config.is_train:
      rpn_label_loss, rpn_box_loss = self.multilevel_rpn_losses(
          multilevel_anchors, multilevel_label_logits, multilevel_box_logits)

      # rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
      fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])

      # for training, maskRCNN only apply on positive box
      # [N_FG, num_class, 14, 14]

      # [N_FG, 4]
      # sampled boxes are at least iou with a gt_boxes
      fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
      fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits,
                                         fg_inds_wrt_sample)

      # [N_FG, 4] # each proposal box assigned gt box, may repeat
      matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

      # fastrcnn also need to regress box (just the FG box)
      encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * \
          tf.constant(config.fastrcnn_bbox_reg_weights)  # [10,10,5,5]?

      # fastrcnn input is fg and bg proposal box, do classification to
      # num_class(include bg) and then regress on fg boxes
      # [N_FG+N_NEG,4] & [N_FG,4]
      fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_losses(
          rcnn_labels, fastrcnn_label_logits, encoded_boxes,
          fg_fastrcnn_box_logits)

      # ---------------------------------------------------------

      # for debug
      self.rpn_label_loss = rpn_label_loss
      self.rpn_box_loss = rpn_box_loss
      self.fastrcnn_label_loss = fastrcnn_label_loss
      self.fastrcnn_box_loss = fastrcnn_box_loss

      losses = [rpn_label_loss, rpn_box_loss, fastrcnn_label_loss,
                fastrcnn_box_loss]

      if config.use_small_object_head:
        # assume we have the small gt boxes and labels
        # so_boxes [C, K, 4]
        # so_label_logits [C, K, 2]
        # so_labels [C, K] # [0, 1]
        so_labels = get_so_labels(self.so_boxes, self.so_gt_boxes,
                                  self.so_gt_labels, config=config)

        so_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=so_labels, logits=so_label_logits)

        so_label_loss = tf.reduce_mean(so_label_loss, name="label_loss")

        self.so_label_loss = so_label_loss
        losses.append(so_label_loss)


      # mask rcnn loss
      if config.add_mask:
        fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])
        fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)

        # NxCx14x14
        # only the fg boxes
        roi_feature_fastrcnn = self.multilevel_roi_align(
            p23456[:4], fg_sampled_boxes, 14)

        mask_logits = self.maskrcnn_up4conv_head(
            fg_feature, config.num_class, scope="maskrcnn")


        # [N_FG, H,W]
        gt_mask = self.gt_mask
        gt_mask_for_fg = tf.gather(gt_mask, fg_inds_wrt_gt)
        # [N_FG, H, W] -> [N_FG, 14, 14]
        target_masks_for_fg = crop_and_resize(
            tf.expand_dims(gt_masks, 1),
            fg_sampled_boxes,
            fg_inds_wrt_gt, 28, pad_border=False) # fg x 1x28x28
        target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1)

        mrcnn_loss = self.maskrcnn_loss(mask_logits, fg_labels,
                                        target_masks_for_fg)

        losses += [mrcnn_loss]

      self.wd = None
      if config.wd is not None:
        wd = wd_cost(".*/W", config.wd, scope="wd_cost")
        self.wd = wd
        losses.append(wd)

      self.loss = tf.add_n(losses, "total_loss")

      # l2loss
    else:

      # inferencing
      # K -> proposal box
      # [K,num_class]
      # image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits

      # get the regressed actual boxes
      if config.use_frcnn_class_agnostic:
        # box regress logits [K, 1, 4], so we tile it to num_class-1 so
        # the rest is the same
        fastrcnn_box_logits = tf.tile(fastrcnn_box_logits,
                                      [1, config.num_class - 1, 1])

      num_class = config.num_class

      # COCO has 81 classes, we only need a few
      if config.use_partial_classes:
        needed_object_classids = [config.classname2id[name]
                                  for name in config.partial_classes]
        needed_object_classids_minus_1 = [o - 1 for o in needed_object_classids]
        # (N, num_class), (N, num_class - 1, 4)
        # -> (num_class, N), (num_class - 1, N, 4)
        label_logits_t = tf.transpose(fastrcnn_label_logits, [1, 0])
        box_logits_t = tf.transpose(fastrcnn_box_logits, [1, 0, 2])

        # [C + 1, N]  # 1 is the BG class
        partial_label_logits_t = tf.gather(label_logits_t,
                                           [0] + needed_object_classids)
        # [C, N, 4]
        partial_box_logits_t = tf.gather(box_logits_t,
                                         needed_object_classids_minus_1)

        partial_label_logits = tf.transpose(partial_label_logits_t, [1, 0])
        partial_box_logits = tf.transpose(partial_box_logits_t, [1, 0, 2])

        fastrcnn_label_logits = partial_label_logits
        fastrcnn_box_logits = partial_box_logits

        num_class = len(needed_object_classids) + 1

      # anchor box [K,4] -> [K, num_class - 1, 4] <-
      # box regress logits [K, num_class-1, 4]
      anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, num_class-1, 1])

      # [K, num_class-1, 4]/ [K, 1, 4]
      decoded_boxes = decode_bbox_target(fastrcnn_box_logits / \
          tf.constant(config.fastrcnn_bbox_reg_weights, dtype=tf.float32),
                                         anchors)

      decoded_boxes = clip_boxes(decoded_boxes, image_shape2d,
                                 name="fastrcnn_all_boxes")

      label_probs = tf.nn.softmax(fastrcnn_label_logits)

      if config.use_small_object_head:
        # so_label_logits: [C, N, 2]
        """
        if config.replace_small_object:
          # replace some of the scores
          small_object_class_ids = [config.classname2id[name]
                                    for name in config.small_objects]
          # [N, num_class]
          # put each label logit for each class then stack
          new_label_logits = []
          for classid in config.classid2name:
            if classid in small_object_class_ids:
              so_idx = small_object_class_ids.index(classid)
              # 1 is the class score and 0 is score for BG
              new_label_logits.append(so_label_logits[so_idx, :, 1])
            else:
              new_label_logits.append(fastrcnn_label_logits[:, classid])
          fastrcnn_label_logits = tf.stack(new_label_logits, axis=1)
        """

        # output the small object boxes separately
        # K is result_per_im=100
        # 1. so_label_logits is [C, K, 2]
        # so_boxes [C, K, 4]
        # reconstruct label logit to be [K, C+1]
        new_label_logits = []
        # BG is ignore anyway
        new_label_logits.append(
            tf.reduce_mean(so_label_logits[:, :, 0], axis=0)) # [K]
        for i in range(len(config.small_objects)):
          new_label_logits.append(so_label_logits[i, :, 1])
        # [K, C+1]
        so_label_logits = tf.stack(new_label_logits, axis=1)

        # [K, C, 4]
        so_boxes = tf.transpose(self.so_boxes, [1, 0, 2])

        so_decoded_boxes = clip_boxes(
            so_boxes, image_shape2d, name="so_all_boxes")

        so_pred_indices, so_final_probs = self.fastrcnn_predictions(
            so_decoded_boxes, so_label_logits,
            no_score_filter=not config.use_so_score_thres)
        so_final_boxes = tf.gather_nd(
            so_decoded_boxes, so_pred_indices, name="so_final_boxes")

        so_final_labels = tf.add(
            so_pred_indices[:, 1], 1, name="so_final_labels")
        # [R,4]
        self.so_final_boxes = so_final_boxes
        # [R]
        self.so_final_labels = so_final_labels
        self.so_final_probs = so_final_probs

      if config.use_cpu_nms:
        boxes = decoded_boxes
        probs = label_probs
        assert boxes.shape[1] == config.num_class - 1, \
            (boxes.shape, config.num_class)
        assert probs.shape[1] == config.num_class, \
            (probs.shape[1], config.num_class)
        # transpose to map_fn along each class
        boxes = tf.transpose(boxes, [1, 0, 2]) # [num_class-1, K,4]
        probs = tf.transpose(probs[:, 1:], [1, 0]) # [num_class-1, K]

        self.final_boxes = boxes
        self.final_probs = probs

        # just used for compatable with none cpu nms mode
        self.final_labels = rcnn_boxes

        return None # so no TF GPU NMS

      # decoded boxes are [K,num_class-1,4]. so from each proposal
      # boxes generate all classses" boxes, with prob, then do nms on these
      # pred_indices: [R,2] , each entry (#proposal[1-K],
      #catid [0,num_class-1])
      # final_probs [R]
      # here do nms,

      pred_indices, final_probs = self.fastrcnn_predictions(
          decoded_boxes, label_probs)

      # [R,4]
      final_boxes = tf.gather_nd(
          decoded_boxes, pred_indices)
      # [R] , each is 1-catogory
      final_labels = tf.add(pred_indices[:, 1], 1)

      if config.add_mask:

        roi_feature_maskrcnn = self.multilevel_roi_align(
            p23456[:4], final_boxes, 14)

        # [R, num_class - 1, 14, 14]
        mask_logits = self.maskrcnn_up4conv_head(
            roi_feature_maskrcnn, config.num_class, scope="maskrcnn")

        if config.use_partial_classes:
          # need to select the classes as final_labels
          mask_logits_t = tf.transpose(mask_logits, [1, 0, 2, 3])

          # [C, R, 14, 14]
          partial_mask_logits_t = tf.gather(
              mask_logits_t, needed_object_classids)
          # [R, C, 14, 14]
          partial_mask_logits = tf.transpose(
              partial_mask_logits_t, [1, 0, 2, 3])

        indices = tf.stack(
            [tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1],
            axis=1)

        final_mask_logits = tf.gather_nd(mask_logits, indices)
        final_masks = tf.sigmoid(final_mask_logits)

        # [R,14,14]
        self.final_masks = final_masks

      # [R,4]
      self.final_boxes = tf.identity(final_boxes, name="final_boxes")
      # [R]
      self.final_labels = tf.identity(final_labels, name="final_labels")
      # add a name so the frozen graph will have that name
      self.final_probs = tf.identity(final_probs, name="final_probs")

      # [R, 256, 7, 7]
      fpn_box_feat = self.multilevel_roi_align(p23456[:4], final_boxes, 7)
      self.fpn_box_feat = tf.identity(fpn_box_feat, name="fpn_box_feat")


  # ----some model component
  # feature map -> [1,1024,FS1,FS2] , FS1 = H/16.0, FS2 = W/16.0
  # channle -> 1024
  def rpn_head(self, featuremap, channel, num_anchors, data_format,
               scope="rpn"):
    with tf.variable_scope(scope):
      # [1, channel, FS1, FS2] # channel = 1024
      # conv0:W -> [3,3,1024,1024]
      h = conv2d(
          featuremap, channel, kernel=3, activation=tf.nn.relu,
          data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="conv0")
      # h -> [1,1024(channel),FS1,FS2]

      # 1x1 kernel conv to classification on each grid
      # [1, 1024, FS1, FS2] -> # [1, num_anchors, FS1, FS2]
      label_logits = conv2d(
          h, num_anchors, 1, data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="class")
      # [1, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
      box_logits = conv2d(
          h, 4*num_anchors, 1, data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="box")

      # [1,1024,FS1, FS2] -> [FS1, FS2,1024]
      label_logits = tf.squeeze(tf.transpose(label_logits, [0, 2, 3, 1]), 0)

      box_shape = tf.shape(box_logits)
      box_logits = tf.transpose(box_logits, [0, 2, 3, 1]) # [1,FS1, FS2,1024*4]
      # [FS1, FS2,1024,4]
      box_logits = tf.reshape(
          box_logits, [box_shape[2], box_shape[3], num_anchors, 4])

      return label_logits, box_logits

  def small_object_classification_head(
      self, feature, num_class, scope="small_object_classification"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):
      hidden = dense(
          feature, dim, W_init=initializer, activation=tf.nn.relu, scope="fc6")
      hidden = dense(
          hidden, dim, W_init=initializer, activation=tf.nn.relu, scope="fc7")

      classification = dense(
          hidden, num_class, W_init=tf.random_normal_initializer(stddev=0.01),
          scope="class") # [K,num_class]

    return classification

  # feature: [K,C,7,7] # feature for each roi
  def fastrcnn_2fc_head(
      self, feature, num_class=None, boxes=None, scope="fastrcnn_head"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):

      if config.use_conv_frcnn_head:
        hidden = self.conv_frcnn_head(
            feature, dim, config.conv_frcnn_head_dim, num_conv=4,
            use_gn=config.use_gn)
      else:
        # dense will reshape to [k,C*7*7] first
        if config.add_relation_nn:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
        else:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")

      # hidden -> [K, dim]
      if config.use_att_frcnn_head:
        # changes: 1. conv2d kernel size; 2. softmax/sigmoid;
        # 3. sum or gating?; 4. convert to dim first then attention?;
        # 5. attend then two fc, no use of previous hidden
        # [K, 7, 7, C]
        feature = tf.transpose(feature, perm=[0, 2, 3, 1])
        H, W, feat_dim = feature.get_shape()[1:]
        # 1. simple conv attention
        # [K, 7, 7, 1]
        attention = conv2d(
            feature, 1, kernel=3, padding="SAME", stride=1,
            activation=tf.nn.softmax, use_bias=True, data_format="NHWC",
            W_init=initializer, scope="attention")
        # [K,7*7, C]
        feature = tf.reshape(feature, [-1, H*W, feat_dim])
        attention = tf.reshape(attention, [-1, H*W, 1])
        # [K, C]
        attended = tf.reduce_sum(feature * attention, 1)

        # match the dimension
        attended_feat = dense(
            attended, dim, W_init=initializer, activation=tf.nn.relu,
            scope="att_trans")

        # sum with original feature
        hidden = hidden + attended_feat

      with tf.variable_scope("outputs"):

        classification = dense(
            hidden, num_class, W_init=tf.random_normal_initializer(stddev=0.01),
            scope="class") # [K,num_class]


        box_regression = dense(
            hidden, num_class*4,
            W_init=tf.random_normal_initializer(stddev=0.001),
            scope="box")
        box_regression = tf.reshape(box_regression, (-1, num_class, 4))

        box_regression = box_regression[:, 1:, :]

        box_regression.set_shape([None, num_class-1, 4])

    return classification, box_regression

  def conv_frcnn_head(self, feature, fc_dim, conv_dim, num_conv, use_gn=False):
    l = feature
    for k in range(num_conv):
      l = conv2d(
          l, conv_dim, kernel=3, activation=tf.nn.relu,
          data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="truncated_normal"),
          scope="conv%s" % (k))
      if use_gn:
        l = group_norm(l, scope="gn%s" % (k))
    l = dense(
        l, fc_dim, W_init=tf.variance_scaling_initializer(),
        activation=tf.nn.relu, scope="fc")
    return l

  def fastrcnn_2fc_head_class_agnostic(
      self, feature, num_class, boxes=None, scope="head"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):
      if config.use_conv_frcnn_head:
        hidden = self.conv_frcnn_head(
            feature, dim, config.conv_frcnn_head_dim, num_conv=4,
            use_gn=config.use_gn)
      else:
        # dense will reshape to [k,C*7*7] first
        if config.add_relation_nn:
          hidden = dense(
              feature, dim, W_init=initializer,
              activation=tf.nn.relu, scope="fc6")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
          hidden = dense(hidden, dim, W_init=initializer,
                         activation=tf.nn.relu, scope="fc7")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
        else:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")


    with tf.variable_scope("outputs"):

      classification = dense(
          hidden, num_class,
          W_init=tf.random_normal_initializer(stddev=0.01),
          scope="class") # [K,num_class]
      num_class = 1 # just for box
      box_regression = dense(
          hidden, num_class*4,
          W_init=tf.random_normal_initializer(stddev=0.001), scope="box")
      box_regression = tf.reshape(box_regression, (-1, num_class, 4))

    return classification, box_regression


  def maskrcnn_up4conv_head(self, feature, num_class, scope="maskrcnn_head"):
    # feature [R, 256, 7, 7]
    config = self.config
    num_conv = 4 # C4 model this is 0
    l = feature
    with tf.variable_scope(scope):
      for k in range(num_conv):
        l = conv2d(
            l, config.mrcnn_head_dim, kernel=3, activation=tf.nn.relu,
            data_format="NCHW",
            W_init=tf.variance_scaling_initializer(
                scale=2.0, mode="fan_out", distribution="truncated_normal"),
            scope="fcn%s"%(k))

      l = deconv2d(
          l, config.mrcnn_head_dim, kernel=2, stride=2, activation=tf.nn.relu,
          data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="truncated_normal"),
          scope="deconv")
      # [R, num_class-1, 14, 14]
      l = conv2d(
          l, num_class - 1, kernel=1, data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="normal"),
          scope="conv")
      return l


  def nms_return_masks(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # [K]
    ids = tf.reshape(tf.where(prob > config.result_score_thres), [-1])
    prob_ = tf.gather(prob, ids)
    box_ = tf.gather(box, ids)
    # NMS
    selection = tf.image.non_max_suppression(
        box_, prob_, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    selection = tf.to_int32(tf.gather(ids, selection))
    sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

    mask = tf.sparse_to_dense(
        sparse_indices=sorted_selection,
        output_shape=output_shape,
        sparse_values=True,
        default_value=False)

    return mask
  def nms_return_masks_no_score_filter(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # NMS
    selection = tf.image.non_max_suppression(
        box, prob, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

    mask = tf.sparse_to_dense(
        sparse_indices=sorted_selection, output_shape=output_shape,
        sparse_values=True, default_value=False)

    return mask

  def nms_return_boxes(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # NMS
    selection = tf.image.non_max_suppression(
        box, prob, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    selected_prob = tf.gather(prob, selection)
    selected_box = tf.gather(box, selection)
    return selected_box, selected_prob

  # given all proposal box prediction, based on score thres , get final
  # NMS resulting box
  # [K,num_class-1,4] -> decoded_boxes
  # [K,num_class] label_probs
  # each proposal box has prob and box to all class
  # here using nms for each class, -> [R]
  def fastrcnn_predictions(self, boxes, probs, no_score_filter=False,
                           scope="fastrcnn_predictions"):
    with tf.variable_scope(scope):
      config = self.config

      if config.use_bg_score: # use the BG score to filter out boxes
        # probs: [K, num_class]
        box_classes = tf.argmax(probs, axis=1) # [K]
        # [N]
        nonBG_box_indices = tf.reshape(
            tf.where(tf.greater(box_classes, 0)), [-1])
        probs = tf.gather(probs, nonBG_box_indices)
        boxes = tf.gather(boxes, nonBG_box_indices)

      # note if use partial class, config.num_class is not the
      # actual num_class here

      # transpose to map_fn along each class
      boxes = tf.transpose(boxes, [1, 0, 2])  # [num_class-1, K,4]
      probs = tf.transpose(probs[:, 1:], [1, 0])  # [num_class-1, K]

      # for each catagory get the top K
      # [num_class-1, K]
      if no_score_filter:
        masks = tf.map_fn(
            self.nms_return_masks_no_score_filter, (probs, boxes),
            dtype=tf.bool, parallel_iterations=10)
      else:
        masks = tf.map_fn(
            self.nms_return_masks, (probs, boxes), dtype=tf.bool,
            parallel_iterations=10)
      # [R*(num_class-1),2], each entry is [cat_id,box_id]
      selected_indices = tf.where(masks)

      # [num_class-1, K] -> [R*(num_class-1)]
      probs = tf.boolean_mask(probs, masks)
      # topk_indices [R]
      topk_probs, topk_indices = tf.nn.top_k(
          probs, tf.minimum(config.result_per_im, tf.size(probs)), sorted=False)

      # [K,2] <- select [act_num,R]
      filtered_selection = tf.gather(selected_indices, topk_indices)
      filtered_selection = tf.reverse(
          filtered_selection, axis=[1], name="filtered")

      # [R,2], [R,]
      return filtered_selection, topk_probs

  # ---- losses
  def maskrcnn_loss(self, mask_logits, fg_labels, fg_target_masks,
                    scope="maskrcnn_loss"):
    with tf.variable_scope(scope):
      # mask_logits: [N_FG, num_cat, 14, 14]
      # fg_labels: [N_FG]
      # fg_target_masks: [N_FG, 14, 14]
      num_fg = tf.size(fg_labels)
      # [N_FG, 2] # these index is used to get the pos cat"s logit
      indices = tf.stack([tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)
      # ignore other class"s logit
      # [N_FG, 14, 14]
      mask_logits = tf.gather_nd(mask_logits, indices)
      mask_probs = tf.sigmoid(mask_logits)

      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=fg_target_masks, logits=mask_logits)
      loss = tf.reduce_mean(loss, name="maskrcnn_loss")

      return loss


  def multilevel_rpn_losses(self, multilevel_anchors, multilevel_label_logits,
                            multilevel_box_logits, scope="rpn_losses"):
    config = self.config
    sliced_anchor_labels = self.sliced_anchor_labels
    sliced_anchor_boxes = self.sliced_anchor_boxes

    num_lvl = len(config.anchor_strides)
    assert num_lvl == len(multilevel_label_logits)
    assert num_lvl == len(multilevel_box_logits)
    assert num_lvl == len(multilevel_anchors)

    losses = []
    with tf.variable_scope(scope):
      for lvl in range(num_lvl):
        anchors = multilevel_anchors[lvl]
        gt_labels = sliced_anchor_labels[lvl]
        gt_boxes = sliced_anchor_boxes[lvl]

        # get the ground truth T_xywh
        encoded_gt_boxes = encode_bbox_target(gt_boxes, anchors)

        label_loss, box_loss = self.rpn_losses(
            gt_labels, encoded_gt_boxes, multilevel_label_logits[lvl],
            multilevel_box_logits[lvl], scope="level%s" % (lvl+2))
        losses.extend([label_loss, box_loss])

      total_label_loss = tf.add_n(losses[::2], name="label_loss")
      total_box_loss = tf.add_n(losses[1::2], name="box_loss")

    return total_label_loss, total_box_loss


  def rpn_losses(self, anchor_labels, anchor_boxes, label_logits,
                 box_logits, scope="rpn_losses"):
    config = self.config
    with tf.variable_scope(scope):
      # anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
      # label_logits [FS,FS,num_anchors]
      # box_logits [FS,FS,num_anchors,4]

      #with tf.device("/cpu:0"):
      # 1,0|pos/neg
      valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
      pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
      nr_valid = tf.stop_gradient(
          tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
      nr_pos = tf.identity(
          tf.count_nonzero(pos_mask, dtype=tf.int32), name="num_pos_anchor")

      # [nr_valid]
      valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)

      # [nr_valid]
      valid_label_logits = tf.boolean_mask(label_logits, valid_mask)


      placeholder = 0.

      # label loss for all valid anchor box
      if config.focal_loss:
        valid_label_logits = tf.reshape(valid_label_logits, [-1, 1])
        valid_anchor_labels = tf.reshape(valid_anchor_labels, [-1, 1])
        label_loss = focal_loss(
            logits=valid_label_logits, labels=tf.to_float(valid_anchor_labels))
      else:
        label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=valid_label_logits, labels=tf.to_float(valid_anchor_labels))

        label_loss = tf.reduce_sum(label_loss) * (1. / config.rpn_batch_per_im)

      label_loss = tf.where(
          tf.equal(nr_valid, 0), placeholder, label_loss, name="label_loss")

      # box loss for positive anchor
      pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
      pos_box_logits = tf.boolean_mask(box_logits, pos_mask)

      delta = 1.0/9

      # the smooth l1 loss
      box_loss = tf.losses.huber_loss(
          pos_anchor_boxes, pos_box_logits, delta=delta,
          reduction=tf.losses.Reduction.SUM) / delta

      box_loss = box_loss * (1. / config.rpn_batch_per_im)
      box_loss = tf.where(
          tf.equal(nr_pos, 0), placeholder, box_loss, name="box_loss")

      return label_loss, box_loss

  def fastrcnn_losses(self, labels, label_logits, fg_boxes, fg_box_logits,
                      scope="fastrcnn_losses"):
    config = self.config
    with tf.variable_scope(scope):
      # label -> label for roi [N_FG + N_NEG], the fg labels are 1-num_class,
      # 0 is bg
      # label_logits [N_FG + N_NEG,num_class]
      # fg_boxes_logits -> [N_FG,num_class-1,4]

      # so the label is int [0-num_class], 0 being background

      if config.focal_loss:
        # [N, num_classes]
        onehot_label = tf.one_hot(labels, label_logits.get_shape()[-1])

        # here uses sigmoid
        label_loss = focal_loss(
            logits=label_logits, labels=tf.to_float(onehot_label))
      else:
        label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=label_logits)

        label_loss = tf.reduce_mean(label_loss, name="label_loss")

      fg_inds = tf.where(labels > 0)[:, 0]
      fg_labels = tf.gather(labels, fg_inds) # [N_FG]

      num_fg = tf.size(fg_inds) # N_FG
      if int(fg_box_logits.shape[1]) > 1:
        # [N_FG, 2]
        indices = tf.stack(
            [tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)
        # gather the logits from [N_FG,num_class-1, 4] to [N_FG,4],
        # only the gt class"s logit
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)
      else:
        # class agnostic for cascade rcnn
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])
      box_loss = tf.losses.huber_loss(
          fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

      # /  N_FG + N_NEG ?
      box_loss = tf.truediv(
          box_loss, tf.to_float(tf.shape(labels)[0]), name="box_loss")

      return label_loss, box_loss


  # given the image path, and the label for it
  # preprocess
  def get_feed_dict(self, batch, is_train=False):

    #{"imgs":[],"gt":[]}
    config = self.config

    N = len(batch.data["imgs"])

    assert N == 1 # only 1 image for now

    feed_dict = {}

    if "imgdata" in batch.data:
      image = batch.data["imgdata"][0]
    else:
      image = batch.data["imgs"][0]
      if config.use_mixup:
        img1, img2 = image
        use_mixup = random.random() <= config.mixup_chance
        if use_mixup:
          weight = batch.data["mixup_weights"][0]
          img1 = Image.open(img1)
          img2 = Image.open(img2)
          trans_alpha = int(255.0*weight)
          for mixup_box in batch.data["gt"][0]["mixup_boxes"]:
            box_img = img2.crop(mixup_box)
            box_img_sizes = [int(a) for a in box_img.size[::-1]]
            # unit8 and "L" are needed
            mask = Image.fromarray(
                np.zeros(box_img_sizes, dtype="uint8") + trans_alpha, mode="L")
            img1.paste(box_img, mixup_box, mask=mask)
          # PIL to cv2 image
          img1 = np.array(img1)
          img1 = img1[:, :, ::-1].copy()

          # now add the annotation
          batch.data["gt"][0]["boxes"] = np.concatenate(
              [batch.data["gt"][0]["boxes"],
               batch.data["gt"][0]["mixup_boxes"]],
              axis=0)
          batch.data["gt"][0]["labels"].extend(
              batch.data["gt"][0]["mixup_labels"])

          image = img1
        else:
          image = cv2.imread(img1, cv2.IMREAD_COLOR)
      else:
        image = cv2.imread(image, cv2.IMREAD_COLOR)

      assert image is not None, image
      image = image.astype("float32")
    h, w = image.shape[:2] # original width/height

    # resize image, boxes
    short_edge_size = config.short_edge_size
    if config.scale_jitter and is_train:
      short_edge_size = random.randint(
          config.short_edge_size_min, config.short_edge_size_max)


    if "resized_image" in batch.data:
      resized_image = batch.data["resized_image"][0]
    else:
      resized_image = resizeImage(image, short_edge_size, config.max_size)
    newh, neww = resized_image.shape[:2]

    if is_train:
      anno = batch.data["gt"][0] # "boxes" -> [K,4], "labels" -> [K]
      # now the box is in [x1,y1,x2,y2] format, not coco box
      o_boxes = anno["boxes"]
      labels = anno["labels"]
      assert len(labels) == len(o_boxes)

      # boxes # (x,y,w,h)
      """
      boxes = o_boxes[:,[0,2,1,3]] #(x,w,y,h)
      boxes = boxes.reshape((-1,2,2)) #
      boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x,w
      boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y,h
      """

      # boxes # (x1,y1,x2,y2)
      boxes = o_boxes[:, [0, 2, 1, 3]] #(x1,x2,y1,y2)
      boxes = boxes.reshape((-1, 2, 2)) # (x1,x2),(y1,y2)
      boxes[:, 0] = boxes[:, 0] * (neww*1.0/w) # x1,x2
      boxes[:, 1] = boxes[:, 1] * (newh*1.0/h) # y1,y2

      # random horizontal flip
      # no flip for surveilance video?
      if config.flip_image:
        prob = 0.5
        rand = random.random()
        if rand > prob:
          resized_image = cv2.flip(resized_image, 1) # 1 for horizontal
          #boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
          boxes[:, 0] = neww - boxes[:, 0]
          boxes[:, 0, :] = boxes[:, 0, ::-1]# (x_min will be x_max after flip)

      boxes = boxes.reshape((-1, 4))
      boxes = boxes[:, [0, 2, 1, 3]] #(x1,y1,x2,y2)

      # visualize?
      if config.vis_pre:
        label_names = [config.classId_to_class[i] for i in labels]
        o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
        boxes_x1x2 = np.asarray([box for box in boxes])
        ori_vis = draw_boxes(image, o_boxes_x1x2, labels=label_names)
        new_vis = draw_boxes(resized_image, boxes_x1x2, labels=label_names)
        imgname = os.path.splitext(os.path.basename(batch.data["imgs"][0]))[0]
        cv2.imwrite(
            "%s.ori.jpg" % os.path.join(config.vis_path, imgname), ori_vis)
        cv2.imwrite(
            "%s.prepro.jpg" % os.path.join(config.vis_path, imgname), new_vis)
        print("viz saved in %s" % config.vis_path)
        sys.exit()

      # get rpn anchor labels
      # [fs_im,fs_im,num_anchor,4]

      multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(
          resized_image, boxes)

      multilevel_anchor_labels = [l for l, b in multilevel_anchor_inputs]
      multilevel_anchor_boxes = [b for l, b in multilevel_anchor_inputs]
      assert len(multilevel_anchor_labels) == len(multilevel_anchor_boxes) \
          == len(self.anchor_labels) == len(self.anchor_boxes), \
              (len(multilevel_anchor_labels), len(multilevel_anchor_boxes),
               len(self.anchor_labels), len(self.anchor_boxes))

      for pl_labels, pl_boxes, in_labels, in_boxes in zip(
          self.anchor_labels, self.anchor_boxes, multilevel_anchor_labels,
          multilevel_anchor_boxes):

        feed_dict[pl_labels] = in_labels
        feed_dict[pl_boxes] = in_boxes

      assert len(boxes) > 0

      feed_dict[self.gt_boxes] = boxes
      feed_dict[self.gt_labels] = labels

      if config.use_small_object_head:
        for si in range(len(config.small_objects)):
          # the class id in the all classes
          small_object_class_id = config.classname2id[config.small_objects[si]]
          # the box ids
          so_ids = [i for i in range(len(labels))
                    if labels[i] == small_object_class_id]
          # small object label id is different
          # so_label is 0/1, so should be all 1s
          feed_dict[self.so_gt_boxes[si]] = boxes[so_ids, :] # could be empty
          feed_dict[self.so_gt_labels[si]] = [1 for i in range(len(so_ids))]
    else:

      pass

    feed_dict[self.image] = resized_image

    feed_dict[self.is_train] = is_train

    return feed_dict

  def get_feed_dict_forward(self, imgdata):
    feed_dict = {}

    feed_dict[self.image] = imgdata

    feed_dict[self.is_train] = False

    return feed_dict

  # anchor related function for training--------------------

  def filter_box_inside(self, im, boxes):
    h, w = im.shape[:2]
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h)
    )[0]
    return indices, boxes[indices, :]


  # for training, given image and box, get anchor box labels
  # [fs_im,fs_im,num_anchor,4] # not fs,
  def get_rpn_anchor_input(self,im,boxes):


    config = self.config

    boxes = boxes.copy()

    # [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
    all_anchors_np = np.copy(get_all_anchors(
        stride=config.anchor_stride, sizes=config.anchor_sizes,
        ratios=config.anchor_ratios, max_size=config.max_size))

    h, w = im.shape[:2]

    # so image may be smaller than the full anchor size
    #featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
    anchorH, anchorW = all_anchors_np.shape[:2]
    featureh, featurew = anchorH, anchorW

    # [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
    #featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
    #print featuremap_anchors.shape #(46,83,15,4)
    #featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))
    featuremap_anchors_flatten = all_anchors_np.reshape((-1, 4))

    # num_in < FS_im*FS_im*num_anchors # [num_in,4]
    inside_ind, inside_anchors = self.filter_box_inside(
        im, featuremap_anchors_flatten) # the anchor box inside the image


    # anchor labels is in {1,-1,0}, -1 means ignore
    # N = num_in
    # [N], [N,4] # only the fg anchor has box value
    anchor_labels, anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

    # fill back to [fs,fs,num_anchor,4]
    # all anchor outside box is ignored (-1)

    featuremap_labels = -np.ones(
        (featureh * featurew*config.num_anchors,), dtype="int32")
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape(
        (featureh, featurew, config.num_anchors))

    featuremap_boxes = np.zeros(
        (featureh * featurew * config.num_anchors, 4), dtype="float32")
    featuremap_boxes[inside_ind, :] = anchor_boxes
    featuremap_boxes = featuremap_boxes.reshape(
        (featureh, featurew, config.num_anchors, 4))

    return featuremap_labels, featuremap_boxes


  def get_multilevel_rpn_anchor_input(self, im, boxes):

    config = self.config

    boxes = boxes.copy()

    # get anchor for each (anchor_stride,anchor_size) pair
    anchors_per_level = self.get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
    # some image may not be resized to max size, could be shorter edge size
    inside_ind, inside_anchors = self.filter_box_inside(im, all_anchors_flatten)
    # given all these anchors, given the ground truth box, and their iou to
    # each anchor, get the label to be 1 or 0.
    anchor_labels, anchor_gt_boxes = self.get_anchor_labels(
        inside_anchors, boxes)

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors, ), dtype="int32")
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype="float32")
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []

    # put back to list for each level

    for level_anchor in anchors_per_level:
      assert level_anchor.shape[2] == len(config.anchor_ratios)
      anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
      num_anchor_this_level = np.prod(anchor_shape)
      end = start + num_anchor_this_level
      multilevel_inputs.append(
          (all_labels[start: end].reshape(anchor_shape),
           all_boxes[start:end, :].reshape(anchor_shape + (4,))))
      start = end

    assert end == num_all_anchors, \
        ("num all anchors:%s, end:%s" % (num_all_anchors, end))
    return multilevel_inputs




  def get_anchor_labels(self, anchors, gt_boxes):
    config = self.config

    # return max_num of index for labels equal val
    def filter_box_label(labels, val, max_num):
      cur_inds = np.where(labels == val)[0]
      if len(cur_inds) > max_num:
        disable_inds = np.random.choice(
            cur_inds, size=(len(cur_inds) - max_num), replace=False)
        labels[disable_inds] = -1
        cur_inds = np.where(labels == val)[0]
      return cur_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0

    #bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
    #box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
    box_ious = np_iou(anchors, gt_boxes)

    #print box_ious.shape #(37607,7)

    #NA, each anchors max iou to any gt box, and the max gt box"s index [0,NB-1]
    iou_argmax_per_anchor = box_ious.argmax(axis=1)
    iou_max_per_anchor = box_ious.max(axis=1)

    # 1 x NB, each gt box"s max iou to any anchor boxes
    #iou_max_per_gt = box_ious.max(axis=1,keepdims=True)
    #print iou_max_per_gt # all zero?
    iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

    # NA x 1? True for anchors that cover all the gt boxes
    anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

    anchor_labels = -np.ones((NA,), dtype="int32")

    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
    anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

    # cap the number of fg anchor and bg anchor
    target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

    # set the label==1 to -1 if the number exceeds
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

    #assert len(fg_inds) > 0
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
      raise Exception("No valid background for RPN!")

    # the rest of 256 is negative
    target_num_bg = config.rpn_batch_per_im - len(fg_inds)

    # set some label to -1 if exceeds
    filter_box_label(anchor_labels, 0, target_num_bg)

    # only the fg anchor_boxes are filled with the corresponding gt_box
    anchor_boxes = np.zeros((NA, 4), dtype="float32")
    anchor_boxes[fg_inds, :] = gt_boxes[iou_argmax_per_anchor[fg_inds], :]
    return anchor_labels, anchor_boxes


#  given the box, just extract feature for each box
class RCNN_FPN_givenbox():
  def __init__(self, config, gpuid=0):
    self.gpuid = gpuid
    # for batch_norm
    global is_training
    is_training = config.is_train # change this before building model

    assert not config.is_train # only for inferencing

    self.config = config

    self.num_class = config.num_class

    self.global_step = tf.get_variable(
        "global_step", shape=[], dtype="int32",
        initializer=tf.constant_initializer(0), trainable=False)

    # current model get one image at a time
    self.image = tf.placeholder(tf.float32, [None, None, 3], name="image")
    # used for dropout switch
    self.is_train = tf.placeholder("bool", [], name="is_train")

    self.boxes = tf.placeholder(tf.float32, [None, 4], name="boxes")

    # the following will be added in the build_forward and loss
    self.logits = None
    self.yp = None
    self.loss = None

    self.build_preprocess()
    self.build_forward()

  # get feature map anchor and preprocess image
  def build_preprocess(self):
    config = self.config
    image = self.image

    bgr = True  # cv2 load image is bgr
    p_image = tf.expand_dims(image, 0)  # [1,H,W,C]

    with tf.name_scope("image_preprocess"):  # tf.device("/cpu:0"):
      if p_image.dtype.base_dtype != tf.float32:
        p_image = tf.cast(p_image, tf.float32)

      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]

      p_image = p_image*(1.0/255)

      if bgr:
        mean = mean[::-1]
        std = std[::-1]
      image_mean = tf.constant(mean, dtype=tf.float32)
      image_std = tf.constant(std, dtype=tf.float32)
      p_image = (p_image - image_mean) / image_std
      p_image = tf.transpose(p_image, [0, 3, 1, 2])

    self.p_image = p_image

  # based on box sizes
  def fpn_map_rois_to_levels(self, boxes):

    def tf_area(boxes):
      x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
      return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))
    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),# problems with ==?
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]

    level_ids = [tf.reshape(x, [-1], name="roi_level%s_id"%(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name="num_roi_level%s"%(i + 2))
                     for i, x in enumerate(level_ids)]

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes

  # output_shape is the output feature HxW
  def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
    config = self.config
    assert len(features) == 4
    # Reassign rcnn_boxes to levels # based on box area size
    level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
      with tf.name_scope("roi_level%s"%(i + 2)):
        boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])
        all_rois.append(roi_align(featuremap, boxes_on_featuremap, output_shape))

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois

  def build_forward(self):
    config = self.config
    image = self.p_image # [1, C, H, W]
    image_shape2d = tf.shape(image)[2:]
    # the feature map shared by RPN and fast RCNN
    # TODO: fix the batch norm mess
    # TODO: fix global param like data_format and
    # [1,C,FS,FS]

    c2345 = resnet_fpn_backbone(
        image, config.resnet_num_block, use_gn=config.use_gn,
        resolution_requirement=config.fpn_resolution_requirement,
        use_dilations=config.use_dilations,
        use_deformable=config.use_deformable, tf_pad_reverse=True,
        freeze=config.freeze, use_basic_block=config.use_basic_block,
        use_se=config.use_se)

    # include lateral 1x1 conv and final 3x3 conv
    # -> [7, 7, 256]
    p23456 = fpn_model(
        c2345, num_channel=config.fpn_num_channel, use_gn=config.use_gn,
        scope="fpn")

    # here we assume N is not so big that the GPU can handle
    rcnn_boxes = self.boxes # N, 4

    # NxCx7x7 # (?, 256, 7, 7)
    roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4], rcnn_boxes, 7)

    # [N, 256]
    self.final_box_features = tf.reduce_mean(roi_feature_fastrcnn, axis=[2, 3])


  # given the image path, and the label for it
  # preprocess
  def get_feed_dict(self, im, boxes, is_train=False):

    #{"imgs":[],"gt":[]}
    config = self.config
    feed_dict = {}

    feed_dict[self.image] = im
    feed_dict[self.boxes] = boxes

    feed_dict[self.is_train] = is_train

    return feed_dict

class Mask_RCNN_FPN_multi():
  def __init__(self, config, gpuid=0):
    self.gpuid = gpuid
    # for batch_norm
    global is_training
    is_training = config.is_train # change this before building model

    self.config = config

    self.num_class = config.num_class

    self.global_step = tf.get_variable(
        "global_step", shape=[], dtype="int32",
        initializer=tf.constant_initializer(0), trainable=False)

    # current model get one image at a time
    self.image = tf.placeholder(tf.float32, [
        config.im_batch_size, None, None, 3], name="image")

    if not config.is_pack_model:
      self.is_train = tf.placeholder("bool", [], name="is_train")

    # for training
    self.anchor_labels = []
    self.anchor_boxes = []
    num_anchors = len(config.anchor_ratios)
    for k in range(len(config.anchor_strides)):
      self.anchor_labels.append(
          tf.placeholder(tf.int32, [None, None, num_anchors],
                         name="anchor_labels_lvl%s" % (k+2)))
      self.anchor_boxes.append(
          tf.placeholder(tf.float32, [None, None, num_anchors, 4],
                         name="anchor_boxes_lvl%s" % (k+2)))

    # TODO: change to [None, 5] for multi-image batch training
    self.gt_boxes = tf.placeholder(tf.float32, [None, 4], name="gt_boxes")
    self.gt_labels = tf.placeholder(tf.int64, [None, ], name="gt_labels")


    # H,W,v -> {0,1}
    self.gt_mask = tf.placeholder(tf.uint8, [None, None, None], name="gt_masks")

    # the following will be added in the build_forward and loss
    self.logits = None
    self.yp = None
    self.loss = None

    self.build_preprocess()
    self.build_forward()

  # get feature map anchor and preprocess image
  def build_preprocess(self):
    config = self.config
    image = self.image

    # get feature map anchors first
    # slower if put on cpu # 1.5it/s vs 1.2it/s
    self.multilevel_anchors = []
    with tf.name_scope("fpn_anchors"):#,tf.device("/cpu:0"):
      #fm_h,fm_w = tf.shape(image)[0] // config.anchor_stride,tf.shape(image)[1]
      #// config.anchor_stride
      # all posible anchor box coordinates for a given max_size image,
      # so for 1920 x 1920 image, 1920/16 = 120, so (120,120,NA,4) box, NA is
      #scale*ratio boxes
      self.multilevel_anchors = self.get_all_anchors_fpn()

    bgr = True  # cv2 load image is bgr
    #p_image = tf.expand_dims(image, 0)  # [1,H,W,C]
    p_image = image

    with tf.name_scope("image_preprocess"):  # tf.device("/cpu:0"):
      if p_image.dtype.base_dtype != tf.float32:
        p_image = tf.cast(p_image, tf.float32)

      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]

      p_image = p_image * (1.0/255)

      if bgr:
        mean = mean[::-1]
        std = std[::-1]
      image_mean = tf.constant(mean, dtype=tf.float32)
      image_std = tf.constant(std, dtype=tf.float32)
      p_image = (p_image - image_mean) / image_std
      p_image = tf.transpose(p_image, [0, 3, 1, 2])

    self.p_image = p_image

  def build_forward(self):
    config = self.config
    image = self.p_image # [1, C, H, W]
    image_shape2d = tf.shape(image)[2:]
    # a list of numpy anchors, not sliced
    #multilevel_anchors = self.multilevel_anchors

    # the feature map shared by RPN and fast RCNN
    # TODO: fix the batch norm mess
    # [B,C,FS,FS]

    c2345 = resnet_fpn_backbone(
        image, config.resnet_num_block, use_gn=config.use_gn,
        resolution_requirement=config.fpn_resolution_requirement,
        use_dilations=config.use_dilations,
        use_deformable=config.use_deformable, tf_pad_reverse=True,
        freeze=config.freeze, use_basic_block=config.use_basic_block,
        use_se=config.use_se, use_resnext=config.use_resnext)
    # include lateral 1x1 conv and final 3x3 conv
    # -> [B, 256, FS, FS]
    p23456 = fpn_model(c2345, num_channel=config.fpn_num_channel,
                       use_gn=config.use_gn, scope="fpn")

    if config.freeze_rpn or config.freeze_fastrcnn:
      p23456 = [tf.stop_gradient(p) for p in p23456]

    # [1, H, W, channel]
    #self.fpn_feature = tf.image.resize_images(tf.transpose(
    #    p23456[3], perm=[0, 2, 3, 1]), (7, 7)) # p5 # default bilinear

    if config.no_obj_detect: # pair with extract_feat, so only extract feature
      print("no object detect branch..")
      return True
    # given the numpy anchor for each stride,
    # slice the anchor box and label against the feature map size on each
    # level.
    # each level get [FS, FS, num_anchors, 4] anchors
    self.slice_feature_and_anchors(image_shape2d, p23456,
                                   self.multilevel_anchors)

    # num_anchors = config.anchor_ratios
    # each level [B, C, H, W] -> [B, H, W, num_anchors], [B, H, W, num_anchors, 4]
    rpn_outputs = [
        self.rpn_head(pi, config.fpn_num_channel, len(config.anchor_ratios),
                      data_format="NCHW", scope="rpn") for pi in p23456]
    # [B, H, W, num_anchors]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    # [B, H, W, num_anchors, 4]
    multilevel_box_logits = [k[1] for k in rpn_outputs]

    if config.freeze_rpn:
      multilevel_label_logits = [tf.stop_gradient(o)
                                 for o in multilevel_label_logits]
      multilevel_box_logits = [tf.stop_gradient(o)
                               for o in multilevel_box_logits]
    # each H,W location has a box regression and classification score,
    # here combine all positive boxes using NMS
    # [B, K, 4] / [B, K] , K is the number of proposal boxes
    # then they are encoded into [M, 5], with a dimension for batch_idxs
    proposal_boxes, proposal_scores = self.generate_fpn_proposals(
        self.multilevel_anchors, multilevel_label_logits, multilevel_box_logits,
        image_shape2d)


    # for getting RPN performance
    # K depend on rpn_test_post_nms_topk during testing
    # K = 1000

    self.proposal_boxes = proposal_boxes  # [M, 5]
    self.proposal_scores = proposal_scores  # [M]

    if config.is_train:
      # get sliced_anchor_labels
      # now multilevel_anchors are sliced and tf type
      # added sliced gt anchor labels and boxes
      # so we have each fpn level"s anchor boxes, and the ground truth anchor
      # boxes & labels if training

      gt_boxes = self.gt_boxes
      gt_labels = self.gt_labels
      # for training, use gt_box and some proposal box as pos and neg
      # rcnn_sampled_boxes [N_FG+N_NEG,4]
      # fg_inds_wrt_gt -> [N_FG], each is index of gt_boxes
      rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
          proposal_boxes, gt_boxes, gt_labels, config=config)
    else:
      rcnn_boxes = proposal_boxes

    # [M, C, FS, FS], rcnn_boxes: [M, 5]
    # NxCx7x7 # (?, 256, 7, 7)
    roi_feature_fastrcnn = self.multilevel_roi_align(p23456[:4], rcnn_boxes, 7)
    box_batch_idxs, rcnn_boxes = tf.split(rcnn_boxes, [1, 4], axis=1)
    box_batch_idxs = tf.cast(tf.reshape(box_batch_idxs, [-1]), "int32")  # [M]

    if config.use_frcnn_class_agnostic:
      # (N,num_class), (N, 1, 4)
      fastrcnn_label_logits, fastrcnn_box_logits = \
          self.fastrcnn_2fc_head_class_agnostic(
              roi_feature_fastrcnn, config.num_class, scope="fastrcnn")
    else:
      # (N,num_class), (N, num_class - 1, 4), (N, 1024)
      fastrcnn_label_logits, fastrcnn_box_logits, box_embedding = self.fastrcnn_2fc_head(
          roi_feature_fastrcnn, config.num_class, scope="fastrcnn")

    if config.freeze_fastrcnn:
      fastrcnn_label_logits, fastrcnn_box_logits = tf.stop_gradient(
          fastrcnn_label_logits), tf.stop_gradient(fastrcnn_box_logits)

    if config.is_train:
      rpn_label_loss, rpn_box_loss = self.multilevel_rpn_losses(
          self.multilevel_anchors, multilevel_label_logits,
          multilevel_box_logits)

      # rcnn_labels [N_FG + N_NEG] <- index in [N_FG]
      fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])

      # for training, maskRCNN only apply on positive box
      # [N_FG, num_class, 14, 14]

      # [N_FG, 4]
      # sampled boxes are at least iou with a gt_boxes
      fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
      fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits,
                                         fg_inds_wrt_sample)

      # [N_FG, 4] # each proposal box assigned gt box, may repeat
      matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

      # fastrcnn also need to regress box (just the FG box)
      encoded_boxes = encode_bbox_target(matched_gt_boxes, fg_sampled_boxes) * \
          tf.constant(config.fastrcnn_bbox_reg_weights)  # [10,10,5,5]?

      # fastrcnn input is fg and bg proposal box, do classification to
      # num_class(include bg) and then regress on fg boxes
      # [N_FG+N_NEG,4] & [N_FG,4]
      fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_losses(
          rcnn_labels, fastrcnn_label_logits, encoded_boxes,
          fg_fastrcnn_box_logits)

      # ---------------------------------------------------------

      # for debug
      self.rpn_label_loss = rpn_label_loss
      self.rpn_box_loss = rpn_box_loss
      self.fastrcnn_label_loss = fastrcnn_label_loss
      self.fastrcnn_box_loss = fastrcnn_box_loss

      losses = [rpn_label_loss, rpn_box_loss, fastrcnn_label_loss,
                fastrcnn_box_loss]

      # mask rcnn loss
      if config.add_mask:
        fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])
        fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)

        # NxCx14x14
        # only the fg boxes
        roi_feature_fastrcnn = self.multilevel_roi_align(
            p23456[:4], fg_sampled_boxes, 14)

        mask_logits = self.maskrcnn_up4conv_head(
            fg_feature, config.num_class, scope="maskrcnn")


        # [N_FG, H,W]
        gt_mask = self.gt_mask
        gt_mask_for_fg = tf.gather(gt_mask, fg_inds_wrt_gt)
        # [N_FG, H, W] -> [N_FG, 14, 14]
        target_masks_for_fg = crop_and_resize(
            tf.expand_dims(gt_masks, 1),
            fg_sampled_boxes,
            fg_inds_wrt_gt, 28, pad_border=False) # fg x 1x28x28
        target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1)

        mrcnn_loss = self.maskrcnn_loss(mask_logits, fg_labels,
                                        target_masks_for_fg)

        losses += [mrcnn_loss]

      self.wd = None
      if config.wd is not None:
        wd = wd_cost(".*/W", config.wd, scope="wd_cost")
        self.wd = wd
        losses.append(wd)

      self.loss = tf.add_n(losses, "total_loss")

      # l2loss
    else:

      # inferencing
      # K -> proposal box
      # [K,num_class]
      # image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits

      # box_batch_idxs are used


      # get the regressed actual boxes
      if config.use_frcnn_class_agnostic:
        # box regress logits [K, 1, 4], so we tile it to num_class-1 so
        # the rest is the same
        fastrcnn_box_logits = tf.tile(fastrcnn_box_logits,
                                      [1, config.num_class - 1, 1])

      num_class = config.num_class

      # COCO has 81 classes, we only need a few
      # [K, num_class-1, 4] -> # [K, reduced_num_class, 4]
      if config.use_partial_classes:
        needed_object_classids = [config.classname2id[name]
                                  for name in config.partial_classes]
        needed_object_classids_minus_1 = [o - 1 for o in needed_object_classids]
        # (N, num_class), (N, num_class - 1, 4)
        # -> (num_class, N), (num_class - 1, N, 4)
        label_logits_t = tf.transpose(fastrcnn_label_logits, [1, 0])
        box_logits_t = tf.transpose(fastrcnn_box_logits, [1, 0, 2])

        # [C + 1, N]  # 1 is the BG class
        partial_label_logits_t = tf.gather(label_logits_t,
                                           [0] + needed_object_classids)
        # [C, N, 4]
        partial_box_logits_t = tf.gather(box_logits_t,
                                         needed_object_classids_minus_1)

        partial_label_logits = tf.transpose(partial_label_logits_t, [1, 0])
        partial_box_logits = tf.transpose(partial_box_logits_t, [1, 0, 2])

        fastrcnn_label_logits = partial_label_logits
        fastrcnn_box_logits = partial_box_logits

        num_class = len(needed_object_classids) + 1

      # anchor box [K,4] -> [K, num_class - 1, 4] <-
      # box regress logits [K, num_class-1, 4]
      anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, num_class-1, 1])

      # [K, num_class-1, 4]/ [K, 1, 4]
      decoded_boxes = decode_bbox_target(fastrcnn_box_logits / \
          tf.constant(config.fastrcnn_bbox_reg_weights, dtype=tf.float32),
                                         anchors)

      decoded_boxes = clip_boxes(decoded_boxes, image_shape2d,
                                 name="fastrcnn_all_boxes")

      label_probs = tf.nn.softmax(fastrcnn_label_logits)


      # decoded boxes are [K, num_class-1, 4]
      # label_probs [K, num_class]

      # need to assemble the boxes based on each image in the batch

      # [B, num, 4], [B, num], [B, num], [B]
      nms_boxes, nms_scores, nms_classes, valid_indices = \
          self.fastrcnn_predictions_multibatch(
              decoded_boxes, label_probs, box_batch_idxs)


      # old interface
      """
      # [R,5]
      self.final_boxes = final_boxes
      # [R]
      self.final_labels = final_labels
      # add a name so the frozen graph will have that name
      self.final_probs = tf.identity(final_probs, name="final_probs")
      """

      # add a name so the frozen graph will have that name
      # [B, num, 4]
      self.final_boxes = tf.identity(nms_boxes, name="final_boxes")
      # [B, num]
      self.final_labels = tf.identity(nms_classes, name="final_labels")
      # [B, num]
      self.final_probs = tf.identity(nms_scores, name="final_probs")
      # [B]
      self.final_valid_indices = tf.identity(
          valid_indices, name="final_valid_indices")

      # need to reassemble [M, 5] boxes for feature extraction for tracking

      # put the boxes into [M, 5], with 1 dimension encoding the batch idx
      #batch_size = self.final_boxes.get_shape()[0]
      batch_size = self.config.im_batch_size
      batch_idxs = tf.range(batch_size)  # [B]
      num_boxes = tf.shape(self.final_boxes)[1]
      batch_idxs = tf.tile(tf.reshape(batch_idxs, [batch_size, 1, 1]),
                           [1, num_boxes, 1])
      batch_idxs = tf.cast(batch_idxs, "float32")
      # [B, topk, 5]
      boxes = tf.concat([batch_idxs, self.final_boxes], axis=2)
      # [M, 5], [M]
      boxes = tf.reshape(boxes, [-1, 5])
      # remove boxes based on valid indices
      # [B] -> [B, num_boxes] /[4, 100]
      #   clever way to get the mask
      #   [B, 1] > [1, num_boxes]
      valid_indice_trues = tf.expand_dims(
          self.final_valid_indices, axis=1) > \
              tf.expand_dims(tf.range(num_boxes), axis=0)
      valid_indice_trues = tf.reshape(valid_indice_trues, [-1])

      """ debug
      with tf.control_dependencies([tf.print(
          self.final_valid_indices,
          valid_indice_trues, summarize=-1)]):
        valid_indice_trues = tf.identity(valid_indice_trues)
      """
      valid_box_idxs = tf.squeeze(tf.where(valid_indice_trues))
      final_boxes_with_batchidxs = tf.gather(boxes, valid_box_idxs)

      # [M, 256, 7, 7]
      fpn_box_feat = self.multilevel_roi_align(
          p23456[:4], final_boxes_with_batchidxs, 7)
      # TODO: directly get box_embedding from before with nms indices
      #_, _, box_embedding = self.fastrcnn_2fc_head(
      #    fpn_box_feat, config.num_class, scope="fastrcnn")

      #self.fpn_box_feat = tf.identity(box_embedding, name="fpn_box_feat")
      self.fpn_box_feat = tf.identity(fpn_box_feat, name="fpn_box_feat")

      if config.add_mask:

        roi_feature_maskrcnn = self.multilevel_roi_align(
            p23456[:4], final_boxes, 14)

        # [R, num_class - 1, 14, 14]
        mask_logits = self.maskrcnn_up4conv_head(
            roi_feature_maskrcnn, config.num_class, scope="maskrcnn")

        if config.use_partial_classes:
          # need to select the classes as final_labels
          mask_logits_t = tf.transpose(mask_logits, [1, 0, 2, 3])

          # [C, R, 14, 14]
          partial_mask_logits_t = tf.gather(
              mask_logits_t, needed_object_classids)
          # [R, C, 14, 14]
          partial_mask_logits = tf.transpose(
              partial_mask_logits_t, [1, 0, 2, 3])

        indices = tf.stack(
            [tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1],
            axis=1)

        final_mask_logits = tf.gather_nd(mask_logits, indices)
        final_masks = tf.sigmoid(final_mask_logits)

        # [R,14,14]
        self.final_masks = final_masks


  def get_all_anchors_fpn(self):
    config = self.config
    anchors = []
    assert len(config.anchor_strides) == len(config.anchor_sizes)
    for stride, size in zip(config.anchor_strides, config.anchor_sizes):
      anchors_np = get_all_anchors(
          stride=stride, sizes=[size], ratios=config.anchor_ratios,
          max_size=config.max_size)

      anchors.append(anchors_np)
    return anchors

  # make the numpy anchor match to the feature shape
  def slice_feature_and_anchors(self, image_shape2d, p23456, anchors):
    # p2345 each is [B, C, FS, FS]
    # anchors is the numpy anchors for different levels
    config = self.config
    batch_size = p23456[0].get_shape()[0]
    # the anchor labels and boxes are grouped into
    gt_anchor_labels = self.anchor_labels
    gt_anchor_boxes = self.anchor_boxes
    self.sliced_anchor_labels = []
    self.sliced_anchor_boxes = []
    for i, stride in enumerate(config.anchor_strides):
      with tf.name_scope("FPN_slice_lvl%s" % (i)):
        if i < 3:
          # Images are padded for p5, which are too large for p2-p4.
          pi = p23456[i]
          target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * \
              (1.0 / stride)))

          p23456[i] = tf.slice(
              pi, [0, 0, 0, 0], tf.concat([[-1, -1], target_shape], axis=0))
          p23456[i].set_shape([batch_size, pi.shape[1], None, None])

        shape2d = tf.shape(p23456[i])[2:] # h,W
        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)

        anchors[i] = tf.slice(anchors[i], [0, 0, 0, 0], slice4d)

        self.sliced_anchor_labels.append(
            tf.slice(gt_anchor_labels[i], [0, 0, 0], slice3d))
        self.sliced_anchor_boxes.append(tf.slice(
            gt_anchor_boxes[i], [0, 0, 0, 0], slice4d))

  def generate_fpn_proposals(self, multilevel_anchors, multilevel_label_logits,
                             multilevel_box_logits, image_shape2d):

    # multilevel_anchors: each [B, H, W, num_anchors]
    config = self.config
    num_lvl = len(config.anchor_strides)
    assert num_lvl == len(multilevel_anchors)
    assert num_lvl == len(multilevel_box_logits)
    assert num_lvl == len(multilevel_label_logits)
    all_boxes = []
    all_scores = []
    fpn_nms_topk = config.rpn_train_post_nms_topk \
        if config.is_train else config.rpn_test_post_nms_topk

    for lvl in range(num_lvl):
      with tf.name_scope("Lvl%s"%(lvl+2)):
        # numpy array of [FS, FS, num_anchors, 4]
        anchors = multilevel_anchors[lvl]
        # [B, FS, FS, num_anchors, 4]
        pred_boxes_decoded = decode_bbox_target_multi(
            multilevel_box_logits[lvl], anchors,
            decode_clip=config.bbox_decode_clip)

        this_fpn_nms_topk = fpn_nms_topk
        # [B, nms_num, 4], [B, nms_num], [B]
        proposal_boxes, proposal_scores, valid_indices = \
            generate_rpn_proposals_multibatch(
                pred_boxes_decoded,
                multilevel_label_logits[lvl], image_shape2d,
                config, pre_nms_topk=this_fpn_nms_topk)
        all_boxes.append(proposal_boxes)
        all_scores.append(proposal_scores)

    # ignore the valid_indices for now, since the scores are zero padded, we
    # can still remove them with top_k
    # [B, nms_num*lvl, 4]
    proposal_boxes = tf.concat(all_boxes, axis=1)
    batch_size = proposal_boxes.get_shape()[0]
    # [B, nms_num*lvl]
    proposal_scores = tf.concat(all_scores, axis=1)
    proposal_topk = tf.minimum(tf.shape(proposal_scores)[-1], fpn_nms_topk)
    # indices: [B, topk]
    proposal_scores, topk_indices = tf.nn.top_k(proposal_scores,
                                                k=proposal_topk, sorted=False)
    # get [B, num_boxes, 4] + [B, topk] -> [B, topk, 4]
    proposal_boxes = tf.gather(proposal_boxes, topk_indices, batch_dims=1)

    # put the boxes into [M, 5], with 1 dimension encoding the batch idx

    batch_idxs = tf.range(batch_size)  # [B]
    num_boxes = tf.shape(proposal_boxes)[1]
    batch_idxs = tf.tile(tf.reshape(batch_idxs, [batch_size, 1, 1]),
                         [1, num_boxes, 1])
    batch_idxs = tf.cast(batch_idxs, "float32")
    # [B, topk, 5]
    proposal_boxes = tf.concat([batch_idxs, proposal_boxes], axis=2)
    # [M, 5], [M]
    proposal_boxes = tf.reshape(proposal_boxes, [-1, 5])
    proposal_scores = tf.reshape(proposal_scores, [-1])
    # remove zero area boxes?
    area = self.tf_area(proposal_boxes)
    not_padded_box_idxs = tf.squeeze(tf.where(area > 0.))
    proposal_boxes = tf.gather(proposal_boxes, not_padded_box_idxs)
    proposal_scores = tf.gather(proposal_scores, not_padded_box_idxs)
    return tf.stop_gradient(proposal_boxes, name="boxes"), \
           tf.stop_gradient(proposal_scores, name="scores")

  def tf_area(self, boxes):
    # given [M, 5], get [M]
    # [M, 1]
    _, x_min, y_min, x_max, y_max = tf.split(boxes, 5, axis=1)
    # [M]
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

  # based on box sizes
  def fpn_map_rois_to_levels(self, boxes):

    # boxes: [K, 5] -> each level got [M, 5]
    # [M]
    sqrtarea = tf.sqrt(self.tf_area(boxes))
    level = tf.to_int32(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * \
        (1.0 / np.log(2))))

    # RoI levels range from 2~5 (not 6)\
    # [M, 1] for [K] shape level
    # [M, 2] for [B, K] level
    level_idxs = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]

    level_idxs = [tf.reshape(x, [-1], name="roi_level%s_id" % (i + 2))
                  for i, x in enumerate(level_idxs)]

    # for each level, get [M*, 5] boxes
    level_boxes = [tf.gather(boxes, ids) for ids in level_idxs]
    return level_idxs, level_boxes


  # output_shape is the output feature HxW
  def multilevel_roi_align(self, features, rcnn_boxes, output_shape):
    # features, each [B, C, H, W]
    # rcnn_boxes [M, 5]
    config = self.config
    assert len(features) == 4
    # Reassign rcnn_boxes to levels # based on box area size
    # [M*] and [M*, 5] for each level
    level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
      with tf.name_scope("roi_level%s" % (i + 2)):
        box_idxs, boxes = tf.split(boxes, [1, 4], axis=1)
        box_idxs = tf.reshape(box_idxs, [-1])
        box_idxs = tf.cast(box_idxs, "int32") # [M]
        boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i])

        all_rois.append(
            roi_align_multi(featuremap, boxes_on_featuremap, box_idxs, output_shape))

    # this can fail if using TF<=1.8 with MKL build
    # [M, C, FS, FS]
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    # all_rois -> rcnn_boxes same order
    return all_rois

  # ----some model component
  # feature map -> [B, C, FS1, FS2] , FS1 = H/16.0, FS2 = W/16.0
  def rpn_head(self, featuremap, channel, num_anchors, data_format,
               scope="rpn"):
    with tf.variable_scope(scope):
      # [B, channel, FS1, FS2] # channel
      # conv0:W -> [3, 3, 1024, 1024]
      #batch_size = featuremap.get_shape()[0]
      batch_size = self.config.im_batch_size
      h = conv2d(
          featuremap, channel, kernel=3, activation=tf.nn.relu,
          data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="conv0")
      # h -> [B,channel,FS1,FS2]

      # 1x1 kernel conv to classification on each grid
      # [B, 1024, FS1, FS2] -> # [B, num_anchors, FS1, FS2]
      label_logits = conv2d(
          h, num_anchors, kernel=1, data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="class")
      # [B, 1024, FS1, FS2] -> # [1, 4 * num_anchors, FS1, FS2]
      box_logits = conv2d(
          h, 4*num_anchors, kernel=1, data_format=data_format,
          W_init=tf.random_normal_initializer(stddev=0.01), scope="box")

      # [B, num_anchors, FS1, FS2] -> [B, FS1, FS2, num_anchors]
      # it used to be [FS1, FS2, num_anchors]
      label_logits = tf.transpose(label_logits, [0, 2, 3, 1])

      box_shape = tf.shape(box_logits)
      # [B, FS1, FS2, num_anchors*4]
      box_logits = tf.transpose(box_logits, [0, 2, 3, 1])
      # [B, FS1, FS2, num_anchors, 4]
      # it used to be [FS1, FS2, num_anchors, 4]
      box_logits = tf.reshape(
          box_logits,
          [batch_size, box_shape[2], box_shape[3], num_anchors, 4])

      return label_logits, box_logits

  def small_object_classification_head(
      self, feature, num_class, scope="small_object_classification"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):
      hidden = dense(
          feature, dim, W_init=initializer, activation=tf.nn.relu, scope="fc6")
      hidden = dense(
          hidden, dim, W_init=initializer, activation=tf.nn.relu, scope="fc7")

      classification = dense(
          hidden, num_class, W_init=tf.random_normal_initializer(stddev=0.01),
          scope="class") # [K,num_class]

    return classification

  # feature: [K,C,7,7] # feature for each roi
  def fastrcnn_2fc_head(
      self, feature, num_class=None, boxes=None, scope="fastrcnn_head"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):

      if config.use_conv_frcnn_head:
        hidden = self.conv_frcnn_head(
            feature, dim, config.conv_frcnn_head_dim, num_conv=4,
            use_gn=config.use_gn)
      else:
        # dense will reshape to [k,C*7*7] first
        if config.add_relation_nn:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
        else:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")

      # hidden -> [K, dim]
      if config.use_att_frcnn_head:
        # changes: 1. conv2d kernel size; 2. softmax/sigmoid;
        # 3. sum or gating?; 4. convert to dim first then attention?;
        # 5. attend then two fc, no use of previous hidden
        # [K, 7, 7, C]
        feature = tf.transpose(feature, perm=[0, 2, 3, 1])
        H, W, feat_dim = feature.get_shape()[1:]
        # 1. simple conv attention
        # [K, 7, 7, 1]
        attention = conv2d(
            feature, 1, kernel=3, padding="SAME", stride=1,
            activation=tf.nn.softmax, use_bias=True, data_format="NHWC",
            W_init=initializer, scope="attention")
        # [K,7*7, C]
        feature = tf.reshape(feature, [-1, H*W, feat_dim])
        attention = tf.reshape(attention, [-1, H*W, 1])
        # [K, C]
        attended = tf.reduce_sum(feature * attention, 1)

        # match the dimension
        attended_feat = dense(
            attended, dim, W_init=initializer, activation=tf.nn.relu,
            scope="att_trans")

        # sum with original feature
        hidden = hidden + attended_feat

      with tf.variable_scope("outputs"):

        classification = dense(
            hidden, num_class, W_init=tf.random_normal_initializer(stddev=0.01),
            scope="class") # [K,num_class]


        box_regression = dense(
            hidden, num_class*4,
            W_init=tf.random_normal_initializer(stddev=0.001),
            scope="box")
        box_regression = tf.reshape(box_regression, (-1, num_class, 4))

        box_regression = box_regression[:, 1:, :]

        box_regression.set_shape([None, num_class-1, 4])

    return classification, box_regression, hidden

  def conv_frcnn_head(self, feature, fc_dim, conv_dim, num_conv, use_gn=False):
    l = feature
    for k in range(num_conv):
      l = conv2d(
          l, conv_dim, kernel=3, activation=tf.nn.relu,
          data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="truncated_normal"),
          scope="conv%s" % (k))
      if use_gn:
        l = group_norm(l, scope="gn%s" % (k))
    l = dense(
        l, fc_dim, W_init=tf.variance_scaling_initializer(),
        activation=tf.nn.relu, scope="fc")
    return l

  def fastrcnn_2fc_head_class_agnostic(
      self, feature, num_class, boxes=None, scope="head"):
    config = self.config
    dim = config.fpn_frcnn_fc_head_dim # 1024
    initializer = tf.variance_scaling_initializer()

    with tf.variable_scope(scope):
      if config.use_conv_frcnn_head:
        hidden = self.conv_frcnn_head(
            feature, dim, config.conv_frcnn_head_dim, num_conv=4,
            use_gn=config.use_gn)
      else:
        # dense will reshape to [k,C*7*7] first
        if config.add_relation_nn:
          hidden = dense(
              feature, dim, W_init=initializer,
              activation=tf.nn.relu, scope="fc6")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r1")
          hidden = dense(hidden, dim, W_init=initializer,
                         activation=tf.nn.relu, scope="fc7")
          hidden = hidden + relation_network(
              hidden, boxes, group=16, geo_feat_dim=64, scope="RM_r2")
        else:
          hidden = dense(
              feature, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc6")
          hidden = dense(
              hidden, dim, W_init=initializer, activation=tf.nn.relu,
              scope="fc7")


    with tf.variable_scope("outputs"):

      classification = dense(
          hidden, num_class,
          W_init=tf.random_normal_initializer(stddev=0.01),
          scope="class") # [K,num_class]
      num_class = 1 # just for box
      box_regression = dense(
          hidden, num_class*4,
          W_init=tf.random_normal_initializer(stddev=0.001), scope="box")
      box_regression = tf.reshape(box_regression, (-1, num_class, 4))

    return classification, box_regression


  def maskrcnn_up4conv_head(self, feature, num_class, scope="maskrcnn_head"):
    # feature [R, 256, 7, 7]
    config = self.config
    num_conv = 4 # C4 model this is 0
    l = feature
    with tf.variable_scope(scope):
      for k in range(num_conv):
        l = conv2d(
            l, config.mrcnn_head_dim, kernel=3, activation=tf.nn.relu,
            data_format="NCHW",
            W_init=tf.variance_scaling_initializer(
                scale=2.0, mode="fan_out", distribution="truncated_normal"),
            scope="fcn%s"%(k))

      l = deconv2d(
          l, config.mrcnn_head_dim, kernel=2, stride=2, activation=tf.nn.relu,
          data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="truncated_normal"),
          scope="deconv")
      # [R, num_class-1, 14, 14]
      l = conv2d(
          l, num_class - 1, kernel=1, data_format="NCHW",
          W_init=tf.variance_scaling_initializer(
              scale=2.0, mode="fan_out", distribution="normal"),
          scope="conv")
      return l


  def nms_return_masks(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # [K]
    ids = tf.reshape(tf.where(prob > config.result_score_thres), [-1])
    prob_ = tf.gather(prob, ids)
    box_ = tf.gather(box, ids)
    # NMS
    selection = tf.image.non_max_suppression(
        box_, prob_, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    selection = tf.to_int32(tf.gather(ids, selection))
    sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

    mask = tf.sparse_to_dense(
        sparse_indices=sorted_selection,
        output_shape=output_shape,
        sparse_values=True,
        default_value=False)

    return mask
  def nms_return_masks_no_score_filter(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # NMS
    selection = tf.image.non_max_suppression(
        box, prob, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]

    mask = tf.sparse_to_dense(
        sparse_indices=sorted_selection, output_shape=output_shape,
        sparse_values=True, default_value=False)

    return mask

  def nms_return_boxes(self, X):
    config = self.config
    prob, box = X # [K], [K,4]
    output_shape = tf.shape(prob)
    # NMS
    selection = tf.image.non_max_suppression(
        box, prob, max_output_size=config.result_per_im,
        iou_threshold=config.fastrcnn_nms_iou_thres)
    selected_prob = tf.gather(prob, selection)
    selected_box = tf.gather(box, selection)
    return selected_box, selected_prob

  # given all proposal box prediction, based on score thres , get final
  # NMS resulting box
  # [K,num_class-1,4] -> decoded_boxes
  # [K,num_class] label_probs
  # each proposal box has prob and box to all class
  # here using nms for each class, -> [R]
  def fastrcnn_predictions(self, boxes, probs, no_score_filter=False,
                           scope="fastrcnn_predictions"):
    with tf.variable_scope(scope):
      config = self.config

      if config.use_bg_score: # use the BG score to filter out boxes
        # probs: [K, num_class]
        box_classes = tf.argmax(probs, axis=1) # [K]
        # [N]
        nonBG_box_indices = tf.reshape(
            tf.where(tf.greater(box_classes, 0)), [-1])
        probs = tf.gather(probs, nonBG_box_indices)
        boxes = tf.gather(boxes, nonBG_box_indices)

      # note if use partial class, config.num_class is not the
      # actual num_class here

      # transpose to map_fn along each class
      boxes = tf.transpose(boxes, [1, 0, 2])  # [num_class-1, K,4]
      probs = tf.transpose(probs[:, 1:], [1, 0])  # [num_class-1, K]

      # for each catagory get the top K
      # [num_class-1, K]
      if no_score_filter:
        masks = tf.map_fn(
            self.nms_return_masks_no_score_filter, (probs, boxes),
            dtype=tf.bool, parallel_iterations=10)
      else:
        masks = tf.map_fn(
            self.nms_return_masks, (probs, boxes), dtype=tf.bool,
            parallel_iterations=10)
      # [R*(num_class-1),2], each entry is [cat_id,box_id]
      selected_indices = tf.where(masks)

      # [num_class-1, K] -> [R*(num_class-1)]
      probs = tf.boolean_mask(probs, masks)
      # topk_indices [R]
      topk_probs, topk_indices = tf.nn.top_k(
          probs, tf.minimum(config.result_per_im, tf.size(probs)), sorted=False)

      # [K,2] <- select [act_num,R]
      filtered_selection = tf.gather(selected_indices, topk_indices)
      filtered_selection = tf.reverse(
          filtered_selection, axis=[1], name="filtered")

      # [R,2], [R,]
      return filtered_selection, topk_probs

  def fastrcnn_predictions_multibatch(self, boxes, probs, box_batch_idxs,
                                      scope="fastrcnn_predictions"):
    with tf.variable_scope(scope):
      config = self.config

      # note if use partial class, config.num_class is not the
      # actual num_class here

      # boxes [K, num_class-1, 4]
      # probs [K, num_class]
      # box_batch_idxs [K]
      # need [K, 2] indexes to send to [B, K, cat_num , 4]

      probs = probs[:, 1:] # [K, num_class - 1]
      K, cat_num = tf.shape(boxes)[0], tf.shape(boxes)[1]
      batch_size = config.im_batch_size

      # gather and pad to [B, num_box, cat_num, 4], [B, num_box, cat_num]
      # [K, 2]
      indices = tf.stack([box_batch_idxs, tf.range(K)], axis=1)

      # TODO: fix this, there are a lot of zero area boxes
      padded_boxes = tf.scatter_nd(indices, boxes, (batch_size, K, cat_num, 4))
      padded_probs = tf.scatter_nd(indices, probs, (batch_size, K, cat_num))

      # change to y1x1y2x2
      boxes_x1y1x2y2 = tf.reshape(padded_boxes, (batch_size, K, cat_num, 2, 2))
      boxes_y1x1y2x2 = tf.reshape(
          tf.reverse(boxes_x1y1x2y2, axis=[4]), (batch_size, K, cat_num, 4),
          name="fastrcnn_nms_input_boxes")

      # inputs: [B, valid, cat_num, 4], [B, valid, cat_num]
      # [B, nms_num, 4], [B, nms_num]
      # valid_indices: [B], each is an int, top k entries in the first 3 are valid
      nms_boxes, nms_scores, nms_classes, valid_indices = \
          tf.image.combined_non_max_suppression(
              boxes_y1x1y2x2, padded_probs,
              max_output_size_per_class=config.result_per_im,
              max_total_size=config.result_per_im,  # over all class
              iou_threshold=config.fastrcnn_nms_iou_thres,
              clip_boxes=False,
              pad_per_class=False)

      # boxes are y1x1y2x2
      nms_boxes_y1x1y2x2 = tf.reshape(nms_boxes, (batch_size, -1, 2, 2))
      nms_boxes_x1y1x2y2 = tf.reverse(nms_boxes_y1x1y2x2, axis=[3])
      # [B, nms_num, 4]
      nms_boxes_x1y1x2y2 = tf.reshape(nms_boxes_x1y1x2y2, (batch_size, -1, 4))

      nms_classes = nms_classes + 1

      # [B, num, 4], [B, num], [B, num], [B]
      return nms_boxes_x1y1x2y2, nms_scores, nms_classes, valid_indices

  # ---- losses
  def maskrcnn_loss(self, mask_logits, fg_labels, fg_target_masks,
                    scope="maskrcnn_loss"):
    with tf.variable_scope(scope):
      # mask_logits: [N_FG, num_cat, 14, 14]
      # fg_labels: [N_FG]
      # fg_target_masks: [N_FG, 14, 14]
      num_fg = tf.size(fg_labels)
      # [N_FG, 2] # these index is used to get the pos cat"s logit
      indices = tf.stack([tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)
      # ignore other class"s logit
      # [N_FG, 14, 14]
      mask_logits = tf.gather_nd(mask_logits, indices)
      mask_probs = tf.sigmoid(mask_logits)

      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=fg_target_masks, logits=mask_logits)
      loss = tf.reduce_mean(loss, name="maskrcnn_loss")

      return loss


  def multilevel_rpn_losses(self, multilevel_anchors, multilevel_label_logits,
                            multilevel_box_logits, scope="rpn_losses"):
    config = self.config
    sliced_anchor_labels = self.sliced_anchor_labels
    sliced_anchor_boxes = self.sliced_anchor_boxes

    num_lvl = len(config.anchor_strides)
    assert num_lvl == len(multilevel_label_logits)
    assert num_lvl == len(multilevel_box_logits)
    assert num_lvl == len(multilevel_anchors)

    losses = []
    with tf.variable_scope(scope):
      for lvl in range(num_lvl):
        anchors = multilevel_anchors[lvl]
        gt_labels = sliced_anchor_labels[lvl]
        gt_boxes = sliced_anchor_boxes[lvl]

        # get the ground truth T_xywh
        encoded_gt_boxes = encode_bbox_target(gt_boxes, anchors)

        label_loss, box_loss = self.rpn_losses(
            gt_labels, encoded_gt_boxes, multilevel_label_logits[lvl],
            multilevel_box_logits[lvl], scope="level%s" % (lvl+2))
        losses.extend([label_loss, box_loss])

      total_label_loss = tf.add_n(losses[::2], name="label_loss")
      total_box_loss = tf.add_n(losses[1::2], name="box_loss")

    return total_label_loss, total_box_loss


  def rpn_losses(self, anchor_labels, anchor_boxes, label_logits,
                 box_logits, scope="rpn_losses"):
    config = self.config
    with tf.variable_scope(scope):
      # anchor_label ~ {-1,0,1} , -1 means ignore, , 0 neg, 1 pos
      # label_logits [FS,FS,num_anchors]
      # box_logits [FS,FS,num_anchors,4]

      #with tf.device("/cpu:0"):
      # 1,0|pos/neg
      valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
      pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
      nr_valid = tf.stop_gradient(
          tf.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
      nr_pos = tf.identity(
          tf.count_nonzero(pos_mask, dtype=tf.int32), name="num_pos_anchor")

      # [nr_valid]
      valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)

      # [nr_valid]
      valid_label_logits = tf.boolean_mask(label_logits, valid_mask)


      placeholder = 0.

      # label loss for all valid anchor box
      if config.focal_loss:
        valid_label_logits = tf.reshape(valid_label_logits, [-1, 1])
        valid_anchor_labels = tf.reshape(valid_anchor_labels, [-1, 1])
        label_loss = focal_loss(
            logits=valid_label_logits, labels=tf.to_float(valid_anchor_labels))
      else:
        label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=valid_label_logits, labels=tf.to_float(valid_anchor_labels))

        label_loss = tf.reduce_sum(label_loss) * (1. / config.rpn_batch_per_im)

      label_loss = tf.where(
          tf.equal(nr_valid, 0), placeholder, label_loss, name="label_loss")

      # box loss for positive anchor
      pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
      pos_box_logits = tf.boolean_mask(box_logits, pos_mask)

      delta = 1.0/9

      # the smooth l1 loss
      box_loss = tf.losses.huber_loss(
          pos_anchor_boxes, pos_box_logits, delta=delta,
          reduction=tf.losses.Reduction.SUM) / delta

      box_loss = box_loss * (1. / config.rpn_batch_per_im)
      box_loss = tf.where(
          tf.equal(nr_pos, 0), placeholder, box_loss, name="box_loss")

      return label_loss, box_loss

  def fastrcnn_losses(self, labels, label_logits, fg_boxes, fg_box_logits,
                      scope="fastrcnn_losses"):
    config = self.config
    with tf.variable_scope(scope):
      # label -> label for roi [N_FG + N_NEG], the fg labels are 1-num_class,
      # 0 is bg
      # label_logits [N_FG + N_NEG,num_class]
      # fg_boxes_logits -> [N_FG,num_class-1,4]

      # so the label is int [0-num_class], 0 being background

      if config.focal_loss:
        # [N, num_classes]
        onehot_label = tf.one_hot(labels, label_logits.get_shape()[-1])

        # here uses sigmoid
        label_loss = focal_loss(
            logits=label_logits, labels=tf.to_float(onehot_label))
      else:
        label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=label_logits)

        label_loss = tf.reduce_mean(label_loss, name="label_loss")

      fg_inds = tf.where(labels > 0)[:, 0]
      fg_labels = tf.gather(labels, fg_inds) # [N_FG]

      num_fg = tf.size(fg_inds) # N_FG
      if int(fg_box_logits.shape[1]) > 1:
        # [N_FG, 2]
        indices = tf.stack(
            [tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)
        # gather the logits from [N_FG,num_class-1, 4] to [N_FG,4],
        # only the gt class"s logit
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)
      else:
        # class agnostic for cascade rcnn
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])
      box_loss = tf.losses.huber_loss(
          fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)

      # /  N_FG + N_NEG ?
      box_loss = tf.truediv(
          box_loss, tf.to_float(tf.shape(labels)[0]), name="box_loss")

      return label_loss, box_loss


  # given the image path, and the label for it
  # preprocess
  def get_feed_dict(self, batch, is_train=False):

    #{"imgs":[],"gt":[]}
    config = self.config

    N = len(batch.data["imgs"])

    assert N == 1 # only 1 image for now

    feed_dict = {}

    if "imgdata" in batch.data:
      image = batch.data["imgdata"][0]
    else:
      image = batch.data["imgs"][0]
      if config.use_mixup:
        img1, img2 = image
        use_mixup = random.random() <= config.mixup_chance
        if use_mixup:
          weight = batch.data["mixup_weights"][0]
          img1 = Image.open(img1)
          img2 = Image.open(img2)
          trans_alpha = int(255.0*weight)
          for mixup_box in batch.data["gt"][0]["mixup_boxes"]:
            box_img = img2.crop(mixup_box)
            box_img_sizes = [int(a) for a in box_img.size[::-1]]
            # unit8 and "L" are needed
            mask = Image.fromarray(
                np.zeros(box_img_sizes, dtype="uint8") + trans_alpha, mode="L")
            img1.paste(box_img, mixup_box, mask=mask)
          # PIL to cv2 image
          img1 = np.array(img1)
          img1 = img1[:, :, ::-1].copy()

          # now add the annotation
          batch.data["gt"][0]["boxes"] = np.concatenate(
              [batch.data["gt"][0]["boxes"],
               batch.data["gt"][0]["mixup_boxes"]],
              axis=0)
          batch.data["gt"][0]["labels"].extend(
              batch.data["gt"][0]["mixup_labels"])

          image = img1
        else:
          image = cv2.imread(img1, cv2.IMREAD_COLOR)
      else:
        image = cv2.imread(image, cv2.IMREAD_COLOR)

      assert image is not None, image
      image = image.astype("float32")
    h, w = image.shape[:2] # original width/height

    # resize image, boxes
    short_edge_size = config.short_edge_size
    if config.scale_jitter and is_train:
      short_edge_size = random.randint(
          config.short_edge_size_min, config.short_edge_size_max)


    if "resized_image" in batch.data:
      resized_image = batch.data["resized_image"][0]
    else:
      resized_image = resizeImage(image, short_edge_size, config.max_size)
    newh, neww = resized_image.shape[:2]
    #print newh,neww, batch.data["imgs"][0]
    #sys.exit()

    if is_train:
      anno = batch.data["gt"][0] # "boxes" -> [K,4], "labels" -> [K]
      # now the box is in [x1,y1,x2,y2] format, not coco box
      o_boxes = anno["boxes"]
      labels = anno["labels"]
      assert len(labels) == len(o_boxes)

      # boxes # (x,y,w,h)
      """
      boxes = o_boxes[:,[0,2,1,3]] #(x,w,y,h)
      boxes = boxes.reshape((-1,2,2)) #
      boxes[:,0] = boxes[:,0] * (neww*1.0/w) # x,w
      boxes[:,1] = boxes[:,1] * (newh*1.0/h) # y,h
      """

      # boxes # (x1,y1,x2,y2)
      boxes = o_boxes[:, [0, 2, 1, 3]] #(x1,x2,y1,y2)
      boxes = boxes.reshape((-1, 2, 2)) # (x1,x2),(y1,y2)
      boxes[:, 0] = boxes[:, 0] * (neww*1.0/w) # x1,x2
      boxes[:, 1] = boxes[:, 1] * (newh*1.0/h) # y1,y2

      # random horizontal flip
      # no flip for surveilance video?
      if config.flip_image:
        prob = 0.5
        rand = random.random()
        if rand > prob:
          resized_image = cv2.flip(resized_image, 1) # 1 for horizontal
          #boxes[:,0,0] = neww - boxes[:,0,0] - boxes[:,0,1] # for (x,y,w,h)
          boxes[:, 0] = neww - boxes[:, 0]
          boxes[:, 0, :] = boxes[:, 0, ::-1]# (x_min will be x_max after flip)

      boxes = boxes.reshape((-1, 4))
      boxes = boxes[:, [0, 2, 1, 3]] #(x1,y1,x2,y2)

      # visualize?
      if config.vis_pre:
        label_names = [config.classId_to_class[i] for i in labels]
        o_boxes_x1x2 = np.asarray([box_wh_to_x1x2(box) for box in o_boxes])
        boxes_x1x2 = np.asarray([box for box in boxes])
        ori_vis = draw_boxes(image, o_boxes_x1x2, labels=label_names)
        new_vis = draw_boxes(resized_image, boxes_x1x2, labels=label_names)
        imgname = os.path.splitext(os.path.basename(batch.data["imgs"][0]))[0]
        cv2.imwrite(
            "%s.ori.jpg" % os.path.join(config.vis_path, imgname), ori_vis)
        cv2.imwrite(
            "%s.prepro.jpg" % os.path.join(config.vis_path, imgname), new_vis)
        print("viz saved in %s" % config.vis_path)
        sys.exit()

      # get rpn anchor labels
      # [fs_im,fs_im,num_anchor,4]

      multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(
          resized_image, boxes)

      multilevel_anchor_labels = [l for l, b in multilevel_anchor_inputs]
      multilevel_anchor_boxes = [b for l, b in multilevel_anchor_inputs]
      assert len(multilevel_anchor_labels) == len(multilevel_anchor_boxes) \
          == len(self.anchor_labels) == len(self.anchor_boxes), \
              (len(multilevel_anchor_labels), len(multilevel_anchor_boxes),
               len(self.anchor_labels), len(self.anchor_boxes))

      for pl_labels, pl_boxes, in_labels, in_boxes in zip(
          self.anchor_labels, self.anchor_boxes, multilevel_anchor_labels,
          multilevel_anchor_boxes):

        feed_dict[pl_labels] = in_labels
        feed_dict[pl_boxes] = in_boxes

      assert len(boxes) > 0

      feed_dict[self.gt_boxes] = boxes
      feed_dict[self.gt_labels] = labels

    else:

      pass

    feed_dict[self.image] = resized_image

    feed_dict[self.is_train] = is_train

    return feed_dict

  def get_feed_dict_forward(self, imgdata):
    feed_dict = {}

    feed_dict[self.image] = imgdata

    feed_dict[self.is_train] = False

    return feed_dict

  def get_feed_dict_forward_multi(self, imgs):
    # imgs: a list of [H, W, 3]
    feed_dict = {}

    # [B, H, W, 3]
    feed_dict[self.image] = np.stack(imgs, axis=0)

    feed_dict[self.is_train] = False

    return feed_dict

  # anchor related function for training--------------------

  def filter_box_inside(self, im, boxes):
    h, w = im.shape[:2]
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h)
    )[0]
    return indices, boxes[indices, :]


  # for training, given image and box, get anchor box labels
  # [fs_im,fs_im,num_anchor,4] # not fs,
  def get_rpn_anchor_input(self, im, boxes):


    config = self.config

    boxes = boxes.copy()

    # [FS,FS,num_anchor,4] all possible anchor boxes given the max image size
    all_anchors_np = np.copy(get_all_anchors(
        stride=config.anchor_stride, sizes=config.anchor_sizes,
        ratios=config.anchor_ratios, max_size=config.max_size))

    h, w = im.shape[:2]

    # so image may be smaller than the full anchor size
    #featureh,featurew = h//config.anchor_stride,w//config.anchor_stride
    anchorH, anchorW = all_anchors_np.shape[:2]
    featureh, featurew = anchorH, anchorW

    # [FS_im,FS_im,num_anchors,4] # the anchor field that the image is included
    #featuremap_anchors = all_anchors_np[:featureh,:featurew,:,:]
    #print featuremap_anchors.shape #(46,83,15,4)
    #featuremap_anchors_flatten = featuremap_anchors.reshape((-1,4))
    featuremap_anchors_flatten = all_anchors_np.reshape((-1, 4))

    # num_in < FS_im*FS_im*num_anchors # [num_in,4]
    inside_ind, inside_anchors = self.filter_box_inside(
        im, featuremap_anchors_flatten) # the anchor box inside the image


    # anchor labels is in {1,-1,0}, -1 means ignore
    # N = num_in
    # [N], [N,4] # only the fg anchor has box value
    anchor_labels, anchor_boxes = self.get_anchor_labels(inside_anchors, boxes)

    # fill back to [fs,fs,num_anchor,4]
    # all anchor outside box is ignored (-1)

    featuremap_labels = -np.ones(
        (featureh * featurew*config.num_anchors,), dtype="int32")
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape(
        (featureh, featurew, config.num_anchors))

    featuremap_boxes = np.zeros(
        (featureh * featurew * config.num_anchors, 4), dtype="float32")
    featuremap_boxes[inside_ind, :] = anchor_boxes
    featuremap_boxes = featuremap_boxes.reshape(
        (featureh, featurew, config.num_anchors, 4))

    return featuremap_labels, featuremap_boxes


  def get_multilevel_rpn_anchor_input(self, im, boxes):

    config = self.config

    boxes = boxes.copy()

    # get anchor for each (anchor_stride,anchor_size) pair
    anchors_per_level = self.get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
    # some image may not be resized to max size, could be shorter edge size
    inside_ind, inside_anchors = self.filter_box_inside(im, all_anchors_flatten)
    # given all these anchors, given the ground truth box, and their iou to
    # each anchor, get the label to be 1 or 0.
    anchor_labels, anchor_gt_boxes = self.get_anchor_labels(
        inside_anchors, boxes)

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors, ), dtype="int32")
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype="float32")
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []

    # put back to list for each level

    for level_anchor in anchors_per_level:
      assert level_anchor.shape[2] == len(config.anchor_ratios)
      anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
      num_anchor_this_level = np.prod(anchor_shape)
      end = start + num_anchor_this_level
      multilevel_inputs.append(
          (all_labels[start: end].reshape(anchor_shape),
           all_boxes[start:end, :].reshape(anchor_shape + (4,))))
      start = end

    assert end == num_all_anchors, \
        ("num all anchors:%s, end:%s" % (num_all_anchors, end))
    return multilevel_inputs




  def get_anchor_labels(self, anchors, gt_boxes):
    config = self.config

    # return max_num of index for labels equal val
    def filter_box_label(labels, val, max_num):
      cur_inds = np.where(labels == val)[0]
      if len(cur_inds) > max_num:
        disable_inds = np.random.choice(
            cur_inds, size=(len(cur_inds) - max_num), replace=False)
        labels[disable_inds] = -1
        cur_inds = np.where(labels == val)[0]
      return cur_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0

    #bbox_iou_float = get_iou_callable() # tf op on cpu, nn.py
    #box_ious = bbox_iou_float(anchors,gt_boxes) #[NA,NB]
    box_ious = np_iou(anchors, gt_boxes)

    #print box_ious.shape #(37607,7)

    #NA, each anchors max iou to any gt box, and the max gt box"s index [0,NB-1]
    iou_argmax_per_anchor = box_ious.argmax(axis=1)
    iou_max_per_anchor = box_ious.max(axis=1)

    # 1 x NB, each gt box"s max iou to any anchor boxes
    #iou_max_per_gt = box_ious.max(axis=1,keepdims=True)
    #print iou_max_per_gt # all zero?
    iou_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB

    # NA x 1? True for anchors that cover all the gt boxes
    anchors_with_max_iou_per_gt = np.where(box_ious == iou_max_per_gt)[0]

    anchor_labels = -np.ones((NA,), dtype="int32")

    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[iou_max_per_anchor >= config.positive_anchor_thres] = 1
    anchor_labels[iou_max_per_anchor < config.negative_anchor_thres] = 0

    # cap the number of fg anchor and bg anchor
    target_num_fg = int(config.rpn_batch_per_im * config.rpn_fg_ratio)

    # set the label==1 to -1 if the number exceeds
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)

    #assert len(fg_inds) > 0
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
      raise Exception("No valid background for RPN!")

    # the rest of 256 is negative
    target_num_bg = config.rpn_batch_per_im - len(fg_inds)

    # set some label to -1 if exceeds
    filter_box_label(anchor_labels, 0, target_num_bg)

    # only the fg anchor_boxes are filled with the corresponding gt_box
    anchor_boxes = np.zeros((NA, 4), dtype="float32")
    anchor_boxes[fg_inds, :] = gt_boxes[iou_argmax_per_anchor[fg_inds], :]
    return anchor_labels, anchor_boxes


def initialize(load,load_best,config,sess):
  tf.global_variables_initializer().run()
  if load:
    print("restoring model...")
    allvars = tf.global_variables()
    allvars = [var for var in allvars if "global_step" not in var.name]
    #restore_vars = allvars
    opts = ["Adam", "beta1_power", "beta1_power_1", "beta2_power",
            "beta2_power_1", "Adam_1", "Adadelta_1", "Adadelta", "Momentum"]


    allvars = [var for var in allvars
               if var.name.split(":")[0].split("/")[-1] not in opts]
    # so allvars is actually the variables except things for training

    if config.ignore_gn_vars:
      allvars = [var for var in allvars if "/gn" not in var.name.split(":")[0]]

    if config.ignore_vars is not None:
      ignore_vars = config.ignore_vars.split(":")
      ignore_vars.extend(opts)
      # also these
      #ignore_vars+=["global_step"]

      restore_vars = []
      for var in allvars:
        ignore_it = False
        for ivar in ignore_vars:
          if ivar in var.name:
            ignore_it = True
            print("ignored %s" % var.name)
            break
        if not ignore_it:
          restore_vars.append(var)

      print("ignoring %s variables, original %s vars, restoring for %s vars" % \
          (len(ignore_vars), len(allvars), len(restore_vars)))

    else:
      restore_vars = allvars

    saver = tf.train.Saver(restore_vars, max_to_keep=5)

    load_from = None

    if config.load_from is not None:
      load_from = config.load_from
    else:
      if load_best:
        load_from = config.save_dir_best
      else:
        load_from = config.save_dir

    ckpt = tf.train.get_checkpoint_state(load_from)
    if ckpt and ckpt.model_checkpoint_path:
      loadpath = ckpt.model_checkpoint_path
      saver.restore(sess, loadpath)
      print("Model:")
      print("\tloaded %s"%loadpath)
      print("")
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
          #param_names = set(params.iterkeys())
          param_names = set(params.keys())

          #variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

          variables = restore_vars

          variable_names = set([k.name for k in variables])

          intersect = variable_names & param_names

          restore_vars = [v for v in variables if v.name in intersect]

          with sess.as_default():
            for v in restore_vars:
              vname = v.name
              v.load(params[vname])

          #print variables # all the model"s params

          not_used = [(one, weights[one].shape) for one in weights.keys()
                      if get_op_tensor_name(one)[1] not in intersect]
          if not_used:
            print("warning, %s/%s in npz not restored:%s" % (
                len(weights.keys()) - len(intersect),
                len(weights.keys()), not_used))

          #if config.show_restore:
          #  print "loaded %s vars:%s"%(len(intersect),intersect)


        else:
          raise Exception("Not recognized model type:%s" % load_from)
      else:
        raise Exception("Model not exists")
    print("done.")
