# coding=utf-8
"""Model wrapper for EfficientDet."""

import argparse
import tensorflow as tf
from efficientdet import efficientdet_arch
from efficientdet import anchors
from efficientdet import dataloader
from class_ids import coco_obj_class_to_id
from nn import crop_and_resize_nhwc

class EfficientDet():
  def __init__(self, config):
    self.config = config
    # input place holders
    # efficientdet needs to now the width
    # note here the image type is uint8, not float32, as efficientdet
    # uses tf.image.convert_image_dtype, which need uint8 to auto scale to 0..1
    self.image = tf.placeholder(
        tf.uint8, [None, config.max_size, 3], name="image")

    # [H, W, 3]  # efficientdet pad to 1920x1920
    p_image, scale = self.build_preprocess(self.image)

    boxes, scores, classes, fpn_box_feat = \
        self.build_model(p_image, scale)

    # add a name so the frozen graph will have that name
    self.final_boxes = tf.identity(boxes, name="final_boxes")
    self.final_labels = tf.identity(classes, name="final_labels")  # [1-90]
    self.final_probs = tf.identity(scores, name="final_probs")

    self.fpn_box_feat = tf.identity(fpn_box_feat, name="fpn_box_feat")
    #self.level_indexes = level_indexes
    #self.boxes_on_fp = boxes_on_fp


  def build_preprocess(self, image):
    config = self.config
    img_width = config.max_size
    img_height = config.short_edge_size

    bgr = True  # cv2 load image is bgr

    p_image = image
    if bgr:
      # to RGB, efficientdet is trained with PIL
      p_image = p_image[:, :, ::-1]
    #input_processor = dataloader.DetectionInputProcessor(p_image, img_size)
    input_processor = dataloader.DetectionInputProcessor(
        p_image, (img_height, img_width))
    # make image [0,1] and -mean/var
    input_processor.normalize_image()
    input_processor.set_scale_factors_to_output_size()
    # here the original efficientdet pad image to (max_size, max_size)
    p_image = input_processor.resize_and_crop_image()
    p_image_scale = input_processor.image_scale_to_original
    p_image = tf.expand_dims(p_image, 0)  # [1, H, W, C]
    return p_image, p_image_scale

  def build_model(self, image, scale):
    """
      image: [H, W, 3]
      Return:
        boxes, labels, probs
    """
    config = self.config
    # [1, H, W, 3] image
    # get all the parameters for the efficient_det
    eff_config = get_efficientdet_config(config)
    #print(image, config.max_size, config.short_edge_size)
    # 2 -> 5 level, [N, H, W, C]
    features = efficientdet_arch.build_backbone(image, eff_config)
    #print(features)
    # 3 -> 7 level, [N, H, W, C]
    fpn_feats = efficientdet_arch.build_feature_network(features, eff_config)
    #(max_size==1280, d5)# [1, 160, 160, 288] -> [1, 10, 10, 288]
    # d0 is 64
    #print(fpn_feats)
    # these are used for frozen graph
    #for lvl in range(eff_config.min_level, eff_config.max_level + 1):
    #  fpn_feats[lvl] = tf.identity(fpn_feats[lvl], name="fpn_feats_lvl%s" % lvl)

    # 3 -> 7 level, [N, H, W, 810/36], 810 = 90 * 9(num_anchors), 36 = 4 * 9
    class_outputs, box_outputs = efficientdet_arch.build_class_and_box_outputs(
        fpn_feats, eff_config)

    cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, \
        classes_all, level_index_all_after_topk = add_metric_fn_inputs(
            eff_config, class_outputs, box_outputs)

    boxes, scores, classes, fpn_box_feat = get_results_tf(
        eff_config, fpn_feats,
        cls_outputs_all_after_topk,
        box_outputs_all_after_topk,
        indices_all,
        classes_all,
        level_index_all_after_topk,
        scale)


    return boxes, scores, classes, fpn_box_feat

  def get_feed_dict_forward(self, imgdata):
    feed_dict = {}

    feed_dict[self.image] = imgdata

    return feed_dict




class EfficientDet_frozen():
  def __init__(self, config, modelpath, gpuid):
    self.graph = tf.get_default_graph()
    eff_config = get_efficientdet_config(config)

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

    # intermedia output
    self.final_boxes = self.graph.get_tensor_by_name(
        "%s/final_boxes:0" % self.var_prefix)
    self.final_labels = self.graph.get_tensor_by_name(
        "%s/final_labels:0" % self.var_prefix)
    self.final_probs = self.graph.get_tensor_by_name(
        "%s/final_probs:0" % self.var_prefix)
    self.fpn_box_feat = self.graph.get_tensor_by_name(
        "%s/fpn_box_feat:0" % self.var_prefix)


  def get_feed_dict_forward(self, imgdata):
    feed_dict = {}

    feed_dict[self.image] = imgdata

    return feed_dict




def get_efficientdet_config(config):
  # so this namespace can be access with []
  class my_namespace(argparse.Namespace):
    def __getitem__(self, key):
      return self.__dict__[key]
  eff_config = my_namespace(
      result_score_thres=config.result_score_thres,
      result_per_im=config.result_per_im,
      batch_size=1,
      name=config.efficientdet_modelname,
      #image_size=640,
      #input_rand_hflip=True,
      #train_scale_min=0.1,
      #train_scale_max=2.0,
      #autoaugment_policy=None,
      num_classes=90,
      #skip_crowd_during_training=True
      # model architecture
      #min_level=3,  # moved this guy to high level config
      #max_level=7,
      num_scales=3,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=4.0,
      is_training_bn=False,
      # optimization
      #momentum=0.9,
      #learning_rate=0.08,
      #lr_warmup_init=0.008,
      #lr_warmup_epoch=1.0,
      #first_lr_drop_epoch=200.0,
      #second_lr_drop_epoch=250.0,
      #clip_gradients_norm=10.0,
      #num_epochs=300,
      #alpha=0.25,
      #gamma=1.5,
      #delta=0.1,
      #box_loss_weight=50.0,
      #weight_decay=4e-5,
      #use_bfloat16=True,
      # For detection.
      box_class_repeats=3,
      fpn_cell_repeats=3,
      fpn_num_filters=88,
      separable_conv=True,
      apply_bn_for_resampling=True,
      conv_after_downsample=False,
      conv_bn_relu_pattern=False,
      use_native_resize_op=False,
      pooling_type=None,

      fpn_name=None,
      fpn_config=None,
      use_tpu=False,
      data_format="channels_last",

      # No stochastic depth in default.
      survival_prob=None,
      fpn_weight_method=None,
      conv_bn_act_pattern=False,
      act_type="swish",

      #lr_decay_method="cosine",
      #moving_average_decay=0.9998,
      #ckpt_var_scope=None,
      backbone_name="efficientnet-b1",
      backbone_config=None,
      # RetinaNet.
      resnet_depth=50)

  replace_params = \
      efficientdet_model_param_dict[config.efficientdet_modelname]
  eff_config.__dict__.update(replace_params)

  eff_config.min_level = config.efficientdet_min_level
  eff_config.max_level = config.efficientdet_max_level

  #eff_config.image_size = config.max_size
  eff_config.image_size = (int(config.short_edge_size), int(config.max_size))
  # needed in biFPN
  #eff_config.img_height = config.short_edge_size

  # original code is 5000, the topk boxes before NMS
  eff_config.max_detection_topk = config.efficientdet_max_detection_topk

  eff_config.partial_class_idxs = []
  if config.use_partial_classes:
    # config.partial_classes: all classnames in coco_obj_to_actev_obj
    # -1 to map to [0-89]
    eff_config.partial_class_idxs = [
        coco_id_mapping_reverse[classname] - 1
        for classname in config.partial_classes]

  return eff_config

def roi_align(featuremap, boxes, output_shape):
  boxes = tf.stop_gradient(boxes)
  # [1,FS,FS,C] -> [K,out_shape*2,out_shape*2,C]
  ret = crop_and_resize_nhwc(
      featuremap, boxes,
      tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32), output_shape * 2)
  ret = tf.nn.avg_pool(
      ret, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
      padding="SAME", data_format="NHWC")
  return ret

def multilevel_roi_align(fpn_feats, boxes, level_indexes, output_shape,
                         eff_config):
  """
    Given [R, 4] boxes and [R] level_indexes indicating the FPN level
    # boxes are x1, y1, x2, y2
  """
  # gather boxes for each feature level
  all_rois = []
  level_ids = []
  # for debuging
  #boxes_on_fp = []
  #1920 -> [160, 80, 40, 20, 10]/{3, 4, 5, 6, 7}
  for level in range(eff_config.min_level, eff_config.max_level + 1):
    this_level_boxes_idxs = tf.where(tf.equal(level_indexes, level))
    # [K, 1] -> [K]
    this_level_boxes_idxs = tf.reshape(this_level_boxes_idxs, [-1])
    level_ids.append(this_level_boxes_idxs)
    this_level_boxes = tf.gather(boxes, this_level_boxes_idxs)
    boxes_on_featuremap = this_level_boxes * (1.0 / (2. ** level))
    featuremap = fpn_feats[level]  # [1, H, W, C]
    # [K, output_shape, output_shape, C]
    box_feats = roi_align(featuremap, boxes_on_featuremap, output_shape)
    box_feats = tf.reduce_mean(box_feats, axis=[1, 2])  #  [K, C]
    all_rois.append(box_feats)

    # for debugging
    #boxes_on_fp.append(boxes_on_featuremap)

  all_rois = tf.concat(all_rois, axis=0)
  # Unshuffle to the original order, to match the original samples
  level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
  level_id_invert_perm = tf.invert_permutation(level_id_perm)
  all_rois = tf.gather(all_rois, level_id_invert_perm)

  #boxes_on_fp = tf.concat(boxes_on_fp, axis=0)
  #boxes_on_fp = tf.gather(boxes_on_fp, level_id_invert_perm)
  return all_rois#, boxes_on_fp


def get_results_tf(eff_config, fpn_feats,
                   cls_outputs_all_after_topk,
                   box_outputs_all_after_topk,
                   indices_all,
                   classes_all,
                   level_index_all_after_topk,
                   scale):
  # Create anchor_label for picking top-k predictions.
  eval_anchors = anchors.Anchors(eff_config["min_level"],
                                 eff_config["max_level"],
                                 eff_config["num_scales"],
                                 eff_config["aspect_ratios"],
                                 eff_config["anchor_scale"],
                                 eff_config["image_size"])

  num_classes = eff_config["num_classes"]
  if eff_config["partial_class_idxs"]:
    num_classes = len(eff_config["partial_class_idxs"])

  anchor_labeler = anchors.AnchorLabeler(
      eval_anchors, num_classes)
  assert eff_config["batch_size"] == 1
  # [5000], prob
  cls_outputs_per_sample = cls_outputs_all_after_topk[0]
  # [5000, 4]
  box_outputs_per_sample = box_outputs_all_after_topk[0]
  # [5000], each is 1-H*W*num_anchors
  indices_per_sample = indices_all[0]
  # [5000], each is 1-90
  classes_per_sample = classes_all[0]
  level_index_per_sample = level_index_all_after_topk[0]
  # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/py_function
  # tf.py_func cannot be saved to .pb
  # [R, 7] [image_id, x, y, width, height, score, class]
  # now it is [R, 8], with last is level_index
  #detections = anchor_labeler.generate_detections(
  #    cls_outputs_per_sample, box_outputs_per_sample, indices_per_sample,
  #    classes_per_sample, image_id=[0], image_scale=[scale],
  #    level_index=level_index_per_sample, use_tf=False)
  # [R, 8] [image_id, x, y, width, height, score, class, feature_level_index]
  # class index is 1-90, within which 80 classes have labels
  #boxes, scores, classes, level_indexes = detections

  # tf version
  boxes, scores, classes, level_indexes = anchor_labeler.generate_detections(
      cls_outputs_per_sample, box_outputs_per_sample, indices_per_sample,
      classes_per_sample, image_id=[0], image_scale=[scale],
      level_index=level_index_per_sample, use_tf=True,
      min_score_thresh=eff_config.result_score_thres,
      max_boxes_to_draw=eff_config.result_per_im)

  # get the detection results and the ROI aligned features for each box
  # now they have shapes
  boxes = tf.reshape(boxes, [-1, 4])
  classes = tf.cast(classes, dtype="int32")
  level_indexes = tf.cast(tf.reshape(level_indexes, [-1]), dtype="int32")
  fpn_box_feat = multilevel_roi_align(
      fpn_feats, boxes, level_indexes, 7, eff_config)
  #print(fpn_box_feat)  # [K, 64] for d0, [K, 288] for d5
  return boxes, scores, classes, fpn_box_feat

# ------------ Modified from efficientDet

def add_metric_fn_inputs(params, cls_outputs, box_outputs):
  """Selects top-k predictions and adds the selected to metric_fn_inputs.

  Args:
    params: a parameter dictionary that includes `min_level`, `max_level`,
      `batch_size`, and `num_classes`.
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    metric_fn_inputs: a dictionary that will hold the top-k selections.
  """
  cls_outputs_all = []
  box_outputs_all = []
  level_index_all = []
  num_anchors = len(params.aspect_ratios) * params.num_scales

  num_classes = params["num_classes"]
  # Concatenates class and box of all levels into one tensor.
  for level in range(params["min_level"], params["max_level"] + 1):
    #print(cls_outputs[level])  # [1, H, W, 9* 90] # 9: num_anchors
    _, H, W, _ = cls_outputs[level].get_shape()

    level_index_all.append(tf.constant(
        level, shape=(params["batch_size"], H*W*num_anchors), dtype="uint8"))
    # [1, H*W*num_anchors, classes]
    this_cls_outputs = tf.reshape(
        cls_outputs[level],
        [params["batch_size"], -1, num_classes])

    if params["partial_class_idxs"]:  # a list of class idx [0 - 89]
      # [classes, batch, -1]
      this_cls_outputs = tf.transpose(this_cls_outputs, [2, 0, 1])
      # select the needed classes
      this_cls_outputs = tf.gather(
          this_cls_outputs, params["partial_class_idxs"])
      this_cls_outputs = tf.transpose(this_cls_outputs, [1, 2, 0])

    cls_outputs_all.append(this_cls_outputs)
    # a list of [1, K, 4]
    box_outputs_all.append(tf.reshape(
        box_outputs[level], [params["batch_size"], -1, 4]))

  if params["partial_class_idxs"]:
    num_classes = len(params["partial_class_idxs"])

  cls_outputs_all = tf.concat(cls_outputs_all, 1)
  box_outputs_all = tf.concat(box_outputs_all, 1)
  level_index_all = tf.concat(level_index_all, 1)
  # put all spatial location and anchor together
  #print(cls_outputs_all)  # (1, 306900, 90)
  #print(level_index_all)  # (1, 306900)

  # cls_outputs_all has a shape of [batch_size, N, num_classes] and
  # box_outputs_all has a shape of [batch_size, N, 4]. The batch_size here
  # is per-shard batch size. Recently, top-k on TPU supports batch
  # dimension (b/67110441), but the following function performs top-k on
  # each sample.
  cls_outputs_all_after_topk = []
  box_outputs_all_after_topk = []
  indices_all = []
  classes_all = []

  level_index_all_after_topk = []
  for index in range(params["batch_size"]):
    # [306900, 90]
    cls_outputs_per_sample = cls_outputs_all[index]
    box_outputs_per_sample = box_outputs_all[index]
    level_index_per_sample = level_index_all[index]
    cls_outputs_per_sample_reshape = tf.reshape(cls_outputs_per_sample,
                                                [-1])
    # top 5000 boxes for all classes
    _, cls_topk_indices = tf.nn.top_k(
        cls_outputs_per_sample_reshape, k=params["max_detection_topk"])
    # Gets top-k class and box scores.
    # [1-306900]
    indices = tf.div(cls_topk_indices, num_classes)
    # [0-89]
    # or [0-5], partial classes
    classes = tf.mod(cls_topk_indices, num_classes)
    cls_indices = tf.stack([indices, classes], axis=1)

    # [5000], each is probability,
    # classes is the class index
    cls_outputs_after_topk = tf.gather_nd(cls_outputs_per_sample,
                                          cls_indices)
    cls_outputs_all_after_topk.append(cls_outputs_after_topk)

    # [5000, 4]
    box_outputs_after_topk = tf.gather_nd(
        box_outputs_per_sample, tf.expand_dims(indices, 1))
    box_outputs_all_after_topk.append(box_outputs_after_topk)

    level_index_after_topk = tf.gather(level_index_per_sample, indices)
    level_index_all_after_topk.append(level_index_after_topk)

    indices_all.append(indices)
    classes_all.append(classes)
  # Concatenates via the batch dimension.
  # this is the prob score
  cls_outputs_all_after_topk = tf.stack(cls_outputs_all_after_topk, axis=0)
  box_outputs_all_after_topk = tf.stack(box_outputs_all_after_topk, axis=0)
  level_index_all_after_topk = tf.stack(level_index_all_after_topk, axis=0)
  indices_all = tf.stack(indices_all, axis=0)
  classes_all = tf.stack(classes_all, axis=0)
  return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, \
         classes_all, level_index_all_after_topk
  """
  # [1, 5000]  # prob score
  metric_fn_inputs["cls_outputs_all"] = cls_outputs_all_after_topk
  # [1, 5000, 4]
  metric_fn_inputs["box_outputs_all"] = box_outputs_all_after_topk
  # [5000], each is [1-306900]
  metric_fn_inputs["indices_all"] = indices_all
  # [5000], each is [0-89]
  metric_fn_inputs["classes_all"] = classes_all
  # [5000], each is min-level to max-level
  metric_fn_inputs["level_index_all"] = level_index_all_after_topk
  """


coco_id_mapping = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}
coco_id_mapping_reverse = {v:k for k, v in coco_id_mapping.items()}


efficientdet_model_param_dict = {
    "efficientdet-d0":
        dict(
            name="efficientdet-d0",
            backbone_name="efficientnet-b0",
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    "efficientdet-d1":
        dict(
            name="efficientdet-d1",
            backbone_name="efficientnet-b1",
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    "efficientdet-d2":
        dict(
            name="efficientdet-d2",
            backbone_name="efficientnet-b2",
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    "efficientdet-d3":
        dict(
            name="efficientdet-d3",
            backbone_name="efficientnet-b3",
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    "efficientdet-d4":
        dict(
            name="efficientdet-d4",
            backbone_name="efficientnet-b4",
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    "efficientdet-d5":
        dict(
            name="efficientdet-d5",
            backbone_name="efficientnet-b5",
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    "efficientdet-d6":
        dict(
            name="efficientdet-d6",
            backbone_name="efficientnet-b6",
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_name="bifpn_sum",  # Use unweighted sum for training stability.
        ),
    'efficientdet-d7':
        dict(
            name='efficientdet-d7',
            backbone_name='efficientnet-b6',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        ),
}
