# coding=utf-8
"""Given the tensorflow frozen graph, use TensorRT to optimize,
 get a new frozen graph."""

from __future__ import print_function

import argparse
import time
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser()
parser.add_argument("pbfile")
parser.add_argument("newpbfile")
parser.add_argument("--precision_mode", default="FP32",
                    help="FP32, FP16, or INT8")


# parameter
# https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
if __name__ == "__main__":
  args = parser.parse_args()

  # not sure what these do, so leave them default
  #max_batch_size = 1
  #minimum_segment_size = 2  # smaller the faster? 5 -60?
  #max_workspace_size_bytes = 1 << 32
  #maximum_cached_engines = 1

  output_names = [
      "final_boxes",
      "final_labels",
      "final_probs",
      "fpn_box_feat",
  ]

  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True

  with tf.Graph().as_default() as tf_graph:
    with tf.Session(config=tf_config) as tf_sess:
      with tf.gfile.GFile(args.pbfile, "rb") as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())

      converter = trt.TrtGraphConverter(
          input_graph_def=frozen_graph,
          nodes_blacklist=output_names,
          is_dynamic_op=False,
          precision_mode=args.precision_mode) #output nodes
      trt_graph = converter.convert()
      #converter.save(args.newpbfile)


  with open(args.newpbfile, "wb") as f:
    f.write(trt_graph.SerializeToString())
