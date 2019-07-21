# coding=utf-8
"""Given the tensorflow frozen graph, use TensorRT to optimize,
 get a new frozen graph."""

from __future__ import print_function

import argparse
import time
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

parser = argparse.ArgumentParser()
parser.add_argument("pbfile")
parser.add_argument("newpbfile")
parser.add_argument("--precision_mode", default="FP32",
                    help="FP32, FP16, or INT8")
parser.add_argument("--maximum_cached_engines", default=100,
                    help="Don't know what this does.")


# parameter
# https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
if __name__ == "__main__":
  args = parser.parse_args()

  max_batch_size = 1
  precision_mode = args.precision_mode
  minimum_segment_size = 2
  max_workspace_size_bytes = 1 << 32
  maximum_cached_engines = args.maximum_cached_engines

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

      graph_size = len(frozen_graph.SerializeToString())
      num_nodes = len(frozen_graph.node)
      start_time = time.time()
      frozen_graph = trt.create_inference_graph(
          input_graph_def=frozen_graph,
          outputs=output_names,
          max_batch_size=max_batch_size,
          max_workspace_size_bytes=max_workspace_size_bytes,
          precision_mode=precision_mode,
          minimum_segment_size=minimum_segment_size,
          is_dynamic_op=True,  # this is needed for FPN
          maximum_cached_engines=maximum_cached_engines)
      end_time = time.time()
      print("graph_size(MB)(native_tf): %.1f" % (float(graph_size)/(1<<20)))
      print("graph_size(MB)(trt): %.1f" % (
          float(len(frozen_graph.SerializeToString()))/(1<<20)))
      print("num_nodes(native_tf): %d" % num_nodes)
      print("num_nodes(tftrt_total): %d" % len(frozen_graph.node))
      print("num_nodes(trt_only): %d" % len(
          [1 for n in frozen_graph.node if str(n.op) == "TRTEngineOp"]))
      print("time(s) (trt_conversion): %.4f" % (end_time - start_time))
  with open(args.newpbfile, "wb") as f:
    f.write(frozen_graph.SerializeToString())
