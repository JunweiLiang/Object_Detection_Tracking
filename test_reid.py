# coding=utf-8
"""Test person_reid and vehicle reid model"""

import argparse
import os
from glob import glob
import numpy as np

from torchreid.feature_extractor import FeatureExtractor
from torchreid.distance import compute_distance_matrix

parser = argparse.ArgumentParser()
parser.add_argument("query_img")
parser.add_argument("test_img_prefix")
parser.add_argument("--gpuid", default=0, type=int,
                    help="gpu id")
parser.add_argument("--vehicle_reid_model", default=None)
parser.add_argument("--person_reid_model", default=None)
parser.add_argument("--p_model_name", default="osnet_x1_0")


if __name__ == "__main__":
  args = parser.parse_args()

  if args.person_reid_model is not None:
    extractor = FeatureExtractor(
        model_name=args.p_model_name,
        model_path=args.person_reid_model,
        device="cuda:%d" % args.gpuid
    )

  elif args.vehicle_reid_model is not None:
    extractor = FeatureExtractor(
        model_name="resnet101",
        model_path=args.vehicle_reid_model,
        device="cuda:%d" % args.gpuid
    )
  else:
    raise Exception("Please provide a model!")

  test_imgs = glob(args.test_img_prefix + "*")
  test_imgs.sort()
  assert test_imgs
  img_list = [args.query_img] + test_imgs
  print(img_list)
  features = extractor(img_list)

  print(features.shape)  # [n, 512]
  # compute nxn distance
  distmat = compute_distance_matrix(features, features, metric='euclidean')
  np.set_printoptions(suppress=True, precision=3)
  print(distmat.cpu().numpy())

