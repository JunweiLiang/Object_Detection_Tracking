# coding=utf-8
# given MOT results of each video, use ReID features to re-associate some tracks
# ignore switcher inside each track
# associate similar tracks within a range of frames, within some IOU range
#    by using pretrained Person-ReID/vehicle-reID features

import argparse
import cv2
import os
import operator
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib
# avoid the warning "gdk_cursor_new_for_display:
# assertion 'GDK_IS_DISPLAY (display)' failed" with Python 3
matplotlib.use('Agg')

import torch

from utils import parse_meva_clip_name, expand_tlwh, tlwh_intersection
from torchreid.feature_extractor import FeatureExtractor
from torchreid.distance import compute_distance_matrix

from enqueuer_thread import VideoEnqueuer
from diva_io.video import VideoReader
from moviepy.editor import VideoFileClip

from utils import sec2time

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="mot track result path")
parser.add_argument("videonamelst")
parser.add_argument("video_path")
parser.add_argument("newfilepath")
parser.add_argument("--gpuid", default=0, type=int,
                    help="gpu id")
parser.add_argument("--vehicle_reid_model", default=None)
parser.add_argument("--person_reid_model", default=None)
parser.add_argument("--use_lijun_video_loader", action="store_true")
parser.add_argument("--use_moviepy", action="store_true")

parser.add_argument("--tol_num_frame", default=200, type=int,
                    help="search for tracklet in the next 200/fps seconds.")
parser.add_argument("--expand_width_p", default=2.0, type=float)
parser.add_argument("--expand_height_p", default=0.5, type=float)

parser.add_argument("--feature_box_num", default=500, type=int,
                    help="maximum box num to use for feature extraction, -1 "
                         "means all")
parser.add_argument("--feature_box_gap", default=20, type=int,
                    help="interval when getting boxes")
parser.add_argument("--reject_dist_thres", default=150, type=float,
                    help="eu dist rejecting dist threshold")
parser.add_argument("--compare_method", default="min_all",
                    help="min_all / avg")

parser.add_argument("--max_size", type=int, default=1920)
parser.add_argument("--short_edge_size", type=int, default=1080)

parser.add_argument("--debug", action="store_true")


def valid_box(tlwh, img, min_area=10):
  x, y, w, h = tlwh
  area = w * h
  height, width, _ = img.shape
  if x >= 0 and y >= 0 and w >= 0 and h >= 0:
    if x <= width and y <= height and x + w <= width and y + h <= height:
      if area > min_area:
        return True
      else:
        return False
    else:
      return False
  else:
    return False

def preprocess(track_file, tol_num_frame=30,
               expand_width_p=0.1, expand_height_p=0.1):

  # 1. read the tracks
  # assuming sorted by frameid
  data = []
  with open(track_file, "r") as f:
    for line in f:
      frame_idx, track_id, left, top, width, height, _, _, _, _ = line.strip().split(",")
      data.append([frame_idx, track_id, left, top, width, height])

  if not data:
    return [], {}

  data = np.array(data, dtype="float32")  # [N, 6]

  track_ids = np.unique(data[:, 1]).tolist()
  track_data = {}  # [num_track, K, 6]
  for track_id in track_ids:
    track_data[track_id] = data[data[:, 1] == track_id, :]


  # 2. find possible linkage tracklet pairs
  # assuming only happens at the end of each tracklet, within a num of frames
  # n by n search
  possible_pairs = {}  # track_id -> a list of track_id
  for idx1, track_id1 in enumerate(track_ids):
    #print(track_id, track_data[idx][0, 0])
    # track_id, first frame in ascending order already

    # check this track's ending
    # against other track's start within tol_num_frame, then check intersection
    # with the expanded box
    end_frame_idx1 = track_data[track_id1][-1, 0]
    tlwh1 = track_data[track_id1][-1, 2:]  # the last box

    expanded_tlwh1 = expand_tlwh(
        tlwh1, expand_width_p, expand_height_p)

    for idx2 in range(idx1+1, len(track_ids)):
      track_id2 = track_ids[idx2]
      start_frame_idx2 = track_data[track_id2][0, 0]
      if start_frame_idx2 <= end_frame_idx1:
        continue
      if start_frame_idx2 - end_frame_idx1 < tol_num_frame:
        # only check the start frame's box for this track
        tlwh2 = track_data[track_id2][0, 2:]  # the first box
        box_inter = tlwh_intersection(expanded_tlwh1, tlwh2)
        if box_inter > 0:
          if track_id1 not in possible_pairs:
            possible_pairs[track_id1] = []
            possible_pairs[track_id1].append(
                [track_id2, start_frame_idx2, end_frame_idx1])
          else:
            # some track's end might be matched to multiple track within
            # tolerance time, we only keep the earliest ones,
            prev_start_frame_idx = possible_pairs[track_id1][0][1]
            if start_frame_idx2 == prev_start_frame_idx:
              possible_pairs[track_id1].append(
                  [track_id2, start_frame_idx2, end_frame_idx1])

  return track_data, possible_pairs


def reid(frame_iter, p_extractor, v_extractor,
         p_track_data, p_candidates,
         v_track_data, v_candidates,
         feature_box_num=5, feature_box_gap=20,
         reject_dist_thres=200,
         compare_method="min_all"):
  # get new track data for two classes

  # track_data: track_id -> [K, 6], candidates: track_id -> a list of track_ids

  # 1. get the frame_id=>boxes needed to extract feature based on candidates
  frame_data = {}

  needed_track_boxes = {}  # "query_track_id"/"gallery_track_id" => boxes

  def get_track_boxes(track_data, candidates):
    for query_track_id in candidates:
      query_key = "query_%s" % query_track_id
      needed_track_boxes[query_key] = []
      if feature_box_num <= 0:
        # get all box (under frame_gap)
        this_box_num_limit = len(track_data[query_track_id])
      else:
        this_box_num_limit = feature_box_num
      # for query, get last few boxes
      idx = 1
      while idx <= len(track_data[query_track_id]) and \
          len(needed_track_boxes[query_key]) < this_box_num_limit:
        # get from the end of the tracklet
        box = track_data[query_track_id][-idx]  # [6]
        needed_track_boxes[query_key].append(box)
        idx += feature_box_gap

      for gallery_track_id, _, _ in candidates[query_track_id]:
        gallery_key = "gallery_%s" % gallery_track_id

        if gallery_key not in needed_track_boxes:

          needed_track_boxes[gallery_key] = []

          if feature_box_num <= 0:
            # get all box (under frame_gap)
            this_box_num_limit = len(track_data[gallery_track_id])
          else:
            this_box_num_limit = feature_box_num

          # for gallery, get first few boxes
          idx = 0
          while idx < len(track_data[gallery_track_id]) and \
              len(needed_track_boxes[gallery_key]) < this_box_num_limit:
            # get from the end of the tracklet
            box = track_data[gallery_track_id][idx]  # [6]
            needed_track_boxes[gallery_key].append(box)
            idx += feature_box_gap
  get_track_boxes(p_track_data, p_candidates)
  get_track_boxes(v_track_data, v_candidates)

  for key in needed_track_boxes:
    for box_idx, box in enumerate(needed_track_boxes[key]):
      # [6],
      frame_idx = box[0]
      tlwh = box[2:]

      if not frame_idx in frame_data:
        frame_data[frame_idx] = []
      frame_data[frame_idx].append((tlwh, key, box_idx))

  # 2. go through the video once and crop all the images (RAM should fit)
  needed_track_box_imgs = {}  # "query_track_id"/"gallery_track_id" => boxes

  for batch in tqdm(frame_iter[0], total=frame_iter[1]):
    image, scale, frame_idx = batch[0]
    image = image.astype("uint8")  # need uint8 type
    if frame_idx in frame_data:
      for tlwh, key, box_idx in frame_data[frame_idx]:

        if key not in needed_track_box_imgs:
          needed_track_box_imgs[key] = {}
        # check box valid
        if valid_box(tlwh, image):
          x, y, w, h = tlwh
          x, y, w, h = int(x), int(y), int(w), int(h)
          #print(x, y, w, h)
          #print(image[y:y+h, x:x+w])
          box_img = cv2.cvtColor(
              image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
          needed_track_box_imgs[key][box_idx] = box_img


  # 3. for each query and gallery tracklets, extract feature and compare

  # add the distance to the candidates
  def get_features_and_compare(candidates, extractor, method="min_all"):
    # method: "min_all" / "avg"
    # 1. minimum among N (query_tracklet) by M (one gallery tracklet) dist
    # 2. avg pool feature and then compute eu dist
    assert method in ["min_all", "avg"]
    remove_track_ids = {}
    for query_track_id in candidates:
      query_key = "query_%s" % query_track_id
      query_boxes = needed_track_box_imgs[query_key]  # box_idx => img [H, W, 3]
      if not query_boxes:
        remove_track_ids[query_track_id] = 1
        continue
      # front more important
      box_idxs = sorted(list(query_boxes.keys()))
      # torch tensor
      query_features = extractor([query_boxes[i] for i in box_idxs])  # [K, 512]

      gallery_featuress = []
      # check the gallery boxes first
      temp_list = []
      for gallery_track_id, o, c in candidates[query_track_id]:
        gallery_key = "gallery_%s" % gallery_track_id
        gallery_boxes = needed_track_box_imgs[gallery_key]  # box_idx => img
        if gallery_boxes:
          temp_list.append([gallery_track_id, o, c])

      if not temp_list:
        remove_track_ids[query_track_id] = 1
        continue
      candidates[query_track_id] = temp_list

      for gallery_track_id, _, _ in candidates[query_track_id]:
        gallery_key = "gallery_%s" % gallery_track_id
        gallery_boxes = needed_track_box_imgs[gallery_key]  # box_idx => img
        assert gallery_boxes
        # front more important
        box_idxs = sorted(list(gallery_boxes.keys()))
        gallery_features = extractor([gallery_boxes[i] for i in box_idxs])
        #features = features.cpu().numpy()
        gallery_featuress.append(gallery_features)

      assert gallery_featuress

      # # N query frames, M gallery tracklet with each K frames
      # query_features [N, 512]
      # gallery_featuress M of [K, 512], K is variable
      if method == "avg":
        # avg pooling
        # [1, 512]
        query_feature = torch.mean(query_features, 0, keepdim=True)
        for i in range(len(gallery_featuress)):
          gallery_featuress[i] = torch.mean(gallery_featuress[i], 0)

        # [K, 512]
        gallery_featuress = torch.stack(gallery_featuress)
        # [1, K]
        distmat = compute_distance_matrix(
            query_feature, gallery_featuress, metric="euclidean")
      elif method == "min_all":
        # compute all query frame with all frames in each gallery track (N by K)
        # take minimum
        distmat = torch.zeros((1, len(gallery_featuress)))
        for i, gallery_features in enumerate(gallery_featuress):
          # [N, K]
          this_distmat = compute_distance_matrix(
              query_features, gallery_features, metric="euclidean")
          distmat[0, i] = torch.min(this_distmat)

      distmat = distmat.cpu().numpy()

      for i in range(distmat.shape[1]):
        candidates[query_track_id][i].append(distmat[0, i])
      # sort the candidate according to the distance
      candidates[query_track_id].sort(key=operator.itemgetter(-1))
    for track_id in remove_track_ids:
      del candidates[track_id]
    if args.debug:
      print("removed %s candidates due to bad boxes" % len(remove_track_ids))

  get_features_and_compare(p_candidates, p_extractor, method=compare_method)
  get_features_and_compare(v_candidates, v_extractor, method=compare_method)


  def merge_candidates(candidates, track_data):
    merge_map = {}  # track_id -> track_id
    # go through the candidate and decide merging or not
    # 1. remove candidates with large dist
    # 2. select longest one if muliple in reverse
    # 3. return new track_data
    reverse_candidates = {}  # track_id -> list of track_id
    for query_track_id in candidates:
      match_list = [(o[0], o[-1])
                    for o in candidates[query_track_id]
                    if o[-1] < reject_dist_thres]
      if match_list:
        # top 1 is used
        match_track_id, dist = match_list[0]
        if match_track_id not in reverse_candidates:
          reverse_candidates[match_track_id] = []

        reverse_candidates[match_track_id].append((
            query_track_id, dist, len(track_data[query_track_id])))
    for track_id in reverse_candidates:
      if len(reverse_candidates[track_id]) > 1:
        reverse_candidates[track_id].sort(key=operator.itemgetter(2),
                                          reverse=True)
      matched_prev_track_id, _, _ = reverse_candidates[track_id][0]

      # find the earliest track_id
      while matched_prev_track_id in merge_map:
        matched_prev_track_id = merge_map[matched_prev_track_id]

      assert track_id not in merge_map
      assert matched_prev_track_id not in merge_map

      merge_map[track_id] = matched_prev_track_id

    if args.debug:
      print(candidates, merge_map)
    new_track_data = {}
    for track_id in track_data:
      this_track_data = track_data[track_id][:, :]  # [K, 6]
      if track_id in merge_map:
        track_id = merge_map[track_id]
        this_track_data[:, 1] = track_id

      if track_id in new_track_data:
        new_track_data[track_id] = np.concatenate(
            [new_track_data[track_id], this_track_data], axis=0)
      else:
        new_track_data[track_id] = this_track_data

    return new_track_data

  p_track_data = merge_candidates(p_candidates, p_track_data)
  v_track_data = merge_candidates(v_candidates, v_track_data)

  return p_track_data, v_track_data


def save_new_track(cat_name, track_data, out_dir, videoname):

  track_results = sorted(
      [l.tolist() for t in track_data for l in track_data[t]],
      key=lambda x: (x[0], x[1]))

  out_file_dir = os.path.join(out_dir, videoname, cat_name)
  if not os.path.exists(out_file_dir):
    os.makedirs(out_file_dir)
  out_file = os.path.join(
      out_file_dir, os.path.splitext(videoname)[0] + ".txt")
  with open(out_file, "w") as fw:
    for row in track_results:
      line = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (
          row[0], row[1], row[2], row[3], row[4], row[5])
      fw.write(line + "\n")

if __name__ == "__main__":
  args = parser.parse_args()

  # still has the .mp4/.avi
  videonames = [os.path.basename(line.strip())
                for line in open(args.videonamelst, "r").readlines()]
  if not os.path.exists(args.newfilepath):
    os.makedirs(args.newfilepath)

  if args.person_reid_model is None or args.vehicle_reid_model is None:
    raise Exception("Please provide models for person and vehicle!")

  # assuming your GPU can fit both model at once

  person_reid_extractor = FeatureExtractor(
      model_name="osnet_x1_0",
      model_path=args.person_reid_model,
      image_size=(256, 128), # (h, w)
      device="cuda:%d" % args.gpuid
  )
  vehicle_reid_extractor = FeatureExtractor(
      model_name="resnet101",
      model_path=args.vehicle_reid_model,
      image_size=(128, 256),
      device="cuda:%d" % args.gpuid
  )

  print("Model loaded.")

  total_p_track = [0, 0]  # before / after
  total_v_track = [0, 0]
  for videoname in tqdm(videonames):
    clip_name = os.path.splitext(videoname)[0]

    if args.debug and clip_name not in [
        "2018-03-11.11-35-01.11-40-01.school.G424",
        "2018-03-05.13-10-00.13-15-00.bus.G340",
        "2018-03-07.17-25-06.17-30-06.school.G339"]:
      continue
    date, hr_slot = parse_meva_clip_name(clip_name)
    video_path = os.path.join(args.video_path, date, hr_slot, videoname)
    # start reading video frames now!
    if args.use_lijun_video_loader:
      vcap = VideoReader(video_path)
      frame_count = int(vcap.length)
    elif args.use_moviepy:
      vcap = VideoFileClip(video_path, audio=False)
      frame_count = int(vcap.fps * vcap.duration)  # uh
      vcap = vcap.iter_frames()
    else:
      try:
        vcap = cv2.VideoCapture(video_path)
        if not vcap.isOpened():
          raise Exception("cannot open %s" % video_path)
      except Exception as e:
        # raise e
        # just move on to the next video
        print("warning, cannot open %s" % video_path)
        continue
      # opencv 3/4
      frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # start reading frames into queues now
    video_queuer = VideoEnqueuer(
        args, vcap, frame_count, frame_gap=1,  # no skipping frames
        prefetch=100,
        start=True, is_moviepy=args.use_moviepy,
        batch_size=1)
    get_frame_batches = video_queuer.get()

    # 1. read the tracklets and identify potential matching tracklets
    #    for each query tracklet
    person_track_file = os.path.join(
        args.filepath, videoname, "Person", "%s.txt" % (
            os.path.splitext(videoname)[0]))
    vehicle_track_file = os.path.join(
        args.filepath, videoname, "Vehicle", "%s.txt" % (
            os.path.splitext(videoname)[0]))

    if not os.path.exists(person_track_file) or not os.path.exists(vehicle_track_file):
      tqdm.write("warning, skipping %s due to track file not exists" % clip_name)
      if not args.use_lijun_video_loader and not args.use_moviepy:
        vcap.release()
      video_queuer.stop()
      continue

    p_track_data, p_candidates = preprocess(
        person_track_file, args.tol_num_frame,
        args.expand_width_p, args.expand_height_p)
    v_track_data, v_candidates = preprocess(
        vehicle_track_file, args.tol_num_frame,
        args.expand_width_p, args.expand_height_p)


    # 2. compute similarities between query tracklet and the candidates. It is
    #    a match if below threshold

    new_p_track_data, new_v_track_data = reid(
        (get_frame_batches, video_queuer.num_batches),
        person_reid_extractor, vehicle_reid_extractor,
        p_track_data, p_candidates, v_track_data, v_candidates,
        feature_box_num=args.feature_box_num,
        feature_box_gap=args.feature_box_gap,
        reject_dist_thres=args.reject_dist_thres,
        compare_method=args.compare_method)

    if args.debug:
      print("person track %s -> %s" % (
          len(p_track_data), len(new_p_track_data)))
      print("vehicle track %s -> %s" % (
          len(v_track_data), len(new_v_track_data)))

    # save new file

    save_new_track("Person", new_p_track_data, args.newfilepath, videoname)
    save_new_track("Vehicle", new_v_track_data, args.newfilepath, videoname)

    total_p_track[0] += len(p_track_data)
    total_p_track[1] += len(new_p_track_data)
    total_v_track[0] += len(v_track_data)
    total_v_track[1] += len(new_v_track_data)

    if not args.use_lijun_video_loader and not args.use_moviepy:
      vcap.release()
    video_queuer.stop()

  print("total person track %s -> %s" % (total_p_track[0], total_p_track[1]))
  print("total vehicle track %s -> %s" % (total_v_track[0], total_v_track[1]))


