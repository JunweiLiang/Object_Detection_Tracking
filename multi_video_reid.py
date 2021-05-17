# coding=utf-8
# multi-camera reid
# given sync camera group (will use homography to check spatial)
# and consecutive camera group
# for reid, use hungarian algo.

import argparse
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import lap  # 0.4.0
#from scipy.optimize import linear_sum_assignment

from utils import parse_camera_file, compute_c1_to_c2_homography
from utils import parse_meva_clip_name
from utils import valid_box, warp_points

import torch
from torchreid.feature_extractor import FeatureExtractor
from torchreid.distance import compute_distance_matrix

from enqueuer_thread import VideoEnqueuer
from diva_io.video import VideoReader
from moviepy.editor import VideoFileClip
# outdoor cameras (10), ignoring "G300", "G301",
#camera_list = ["G505", "G506", "G638", "G424", "G339", "G328",
#               "G341", "G436", "G336", "G340"]

# for G339, after 03-07.11-10, it enters patrol mode (03-07.16-50)
# only train set has G339
#exclude_list = {"G339": ("2018-03-07", "11-10-07")}
"""
  "sync_groups": {
      "2018-03-11.16-40-08.16-45-08": [
          "2018-03-11.16-40-08.16-45-08.hospital.G341",
          "2018-03-11.16-40-02.16-45-02.school.G424",
          "2018-03-11.16-40-01.16-45-00.school.G328",
          "2018-03-11.16-40-01.16-45-01.school.G336",
          "2018-03-11.16-40-01.16-45-01.bus.G506",
          ..
        ],
      ...,
  "consecutive_groups": [
      [
          "2018-03-11.16-40-08.16-45-08"
      ],
      [
          "2018-03-11.16-30-02.16-35-02",
          "2018-03-11.16-35-01.16-40-01"
      ],
  ...
"""

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="mot track result path")
parser.add_argument("camera_group")
parser.add_argument("camera_model_path")
parser.add_argument("topdown_camera")
parser.add_argument("video_path")
parser.add_argument("newfilepath")
parser.add_argument("--gpuid", default=0, type=int,
                    help="gpu id")
parser.add_argument("--vehicle_reid_model", default=None)
parser.add_argument("--person_reid_model", default=None)
parser.add_argument("--use_lijun_video_loader", action="store_true")
parser.add_argument("--use_moviepy", action="store_true")

parser.add_argument("--max_size", type=int, default=1920)
parser.add_argument("--short_edge_size", type=int, default=1080)


parser.add_argument("--use_avg_pool", action="store_true",
                    help="use average pooling on each track's features")
parser.add_argument("--feature_box_num", default=100, type=int,
                    help="maximum box num to use for feature extraction, -1 "
                         "means all")
parser.add_argument("--feature_box_gap", default=20, type=int,
                    help="interval when getting boxes")
parser.add_argument("--spatial_dist_tol", default=50., type=float,
                    help="pixel distance tolerance")

camera_model = {
    "G505": "2018-03-05.13-20-01.13-25-00.bus.G505.krtd",
    "G506": "2018-03-05.13-15-00.13-20-00.bus.G506.krtd",
    "G638": "2018-03-07.13-15-01.13-20-01.school.G638.krtd",
    "G424": "2018-03-05.18-25-00.18-29-31.school.G424.krtd",
    "G339": "2018-03-05.11-15-00.11-20-00.school.G339.krtd",
    "G328": "2018-03-05.13-25-01.13-30-01.school.G328.krtd",
    "G341": "2018-03-05.15-55-00.16-00-00.hospital.G341.krtd",
    "G436": "2018-03-05.18-10-00.18-15-00.hospital.G436.krtd",
    "G336": "2018-03-06.15-05-02.15-10-02.school.G336.krtd",
    "G340": "2018-03-05.11-20-00.11-25-00.bus.G340.krtd"
}

def compute_homographys(topdown_camera, camera_path, camera_files):
  hs = {}

  c2_r, c2_t, c2_k = parse_camera_file(topdown_camera)

  for camera in camera_files:
    c1_r, c1_t, c1_k = parse_camera_file(
        os.path.join(camera_path, camera_files[camera]))

    homography = compute_c1_to_c2_homography(c1_r, c1_t, c1_k, c2_r, c2_t, c2_k)

    hs[camera] = homography

  return hs

def compute_frame_offset(v1, v2, fps):
  date1, start_time1, end_time, location, camera = v1.split(".")
  date2, start_time2, end_time, location, camera = v2.split(".")
  assert date1 == date2
  def time2sec(time_str):
    # hour-minutes-seconds
    hours, minutes, seconds = time_str.split("-")
    return float(hours)*60.*60. + float(minutes)*60. + float(seconds)

  time_offset = time2sec(start_time2) - time2sec(start_time1)
  return time_offset * fps

def load_track_and_features(args, video_name, p_file, v_file, p_extractor,
                            v_extractor, hs):
  date, hr_slot, camera = parse_meva_clip_name(video_name)
  # start loading video_frames first
  video_path = os.path.join(args.video_path, date, hr_slot, video_name + ".avi")
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
      raise Exception("warning, cannot open %s" % video_path)
    # opencv 3/4
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

  # start reading frames into queues now
  video_queuer = VideoEnqueuer(
      args, vcap, frame_count, frame_gap=1,  # no skipping frames
      prefetch=100,
      start=True, is_moviepy=args.use_moviepy,
      batch_size=1)
  get_frame_batches = video_queuer.get()

  def load_track_file(file_path, homography):
    """load a tracking file into dict of numpy arrays."""
    # assuming sorted by frameid
    data = []
    with open(file_path, "r") as f:
      for line in f:
        frame_idx, track_id, left, top, width, height, conf, _, _, _ = line.strip().split(",")
        data.append([frame_idx, track_id, left, top, width, height, conf])

    if not data:
      return {}

    data = np.array(data, dtype="float32")  # [N, 7]

    # compute topdown points
    foot_points_x = data[:, 2] + data[:, 4] / 2.  # [N]
    foot_points_y = data[:, 3] + data[:, 5]
    foot_points = np.stack([foot_points_x, foot_points_y], axis=0)  # [2, N]
    # [2, N]
    top_down_points = warp_points(foot_points, homography)
    top_down_points = np.transpose(top_down_points, [1, 0])  # [N, 2]

    # [N, 9]
    data = np.concatenate([data, top_down_points], axis=1)

    track_ids = np.unique(data[:, 1]).tolist()
    track_data = {}  # [num_track, K, 9]
    for track_id in track_ids:
      track_data[track_id] = data[data[:, 1] == track_id, :]
    return track_data

  # track_id -> data
  p_tracks = load_track_file(p_file, hs[camera])
  v_tracks = load_track_file(v_file, hs[camera])

  # get each frame's boxes to extract
  frame_data = {}  # frame_idx -> a list of boxes,
  def get_track_boxes(tracks, cat_name):
    for track_id in tracks:
      idxs = list(range(0, len(tracks[track_id]), args.feature_box_gap))
      idxs = idxs[:args.feature_box_num]
      boxes = tracks[track_id][idxs, :]  # [k, 7]

      for box_idx, box in enumerate(boxes):
        frame_idx = box[0]
        tlwh = box[2:6]
        if not frame_idx in frame_data:
          frame_data[frame_idx] = []
        frame_data[frame_idx].append((tlwh, track_id, box_idx, cat_name))
  get_track_boxes(p_tracks, "Person")
  get_track_boxes(v_tracks, "Vehicle")

  # 2. go through the video once and crop all the images to extract features
  # assuming not conflict between person/vehicle track_id
  p_track_to_feat = {}  # "track_id" => features
  v_track_to_feat = {}  # "track_id" => features

  for batch in tqdm(get_frame_batches, total=video_queuer.num_batches):
    image, scale, frame_idx = batch[0]
    image = image.astype("uint8")  # need uint8 type
    if frame_idx in frame_data:
      for tlwh, track_id, box_idx, cat_name in frame_data[frame_idx]:

        # check box valid
        if valid_box(tlwh, image):

          x, y, w, h = tlwh
          x, y, w, h = int(x), int(y), int(w), int(h)
          #print(x, y, w, h)
          #print(image[y:y+h, x:x+w])
          box_img = cv2.cvtColor(
              image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
          if cat_name == "Person":
            if track_id not in p_track_to_feat:
              p_track_to_feat[track_id] = []
            p_track_to_feat[track_id].append(box_img)
          elif cat_name == "Vehicle":
            if track_id not in v_track_to_feat:
              v_track_to_feat[track_id] = []
            v_track_to_feat[track_id].append(box_img)
  # extract features
  def get_features(track_to_imgs, extractor):
    for track_id in track_to_imgs:
      box_imgs = track_to_imgs[track_id]
      track_to_imgs[track_id] = extractor(box_imgs).cpu().numpy()  # [K, 512]
      if args.use_avg_pool:
        # [1, 512]
        track_to_imgs[track_id] = np.mean(
            track_to_imgs[track_id], axis=0, keepdims=True)

  get_features(p_track_to_feat, p_extractor)
  get_features(v_track_to_feat, v_extractor)

  data = {}
  def gather_data(track_data, track_features, cat_name):
    data[cat_name] = {}
    for track_id in track_data:
      # ignore track with no valid boxes
      if track_id in track_features:
        data[cat_name][track_id] = (
            track_data[track_id], track_features[track_id])
  gather_data(p_tracks, p_track_to_feat, "Person")
  gather_data(v_tracks, v_track_to_feat, "Vehicle")

  return data


def compute_spatial_dist(tracks1, tracks2, frame_offset=0, tol=50, ignore_pairs=[[], []]):
  # frameoffset: all frames in tracks2 add this
  # tol: tolerance pixels
  N = len(tracks1)
  M = len(tracks2)
  frame_offset = int(frame_offset)
  spatial_dist = np.ones((N, M), dtype="float") * 9999.
  for i, track_id1 in enumerate(sorted(tracks1.keys())):
    track1 = tracks1[track_id1][0] # (K, 9)
    frame_to_points1 = {int(p[0]): p[-2:] for p in track1}
    frame_set1 = set([int(p[0]) for p in track1])
    for j, track_id2 in enumerate(sorted(tracks2.keys())):
      track2 = tracks2[track_id2][0]
      frame_to_points2 = {(int(p[0]) + frame_offset): p[-2:] for p in track2}
      frame_set2 = set([int(p[0]) + frame_offset for p in track2])

      intersected_frame_ids = list(frame_set1 & frame_set2)
      if intersected_frame_ids:
        # [K, 2]
        track1_points_to_compare = np.array(
            [frame_to_points1[fid] for fid in intersected_frame_ids])
        track2_points_to_compare = np.array(
            [frame_to_points2[fid] for fid in intersected_frame_ids])
        # pixel dist of the intersected frame part [K]
        dist = np.linalg.norm(
            track1_points_to_compare - track2_points_to_compare, axis=1)

        # check how many are above the tolerance
        # the tolerance should be taken into account the synchronize error,
        # and the homography errors
        #good = [1. if d <= tol else 0. for d in dist]
        #spatial_dist[i, j] = np.sum(good)
        mean_dist = np.mean(dist)
        if mean_dist <= tol:
          spatial_dist[i, j] = mean_dist
        # TODO: the above does not consider intersected length

  # reset the ignore pairs dist to large
  for i, track_id1 in enumerate(sorted(tracks1.keys())):
    for j, track_id2 in enumerate(sorted(tracks2.keys())):
      if track_id1 in ignore_pairs[0] and track_id2 in ignore_pairs[1]:
        spatial_dist[i, j] = 9999.

  return spatial_dist


def compute_feature_dist(tracks1, tracks2, spatial_dist):
  """Compute squared l2 distance, save time on the sqrt op"""
  N = len(tracks1)
  M = len(tracks2)
  feature_dist = np.ones((N, M), dtype="float") * 999

  for i, track_id1 in enumerate(sorted(tracks1.keys())):
    track1 = tracks1[track_id1][1] # features [K1, 512]
    for j, track_id2 in enumerate(sorted(tracks2.keys())):
      track2 = tracks2[track_id2][1] # features [K2, 512]
      if spatial_dist[i, j] < 9999.:
        # [K1, K2]
        dist_mat = euclidean_distances(track1, track2, squared=True)
        min_dist = dist_mat.min()

        feature_dist[i, j] = min_dist
  return feature_dist

def get_cur_links(tracks1, tracks2, video_name1, video_name2,
                  global_track_ids, cat_name):

  # 1. get trackid pairs that already in the same global track
  linked_pairs = [[], []]

  for gid in global_track_ids[cat_name]:
    track_id_set = global_track_ids[cat_name][gid]
    for track_id1 in tracks1:
      for track_id2 in tracks2:
        key1 = (video_name1, track_id1)
        key2 = (video_name2, track_id2)
        if key1 in track_id_set and key2 in track_id_set:
          linked_pairs[0].append(track_id1)
          linked_pairs[1].append(track_id2)
  # 2. get trackid pairs [at most NxM] that are in separate global tracks
  # so we don't want to accidentally match them
  track1_ids_in_global = []
  track2_ids_in_global = []
  for track_id in tracks1:
    key = (video_name1, track_id)
    for gid in global_track_ids[cat_name]:
      track_id_set = global_track_ids[cat_name][gid]
      if key in track_id_set:
        track1_ids_in_global.append(track_id)
  for track_id in tracks2:
    key = (video_name2, track_id)
    for gid in global_track_ids[cat_name]:
      track_id_set = global_track_ids[cat_name][gid]
      if key in track_id_set:
        track2_ids_in_global.append(track_id)


  return linked_pairs, (track1_ids_in_global, track2_ids_in_global)



def create_or_merge_global_id(global_track_ids, cat_name,
                              video_name1, track_id1,
                              video_name2, track_id2):
  key1 = (video_name1, track_id1)
  key2 = (video_name2, track_id2)
  found = None
  for gid in global_track_ids[cat_name]:
    track_id_set = global_track_ids[cat_name][gid]
    if key1 in track_id_set or key2 in track_id_set:
      found = gid
      break
  if found is None:
    # global track Id start from 1
    new_gid = len(global_track_ids[cat_name]) + 1
    global_track_ids[cat_name][new_gid] = set([key1, key2])
  else:
    global_track_ids[cat_name][found].add(key1)
    global_track_ids[cat_name][found].add(key2)


def save_new_track(cat_name, track_data, global_track, out_dir, video_name):
  # save the global track id in the x,y,z
  track_results = sorted(
      [b.tolist() for t in track_data for b in track_data[t][0]],
      key=lambda x: (x[0], x[1]))
  # make a reverse index first
  local_to_global_track_ids = {tid: gid
                               for gid in global_track
                               for (vn, tid) in global_track[gid]
                               if vn == video_name}
  out_file_dir = os.path.join(out_dir, video_name + ".avi", cat_name)
  if not os.path.exists(out_file_dir):
    os.makedirs(out_file_dir)
  out_file = os.path.join(
      out_file_dir, video_name + ".txt")
  with open(out_file, "w") as fw:
    for row in track_results:
      # replace all local track_id with global track
      local_track_id = row[1]
      global_track_id = -1
      if local_track_id in local_to_global_track_ids:
        global_track_id = local_to_global_track_ids[local_track_id]
      line = "%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,-1,-1" % (
          row[0], local_track_id, row[2], row[3], row[4], row[5], row[6],
          global_track_id)
      fw.write(line + "\n")

if __name__ == "__main__":
  args = parser.parse_args()
  np.set_printoptions(precision=2, suppress=True)

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

  # compute homography first
  # camera -> H
  hs = compute_homographys(
      args.topdown_camera, args.camera_model_path, camera_model)

  with open(args.camera_group, "r") as f:
    camera_data = json.load(f)

  # reid among synchronized videos

  print("reid in sync groups...")
  for time_slot in tqdm(camera_data["sync_groups"]):
    global_track_ids = {
        "Person": {},
        "Vehicle": {},
    }  # id -> a set of (video_name, track_id)
    # 1. extract track data and features from each video
    # video_name -> object -> track_id -> boxes [N, 9] and features [<M, 512]
    tracks = {}
    for video_name in camera_data["sync_groups"][time_slot]:
      person_track_file = os.path.join(
          args.filepath, video_name + ".avi", "Person", video_name + ".txt")
      vehicle_track_file = os.path.join(
          args.filepath, video_name + ".avi", "Vehicle", video_name + ".txt")
      if not os.path.exists(person_track_file) or not os.path.exists(vehicle_track_file):
        tqdm.write("skipping %s due to track not exists" % video_name)
        continue

      # compute the top-down coordinates as well
      tracks[video_name] = load_track_and_features(
          args, video_name, person_track_file, vehicle_track_file,
          person_reid_extractor, vehicle_reid_extractor, hs)


    video_names = sorted(tracks.keys())
    for cat_name in ["Person", "Vehicle"]:
      # bubble compare
      for i in range(len(video_names) - 1):
        # compare to all other video's tracks
        for j in range(i + 1, len(video_names)):
          tracks1 = tracks[video_names[i]][cat_name]
          tracks2 = tracks[video_names[j]][cat_name]
          # some pairs in tracks1 and tracks2 might already be linked in the
          # global track in preivous a -> b, a -> c, so b -> c mapping
          # check and remove the already linked tracks
          linked_pairs, dont_match_pairs = get_cur_links(
              tracks1, tracks2,
              video_names[i], video_names[j],
              global_track_ids, cat_name)
          tracks1 = {tid: tracks1[tid]
                     for tid in tracks1 if tid not in linked_pairs[0]}
          tracks2 = {tid: tracks2[tid]
                     for tid in tracks2 if tid not in linked_pairs[1]}
          if not tracks1 or not tracks2:
            continue
          # theses time sync are only accurate within 1-2 seconds
          frame_offset = compute_frame_offset(
              video_names[i], video_names[j], 30.0)

          # [N, M]
          # the number of time-intersected trajectory that is within tol
          spatial_dist = compute_spatial_dist(
              tracks1, tracks2, frame_offset, tol=args.spatial_dist_tol,
              ignore_pairs=dont_match_pairs)


          #cost, x, y = lap.lapjv(spatial_dist, extend_cost=True, cost_limit=998.)

          # ignoring large spatial dist items
          feat_dist = compute_feature_dist(tracks1, tracks2, spatial_dist)

          # minimize the total cost
          cost, x, y = lap.lapjv(feat_dist, extend_cost=True, cost_limit=998.)

          tracks1_ids = sorted(tracks1.keys())
          tracks2_ids = sorted(tracks2.keys())

          """
          print(video_names[i], video_names[j])
          print(feat_dist)
          print(x, y)
          for ix, match_y in enumerate(x):
            if match_y >= 0:
              print("track 1 %s -> %s in track 2" % (
                  tracks1_ids[ix], tracks2_ids[match_y]))
          sys.exit()
          """
          for ix, match_y in enumerate(x):
            if match_y >= 0:
              matched_track1_id = tracks1_ids[ix]
              matched_track2_id = tracks2_ids[match_y]
              create_or_merge_global_id(
                  global_track_ids, cat_name,
                  video_names[i], matched_track1_id,
                  video_names[j], matched_track2_id)

    for cat_name in ["Person", "Vehicle"]:
      tqdm.write("group %s %s videos total %s %s track, %s got into %s global track" % (
          time_slot, len(camera_data["sync_groups"][time_slot]),
          sum([len(tracks[vn][cat_name]) for vn in tracks]), cat_name,
          sum([len(global_track_ids[cat_name][gid]) for gid in global_track_ids[cat_name]]),
          len(global_track_ids[cat_name])))

    # save the results
    for video_name in camera_data["sync_groups"][time_slot]:
      for cat_name in ["Person", "Vehicle"]:
        save_new_track(
            cat_name, tracks[video_name][cat_name], global_track_ids[cat_name],
            args.newfilepath, video_name)


  print("Done reid in sync group.")

  # TODO: multi-reid in consecutive camera groups



