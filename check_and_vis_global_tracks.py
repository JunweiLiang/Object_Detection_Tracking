# coding=utf-8
# given the multi-reid track output, compute stats, visualize all global tracks
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import cv2
import gc

from utils import sec2time, tlbr2tlwh, valid_box
from utils import parse_meva_clip_name
from vis_tracks import draw_boxes

from enqueuer_thread import VideoEnqueuer
from diva_io.video import VideoReader
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument("camera_group")
parser.add_argument("track_out")
parser.add_argument("--video_path")
parser.add_argument("--vis_out_path")
parser.add_argument("--vis_w", type=int, default=1280)
parser.add_argument("--vis_h", type=int, default=720)
parser.add_argument("--vis_cats", default="Person,Vehicle")

parser.add_argument("--job", type=int, default=1, help="total job")
parser.add_argument("--curJob", type=int, default=1,
                    help="this script run job Num")

parser.add_argument("--use_lijun_video_loader", action="store_true")
parser.add_argument("--use_moviepy", action="store_true")
parser.add_argument("--max_size", type=int, default=1920)
parser.add_argument("--short_edge_size", type=int, default=1080)

def load_track_file(file_path):
  """load a tracking file into dict of numpy arrays."""
  # assuming sorted by frameid
  data = []
  with open(file_path, "r") as f:
    for line in f:
      frame_idx, track_id, left, top, width, height, conf, gid, _, _ = line.strip().split(",")
      # save the track data as tlbr
      left, top, width, height = float(left), float(top), float(width), float(height)
      data.append([frame_idx, track_id, left, top, left + width, top + height, conf, gid])

  if not data:
    return {}, []

  data = np.array(data, dtype="float32")  # [N, 8]

  track_ids = np.unique(data[:, 1]).tolist()
  track_data = {}  # [num_track, K, 9]
  for track_id in track_ids:
    track_data[track_id] = data[data[:, 1] == track_id, :]
  # check for global track ids
  gids = [(tid, track_data[tid][0, 7])
          for tid in track_data
          if track_data[tid][0, 7] != -1]
  return track_data, gids


def compute_frame_offset(time_slot, v2, fps):
  date1, start_time1, end_time = time_slot.split(".")
  date2, start_time2, end_time, location, camera = v2.split(".")
  assert date1 == date2
  def time2sec(time_str):
    # hour-minutes-seconds
    hours, minutes, seconds = time_str.split("-")
    return float(hours)*60.*60. + float(minutes)*60. + float(seconds)

  time_offset = time2sec(start_time2) - time2sec(start_time1)
  return time_offset * fps


def get_video_reader(args, video_name):
  date, hr_slot, camera = parse_meva_clip_name(video_name)
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
  return get_frame_batches, video_queuer.num_batches

def expand_tlbr(tlbr, pixels=50, img_size=(1920, 1080)):
  x1, y1, x2, y2 = tlbr
  x1 -= pixels
  y1 -= pixels
  x2 += pixels
  y2 += pixels
  x1 = 0 if x1 < 0 else x1
  y1 = 0 if y1 < 0 else y1
  x2 = img_size[0] - 1 if x2 >= img_size[0] else x2
  y2 = img_size[1] - 1 if y2 >= img_size[1] else y2
  return [x1, y1, x2, y2]

def draw_box(img, local_tlbr, track_id, gid):
  boxes = [local_tlbr]
  labels = ["%d#%d" % (gid, track_id)]
  box_colors = [(0, 0, 255)]  # red

  img = draw_boxes(img, boxes, labels, box_colors, font_scale=0.6,
                   font_thick=2, box_thick=1, bottom_text=False)
  # write the text
  #img = cv2.putText(img, "%s # %d" % (video_name, frame_idx),
  #                  (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
  #                  1, (0, 255, 0), 2)
  return img

def arrange_vis(vns, base_w=1920, base_h=1080):
  # return each video's wh, and their top-left corner
  assert len(vns) > 1
  if len(vns) > 9:
    print("warning, ignoring %s videos" % (len(vns) - 9))
    vns = vns[:9]
  if len(vns) == 2:
    block_wh = (int(base_w/2), int(base_h/2))
    total_wh = (int(base_w), int(base_h/2))
    return total_wh, block_wh, {vns[0]: (0, 0), vns[1]: (block_wh[0], 0)}
  elif len(vns) > 2 and len(vns) <= 4:
    block_wh = (int(base_w/2), int(base_h/2))
    total_wh = (int(base_w), int(base_h))
    base = {
        vns[0]: (0, 0),
        vns[1]: (block_wh[0], 0),
        vns[2]: (0, block_wh[1])}
    if len(vns) == 4:
      base.update({
          vns[3]: (block_wh[0], block_wh[1])})
    return total_wh, block_wh, base
  elif len(vns) > 4 and len(vns) <= 9:
    # for 9 videos, we will keep using 1920x1080
    block_wh = (640, 360)
    total_wh = (1920, 1080)
    # for 5 videos
    base = {
        vns[0]: (0, 0),
        vns[1]: (block_wh[0], 0),
        vns[2]: (block_wh[0]*2, 0),
        vns[3]: (0, block_wh[1]),
        vns[4]: (block_wh[0], block_wh[1])}
    if len(vns) >= 6:
      base.update({
          vns[5]: (block_wh[0]*2, block_wh[1])})
    if len(vns) >= 7:
      base.update({
          vns[6]: (0, block_wh[1]*2)})
    if len(vns) >= 8:
      base.update({
          vns[7]: (block_wh[0], block_wh[1]*2)})
    if len(vns) == 9:
      base.update({
          vns[8]: (block_wh[0]*2, block_wh[1]*2)})

    return total_wh, block_wh, base
  else:
    raise Exception(vns)


def keep_ratio_resize(img, target_wh):
  # resize the img to target_wh by keeping original ratio, filled with black
  target_w, target_h = target_wh
  ori_h, ori_w = img.shape[:2]

  target_ratio = target_h / target_w
  ori_ratio = ori_h / ori_w

  if target_ratio > ori_ratio:
    resize_w = target_w
    resize_h = int(round(resize_w * ori_ratio))
  else:
    resize_h = target_h
    resize_w = int(round(resize_h / ori_ratio))
  img_resized = cv2.resize(img, (resize_w, resize_h))

  new_img = np.zeros((target_h, target_w, 3), dtype="uint8")
  new_img[:resize_h, :resize_w] = img_resized

  return new_img



if __name__ == "__main__":
  np.set_printoptions(precision=2, suppress=True)
  args = parser.parse_args()

  with open(args.camera_group) as f:
    camera_data = json.load(f)

  # some stats
  total_person_num = [0, 0]  # from track/total global track
  total_vehicle_num = [0, 0]
  fps = 30.0
  global_track_lengths = {
      "Person": [],  # -> a list of seconds
      "Vehicle": [],
  }
  global_track_cams = {
      "Person": [],  # -> a list of number of cameras
      "Vehicle": [],
  }
  longest_track = {
      "Person": [0, None, None],
      "Vehicle": [0, None, None],
  }
  most_camera_track = {
      "Person": [0, None, None],
      "Vehicle": [0, None, None],
  }

  if args.vis_out_path is not None and args.video_path is not None:
    if not os.path.exists(args.vis_out_path):
      os.makedirs(args.vis_out_path)

    vis_cats = args.vis_cats.split(",")
    print("Will visualize %s" % vis_cats)

  count = 0
  for time_slot in tqdm(camera_data["sync_groups"]):
    count += 1
    if (count % args.job) != (args.curJob-1):
      continue
    global_tracks = {
        "Person": {},
        "Vehicle": {},
    }  # -> global_track_id -> a list of (video_name, track_id)
    tracks = {
        "Person": {},
        "Vehicle": {},
    }  # -> video_name -> track_id -> [N, 9]
    for video_name in camera_data["sync_groups"][time_slot]:
      person_track_file = os.path.join(
          args.track_out, video_name + ".avi", "Person", video_name + ".txt")
      vehicle_track_file = os.path.join(
          args.track_out, video_name + ".avi", "Vehicle", video_name + ".txt")
      if not os.path.exists(person_track_file) or not os.path.exists(vehicle_track_file):
        tqdm.write("skipping %s due to track not exists" % video_name)
        continue
      # gids: a list of (track_id , global_id)
      p_track_data, p_gids = load_track_file(person_track_file)
      v_track_data, v_gids = load_track_file(vehicle_track_file)

      # save the results
      tracks["Person"][video_name] = p_track_data
      tracks["Vehicle"][video_name] = v_track_data
      for p_tid, p_gid in p_gids:
        if p_gid not in global_tracks["Person"]:
          global_tracks["Person"][p_gid] = []
        global_tracks["Person"][p_gid].append((video_name, p_tid))
      for v_tid, v_gid in v_gids:
        if v_gid not in global_tracks["Vehicle"]:
          global_tracks["Vehicle"][v_gid] = []
        global_tracks["Vehicle"][v_gid].append((video_name, v_tid))

    for cat_name in ["Person", "Vehicle"]:

      print("%s %s videos has total %s %s track, %s track got into %s global track" % (
          time_slot, len(camera_data["sync_groups"][time_slot]),
          sum([len(tracks[cat_name][vn]) for vn in camera_data["sync_groups"][time_slot]]),
          cat_name,
          sum([len(global_tracks[cat_name][gid]) for gid in global_tracks[cat_name]]),
          len(global_tracks[cat_name])))

      # compute global track length stats
      # 1. compute the time span of each track
      for gid in global_tracks[cat_name]:
        g_start, g_end = 99999, -1
        # ignore the time offset of the videos for now
        # all videos are in a 5-min group
        for vn, tid in global_tracks[cat_name][gid]:
          start_sec = tracks[cat_name][vn][tid][0, 0] / fps
          end_sec = tracks[cat_name][vn][tid][-1, 0] / fps
          if start_sec < g_start:
            g_start = start_sec
          if end_sec > g_end:
            g_end = end_sec
        track_length = g_end - g_start
        if track_length >= longest_track[cat_name][0]:
          longest_track[cat_name][0] = track_length
          longest_track[cat_name][1] = (gid, time_slot)
          longest_track[cat_name][2] = global_tracks[cat_name][gid]
        cam_num = len(global_tracks[cat_name][gid])
        if cam_num > most_camera_track[cat_name][0]:
          most_camera_track[cat_name][0] = cam_num
          most_camera_track[cat_name][1] = (gid, time_slot, track_length)
          most_camera_track[cat_name][2] = global_tracks[cat_name][gid]

        global_track_lengths[cat_name].append((track_length, gid))
        global_track_cams[cat_name].append((
            cam_num, gid))


    total_person_num[0] += sum([len(global_tracks["Person"][gid]) for gid in global_tracks["Person"]])
    total_vehicle_num[0] += sum([len(global_tracks["Vehicle"][gid]) for gid in global_tracks["Vehicle"]])
    total_person_num[1] += len(global_tracks["Person"])
    total_vehicle_num[1] += len(global_tracks["Vehicle"])

    # now, visualize if args are given
    if args.vis_out_path is not None and args.video_path is not None:
      if len(camera_data["sync_groups"][time_slot]) == 1:
        continue
      # for each global track, visualize into a single video

      # separate cat_name to save memory
      #for cat_name in ["Person", "Vehicle"]:
      for cat_name in vis_cats:
        # 1. go through all the data, remember all bbox and frames to be extracted
        # gid -> a dict of
        # (videoname, track_id, track_data, track_imgs, converted_data)
        cube_track_datas = {}

        for gid in global_tracks[cat_name]:
          # 1. get each local track's cube representation
          cube_track_datas[gid] = {}

          for video_name, track_id in global_tracks[cat_name][gid]:
            # [N, 8]
            track_data = tracks[cat_name][video_name][track_id][:, :]
            more_data = np.zeros((len(track_data), 5), dtype="float")
            # get the largest box
            cube_tlbr = [track_data[:, 2].min(), track_data[:, 3].min(),
                         track_data[:, 4].max(), track_data[:, 5].max()]
            cube_tlbr = expand_tlbr(cube_tlbr, 50, (1920, 1080))
            # convert the boxes to the cube local coordinates
            more_data[:, 0] = track_data[:, 2] - cube_tlbr[0]
            more_data[:, 1] = track_data[:, 3] - cube_tlbr[1]
            more_data[:, 2] = track_data[:, 4] - cube_tlbr[0]
            more_data[:, 3] = track_data[:, 5] - cube_tlbr[1]

            # convert to the synchronizing frame_id in the group
            frame_offset = compute_frame_offset(time_slot, video_name, 30.0)
            more_data[:, 4] = track_data[:, 0] + frame_offset

            cube_track_datas[gid][(video_name, track_id)] = [
                track_id, track_data, {}, # frame_idx -> img
                more_data, cube_tlbr]
        # now go through the data and get a reversed frameId to box format
        frame_data = {}  # video_name -> frame_idx -> a list of box with gid, trackid
        for gid in cube_track_datas:
          for video_name, track_id in cube_track_datas[gid]:
            _, track_data, _, more_data, cube_tlbr = \
                cube_track_datas[gid][(video_name, track_id)]

            if video_name not in frame_data:
              frame_data[video_name] = {}
            for box, more in zip(track_data, more_data):
              frame_idx = box[0]
              local_tlbr = more[:4]

              if frame_idx not in frame_data[video_name]:
                frame_data[video_name][frame_idx] = []

              frame_data[video_name][frame_idx].append((
                  local_tlbr, cube_tlbr, gid, track_id))

        # move the data back to cube_track_datas
        for video_name in list(frame_data.keys()):
          # read the video and get images with bbox
          vcap, num_frames = get_video_reader(args, video_name)
          for batch in tqdm(vcap, total=num_frames):
            image, scale, frame_idx = batch[0]
            image = image.astype("uint8")  # need uint8 type
            if frame_idx in frame_data[video_name]:
              for local_tlbr, cube_tlbr, gid, track_id in frame_data[video_name][frame_idx]:
                if valid_box(tlbr2tlwh(cube_tlbr), image):
                  x1, y1, x2, y2 = cube_tlbr
                  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                  cube_img = image[y1:y2, x1:x2]
                  # draw bbox
                  cube_img = draw_box(cube_img, local_tlbr, track_id, gid)
                  # this will consume a lot of RAM (70GB) for 5 videos
                  cube_track_datas[gid][(video_name, track_id)][2][frame_idx] = cube_img
            del image
          del frame_data[video_name]
        gc.collect()  # this helps

        # now we can write the video for each cat_name each gid
        for gid in sorted(list(cube_track_datas.keys())):
          video_names = sorted([vn for vn, tid in cube_track_datas[gid]])
          # vis format: 1-1, 2-2, 3-3, so max 9 video
          total_wh, block_wh, vn2tl = arrange_vis(
              video_names, args.vis_w, args.vis_h)

          target_video = os.path.join(args.vis_out_path, "%s.%s.g%d.cam%d.mp4" % (
              time_slot, cat_name, gid, len(video_names)))

          frame_data = {}  # frame_idx -> img

          for vn, track_id in cube_track_datas[gid]:

            _, track_data, f2img, more_data, cube_tlbr = \
                cube_track_datas[gid][(vn, track_id)]
            for frame_idx, sync_fidx in zip(track_data[:, 0], more_data[:, 4]):
              if frame_idx in f2img:
                cube_img = f2img[frame_idx]
                # resize to what we want
                cube_img = keep_ratio_resize(cube_img, block_wh)
                # put some text on
                cube_img = cv2.putText(cube_img, "%s" % (vn),
                                       (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=0.6, color=(0, 255, 0),
                                       thickness=2)
                cube_img = cv2.putText(cube_img, "# %d" % (frame_idx),
                                       (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=0.6, color=(255, 0, 0),
                                       thickness=2)

                # save this to the big img
                if sync_fidx not in frame_data:
                  frame_data[sync_fidx] = np.zeros(
                      (total_wh[1], total_wh[0], 3), dtype="uint8")
                tl = vn2tl[vn]
                frame_data[sync_fidx][tl[1]:tl[1]+block_wh[1], tl[0]:tl[0]+block_wh[0]] = \
                    cube_img

          # start writing at the earliest frame
          frame_idxs = sorted(frame_data.keys())
          if not frame_idxs:
            print(cube_track_datas[gid])
            print("warning, %s has no frame data to write, skipped." % target_video)
            continue

          frame_range = int(min(frame_idxs)), int(max(frame_idxs))
          # the frame idx could be negative
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          fps = 30.0
          video_writer = cv2.VideoWriter(
              target_video, fourcc, fps, total_wh, True)
          for frame_idx in tqdm(range(frame_range[0], frame_range[1])):
            frame_idx = float(frame_idx)  # originally it is float type
            if frame_idx in frame_data:
              frame = frame_data[frame_idx]
            else:
              frame = np.zeros(
                  (total_wh[1], total_wh[0], 3), dtype="uint8")
            video_writer.write(frame)
          video_writer.release()
          cv2.destroyAllWindows()

          del cube_track_datas[gid]
          del frame_data
          gc.collect()
  print("Total [ori_track/global]: Person %s, Vehicle %s" % (
      total_person_num, total_vehicle_num))
  for cat_name in ["Person", "Vehicle"]:
    g_lengths_sorted = sorted(
        global_track_lengths[cat_name], key=lambda x: x[0])
    print("\t %s: %s global track total length %s. shortest %.1f, longest %.1f (g%s), median %.1f" % (
        cat_name,
        len(global_track_lengths[cat_name]),
        sec2time(sum([o[0] for o in global_track_lengths[cat_name]])),
        g_lengths_sorted[0][0], g_lengths_sorted[-1][0], g_lengths_sorted[-1][1],
        np.median([o[0] for o in global_track_lengths[cat_name]])))
    print("\tlongest track gid %s, from %s: %s" % (
        longest_track[cat_name][1][0], longest_track[cat_name][1][1],
        longest_track[cat_name][2]))
    g_cams_sorted = sorted(
        global_track_cams[cat_name], key=lambda x: x[0])
    print("\t %s: %s global track total local tracks %s. least cams %d, most %d (g%s), median %.1f" % (
        cat_name,
        len(global_track_cams[cat_name]),
        sum([o[0] for o in global_track_cams[cat_name]]),
        g_cams_sorted[0][0], g_cams_sorted[-1][0], g_cams_sorted[-1][1],
        np.median([o[0] for o in global_track_cams[cat_name]])))
    print("\tmost_camera track gid %s, length %.1f, from %s: %s" % (
        most_camera_track[cat_name][1][0],
        most_camera_track[cat_name][1][2],
        most_camera_track[cat_name][1][1],
        most_camera_track[cat_name][2]))

