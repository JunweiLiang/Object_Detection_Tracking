# coding=utf-8
# given MOT track file path, visualize into videos
import argparse
import cv2
import random
import os
import sys

from tqdm import tqdm
from glob import glob
import numpy as np

import matplotlib.colors as mcolors  # to get a list of colors

parser = argparse.ArgumentParser()
parser.add_argument("track_path")
parser.add_argument("frame_path")
parser.add_argument("video_name_lst")
parser.add_argument("out_path")
parser.add_argument("--show_only_global", action="store_true")

def hex_color_to_rgb(s):
  r = int(s[1:3], 16)
  g = int(s[3:5], 16)
  b = int(s[5:7], 16)
  return (r, g, b)  # (0-255)

def load_track_file(file_path, cat_names):

  track_data = {}  # frame_id -> {cat_name: }
  video_name = os.path.splitext(os.path.basename(file_path))[0]
  for cat_name in cat_names:
    track_file_path = os.path.join(file_path, cat_name, video_name + ".txt")
    data = []
    with open(track_file_path, "r") as f:
      for line in f:
        frame_idx, track_id, left, top, width, height, conf, gid, _, _ = line.strip().split(",")
        data.append([frame_idx, track_id, left, top, width, height, conf, gid])

    data = np.array(data, dtype="float32")  # [N, 8]
    frame_ids = np.unique(data[:, 0]).tolist()

    for frame_id in frame_ids:
      if frame_id not in track_data:
        track_data[frame_id] = {}
      track_data[frame_id][cat_name] = data[data[:, 0] == frame_id, :]
  return track_data


def get_or_create_color_from_dict(key, color_dict, color_list):
  if key not in color_dict:
    this_color = color_list.pop()

    color_dict[key] = hex_color_to_rgb(color_name_to_hex[this_color])
    # recycle it
    color_list.insert(0, this_color)
  color = color_assign[key]
  return color

def draw_boxes(im, boxes, labels=None, colors=None, font_scale=0.6,
               font_thick=1, box_thick=1, bottom_text=False, offsets=None):
  if not boxes:
    return im

  boxes = np.asarray(boxes, dtype="int")

  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = font_scale

  if labels is not None:
    assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
  if colors is not None:
    assert len(labels) == len(colors)

  areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
  sorted_inds = np.argsort(-areas)  # draw large ones first
  assert areas.min() > 0, areas.min()

  im = im.copy()

  for i in sorted_inds:
    box = boxes[i, :]
    if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
      continue

    color = (218, 218, 218)
    if colors is not None:
      color = colors[i]

    best_color = color

    lineh = 2 # for box enlarging, replace with text height if there is label
    if labels is not None:
      label = labels[i]

      # find the best placement for the text
      ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, font_thick)
      bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
      top_left = [box[0] + 1, box[1] - 1.3 * lineh]
      if top_left[1] < 0:   # out of image
        top_left[1] = box[3] - 1.3 * lineh
        bottom_left[1] = box[3] - 0.3 * lineh

      textbox = [int(top_left[0]), int(top_left[1]),
                 int(top_left[0] + linew), int(top_left[1] + lineh)]
      #textbox.clip_by_shape(im.shape[:2])

      offset = 0
      if offsets is not None:
        offset = lineh * offsets[i]

      if bottom_text:
        cv2.putText(im, label, (box[0] + 2, box[3] - 4 + offset),
                    FONT, FONT_SCALE, color=best_color, thickness=font_thick)
      else:
        cv2.putText(im, label, (textbox[0], textbox[3] - offset),
                    FONT, FONT_SCALE, color=best_color, thickness=font_thick)

    # expand the box on y axis for overlapping results
    offset = 0
    if offsets is not None:
      offset = lineh * offsets[i]
      box[0] -= box_thick * offsets[i] + 1
      box[2] += box_thick * offsets[i] + 1
      if bottom_text:
        box[1] -= box_thick * offsets[i] + 1
        box[3] += offset
      else:
        box[3] += box_thick * offsets[i] + 1
        box[1] -= offset

    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                  color=best_color, thickness=box_thick)
  return im

color_name_to_hex = mcolors.CSS4_COLORS.copy()  # {'whitesmoke': '#F5F5F5', ...}
if __name__ == "__main__":
  args = parser.parse_args()

  color_name_list = sorted(list(color_name_to_hex.keys()))[:]
  random.seed(69)
  random.shuffle(color_name_list)

  color_assign = {} # global track id, obj -> name

  if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

  video_names = [os.path.basename(line.strip())  # with .avi
                 for line in open(args.video_name_lst, "r").readlines()]
  for video_name in tqdm(video_names):
    video_name_no_appendix = os.path.splitext(video_name)[0]
    frames = glob(os.path.join(args.frame_path, video_name_no_appendix, "*.jpg"))
    frames.sort()

    # frame_id -> {cat_name: ..}
    track_data = load_track_file(
        os.path.join(args.track_path, video_name),
        ["Person", "Vehicle"])

    target_file = os.path.join(args.out_path, "%s.mp4" % video_name_no_appendix)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    video_writer = cv2.VideoWriter(target_file, fourcc, fps, (1920, 1080), True)

    count_global_ids = {}
    for frame in frames:
      filename = os.path.splitext(os.path.basename(frame))[0]
      frame_id = int(filename.split("_F_")[-1])

      boxes = []
      labels = []
      box_colors = []
      if frame_id in track_data:
        this_track_data = track_data[frame_id]
        for cat_name in this_track_data:
          for box_data in this_track_data[cat_name]:  # [N, 8]
            # get color and label
            local_track_id = box_data[1]
            global_track_id = box_data[7]
            if global_track_id != -1:
              color_key = (global_track_id, cat_name)
              count_global_ids[color_key] = 1
              track_id = "g%s" % global_track_id
            else:
              if args.show_only_global:
                continue
              color_key = (video_name, local_track_id, cat_name)
              track_id = local_track_id
            color = get_or_create_color_from_dict(
                color_key, color_assign, color_name_list)
            box_colors.append(color)

            conf = box_data[6]
            conf_str = ""
            if conf != 1.:
              conf_str = "%.2f" % conf
            labels.append("%s #%s %s"%(cat_name, track_id, conf_str))

            tlwh = box_data[2:6]
            tlbr = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            boxes.append(tlbr)

      new_im = cv2.imread(frame, cv2.IMREAD_COLOR)
      new_im = draw_boxes(new_im, boxes, labels, box_colors, font_scale=0.8,
                          font_thick=2, box_thick=2, bottom_text=False)
      # write the frame idx
      new_im = cv2.putText(new_im, "# %d" % frame_id,
                           (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
      # the frames might not be 1920x1080
      new_im = cv2.resize(new_im, (1920, 1080))
      video_writer.write(new_im)

    video_writer.release()
    tqdm.write("%s has %s global tracks:%s" % (
        video_name, len(count_global_ids), count_global_ids.keys()))
  cv2.destroyAllWindows()
