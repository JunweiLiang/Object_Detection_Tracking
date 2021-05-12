import numpy as np
from deep_sort.detection import Detection
from bisect import bisect

def create_obj_infos(cur_frame, final_boxes, final_probs, final_labels,
                     box_feats, targetid2class, tracking_objs, min_confidence,
                     min_detection_height, scale, is_coco_model=False,
                     coco_to_actev_mapping=None):

  # tracking_objs is a single item
  obj_infos = []
  tracking_boxes = final_boxes / scale
  for j, (box, prob, label) in enumerate(zip(tracking_boxes, final_probs, final_labels)):
    cat_name = targetid2class[label]
    if is_coco_model:
      if cat_name not in coco_to_actev_mapping:
        continue
      else:
        cat_name = coco_to_actev_mapping[cat_name]

    confidence_socre = float(round(prob, 7))
    if cat_name not in tracking_objs or confidence_socre < min_confidence:
      continue
    box[2] -= box[0]
    box[3] -= box[1]  # x, y, w, h
    avg_feat = box_feats[j]
    if len(avg_feat.shape) > 2:  # [C, H, W]
      avg_feat = np.mean(box_feats[j], axis=(1, 2))


    #norm_feat = avg_feat / np.linalg.norm(avg_feat)  # will be normed later

    list_feat = avg_feat.tolist()
    # frameIdx, xywh, conf, feature
    bbox_data = [cur_frame, box[0], box[1], box[2], box[3], confidence_socre] + list_feat
    obj_infos.append(bbox_data)

  detections = []
  for row in obj_infos:
    bbox, confidence, feature = row[1:5], row[5], row[6:]
    if bbox[3] < min_detection_height:
      continue
    detections.append(Detection(bbox, confidence, feature))
  return detections


# 1
def linear_inter_bbox(tracking_data, frame_gap):
  # print tracking_data.shape
  if tracking_data.shape[0] == 0:
    return tracking_data
  obj_indices = tracking_data[:, 1].astype(np.int)
  obj_ids = set(obj_indices.tolist())
  tracking_data_list = tracking_data.tolist()
  # if len(tracking_data_list) == 0:
  #   return tracking_data

  # for each track
  for obj_index in obj_ids:
    mask = obj_indices == obj_index
    # all the frames for this track
    tracked_frames = tracking_data[mask][:, 0].tolist()

    min_frame_idx = int(min(tracked_frames))
    max_frame_idx = int(max(tracked_frames))
    whole_frames = range(min_frame_idx, max_frame_idx)
    missing_frames = list(set(whole_frames).difference(tracked_frames))
    if not missing_frames:
      continue
    for missing_frame in missing_frames:
      insert_index = bisect(tracked_frames, missing_frame)
      if insert_index == 0 or insert_index == len(whole_frames):
        continue
      selected_data = tracking_data[mask]
      prev_frame = selected_data[insert_index-1, 0]
      next_frame = selected_data[insert_index, 0]
      # tolerate some occlusion?
      if next_frame - prev_frame > 10*frame_gap:
        continue
      prev_data = selected_data[insert_index-1, 2:]
      next_data = selected_data[insert_index, 2:]

      ratio = 1.0 * (missing_frame - prev_frame) / (next_frame - prev_frame)
      cur_data = prev_data + (next_data - prev_data) * ratio
      cur_data = np.around(cur_data, decimals=2)
      missing_data = [missing_frame, obj_index] + cur_data.tolist()
      tracking_data_list.append(missing_data)

  tracking_data_list = sorted(tracking_data_list, key=lambda x: (x[0], x[1]))
  tracking_data = np.asarray(tracking_data_list)
  return tracking_data


# 3
def filter_short_objs(tracking_data):
  # print tracking_data.shape
  if tracking_data.shape[0] == 0:
    return tracking_data
  obj_indices = tracking_data[:, 1].astype(np.int)
  obj_ids = set(obj_indices.tolist())
  filter_objs = set()

  for obj_index in obj_ids:
    mask = obj_indices == obj_index
    num_frames = np.sum(mask)
    if num_frames < 2:
      filter_objs.add(obj_index)

  tracking_data_list = tracking_data.tolist()
  tracking_data_list = [tracklet for tracklet in tracking_data_list if int(tracklet[1]) not in filter_objs]
  tracking_data_list = sorted(tracking_data_list, key=lambda x: (x[0], x[1]))
  tracking_data = np.asarray(tracking_data_list)
  return tracking_data
