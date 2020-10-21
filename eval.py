# coding=utf-8
"""given a file lst, ground truth and detection output, get the eval result."""

from __future__ import print_function

import argparse
import os
import json
import operator
from tqdm import tqdm
import numpy as np
from class_ids import bupt_act_mapping, meva_act_mapping, coco_obj_to_actev_obj
from utils import match_dt_gt, aggregate_eval



parser = argparse.ArgumentParser()
parser.add_argument("filelst")
parser.add_argument("gtpath")
parser.add_argument("outpath")
parser.add_argument("--not_coco_box", action="store_true")
parser.add_argument("--merge_prop", action="store_true",
                    help="this means put all Push_Pulled_Object anno into prop")
parser.add_argument("--skip", type=int, default=1)
parser.add_argument("--skip_not_exist_out", action="store_true")
parser.add_argument("--scene", default=None)

parser.add_argument("--limit", type=int, default=None,
                    help="limit top k per json")
parser.add_argument("--conf_thres", type=float, default=None,
                    help="filter out detection thres <")

parser.add_argument("--bupt_exp", action="store_true",
                    help="bupt act box experiment")
parser.add_argument("--meva_exp", action="store_true",
                    help="meva act box experiment")

parser.add_argument("--is_coco_model", action="store_true",
                    help="The output is in coco class, will map to actev class")

def get_scene(videoname):
  """some decoding of the videoname."""
  s = videoname.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]

def gather_dt(boxes_, probs_, labels_, eval_target_, not_coco_box=False):
  """Gather detection boxes."""
  target_dt_boxes_ = {one:[] for one in eval_target_}
  for box, prob, label in zip(boxes_, probs_, labels_):

    if not_coco_box:
      box[2] -= box[0]
      box[3] -= box[1]

    target_class = None


    if label in eval_target:
      target_class = label

    if target_class is None: # box from other class of mscoco/diva
      continue

    prob = float(round(prob, 4))
    box = [float(round(x, 4)) for x in box]

    target_dt_boxes_[target_class].append((box, prob))
  return target_dt_boxes_


def gather_gt(anno_boxes, anno_labels, eval_target_):
  """Gather ground truth boxes."""
  gt_boxes_ = {one:[] for one in eval_target_}
  for box, label in zip(anno_boxes, anno_labels):
    if label in eval_target:
      gt_box = [float(round(x, 4)) for x in box]
      # gt_box is in (x1,y1,x2,y2)
      # convert to coco box
      gt_box[2] -= gt_box[0]
      gt_box[3] -= gt_box[1]

      gt_boxes_[label].append(gt_box)
  return gt_boxes_


if __name__ == "__main__":
  args = parser.parse_args()

  files = [os.path.splitext(os.path.basename(line.strip()))[0]
           for line in open(args.filelst, "r").readlines()]
  files.sort()
  files = files[::args.skip]

  if args.scene is not None:
    new_files = []
    for f in files:
      scene = get_scene(f)
      if scene == args.scene:
        new_files.append(f)
    print("only eval scene %s, got %s/%s files for eval" % (
        args.scene, len(new_files), len(files)))
    files = new_files

  # previous classes before annotation refining
  #eval_target = ["Vehicle","Person","Construction_Barrier","Construction_Vehicle", "Door","Dumpster","Prop","Push_Pulled_Object","Bike","Parking_Meter", "Prop_plus_Push_Pulled_Object"]
  eval_target = [
      "Vehicle",
      "Person",
      "Construction_Barrier",
      "Construction_Vehicle",
      "Door",
      "Dumpster",
      "Prop",
      "Push_Pulled_Object",
      "Bike",
      "Parking_Meter",
      "Skateboard",
      "Prop_Overshoulder",
  ]

  if args.bupt_exp:
    eval_target = [
        "Person-Vehicle",
        "Vehicle-Turning",
        "activity_carrying",
        "Transport_HeavyCarry",
        "Talking",
        "Pull",
        "Riding",
        "specialized_texting_phone",
        "specialized_talking_phone",
    ]
  if args.meva_exp:
    # removed some classes that we dont have any annotations
    eval_target = [
        "Person-Vehicle",
        "Person-Structure",
        "Vehicle-Turning",

        # "Person_Heavy_Carry",
        "People_Talking",
        # "Riding",
        "Person_Texting_on_Phone",
        "Person_Talking_on_Phone",
        "Person_Sitting_Down",
        "Person_Sets_Down_Object",
        "Person_Standing_Up",
        "Person_Picks_Up_Object",
        # "Person_Purchasing",
        "Person_Reading_Document",
        "Object_Transfer",
        # "Hand_Interaction",
        "Person-Person_Embrace",
        # "Person-Laptop_Interaction",

        "Vehicle_Stopping",
        "Vehicle_Starting",
        "Vehicle_Reversing",
    ]



  eval_target = {one:1 for one in eval_target}

  e = {one:{} for one in eval_target} # cat_id -> imgid -> {"dm","dscores"}

  count_no_out = 0

  gt_has_none = {one: True for one in eval_target}

  for filename in tqdm(files, ascii=True):
    gtfile = os.path.join(args.gtpath, "%s.npz"%filename)
    outfile = os.path.join(args.outpath, "%s.json"%filename)

    # load annotation first
    if not os.path.exists(gtfile):
      continue
    anno = dict(np.load(gtfile, allow_pickle=True))

    if not os.path.exists(outfile):
      count_no_out += 1
      out = []
      if args.skip_not_exist_out:
        continue
    else:
      with open(outfile, "r") as f:
        out = json.load(f)

    if args.conf_thres is not None:
      out = [one for one in out if one["score"] >= args.conf_thres]

    if args.merge_prop:

      for i, one in enumerate(out):
        if one["cat_name"] == "Push_Pulled_Object" or one["cat_name"] == "Prop":
          out[i]["cat_name"] = "Prop_plus_Push_Pulled_Object"

    if args.is_coco_model:
      newout = []
      for one in out:
        if one["cat_name"] in coco_obj_to_actev_obj:
          one["cat_name"] = coco_obj_to_actev_obj[one["cat_name"]]
          newout.append(one)
      out = newout

      # change ground truth, too # groung truth already has it
      # v1-validate_actgt_allsingle_mergeprop_npz
      #for i,one in enumerate(anno["labels"]):
      #  if one == "Push_Pulled_Object" or one == "Prop":
      #    anno["labels"][i] = "Prop_plus_Push_Pulled_Object"

    if args.limit is not None:
      out.sort(key=operator.itemgetter("score"), reverse=True)
      out = out[:args.limit]
    boxes = [one["bbox"] for one in out]
    probs = [one["score"] for one in out]
    labels = [one["cat_name"] for one in out]

    target_dt_boxes = gather_dt(
        boxes, probs, labels, eval_target, not_coco_box=args.not_coco_box)

    if args.bupt_exp:
      anno["labels"] = anno["actlabels"]
      anno["boxes"] = anno["actboxes"]

      anno["labels"] = [
          bupt_act_mapping[one] if one in bupt_act_mapping else one
          for one in anno["labels"]]
    if args.meva_exp:
      anno["labels"] = anno["actlabels"]
      anno["boxes"] = anno["actboxes"]

      anno["labels"] = [
          meva_act_mapping[one] if one in meva_act_mapping else one
          for one in anno["labels"]]

    gt_boxes = gather_gt(anno["boxes"], anno["labels"], eval_target)

    match_dt_gt(e, filename, target_dt_boxes, gt_boxes, eval_target)

    # check for gt class that doesn"t exists in this file list
    for one in anno["labels"]:
      gt_has_none[one] = False

  print("%s/%s out file not exists" % (count_no_out, len(files)))
  no_gt_classes = [one for one in gt_has_none if gt_has_none[one]]
  print("%s class has no ground truth: %s" % (
      len(no_gt_classes), no_gt_classes))
  aps, ars = aggregate_eval(e, maxDet=100)
  aps_str = "|".join(["%s:%.5f" % (class_, aps[class_]) for class_ in aps])
  ars_str = "|".join(["%s:%.5f" % (class_, ars[class_]) for class_ in ars])
  classes = sorted(aps.keys())
  headers = ["metric"] + classes
  print(",".join(headers))
  print(",".join(["AP"] + ["%.6f"%aps[c] for c in classes]))
  print(",".join(["AR"] + ["%.6f"%ars[c] for c in classes]))
