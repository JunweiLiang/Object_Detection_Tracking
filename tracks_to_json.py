# coding=utf-8
# convert the detections or mtsc txt file into json for each frame
import sys, os, json, argparse

from tqdm import tqdm
from glob import glob

from class_ids import targetClass2id_new_nopo, targetAct2id_bupt

targetClass2id = targetClass2id_new_nopo

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="all txt files for each video")
parser.add_argument("videonamelst")
parser.add_argument("despath", help="despath/videoname_F_08d.json, index from 0")
parser.add_argument("--bupt_exp", action="store_true")


if __name__ == "__main__":
	args = parser.parse_args()

	# leave the .mp4
	videonames = [os.path.basename(line.strip()) for line in open(args.videonamelst,"r").readlines()]

	if not os.path.exists(args.despath):
		os.makedirs(args.despath)

	if args.bupt_exp:
		targetClass2id = targetAct2id_bupt

	for videoname in tqdm(videonames, ascii=True):
		#detfile = os.path.join(args.filepath, "%s.txt"%videoname)
		detfiles = glob(os.path.join(args.filepath, videoname, "*", "%s.txt" % (os.path.splitext(videoname)[0])))

		data = {} # frame -> boxes
		for detfile in detfiles:
			cat_name = detfile.split("/")[-2]
			for line in open(detfile, "r").readlines():
				# note the frameIdx start from 1
				frameIdx, track_id, left, top, width, height, conf, _, _, _ = line.strip().split(",")
				frameIdx = int(frameIdx) - 1  # note here I made a mistake, gt is 1-indexed, but out obj_tracking output is 0-indexed

				track_id = int(track_id)

				box = [float(left), float(top), float(width), float(height)]

				if not data.has_key(frameIdx):
					data[frameIdx] = []
				data[frameIdx].append({
					"category_id": targetClass2id[cat_name],
					"cat_name": cat_name,
					"score": float(round(float(conf), 7)),
					"bbox": box,
					"segmentation": None,
					"trackId": track_id
				})

		for frameIndex in data:
			
			annofile = os.path.join(args.despath, "%s_F_%08d.json"%(os.path.splitext(videoname)[0], frameIndex))

			with open(annofile, "w") as f:
				json.dump(data[frameIndex], f)

