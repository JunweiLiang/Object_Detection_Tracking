# coding=utf-8
# convert the detections or mtsc txt file into json for each frame
import sys, os, json, argparse

from tqdm import tqdm

from class_ids import targetClass2id_new_nopo

targetClass2id = targetClass2id_new_nopo

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="all  txt files for each video")
parser.add_argument("videonamelst")
parser.add_argument("cat_name")
parser.add_argument("despath", help="despath/videoname_F_08d.json, index from 0")


if __name__ == "__main__":
	args = parser.parse_args()

	videonames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(args.videonamelst,"r").readlines()]

	if not os.path.exists(args.despath):
		os.makedirs(args.despath)

	for videoname in tqdm(videonames, ascii=True):
		detfile = os.path.join(args.filepath, "%s.txt"%videoname)

		data = {} # frame -> boxes

		for line in open(detfile, "r").readlines():
			# note the frameIdx start from 1
			frameIdx, track_id, left, top, width, height, conf, _, _, _ = line.strip().split(",")
			frameIdx = int(frameIdx) - 1  # note here I made a mistake, gt is 1-indexed, but out obj_tracking output is 0-indexed

			track_id = int(track_id)

			box = [float(left), float(top), float(width), float(height)]

			if not data.has_key(frameIdx):
				data[frameIdx] = []
			data[frameIdx].append({
				"category_id": targetClass2id[args.cat_name],
				"cat_name": args.cat_name,
				"score":float(round(float(conf), 7)),
				"bbox": box,
				"segmentation": None,
				"trackId": track_id
			})

		for frameIndex in data:
			
			annofile = os.path.join(args.despath, "%s_F_%08d.json"%(videoname, frameIndex))

			with open(annofile, "w") as f:
				json.dump(data[frameIndex], f)

