# coding=utf-8
# visualize detection or mtsc jsons

import sys, os, argparse, cv2, copy, json
from tqdm import tqdm
from glob import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("videonamelst")
parser.add_argument("framepath")
parser.add_argument("jsonpath")
parser.add_argument("despath")
parser.add_argument("--score_thres", default=0.0, type=float)

PALETTE_HEX = [
	"#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
	"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
	"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
	"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
	"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
	"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
	"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
	"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
	"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
	"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
	"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
	"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
	"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
	"#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
	"#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
	"#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
	"#7ED379", "#012C58"]


def _parse_hex_color(s):
	r = int(s[1:3], 16)
	g = int(s[3:5], 16)
	b = int(s[5:7], 16)
	return (r, g, b)

COLORS = list(map(_parse_hex_color, PALETTE_HEX))
PALETTE_RGB = np.asarray(COLORS,dtype="int32")

class BoxBase(object):
	__slots__ = ['x1', 'y1', 'x2', 'y2']

	def __init__(self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def copy(self):
		new = type(self)()
		for i in self.__slots__:
			setattr(new, i, getattr(self, i))
		return new

	def __str__(self):
		return '{}(x1={}, y1={}, x2={}, y2={})'.format(
			type(self).__name__, self.x1, self.y1, self.x2, self.y2)

	__repr__ = __str__

	def area(self):
		return self.w * self.h

	def is_box(self):
		return self.w > 0 and self.h > 0


class IntBox(BoxBase):
	def __init__(self, x1, y1, x2, y2):
		for k in [x1, y1, x2, y2]:
			assert isinstance(k, int)
		super(IntBox, self).__init__(x1, y1, x2, y2)

	@property
	def w(self):
		return self.x2 - self.x1 + 1

	@property
	def h(self):
		return self.y2 - self.y1 + 1

	def is_valid_box(self, shape):
		"""
		Check that this rect is a valid bounding box within this shape.
		Args:
			shape: int [h, w] or None.
		Returns:
			bool
		"""
		if min(self.x1, self.y1) < 0:
			return False
		if min(self.w, self.h) <= 0:
			return False
		if self.x2 >= shape[1]:
			return False
		if self.y2 >= shape[0]:
			return False
		return True

	def clip_by_shape(self, shape):
		"""
		Clip xs and ys to be valid coordinates inside shape
		Args:
			shape: int [h, w] or None.
		"""
		self.x1 = np.clip(self.x1, 0, shape[1] - 1)
		self.x2 = np.clip(self.x2, 0, shape[1] - 1)
		self.y1 = np.clip(self.y1, 0, shape[0] - 1)
		self.y2 = np.clip(self.y2, 0, shape[0] - 1)

	def roi(self, img):
		assert self.is_valid_box(img.shape[:2]), "{} vs {}".format(self, img.shape[:2])
		return img[self.y1:self.y2 + 1, self.x1:self.x2 + 1]

# from tensorpack
def draw_boxes(im, boxes, labels=None, colors=None,font_scale=0.6,font_thick=1,box_thick=1,bottom_text=False,offsets=None):
	if len(boxes) == 0:
		return im

	boxes = np.asarray(boxes,dtype="int")

	FONT = cv2.FONT_HERSHEY_SIMPLEX
	FONT_SCALE = font_scale


	if labels is not None:
		assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
	if colors is not None:
		assert len(labels) == len(colors)
	areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
	sorted_inds = np.argsort(-areas)	# draw large ones first
	assert areas.min() > 0, areas.min()
	# allow equal, because we are not very strict about rounding error here
	#assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
	#	and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
	#	"Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

	im = im.copy()
	COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2), dtype='int32')	# https://www.wikiwand.com/en/Color_difference
	COLOR_CANDIDATES = PALETTE_RGB[:, ::-1]
	if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
		im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	for i in sorted_inds:
		box = boxes[i, :]
		# for cropped visualization
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
			if top_left[1] < 0:	 # out of image
				top_left[1] = box[3] - 1.3 * lineh
				bottom_left[1] = box[3] - 0.3 * lineh

			textbox = IntBox(int(top_left[0]), int(top_left[1]),
							 int(top_left[0] + linew), int(top_left[1] + lineh))
			textbox.clip_by_shape(im.shape[:2])

			offset = 0
			if offsets is not None:
				offset = lineh * offsets[i]

			if color is None:
				# find the best color
				mean_color = textbox.roi(im).mean(axis=(0, 1))
				best_color_ind = (np.square(COLOR_CANDIDATES - mean_color) *
								  COLOR_DIFF_WEIGHT).sum(axis=1).argmax()
				best_color = COLOR_CANDIDATES[best_color_ind].tolist()

			if bottom_text:
				cv2.putText(im, label, (box[0] + 2,box[3] - 4 + offset),
						FONT, FONT_SCALE, color=best_color)
			else:
				cv2.putText(im, label, (textbox.x1, textbox.y2 - offset),
						FONT, FONT_SCALE, color=best_color)#, lineType=cv2.LINE_AA)
		# expand the box on y axis for overlapping results
		offset = 0
		if offsets is not None:
			offset = lineh * offsets[i]
			box[0]-= box_thick * offsets[i] + 1
			box[2]+= box_thick * offsets[i] + 1
			if bottom_text:
				box[1]-= box_thick * offsets[i] + 1
				box[3]+=offset
			else:
				box[3]+= box_thick * offsets[i] + 1
				box[1]-=offset

		cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
					  color=best_color, thickness=box_thick)
	return im

if __name__ == "__main__":
	args = parser.parse_args()

	videonames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in open(args.videonamelst, "r").readlines()]

	color_queue = copy.deepcopy(COLORS)
	color_assign = {} # track Id -> / "cat_name" ->

	for videoname in tqdm(videonames, ascii=True):
		frames = glob(os.path.join(args.framepath, videoname, "*.jpg"))

		target_path = os.path.join(args.despath, videoname)
		if not os.path.exists(target_path):
			os.makedirs(target_path)

		for frame in frames:
			filename = os.path.splitext(os.path.basename(frame))[0]

			jsonfile = os.path.join(args.jsonpath, "%s.json"%filename)

			boxes = []
			labels = []
			box_colors = []
			if os.path.exists(jsonfile):

				with open(jsonfile, "r") as f:
					data = json.load(f)
				for one in data:
					if one['score'] < args.score_thres:
						continue
					box = one['bbox'] # [x, y, w, h]
					box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
					boxes.append(box)
					if one.has_key("trackId"):
						trackId = int(one['trackId'])
						color_key = (trackId, one['cat_name'])
						labels.append("%s: #%s"%(one['cat_name'], trackId))
						if not color_assign.has_key(color_key):
							this_color = color_queue.pop()
							color_assign[color_key] = this_color
							# recycle it
							color_queue.insert(0, this_color)
						color = color_assign[color_key]
						box_colors.append(color)
					else:
						# no trackId, just visualize the boxes
						cat_name = one['cat_name']
						labels.append("%s: %.2f"%(cat_name, float(one['score'])))
						if not color_assign.has_key(cat_name):
							this_color = color_queue.pop()
							color_assign[cat_name] = this_color
							# recycle it
							color_queue.insert(0, this_color)
						color = color_assign[cat_name]
						box_colors.append(color)

			ori_im = cv2.imread(frame, cv2.IMREAD_COLOR)

			new_im = draw_boxes(ori_im, boxes, labels, box_colors, font_scale=0.8, font_thick=10, box_thick=2,bottom_text=False)

			target_file = os.path.join(target_path, "%s.jpg"%filename)

			cv2.imwrite(target_file, new_im)
