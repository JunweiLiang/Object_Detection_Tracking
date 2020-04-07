# coding=utf-8
# given the maskrcnn json output and the image, visualize

import sys,os,argparse
import cv2

import json
import numpy as np
import pycocotools.mask as cocomask

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("img",help="img has to be the same as the resultjson")
	parser.add_argument("resultjson")
	parser.add_argument("newimg")
	parser.add_argument("--mask",action="store_true",help="whether there is mask in the result")
	parser.add_argument("--kp",action="store_true",help="vis keypoint")
	parser.add_argument("--nobox",action="store_true",help="no bounding box")
	parser.add_argument("--only",default=None,help="only visualize certain class")
	parser.add_argument("--thres",default=0.05,type=float,help="confidence score thresold")
	parser.add_argument("--kp_thres",default=2.0,type=float,help="kp vis threshold, apply to logit")
	parser.add_argument("--ox",default=0,type=int,help="img offset")
	parser.add_argument("--oy",default=0,type=int)
	parser.add_argument("--oxmax",default=-1,type=int)
	parser.add_argument("--oymax",default=-1,type=int)
	return parser.parse_args()

# for cv3
try:
	a = cv2.CV_AA
except Exception as e:
	cv2.CV_AA = cv2.LINE_AA

# copied from https://stackoverflow.com/questions/2328339/how-to-generate-n-different-colors-for-any-natural-number-n
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


PALETTE_RGB = np.asarray(
	list(map(_parse_hex_color, PALETTE_HEX)),
	dtype='int32')


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
def draw_boxes(im, boxes, labels=None, color=None,font_scale=0.3,thickness=1):
	if len(boxes) == 0:
		return im
	"""
	Args:
		im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
		boxes (np.ndarray or list[BoxBase]): If an ndarray,
			must be of shape Nx4 where the second dimension is [x1, y1, x2, y2].
		labels: (list[str] or None)
		color: a 3-tuple (in range [0, 255]). By default will choose automatically.
	Returns:
		np.ndarray: a new image.
	"""
	FONT = cv2.FONT_HERSHEY_SIMPLEX
	FONT_SCALE = font_scale
	if isinstance(boxes, list):
		arr = np.zeros((len(boxes), 4), dtype='int32')
		for idx, b in enumerate(boxes):
			assert isinstance(b, BoxBase), b
			arr[idx, :] = [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
		boxes = arr
	else:
		boxes = boxes.astype('int32')
	if labels is not None:
		assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
	areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
	sorted_inds = np.argsort(-areas)	# draw large ones first
	assert areas.min() > 0, areas.min()
	# allow equal, because we are not very strict about rounding error here
	#assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
	#	and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
	#	"Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

	im = im.copy()
	COLOR = (218, 218, 218) if color is None else color
	COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2), dtype='int32')	# https://www.wikiwand.com/en/Color_difference
	COLOR_CANDIDATES = PALETTE_RGB[:, ::-1]
	if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
		im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	for i in sorted_inds:
		box = boxes[i, :]
		# for cropped visualization
		if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
			continue

		cat_name = labels[i].split(",")[0]
		#color = None
		#if cat2color.has_key(cat_name):
		if cat_name in cat2color:
			color = cat2color[cat_name]

		best_color = COLOR if color is None else color
		if labels is not None:
			label = labels[i]

			# find the best placement for the text
			((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
			bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
			top_left = [box[0] + 1, box[1] - 1.3 * lineh]
			if top_left[1] < 0:	 # out of image
				top_left[1] = box[3] - 1.3 * lineh
				bottom_left[1] = box[3] - 0.3 * lineh
			textbox = IntBox(int(top_left[0]), int(top_left[1]),
							 int(top_left[0] + linew), int(top_left[1] + lineh))
			textbox.clip_by_shape(im.shape[:2])
			if color is None:
				# find the best color
				mean_color = textbox.roi(im).mean(axis=(0, 1))
				best_color_ind = (np.square(COLOR_CANDIDATES - mean_color) *
								  COLOR_DIFF_WEIGHT).sum(axis=1).argmax()
				best_color = COLOR_CANDIDATES[best_color_ind].tolist()
			best_color = list(np.array(best_color, dtype="float"))
			cv2.putText(im, label, (textbox.x1, textbox.y2),
						FONT, FONT_SCALE, color=best_color)#, lineType=cv2.LINE_AA)
		cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
					  color=best_color, thickness=thickness)
	return im

def get_keypoints():
	"""Get the COCO keypoints and their left/right flip coorespondence map."""
	# Keypoints are not available in the COCO json for the test split, so we
	# provide them here.
	keypoints = [
		'nose',
		'left_eye',
		'right_eye',
		'left_ear',
		'right_ear',
		'left_shoulder',
		'right_shoulder',
		'left_elbow',
		'right_elbow',
		'left_wrist',
		'right_wrist',
		'left_hip',
		'right_hip',
		'left_knee',
		'right_knee',
		'left_ankle',
		'right_ankle'
	]
	keypoint_flip_map = {
		'left_eye': 'right_eye',
		'left_ear': 'right_ear',
		'left_shoulder': 'right_shoulder',
		'left_elbow': 'right_elbow',
		'left_wrist': 'right_wrist',
		'left_hip': 'right_hip',
		'left_knee': 'right_knee',
		'left_ankle': 'right_ankle'
	}
	return keypoints, keypoint_flip_map
def kp_connections(keypoints):
	kp_lines = [
		[keypoints.index('left_eye'), keypoints.index('right_eye')],
		[keypoints.index('left_eye'), keypoints.index('nose')],
		[keypoints.index('right_eye'), keypoints.index('nose')],
		[keypoints.index('right_eye'), keypoints.index('right_ear')],
		[keypoints.index('left_eye'), keypoints.index('left_ear')],
		[keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
		[keypoints.index('right_elbow'), keypoints.index('right_wrist')],
		[keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
		[keypoints.index('left_elbow'), keypoints.index('left_wrist')],
		[keypoints.index('right_hip'), keypoints.index('right_knee')],
		[keypoints.index('right_knee'), keypoints.index('right_ankle')],
		[keypoints.index('left_hip'), keypoints.index('left_knee')],
		[keypoints.index('left_knee'), keypoints.index('left_ankle')],
		[keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
		[keypoints.index('right_hip'), keypoints.index('left_hip')],
	]
	return kp_lines
def int_it(w):
	return tuple(int(one) for one in w)
def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
	"""Visualizes keypoints (adapted from vis_one_image).
	kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
	"""
	dataset_keypoints, _ = get_keypoints()
	kp_lines = kp_connections(dataset_keypoints)

	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
	colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

	# Perform the drawing on a copy of the image, to allow for blending.
	kp_mask = np.copy(img)

	# Draw mid shoulder / mid hip first for better visualization.
	mid_shoulder = (
		kps[:2, dataset_keypoints.index('right_shoulder')] +
		kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
	sc_mid_shoulder = np.minimum(
		kps[2, dataset_keypoints.index('right_shoulder')],
		kps[2, dataset_keypoints.index('left_shoulder')])
	mid_hip = (
		kps[:2, dataset_keypoints.index('right_hip')] +
		kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
	sc_mid_hip = np.minimum(
		kps[2, dataset_keypoints.index('right_hip')],
		kps[2, dataset_keypoints.index('left_hip')])
	nose_idx = dataset_keypoints.index('nose')
	if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:

		cv2.line(
			kp_mask, int_it(tuple(mid_shoulder)), int_it(tuple(kps[:2, nose_idx])),
			color=colors[len(kp_lines)], thickness=2, lineType=cv2.CV_AA)
	if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
		cv2.line(
			kp_mask, int_it(tuple(mid_shoulder)), int_it(tuple(mid_hip)),
			color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.CV_AA)

	# Draw the keypoints.
	for l in range(len(kp_lines)):
		i1 = kp_lines[l][0]
		i2 = kp_lines[l][1]
		p1 = int(kps[0, i1]), int(kps[1, i1])
		p2 = int(kps[0, i2]), int(kps[1, i2])
		if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:

			cv2.line(
				kp_mask, p1, p2,
				color=colors[l], thickness=2, lineType=cv2.CV_AA)
		if kps[2, i1] > kp_thresh:
			cv2.circle(
				kp_mask, p1,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.CV_AA)
		if kps[2, i2] > kp_thresh:
			cv2.circle(
				kp_mask, p2,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.CV_AA)

	# Blend the keypoints.
	return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def draw_mask(im, mask, alpha=0.5, color=None, show_border=True,border_thick=1):
	"""
	Overlay a mask on top of the image.

	Args:
		im: a 3-channel uint8 image in BGR
		mask: a binary 1-channel image of the same size
		color: if None, will choose automatically
	"""
	if color is None:
		color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]


	im = np.where(np.squeeze(np.repeat((mask > 0)[:, :, None], 3, axis=2)),
				  im * (1 - alpha) + color * alpha, im)
	if show_border:
		if cv2.__version__.startswith("2"):
			contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		else: # cv 3
			_,contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(im, contours, -1, (255,255,255), border_thick, lineType=cv2.CV_AA)

	im = im.astype('uint8')
	return im

def decode_mask(mask_obj):

	mask = cocomask.decode([mask_obj])
	#print mask.shape
	return mask

# conver from COCO format (x,y,w,h) to (x1,y1,x2,y2)
def convert_box(box):
	return [box[0],box[1],box[0]+box[2],box[1]+box[3]]


# conver from (x1,y1,x2,y2)  to coco (x,y,w,h)
def to_coco_box(box):
	return [box[0],box[1],box[2]-box[0],box[3]-box[1]]

# BGR
cat2color = {
	"car":np.array([255,0,0]),
	"person":np.array([0,255,0])
}

# need box format: (x1,y1,x2,y2)
def draw_result(im,data,hasmask=False,haskp=False,nobox=False,kp_thresh=2.0,font_scale=0.3,thickness=1):
	if len(data) == 0:
		return im
	tags = []
	for one in data:
		tags.append("%s,%.2f"%(one['cat_name'],one['score']))
	boxes = np.asarray([one['bbox'] for one in data])

	if not nobox:
		newim = draw_boxes(im,boxes,tags,color=np.array([255,0,0]),font_scale=font_scale,thickness=thickness)
	else:
		newim = im

	if hasmask:
		for one in data:
			# ---- specially for site visit
			cat_name = one['cat_name']
			color = None
			#if cat2color.has_key(cat_name):
			if cat_name in cat2color:
				color = cat2color[cat_name]
			# --------------------------

			mask = decode_mask(one['segmentation']) # (imgh,imgw,1)

			newim = draw_mask(newim,mask,color=color)

	if haskp:
		#kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
		for one in data:
			newim = vis_keypoints(newim,np.array(one['kps']).reshape(4,17),kp_thresh=kp_thresh)

	return newim

import matplotlib.pyplot as plt



if __name__ == "__main__":
	args = get_args()

	img = cv2.imread(args.img,cv2.IMREAD_COLOR) # (H,W,C)
	h,w = img.shape[:2]
	if args.oxmax < 0:
		args.oxmax = w
	if args.oymax < 0:
		args.oymax = h

	with open(args.resultjson,"r") as f:
		data = json.load(f)

	# --------------------- specially for site visit 02152018
	"""
	cat2thres = {
		"car":0.05,
		"person":0.5
	}

	newdata = []
	for one in data:
		cat_name = one['cat_name']
		if cat2thres.has_key(cat_name):
			if one['score'] >= cat2thres[cat_name]:
				newdata.append(one)

	data = newdata
	# ---------------------

	data = [one for one in data if one['score'] >= args.thres]

	onlys = ["person","car"]
	data = [one for one in data if one['cat_name'] in onlys]
	"""
	# --------------

	data = [one for one in data if one['score'] >= args.thres]

	if args.only is not None:
		data = [one for one in data if one['cat_name'].lower() == args.only.lower()]

	# convert the boexs format from COCO
	for i in range(len(data)):
		data[i]['bbox'] = convert_box(data[i]['bbox'])

	newimg = draw_result(img,data,hasmask=args.mask,haskp=args.kp,nobox=args.nobox,kp_thresh=args.kp_thres)

	newimg = newimg[args.oy:args.oymax,args.ox:args.oxmax,:]

	cv2.imwrite(args.newimg,newimg)
	#plt.imshow(newimg)
	#plt.show()
