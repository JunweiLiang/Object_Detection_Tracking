# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

from six.moves import range
import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#	>> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#	>> anchors
#
#	anchors =
#
#	   -83   -39   100	56
#	  -175   -87   192   104
#	  -359  -183   376   200
#	   -55   -55	72	72
#	  -119  -119   136   136
#	  -247  -247   264   264
#	   -35   -79	52	96
#	   -79  -167	96   184
#	  -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#	   [-175.,  -87.,  192.,  104.],
#	   [-359., -183.,  376.,  200.],
#	   [ -55.,  -55.,   72.,   72.],
#	   [-119., -119.,  136.,  136.],
#	   [-247., -247.,  264.,  264.],
#	   [ -35.,  -79.,   52.,   96.],
#	   [ -79., -167.,   96.,  184.],
#	   [-167., -343.,  184.,  360.]])
# base_size -> anchor_stride=16,
# scales -> scales=np.array((32, 64, 128, 256, 512), dtype=np.float) / 16,
# generate anchor for one position
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
					 scales=2**np.arange(3, 6)):
	"""
	Generate anchor (reference) windows by enumerating aspect ratios X
	scales wrt a reference (0, 0, 15, 15) window.
	"""
	# anchor box, 0-indexed, x1,y1,x2,y2
	base_anchor = np.array([1, 1, base_size, base_size], dtype='float32') - 1
	# with the same center, same size, -> [0.5,1.0,2.0] boxes
	# [[0,0,15,15],[0,0,22,11.],..]
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	# -> [[0,0,31,31],....]
	anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
						 for i in range(ratio_anchors.shape[0])])
	return anchors

def _whctrs(anchor): # x1,y1,x2,y2: (0,0,15,15) -> (16,16,8,8)
	"""
	Return width, height, x center, and y center for an anchor (window).
	"""

	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5 * (w - 1)
	y_ctr = anchor[1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
	"""
	Given a vector of widths (ws) and heights (hs) around a center
	(x_ctr, y_ctr), output a set of anchors (windows).
	"""

	ws = ws[:, np.newaxis] # [k] -> [k,1]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
						 y_ctr - 0.5 * (hs - 1),
						 x_ctr + 0.5 * (ws - 1),
						 y_ctr + 0.5 * (hs - 1)))
	return anchors

def _ratio_enum(anchor, ratios):
	"""
	Enumerate a set of anchors for each aspect ratio wrt an anchor.
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor) # 0,0,15,15 -> # 16,16, 8,8,
	size = w * h # 16 * 16 = 256
	# given the same size, get the box with different ratio
	size_ratios = size / ratios # ratios: [0.5,1,2] -> [512,256,128]
	ws = np.round(np.sqrt(size_ratios)) # np_round to a int, -> [sqrt(512),16,sqrt(128)]
	hs = np.round(ws * ratios)  # [sqrt(512)*0.5, 16 * 1, sqrt(128)*2]
	# ws*hs == w*h
	# get anchors with the same x,y,center
	# a list of [x1,y1,x2,y2]
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def _scale_enum(anchor, scales):
	"""
	Enumerate a set of anchors for each scale wrt an anchor.
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor)
	ws = w * scales
	hs = h * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

if __name__ == '__main__':
	#import time
	#t = time.time()
	#a = generate_anchors()
	#print(time.time() - t)
	#print(a)
	#from IPython import embed; embed()

	print(generate_anchors(
				16, scales=np.asarray((2, 4, 8, 16, 32), 'float32'),
				ratios=[0.5,1,2]))
