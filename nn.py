# coding=utf-8
import tensorflow as tf
import math,re,cv2
from operator import mul
import numpy as np
from deformable_helper import _tf_batch_map_offsets

VERY_NEGATIVE_NUMBER = -1e30


# f(64,2) -> 32, f(32,2) -> 16 
def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))


# softnms
try:
	from softnms.nms import cpu_soft_nms, cpu_nms
except Exception as e:
	from softnms.cpu_nms import cpu_soft_nms, cpu_nms # Uhuh

# given a list of boxes, 
# dets : [N, 5], in (x1,y1,x2,y2, score)
# kevin sigma uses 1.0
# method 1 -> linear, 2 -> gaussian(paper)
#def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
def soft_nms(dets, sigma=0.3, Nt=0.4, threshold=0.001, method=2):
	keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
						np.float32(sigma), np.float32(Nt),
						np.float32(threshold),
						np.uint8(method))
	return keep

# cpu nms
def nms(dets, thresh):
	return cpu_nms(dets, thresh)

# given regex to get the parameter to do regularization
def wd_cost(regex,wd,scope):
	params = tf.trainable_variables()
	with tf.name_scope(scope):
		costs = []
		names = []
		for p in params:
			para_name = p.op.name
			# freeze backbone, temp fix
			if para_name.startswith("conv0") or para_name.startswith("group0"):
				continue
			if re.search(regex, para_name):
				regloss = tf.multiply(tf.convert_to_tensor(wd, dtype=p.dtype.base_dtype,name="scale"), tf.nn.l2_loss(p), name="%s/wd"%p.op.name)
				assert regloss.dtype.is_floating, regloss
				# Some variables may not be fp32, but it should
				# be fine to assume regularization in fp32
				if regloss.dtype != tf.float32:
					regloss = tf.cast(regloss, tf.float32)
				costs.append(regloss)
				names.append(para_name)

		# print the names?
		print "found %s variables for weight reg"%(len(costs))
		if len(costs) == 0:
			return tf.constant(0, dtype=tf.float32, name="empty_"+scope)
		else:
			return tf.add_n(costs,name=scope)


def group_norm(x, group=32, gamma_init=tf.constant_initializer(1.),scope="gn"):
	with tf.variable_scope(scope):
		shape = x.get_shape().as_list()
		ndims = len(shape)
		assert ndims == 4, shape
		chan = shape[1]
		assert chan % group == 0, chan
		group_size = chan // group

		orig_shape = tf.shape(x)
		h, w = orig_shape[2], orig_shape[3]

		x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

		mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

		new_shape = [1, group, group_size, 1, 1]

		beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
		beta = tf.reshape(beta, new_shape)

		gamma = tf.get_variable('gamma', [chan], initializer=gamma_init)
		gamma = tf.reshape(gamma, new_shape)

		out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
		return tf.reshape(out, orig_shape, name='output')


# relation network head, for enhancing the boxes features
# boxe_feat: [K,1024]
# boxes:[K,4]
# group is the same as multi-head in the self attention paper
def relation_network(box_appearance_feat, boxes, group=16, geo_feat_dim=64, scope="RM"):
	fc_dim = group # the geo feature for each group
	with tf.variable_scope(scope):
		box_feat_dim = box_appearance_feat.get_shape().as_list()[-1] # 1024
		group_feat_dim = box_feat_dim / group
		# [K,4] -> [K,K,4]
		# given the absolute box, get the pairwise relative geometric coordinates
		box_geo_encoded = geometric_encoding(boxes, scope="geometric_encoding")
		# [K,K,4] -> [K,K,geo_feat_dim]
		box_geo_feat = dense(box_geo_encoded, geo_feat_dim, activation=tf.nn.tanh, use_bias=True, wd=None, keep_first=False, scope="geo_emb")

		# [K,K,geo_feat_dim]
		box_geo_feat = tf.transpose(box_geo_feat, perm=[2,0,1])
		box_geo_feat = tf.expand_dims(box_geo_feat, axis=0) # [1,geo_feat_dim,K,K]
		# [1,fc_dim,K,K]
		box_geo_feat_wg = conv2d(box_geo_feat,fc_dim,kernel=1,stride=1,data_format="NCHW",scope="geo_conv")
		box_geo_feat_wg = tf.squeeze(box_geo_feat_wg)
		box_geo_feat_wg = tf.transpose(box_geo_feat_wg,perm=[1,2,0])

		# -> [K,K,fc_dim]
		box_geo_feat_wg_relu = tf.nn.relu(box_geo_feat_wg)
		# [K,fc_dim,K]
		box_geo_feat_wg_relu = tf.transpose(box_geo_feat_wg_relu,perm=[0,2,1])

		# now we get the appearance stuff
		#[K,1024]
		query = dense(box_appearance_feat, box_feat_dim, activation=tf.identity, use_bias=False, wd=None, keep_first=False, scope="query_linear")
		# split head
		#[K,16,1024/16]
		query = tf.reshape(query, (-1, group, group_feat_dim))
		query = tf.transpose(query, perm=[1, 0, 2]) # [16,K,1024/16]

		key = dense(box_appearance_feat,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=False, scope="key_linear")
		# split head
		#[K,16,1024/16]
		key = tf.reshape(key,(-1,group,group_feat_dim))
		key = tf.transpose(key,perm=[1,0,2]) # [16,K,1024/16]

		value = box_appearance_feat

		# [16,K,K]
		logits = tf.matmul(query,key,transpose_b=True)
		logits_scaled = (1.0 / math.sqrt(float(group_feat_dim))) * logits
		logits_scaled = tf.transpose(logits_scaled,perm=[1,0,2]) # [K,16,K]

		# [K,16,K]
		weighted_logits = tf.log(tf.maximum(box_geo_feat_wg_relu,1e-6)) + logits_scaled
		weighted_softmax = tf.nn.softmax(weighted_logits)

		# need to reshape for matmul
		weighted_softmax = tf.reshape(weighted_softmax,(tf.shape(weighted_softmax)[0]*group, tf.shape(weighted_softmax)[-1]))

		#[K*16,K] * [K,1024] -> [K*16,1024]
		output = tf.matmul(weighted_softmax,value)

		#[K,16,1024]
		output = tf.reshape(output,(-1,group,box_feat_dim))

		#[K,1024]
		output = dense(output,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=True, scope="output_linear")

		return output

# [K, 1024] / [K, 4] / [R, 4] / [R, 1024]
# returns [K, 1024], attended for each K to all R
def person_object_relation(box_appearance_feat, boxes, ref_boxes, ref_feat, group=16, geo_feat_dim=64, scope="RM"):
	fc_dim = group # the geo feature for each group
	with tf.variable_scope(scope):
		box_feat_dim = box_appearance_feat.get_shape().as_list()[-1] # 1024
		group_feat_dim = box_feat_dim / group
		# [K,4], [R, 4] -> [K, R, 4]
		# given the absolute box, get the pairwise relative geometric coordinates
		box_geo_encoded = geometric_encoding_pair(boxes, ref_boxes, scope="geometric_encoding_pair")
		# [K, R, 4] -> [K, R, geo_feat_dim]
		box_geo_feat = dense(box_geo_encoded, geo_feat_dim, activation=tf.nn.tanh, use_bias=True, wd=None, keep_first=False, scope="geo_emb")

		# [geo_feat_dim, K, R]
		box_geo_feat = tf.transpose(box_geo_feat, perm=[2,0,1])
		box_geo_feat = tf.expand_dims(box_geo_feat, axis=0) # [1,geo_feat_dim,K,R]
		# [1,fc_dim,K,R]
		box_geo_feat_wg = conv2d(box_geo_feat,fc_dim,kernel=1,stride=1,data_format="NCHW",scope="geo_conv")
		box_geo_feat_wg = tf.squeeze(box_geo_feat_wg)
		box_geo_feat_wg = tf.transpose(box_geo_feat_wg,perm=[1,2,0])

		# -> [K,R,fc_dim]
		box_geo_feat_wg_relu = tf.nn.relu(box_geo_feat_wg)
		# [K,fc_dim,R]
		box_geo_feat_wg_relu = tf.transpose(box_geo_feat_wg_relu,perm=[0,2,1])

		# now we get the appearance stuff
		#[K,1024]
		query = dense(box_appearance_feat, box_feat_dim, activation=tf.identity, use_bias=False, wd=None, keep_first=False, scope="query_linear")
		# split head
		#[K,16,1024/16]
		query = tf.reshape(query, (-1, group, group_feat_dim))
		query = tf.transpose(query, perm=[1, 0, 2]) # [16,K,1024/16]

		#[R, 1024]
		key = dense(ref_feat, box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=False, scope="key_linear")
		# split head
		#[R,16,1024/16]
		key = tf.reshape(key,(-1,group,group_feat_dim))
		key = tf.transpose(key,perm=[1,0,2]) # [16,R,1024/16]

		value = ref_feat

		# [16,K,R]
		logits = tf.matmul(query, key,transpose_b=True)
		logits_scaled = (1.0 / math.sqrt(float(group_feat_dim))) * logits
		logits_scaled = tf.transpose(logits_scaled,perm=[1,0,2]) # [K,16,R]

		# [K,16,R]
		weighted_logits = tf.log(tf.maximum(box_geo_feat_wg_relu,1e-6)) + logits_scaled
		weighted_softmax = tf.nn.softmax(weighted_logits)

		# need to reshape for matmul
		weighted_softmax = tf.reshape(weighted_softmax,(tf.shape(weighted_softmax)[0]*group, tf.shape(weighted_softmax)[-1]))

		#[K*16,R] * [R,1024] -> [K*16,1024]
		output = tf.matmul(weighted_softmax,value)

		#[K,16,1024]
		output = tf.reshape(output,(-1,group,box_feat_dim))

		#[K,1024]
		output = dense(output,box_feat_dim,activation=tf.identity,use_bias=False,wd=None,keep_first=True, scope="output_linear")

		return output

# [K,4] -> [K,K,4] # get the pairwise box geometric feature
def geometric_encoding(boxes,scope="geometric_encoding"):
	with tf.variable_scope(scope):

		x1,y1,x2,y2 = tf.split(boxes, 4, axis=1)
		w = x2 - x1
		h = y2 - y1
		center_x = 0.5 * (x1+x2)
		center_y = 0.5 * (y1+y2)

		# [K,K]
		delta_x = center_x - tf.transpose(center_x)
		delta_x = delta_x / w
		delta_x = tf.log(tf.maximum(tf.abs(delta_x),1e-3))

		delta_y = center_y - tf.transpose(center_y)
		delta_y = delta_y / w
		delta_y = tf.log(tf.maximum(tf.abs(delta_y),1e-3))

		delta_w = tf.log(w / tf.transpose(w))

		delta_h = tf.log(h / tf.transpose(h))

		#[K,K,4]
		output = tf.stack([delta_x,delta_y,delta_w,delta_h],axis=2)
		
		return output


def geometric_encoding_pair(boxes1, boxes2,scope="geometric_encoding_pair"):
	with tf.variable_scope(scope):

		x11,y11,x12,y12 = tf.split(boxes1, 4, axis=1)
		w1 = x12 - x11
		h1 = y12 - y11
		center1_x = 0.5 * (x11+x12)
		center1_y = 0.5 * (y11+y12)

		x21,y21,x22,y22 = tf.split(boxes2, 4, axis=1)
		w2 = x22 - x21
		h2 = y22 - y21
		center2_x = 0.5 * (x21+x22)
		center2_y = 0.5 * (y21+y22)	

		# [K, R]
		delta_x = center1_x - tf.transpose(center2_x)
		delta_x = delta_x / tf.tile(tf.transpose(w2), [tf.shape(delta_x)[0], 1])
		delta_x = tf.log(tf.maximum(tf.abs(delta_x),1e-3))

		delta_y = center1_y - tf.transpose(center2_y)
		delta_y = delta_y / tf.tile(tf.transpose(w2), [tf.shape(delta_y)[0], 1])
		delta_y = tf.log(tf.maximum(tf.abs(delta_y),1e-3))

		delta_w = tf.log(w1 / tf.transpose(w2))

		delta_h = tf.log(h1 / tf.transpose(h2))

		#[K, R,4]
		output = tf.stack([delta_x,delta_y,delta_w,delta_h],axis=2)
		
		return output
def GlobalAvgPooling(x, data_format='NHWC'):
	axis = [1, 2] if data_format == 'NHWC' else [2, 3]
	return tf.reduce_mean(x, axis, name='output')

def conv2d(x, out_channel, kernel, padding="SAME", stride=1, activation=tf.identity, dilations=1, use_bias=True,data_format="NHWC", W_init=None, scope="conv"):
	with tf.variable_scope(scope):
		in_shape = x.get_shape().as_list()

		channel_axis = 3 if data_format == "NHWC" else 1
		in_channel = in_shape[channel_axis]

		assert in_channel is not None

		kernel_shape = [kernel,kernel]

		filter_shape = kernel_shape + [in_channel,out_channel]

		if data_format == "NHWC":
			stride = [1, stride, stride,1]
			dilations = [1, dilations, dilations, 1]
		else:
			stride = [1,1,stride,stride]
			dilations = [1, 1, dilations, dilations]


		if W_init is None:
			W_init = tf.variance_scaling_initializer(scale=2.0)
		W = tf.get_variable('W', filter_shape, initializer=W_init)

		conv = tf.nn.conv2d(x, W, stride, padding, dilations=dilations, data_format=data_format)

		if use_bias:
			b_init = tf.constant_initializer()
			b = tf.get_variable('b', [out_channel], initializer=b_init)
			conv = tf.nn.bias_add(conv,b,data_format=data_format)

		ret = activation(conv,name="output")

	return ret

def deconv2d(x,out_channel, kernel,padding="SAME",stride=1, activation=tf.identity,use_bias=True,data_format="NHWC",W_init=None,scope="deconv"):

	with tf.variable_scope(scope):
		in_shape = x.get_shape().as_list()

		channel_axis = 3 if data_format == "NHWC" else 1
		in_channel = in_shape[channel_axis]

		assert in_channel is not None
		kernel_shape = [kernel,kernel]


		# TODO: change the following to tf.nn.conv2d_transpose
		if W_init is None:
			W_init = tf.variance_scaling_initializer(scale=2.0)
		b_init = tf.constant_initializer()
		
		with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
			layer = tf.layers.Conv2DTranspose(
				out_channel, kernel_shape,
				strides=stride, padding=padding,
				data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
				activation=lambda x: activation(x, name='output'),
				use_bias=use_bias,
				kernel_initializer=W_init,
				bias_initializer=b_init,
				trainable=True)
			ret = layer.apply(x, scope=tf.get_variable_scope())
		"""
		if data_format == "NHWC":
			stride = [1,stride,stride,1]
		else:
			stride = [1,1,stride,stride]

		filter_shape = kernel_shape + [in_channel,out_channel]
		W = tf.get_variable('W', filter_shape, initializer=W_init)

		deconv = tf.nn.conv2d_transpose(x,W,output_shape=out_channel,strides=stride,padding=padding,data_format=data_format)

		if use_bias:
			b = tf.get_variable('b', [out_channel], initializer=b_init)
			deconv = tf.nn.bias_add(deconv,b,data_format=data_format)
		
		return deconv
		"""
		return ret

def rename_get_variable(mapping):
	"""
	Args:
		mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
	"""
	def custom_getter(getter, name, *args, **kwargs):
		splits = name.split('/')
		basename = splits[-1]
		if basename in mapping:
			basename = mapping[basename]
			splits[-1] = basename
			name = '/'.join(splits)
		return getter(name, *args, **kwargs)
	return custom_getter_scope(custom_getter)

from contextlib import contextmanager

@contextmanager
def custom_getter_scope(custom_getter):
	scope = tf.get_variable_scope()
	with tf.variable_scope(scope, custom_getter=custom_getter):
		yield

def resnet_basicblock(l, ch_out, stride,  dilations=1, deformable=False, tf_pad_reverse=False, use_gn=False, use_se=False):
	shortcut = l
	if use_gn:
		NormReLU = GNReLU
	else:
		NormReLU = BNReLU
	l = conv2d(l, ch_out, 3, stride=stride, activation=NormReLU, use_bias=False, data_format="NCHW", scope='conv1')
	l = conv2d(l, ch_out, 3, use_bias=False, activation=get_bn(use_gn, zero_init=True), data_format="NCHW", scope='conv2')
	out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(use_gn, zero_init=False), data_format="NCHW")
	return tf.nn.relu(out)


def resnet_bottleneck(l, ch_out, stride, dilations=1, deformable=False, tf_pad_reverse=False, use_gn=False, use_se=False):
	l, shortcut = l, l
	if use_gn:
		NormReLU = GNReLU
	else:
		NormReLU = BNReLU
	l = conv2d(l, ch_out, 1, activation=NormReLU,scope='conv1',use_bias=False,data_format="NCHW")
	if stride == 2:	
		
		if deformable:
			# l [1, C, H, W]
			# 1. get the offset from conv2d [1, 18, H, W]
			offset = conv2d(l, 2*3*3, 3, stride=1, padding="SAME", scope="conv2_offset", data_format="NCHW")
			# for testing, use all zero offset, so this should be the same as regular conv2d
			#input_h = tf.shape(l)[2]
			#input_w = tf.shape(l)[3]
			#offset = tf.fill(value=0.0, dims=[1, 2*3*3, input_h, input_w])
			# get [1, ch_out, H/2, w/2]
			l = deformable_conv2d(l, offset, ch_out, 3, scope="conv2", data_format="NCHW", use_bias=False)
		else:
			l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1, tf_pad_reverse), maybe_reverse_pad(0,1,tf_pad_reverse)])
			l = conv2d(l, ch_out, 3, dilations=dilations, stride=2, activation=NormReLU, padding='VALID',scope='conv2',use_bias=False,data_format="NCHW")
		if dilations != 1: # weird shit
		# [H+1, W+1]
			l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1, tf_pad_reverse), maybe_reverse_pad(0,1,tf_pad_reverse)])
	else:
		l = conv2d(l, ch_out, 3, dilations=dilations, stride=stride, activation=NormReLU,scope='conv2',use_bias=False,data_format="NCHW")
	l = conv2d(l, ch_out * 4, 1, activation=get_bn(use_gn,zero_init=True),scope='conv3',use_bias=False,data_format="NCHW")

	if use_se:
		squeeze = GlobalAvgPooling(l, data_format='NCHW')
		squeeze = dense(squeeze, ch_out // 4, activation=tf.nn.relu, W_init=tf.variance_scaling_initializer(), scope='fc1')
		squeeze = dense(squeeze, ch_out * 4, activation=tf.nn.sigmoid, W_init=tf.variance_scaling_initializer(), scope='fc2')
		ch_ax = 1 
		shape = [-1, 1, 1, 1]
		shape[ch_ax] = ch_out * 4
		l = l * tf.reshape(squeeze, shape)

	return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(use_gn,zero_init=False),data_format="NCHW")

def resnet_shortcut(l, n_out, stride, activation=tf.identity,data_format="NCHW",use_gn=False):
	n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
	if n_in != n_out:   # change dimension when channel is not the same
		if stride == 2:
			l = l[:, :, :-1, :-1]
			return conv2d(l, n_out, 1,
						  stride=stride, padding='VALID', activation=activation,use_bias=False,data_format=data_format,scope='convshortcut')
		else:
			return conv2d(l, n_out, 1,
						  stride=stride, activation=activation,use_bias=False,data_format=data_format,scope='convshortcut')
	else:
		return l

def resnet_group(l, name, block_func, features, count, stride, dilations=1, use_deformable=False, modified_block_num=3, reuse=False, tf_pad_reverse=False, use_gn=False, use_se=False):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		for i in range(0, count):
			with tf.variable_scope('block{}'.format(i)):
				dilations_ = 1
				deformable_ = False
				if i in range(count)[-modified_block_num:]:
					dilations_ = dilations
					deformable_ = use_deformable
				
				l = block_func(l, features,
							   stride if i == 0 else 1, dilations=dilations_, deformable=deformable_, use_gn=use_gn, tf_pad_reverse=tf_pad_reverse, use_se=use_se)
				# end of each block need an activation
				l = tf.nn.relu(l)
	return l

def get_bn(use_gn=False,zero_init=False):
	if use_gn:
		if zero_init:
			return lambda x, name: group_norm(x, gamma_init=tf.zeros_initializer(),scope="gn")
		else:
			return lambda x, name: group_norm(x,scope="gn")
	else:
		if zero_init:
			return lambda x, name: BatchNorm(x, gamma_init=tf.zeros_initializer(),scope="bn")
		else:
			return lambda x, name: BatchNorm(x,scope="bn")

def BNReLU(x, name=None):
	"""
	A shorthand of Normalization + ReLU.
	"""

	x = BatchNorm(x,scope="bn")
	x = tf.nn.relu(x, name=name)
	return x

def GNReLU(x, name=None):
	"""
	A shorthand of Normalization + ReLU.
	"""

	x = group_norm(x,scope="gn")
	x = tf.nn.relu(x, name=name)
	return x

# TODO: replace these
#-----------------------------------

from tensorflow.contrib.framework import add_model_variable
is_training = False
def BatchNorm(x, use_local_stat=False, decay=0.9, epsilon=1e-5,
			  use_scale=True, use_bias=True,
			  gamma_init=tf.constant_initializer(1.0), data_format='NCHW',
			  internal_update=False,scope="bn"):
	global is_training
	with tf.variable_scope(scope):
		shape = x.get_shape().as_list()
		ndims = len(shape)
		assert ndims in [2, 4]
		if ndims == 2:
			data_format = 'NHWC'
		if data_format == 'NCHW':
			n_out = shape[1]
		else:
			n_out = shape[-1]  # channel
		assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
		beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias, gamma_init)

		use_local_stat = bool(use_local_stat)

		if use_local_stat:
			if ndims == 2:
				x = tf.reshape(x, [-1, 1, 1, n_out])	# fused_bn only takes 4D input
				# fused_bn has error using NCHW? (see #190)

			xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
				x, gamma, beta, epsilon=epsilon,
				is_training=True, data_format=data_format)

			if ndims == 2:
				xn = tf.squeeze(xn, [1, 2])
		else:
			if is_training: # so ugly
				#assert get_tf_version_number() >= 1.4, \
				#	"Fine tuning a BatchNorm model with fixed statistics is only " \
				#	"supported after https://github.com/tensorflow/tensorflow/pull/12580 "
				#if ctx.is_main_training_tower:  # only warn in first tower
				#	logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
				# Using moving_mean/moving_variance in training, which means we
				# loaded a pre-trained BN and only fine-tuning the affine part.
				xn, _, _ = tf.nn.fused_batch_norm(
					x, gamma, beta,
					mean=moving_mean, variance=moving_var, epsilon=epsilon,
					data_format=data_format, is_training=False)
			else:
				# non-fused op is faster for inference  # TODO test if this is still true
				if ndims == 4 and data_format == 'NCHW':
					[g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
									  for _ in [gamma, beta, moving_mean, moving_var]]
					xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
				else:
					# avoid the reshape if possible (when channel is the last dimension)
					xn = tf.nn.batch_normalization(
						x, moving_mean, moving_var, beta, gamma, epsilon)

		# maintain EMA only on one GPU is OK, even in replicated mode.
		# because training time doesn't use EMA
		#if ctx.is_main_training_tower:
		add_model_variable(moving_mean)
		add_model_variable(moving_var)
		if use_local_stat: # and ctx.is_main_training_tower:
			ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay, internal_update)
		else:
			ret = tf.identity(xn, name='output')

		return ret

def update_bn_ema(xn, batch_mean, batch_var,
				  moving_mean, moving_var, decay, internal_update):
	# TODO is there a way to use zero_debias in multi-GPU?
	update_op1 = moving_averages.assign_moving_average(
		moving_mean, batch_mean, decay, zero_debias=False,
		name='mean_ema_op')
	update_op2 = moving_averages.assign_moving_average(
		moving_var, batch_var, decay, zero_debias=False,
		name='var_ema_op')

	if internal_update:
		with tf.control_dependencies([update_op1, update_op2]):
			return tf.identity(xn, name='output')
	else:
		tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
		tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
		return xn


def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
	if use_bias:
		beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
	else:
		beta = tf.zeros([n_out], name='beta')
	if use_scale:
		gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
	else:
		gamma = tf.ones([n_out], name='gamma')
	# x * gamma + beta

	moving_mean = tf.get_variable('mean/EMA', [n_out],
								  initializer=tf.constant_initializer(), trainable=False)
	moving_var = tf.get_variable('variance/EMA', [n_out],
								 initializer=tf.constant_initializer(1.0), trainable=False)
	return beta, gamma, moving_mean, moving_var

def reshape_for_bn(param, ndims, chan, data_format):
	if ndims == 2:
		shape = [1, chan]
	else:
		shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
	return tf.reshape(param, shape)
#----------------------------------------------------


# add weight decay to the current varaible scope
def add_wd(wd,scope=None):
	if wd != 0.0:
		# for all variable in the current scope
		scope = scope or tf.get_variable_scope().name
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		with tf.name_scope("weight_decay"):
			for var in variables:
				weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="%s/wd"%var.op.name)
				tf.add_to_collection('losses', weight_decay)


# flatten a tensor
# [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
def flatten(tensor, keep): # keep how many dimension in the end, so final rank is keep + 1
	# get the shape
	fixed_shape = tensor.get_shape().as_list() #[N, JQ, di] # [N, M, JX, di] 
	start = len(fixed_shape) - keep # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
	# each num in the [] will a*b*c*d...
	# so [0] -> just N here for left
	# for [N, M, JX, di] , left is N*M
	left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
	# [N, JQ,di]
	# [N*M, JX, di] 
	out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
	# reshape
	flat = tf.reshape(tensor, out_shape)
	return flat

def reconstruct(tensor, ref, keep): # reverse the flatten function
	ref_shape = ref.get_shape().as_list()
	tensor_shape = tensor.get_shape().as_list()
	ref_stop = len(ref_shape) - keep
	tensor_start = len(tensor_shape) - keep
	pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
	keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
	# pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
	# keep_shape = tensor.get_shape().as_list()[-keep:]
	target_shape = pre_shape + keep_shape
	out = tf.reshape(tensor, target_shape)
	return out

# boxes are x1y1,x2y2
def pairwise_iou(boxes1,boxes2):
	def area(boxes): # [N,4] -> [N]
		x1,y1,x2,y2 = tf.split(boxes,4,axis=1)
		return tf.squeeze((y2-y1)*(x2-x1),[1])

	# two box list,  get intersected boxes area [N,M] 
	def pairwise_intersection(b1,b2):
		x_min1, y_min1, x_max1, y_max1 = tf.split(b1, 4, axis=1)
		x_min2, y_min2, x_max2, y_max2 = tf.split(b2, 4, axis=1)
		all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
		all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
		intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
		all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
		all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
		intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
		return intersect_heights * intersect_widths

	interarea = pairwise_intersection(boxes1,boxes2)
	areas1 = area(boxes1)#[N]
	areas2 = area(boxes2)#[M]
	unions = tf.expand_dims(areas1,1) + tf.expand_dims(areas2,0) - interarea

	# avoid zero divide?
	return tf.truediv(interarea,unions)


import pycocotools.mask as cocomask

def np_iou(A,B):
	def to_xywh(box):
		box = box.copy()
		box[:, 2] -= box[:, 0]
		box[:, 3] -= box[:, 1]
		return box

	ret = cocomask.iou(
		to_xywh(A), to_xywh(B),
		np.zeros((len(B),), dtype=np.bool))
	# can accelerate even more, if using float32
	return ret.astype('float32')


#@memorized
def get_iou_callable():
	with tf.Graph().as_default(),tf.device("/cpu:0"):
		A = tf.placeholder(tf.float32,shape=[None,4])
		B = tf.placeholder(tf.float32,shape=[None,4])
		iou = pairwise_iou(A,B)
		sess = tf.Session()
		return sess.make_callable(iou,[A,B])



# simple linear layer, without activatation # remember to add it
def dense(x,output_size,W_init=None,b_init=None,activation=tf.identity,use_bias=True,wd=None,keep_first=True, scope="dense"):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		# tensorpack's fully connected keep the first dim and flatten the rest, apply W on the rest
		if keep_first:
			shape = x.get_shape().as_list()[1:]
			if None not in shape:
				flat_x = tf.reshape(x,[-1,int(np.prod(shape))])
			else:
				flat_x = tf.reshape(x,tf.stack([tf.shape(x)[0],-1]))
		else:
			# we used to apply W on the last dimention
			# since the input here is not two rank, we flat the input while keeping the last dims
			keep = 1
			#print x.get_shape().as_list()
			flat_x = flatten(x,keep) # keeping the last one dim # [N,M,JX,JQ,d] => [N*M*JX*JQ,d]
		
		if W_init is None:
			W_init = tf.variance_scaling_initializer(2.0)

		W = tf.get_variable("W",[flat_x.get_shape().as_list()[-1],output_size],initializer=W_init)

		flat_out = tf.matmul(flat_x,W)

		if use_bias:
			if b_init is None:
				b_init = tf.constant_initializer()
			b = tf.get_variable('b', [output_size], initializer=b_init)
			flat_out = tf.nn.bias_add(flat_out, b)

		flat_out = activation(flat_out)

		if wd is not None:
			add_wd(wd)

		if not keep_first:
			out = reconstruct(flat_out,x,keep)
		else:
			out = flat_out
		return out

def maybe_reverse_pad(topleft, bottomright,reverse=False):
	if reverse:
		return [bottomright, topleft]
	else:
		return [topleft, bottomright]

def MaxPooling(x, shape, stride=None, padding='VALID', data_format='NHWC',scope="maxpooling"):
	with tf.variable_scope(scope):
		if stride is None:
			stride = shape
		ret = tf.layers.max_pooling2d(x, shape, stride, padding,
									  'channels_last' if data_format == 'NHWC' else 'channels_first')
		return tf.identity(ret, name='output')

def pretrained_resnet_conv4(image, num_blocks,tf_pad_reverse=False):
	assert len(num_blocks) == 3
	# pad 2 zeros to front of H and 3 zeros to back of H, same for W
	# so 2-3lines of zeros outside the data center
	# original H,W will be H+5,W+5
	l = tf.pad(image,[[0, 0], [0, 0], maybe_reverse_pad(2, 3,tf_pad_reverse), maybe_reverse_pad(2, 3,tf_pad_reverse)])

	l = conv2d(l, 64, 7, stride=2, activation=BNReLU, padding='VALID',scope="conv0",use_bias=False,data_format="NCHW")

	#print l.get_shape()# (1,64,?,?)
	l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1,tf_pad_reverse), maybe_reverse_pad(0, 1,tf_pad_reverse)])
	l = MaxPooling(l, shape=3, stride=2, padding='VALID',scope='pool0',data_format="NCHW")

	#print l.get_shape()# (1,64,?,?)
	l = resnet_group(l, 'group0', resnet_bottleneck, 64, num_blocks[0], stride=1,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,256,?,?)
	# TODO replace var by const to enable folding
	#l = tf.stop_gradient(l) # froze outside
	l = resnet_group(l, 'group1', resnet_bottleneck, 128, num_blocks[1], stride=2,tf_pad_reverse=tf_pad_reverse)
	#print l.get_shape()# (1,512,?,?)
	l = resnet_group(l, 'group2', resnet_bottleneck, 256, num_blocks[2], stride=2,tf_pad_reverse=tf_pad_reverse)
	return l


def resnet_conv5(image,num_block,reuse=False,tf_pad_reverse=False):
	l = resnet_group(image,"group3",resnet_bottleneck,512,num_block,stride=2,reuse=reuse,tf_pad_reverse=tf_pad_reverse)
	return l

# fpn_resolution_requirement is 32 by default FPN
def resnet_fpn_backbone(image, num_blocks,resolution_requirement, use_deformable=False, use_dilations=False, tf_pad_reverse=False,finer_resolution=False,freeze=0,use_gn=False,use_basic_block=False, use_se=False):
	assert len(num_blocks) == 4
	shape2d = tf.shape(image)[2:]

	# this gets the nearest H, W that can be zheng chu by 32
	# 720 -> 736, 1080 -> 1088
	mult = resolution_requirement * 1.0
	new_shape2d = tf.to_int32(tf.ceil(tf.to_float(shape2d) / mult) * mult)
	pad_shape2d = new_shape2d - shape2d

	channel = image.shape[1]

	if use_gn:
		NormReLU = GNReLU
	else:
		NormReLU = BNReLU

	block_func = resnet_bottleneck
	if use_basic_block:
		block_func = resnet_basicblock

	pad_base = maybe_reverse_pad(2, 3,tf_pad_reverse)
	
	l = tf.pad(image, [
			[0, 0], [0, 0], 
			[pad_base[0], pad_base[1] + pad_shape2d[0]], 
			[pad_base[0], pad_base[1] + pad_shape2d[1]]])
	l.set_shape([None,channel,None,None])

	# 720, 1280 -> 736, 1280 / 1080, 1920 -> 1088, 1920, zhengchu by 32
	# actually 741/1093 due to the pad_base of [2, 3]
	# pad_base is for first conv and max pool
	#l = tf.Print(l, data=[tf.shape(l)], summarize=10) 

	# rest is the same as c4 backbone
	l = conv2d(l, 64, 7, stride=2, activation=NormReLU, padding='VALID',scope="conv0",use_bias=False,data_format="NCHW")
	c1 = l
	l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1,tf_pad_reverse), maybe_reverse_pad(0, 1,tf_pad_reverse)]) # H+1,W+1
	
	l = MaxPooling(l, shape=3, stride=2, padding='VALID',scope='pool0',data_format="NCHW")
	# here 4x down already, so smallest anchor box can on use these
	#l = tf.Print(l, data=[tf.shape(l)], summarize=10) # [1 64 272 480] # 4X down from 1088 and 1920
	
	# resnet_group output channel is 64*4
	c2 = resnet_group(l, 'group0', block_func, 64, num_blocks[0], stride=1, tf_pad_reverse=tf_pad_reverse,use_gn=use_gn, use_se=use_se)
	if freeze >= 0: # tensorpack setting
		c2 = tf.stop_gradient(c2)

	#c2 = tf.Print(c2, data=[tf.shape(c2)], summarize=10) # [1 256 272 480] # 

	# for dilated conv and deformable conv, we will add to the last 3 block in each group
	mbn = 3 # modify the last 3 conv block
	# [4]
	c3 = resnet_group(c2, 'group1', block_func, 128, num_blocks[1], dilations=1, modified_block_num=mbn, stride=2, use_deformable=use_deformable, tf_pad_reverse=tf_pad_reverse,use_gn=use_gn, use_se=use_se)
	if freeze >= 1:
		c3 = tf.stop_gradient(c3)

	# [23]
	c4 = resnet_group(c3, 'group2', block_func, 256, num_blocks[2], dilations=1, modified_block_num=mbn, stride=2, use_deformable=use_deformable, tf_pad_reverse=tf_pad_reverse,use_gn=use_gn, use_se=use_se)
	if freeze >= 2:
		c4 = tf.stop_gradient(c4)
	#c4 = tf.Print(c4, data=[tf.shape(c4)], summarize=10) # [1 1024 68 120]

	# [3]
	# people change the last stride to 1 and with dilations 2?
	c5 = resnet_group(c4, "group3", block_func, 512, num_blocks[3], dilations=2 if use_dilations else 1, use_deformable=use_deformable, modified_block_num=mbn, stride=2, tf_pad_reverse=tf_pad_reverse,use_gn=use_gn, use_se=use_se)
	#c5 = tf.Print(c5, data=[tf.shape(c5)], summarize=10) # [1 2048 34 60] same for dilation or not

	if freeze >= 3:
		c5 = tf.stop_gradient(c5)
	## 32x downsampling up to now
	# size of c5: ceil(input/32)
	return c2,c3,c4,c5

# the FPN model
def fpn_model(c2345,num_channel,scope,use_gn=False):

	def upsample2x(x,scope):
		with tf.name_scope(scope):
			# FPN paper uses nearest neighbour
			# a outer product with 2x2 , makes x upsampled
			unpool_mat = np.ones((2,2),dtype="float32")
			shape = (2,2)
			output_shape = x.get_shape().as_list() # [N,C,H,W]

			unpool_mat = tf.constant(unpool_mat,name="unpool_mat")

			assert unpool_mat.get_shape().as_list() == list(shape)

			# outer product with ones, so just duplicate stuff
			x = tf.expand_dims(x, -1) # NxCxHxWx1
			mat = tf.expand_dims(unpool_mat,0) # 1xSHxSW
			ret = tf.tensordot(x, mat, axes=1) # NxCxHxWxSHxSW

			ret = tf.transpose(ret, [0,1,2,4,3,5]) # NxCxHxSHxWxSW

			ret = tf.reshape(ret, tf.stack([-1, output_shape[1],tf.shape(x)[2] * shape[0], tf.shape(x)[3] * shape[1] ]))# [N,C,H*2,W*2]

			return ret


	with tf.variable_scope(scope):
		# each conv feature go through 1x1 conv, then add to 2x upsampled feature, then add 3x3 conv to get final feature
		lat_2345 = [conv2d(c, num_channel, 1, stride=1, activation=tf.identity, padding='SAME',scope="lateral_1x1_c%s"%(i+2), use_bias=True, data_format="NCHW", W_init=tf.variance_scaling_initializer(scale=1.0)) for i,c in enumerate(c2345)]


		if use_gn:
			lat_2345 = [group_norm(c, scope='gn_c{}'.format(i + 2)) for i, c in enumerate(lat_2345)]

		lat_sum_5432 = []
		for idx, lat in enumerate(lat_2345[::-1]):
			if idx == 0:
				lat_sum_5432.append(lat)
			else:
				lat = lat + upsample2x(lat_sum_5432[-1],scope="upsample_lat%s"%(6 - idx))
				lat_sum_5432.append(lat)

		p2345 = [conv2d(c, num_channel, 3, stride=1, activation=tf.identity, padding='SAME',scope="posthoc_3x3_p%s"%(i+2),use_bias=True,data_format="NCHW",W_init=tf.variance_scaling_initializer(scale=1.0)) for i,c in enumerate(lat_sum_5432[::-1])]

		if use_gn:
			p2345 = [group_norm(c,scope='gn_p{}'.format(i + 2)) for i, c in enumerate(p2345)]

		p6 = MaxPooling(p2345[-1], shape=1, stride=2, padding='VALID',scope='maxpool_p6',data_format="NCHW")
		return p2345+[p6]



# -------------------------all model function
def sample_fast_rcnn_targets_plus_act(boxes, gt_boxes, gt_labels, act_single_labels,act_pair_labels, config):
	# act_single_labels is [N,num_act_class]

	iou = pairwise_iou(boxes,gt_boxes)

	# gt_box directly used as proposal
	#boxes = tf.concat([boxes,gt_boxes],axis=0)
	#iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])],axis=0)
	# [N+M,M]
	# gt_box in front, so the act single label can match
	boxes = tf.concat([gt_boxes,boxes],axis=0)
	iou = tf.concat([tf.eye(tf.shape(gt_boxes)[0]),iou],axis=0)

	def sample_fg_bg(iou):
		# [K,M] # [M] is the ground truth
		# [K] # max iou for each proposal to the ground truth
		fg_mask = tf.reduce_max(iou,axis=1) >= config.fastrcnn_fg_thres
		fg_inds = tf.reshape(tf.where(fg_mask),[-1]) # [K_FG] # index of fg_mask true element

		x = tf.where(tf.equal(act_single_labels[:,0],0)) # c==0 will fail
		
		act_single_fg_inds = tf.reshape(x,[-1])

		num_act_fg = tf.size(act_single_fg_inds)

		num_fg = tf.minimum(int(config.fastrcnn_batch_per_im * config.fastrcnn_fg_ratio),tf.size(fg_inds))
		# during train time, each time random sample
		fg_inds = tf.random_shuffle(fg_inds)[:(num_fg-num_act_fg)]# so the pos box is at least > fg_thres iou
		fg_inds = tf.concat([fg_inds,act_single_fg_inds],axis=0)
		num_fg = num_fg+num_act_fg

		bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
		num_bg = tf.minimum(config.fastrcnn_batch_per_im - num_fg,tf.size(bg_inds))
		bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

		return fg_inds,bg_inds

	# get random pos neg from over some iou thres from [N+M]
	fg_inds, bg_inds = sample_fg_bg(iou)

	best_iou_ind = tf.argmax(iou, axis=1) #[N+M],# proposal -> gt best matched# so each proposal has the gt's index
	# [N_FG] -> gt Index, so 0-M-1
	# each pos proposal box assign to the best gt box
	# indexes of gt_boxes that matched to fg_box
	fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds) # get the pos's gt box indexes

	all_indices = tf.concat([fg_inds,bg_inds],axis=0)

	# selected proposal boxes
	ret_boxes = tf.gather(boxes, all_indices, name="sampled_proposal_boxes")
	# [K] -> [N_FG+N_BG]
	ret_labels = tf.concat([tf.gather(gt_labels, fg_inds_wrt_gt),tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name="sampled_labels")

	# pad the activity box labels to the proposal boxes
	# act_single_labels: [N,num_act_class] # included BG class at index 0

	#[N_FG, num_class]
	# there is 1:10 negative in N_FG
	act_single_ret_labels = tf.gather(act_single_labels, fg_inds_wrt_gt)

	# for pair box, [K,K,num_act_class] -> [N_FG+N_BG,N_FG+N_BG, num_act_class]
	# 1, get [N_FG,N_FG,num_act_class]
	# [N_FG, 2] 
	"""
	tiled_fg_inds_wrt_gt = tf.tile(tf.expand_dims(fg_inds_wrt_gt,1),[1,2])
	act_pair_fg_labels = tf.gather_nd(act_pair_labels,tiled_fg_inds_wrt_gt)

	# [N_BG,N_FG,num_act_class]
	act_pad_bg = tf.zeros((tf.size(bg_inds),tf.size(fg_inds),tf.shape(act_single_labels)[-1]), dtype=tf.int64)
	act_pad_bg[:,:,0] = 1 # BG class should be 1
	# [N_FG+N_BG,N_BG,num_act_class]
	act_pad_bg2 = tf.zeros((tf.size(bg_inds)+tf.size(fg_inds),tf.size(bg_inds),tf.shape(act_single_labels)[-1]), dtype=tf.int64)
	act_pad_bg2[:,:,0] = 1 # BG class should be 1

	# [N_FG+N_BG,N_FG,num_act_class]
	act_pair_ret_labels = tf.concat([act_pair_fg_labels,act_pad_bg],axis=0)
	# [N_FG+N_BG,N_FG+N_BG,num_act_class]
	act_pair_ret_labels = tf.concat([act_pair_ret_labels,act_pad_bg2],axis=0)
	"""

	return tf.stop_gradient(ret_boxes),tf.stop_gradient(ret_labels),tf.stop_gradient(act_single_ret_labels),None, fg_inds_wrt_gt


# given the proposal box, decide the positive and negatives
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, config,fg_ratio=None):

	iou = pairwise_iou(boxes,gt_boxes)

	# gt_box directly used as proposal
	boxes = tf.concat([boxes,gt_boxes],axis=0)
	iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])],axis=0)
	# [N+M,M]

	def sample_fg_bg(iou,fg_ratio):
		# [K,M] # [M] is the ground truth
		# [K] # max iou for each proposal to the ground truth
		fg_mask = tf.reduce_max(iou,axis=1) >= config.fastrcnn_fg_thres # 0.5
		fg_inds = tf.reshape(tf.where(fg_mask),[-1]) # [K_FG] # index of fg_mask true element
		num_fg = tf.minimum(int(config.fastrcnn_batch_per_im * fg_ratio),tf.size(fg_inds))
		# during train time, each time random sample
		fg_inds = tf.random_shuffle(fg_inds)[:num_fg]# so the pos box is at least > fg_thres iou

		bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
		num_bg = tf.minimum(config.fastrcnn_batch_per_im - num_fg,tf.size(bg_inds))
		bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

		return fg_inds,bg_inds

	if fg_ratio is None:
		fg_ratio = config.fastrcnn_fg_ratio
	# get random pos neg from over some iou thres from [N+M]
	fg_inds,bg_inds = sample_fg_bg(iou,fg_ratio)

	best_iou_ind = tf.argmax(iou, axis=1) #[N+M],# proposal -> gt best matched# so each proposal has the gt's index
	# [N_FG] -> gt Index, so 0-M-1
	# each pos proposal box assign to the best gt box
	# indexes of gt_boxes that matched to fg_box
	fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds) # get the pos's gt box indexes

	all_indices = tf.concat([fg_inds,bg_inds],axis=0)

	# selected proposal boxes
	ret_boxes = tf.gather(boxes, all_indices, name="sampled_proposal_boxes")

	ret_labels = tf.concat([tf.gather(gt_labels, fg_inds_wrt_gt),tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name="sampled_labels")
	return tf.stop_gradient(ret_boxes),tf.stop_gradient(ret_labels), fg_inds_wrt_gt


# sample small object training
# boxes: [C, N, 4]
# gt_boxes: [C], [G, 4]
# gt_labels: [C], [G] # [0, 1]
# return box_labels: [C, N_] # 0 or 1
def get_so_labels(boxes, gt_boxes, gt_labels, config):

	box_labels = []
	for i in xrange(len(config.small_objects)):
		iou = pairwise_iou(boxes[i], gt_boxes[i])
		#print iou.get_shape()  # [1536,0] # gt_boxes could be empty

		def sample_fg_bg(iou):
			#fg_ratio = 0.2
			# [K,M] # [M] is the ground truth
			# [K] # max iou for each proposal to the ground truth
			fg_mask = tf.reduce_max(iou,axis=1) >= config.fastrcnn_fg_thres # iou 0.5
			fg_inds = tf.reshape(tf.where(fg_mask),[-1]) # [K_FG] # index of fg_mask true element
			# sometimes this does not add up to 512, then the stacking will raise error
			#num_fg = tf.minimum(int(config.fastrcnn_batch_per_im * fg_ratio),tf.size(fg_inds))
			#fg_inds = tf.random_shuffle(fg_inds)[:num_fg]# so the pos box is at least > fg_thres iou
			# use all fg
			
			bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
			#num_bg = tf.minimum(config.fastrcnn_batch_per_im - num_fg, tf.size(bg_inds))
			#bg_inds = tf.random_shuffle(bg_inds)[:num_bg]
			return fg_inds, bg_inds  

		fg_inds, bg_inds = sample_fg_bg(iou)

		# handle when there is no ground truth small object in the image
		best_iou_ind = tf.cond(tf.equal(tf.size(gt_boxes[i]), 0), 
										lambda: tf.zeros_like([], dtype=tf.int64),
										lambda: tf.argmax(iou, axis=1)) #[N+M],# proposal -> gt best matched# so each proposal has the gt's index
		# [N_FG] -> gt Index, so 0-M-1
		# each pos proposal box assign to the best gt box
		# indexes of gt_boxes that matched to fg_box
		fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds) # get the pos's gt box indexes

		this_labels = tf.concat([tf.gather(gt_labels[i], fg_inds_wrt_gt), tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name="sampled_labels")
		box_labels.append(this_labels)
	box_labels = tf.stack(box_labels, axis=0)
	return tf.stop_gradient(box_labels)



# fix the tf.image.crop_and_resize to do roi_align
def crop_and_resize(image, boxes, box_ind, crop_size,pad_border=False):
	# image feature [1,C,FS,FS] # for mask gt [N_FG, 1, H, W]
	# boxes [N,4]
	# box_ind [N] all zero?

	if pad_border:
		image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
		boxes = boxes + 1

	# return [N,C,crop_size,crop_size]
	def transform_fpcoor_for_tf(boxes, image_shape,crop_shape):
		"""
		The way tf.image.crop_and_resize works (with normalized box):
		Initial point (the value of output[0]): x0_box * (W_img - 1)
		Spacing: w_box * (W_img - 1) / (W_crop - 1)
		Use the above grid to bilinear sample.

		However, what we want is (with fpcoor box):
		Spacing: w_box / W_crop
		Initial point: x0_box + spacing/2 - 0.5
		(-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))

		This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

		Returns:
			y1x1y2x2
		"""
		x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

		spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
		spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

		nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
		ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

		nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
		nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

		return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

	image_shape = tf.shape(image)[2:]
	boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
	image = tf.transpose(image, [0, 2, 3, 1])   # 1hwc
	ret = tf.image.crop_and_resize(
		image, boxes, box_ind,
		crop_size=[crop_size, crop_size])
	ret = tf.transpose(ret, [0, 3, 1, 2])   # Ncss
	return ret


# given [1,C,FS,FS] featuremap, and the boxes [K,4], where coordiates are in FS
# get fixed size feature for each box [K,C,output_shape,output_shape]
# crop the box and resize to a shape
# here resize with bilinear pooling to twice large box, then average pooling
def roi_align(featuremap, boxes, output_shape):
	boxes = tf.stop_gradient(boxes)
	# [1,C,FS,FS] -> [K,C,out_shape*2,out_shape*2]
	ret = crop_and_resize(featuremap, boxes, tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32), output_shape * 2)
	ret = tf.nn.avg_pool(ret, ksize=[1,1,2,2],strides=[1,1,2,2],padding='SAME', data_format="NCHW")
	return ret



# given boxes, clip the box to be within the image
def clip_boxes(boxes, image_shape, name=None):
	boxes = tf.maximum(boxes, 0.0) # lower bound
	# image_shape is HW, 
	# HW -> [W, H, W, H] # <- box
	m = tf.tile(tf.reverse(image_shape, [0]), [2])
	boxes = tf.minimum(boxes, tf.to_float(m), name=name) # upper bound
	return boxes


# given all the anchor box and their logits, get the proposal box
# rank and filter, then nms
# boxes [-1,4], scores [-1]
def generate_rpn_proposals(boxes, scores, img_shape,config,pre_nms_topk=None): # image shape : HW
	# for FPN
	if pre_nms_topk is not None:
		post_nms_topk = pre_nms_topk
	else:
		if config.is_train:
			pre_nms_topk = config.rpn_train_pre_nms_topk
			post_nms_topk = config.rpn_train_post_nms_topk
		else:
			pre_nms_topk = config.rpn_test_pre_nms_topk
			post_nms_topk = config.rpn_test_post_nms_topk


	# clip [FS*FS*num_anchors] at the beginning
	topk = tf.minimum(pre_nms_topk, tf.size(scores))
	topk_scores,topk_indices = tf.nn.top_k(scores,k=topk,sorted=False)
	# top_k indices -> [topk]
	# get [topk,4]
	topk_boxes = tf.gather(boxes, topk_indices)
	topk_boxes = clip_boxes(topk_boxes, img_shape)

	topk_boxes_x1y1,topk_boxes_x2y2 = tf.split(topk_boxes, 2, axis=1)

	topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes,(-1,2,2))

	# rpn min size
	wbhb = topk_boxes_x2y2 - topk_boxes_x1y1
	valid = tf.reduce_all(wbhb > config.rpn_min_size, axis=1)
	topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
	topk_valid_scores = tf.boolean_mask(topk_scores, valid)


	# for nms input
	topk_valid_boxes_y1x1y2x2 = tf.reshape(tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),(-1,4),name="nms_input_boxes")
	# [TOPK]
	nms_indices = tf.image.non_max_suppression(topk_valid_boxes_y1x1y2x2,topk_valid_scores,max_output_size=post_nms_topk,iou_threshold=config.rpn_proposal_nms_thres)

	topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1,4))
	# (TOPK,4)
	final_boxes = tf.gather(topk_valid_boxes, nms_indices,name="boxes")
	final_scores = tf.gather(topk_valid_scores, nms_indices,name="scores")

	return final_boxes, final_scores


# given the anchor regression prediction, 
# get the refined anchor boxes
def decode_bbox_target(box_predictions, anchors,decode_clip=np.log(1333/16.0)):
	box_pred_txtytwth = tf.reshape(box_predictions, (-1,4)) 
	# [FS,FS,num_anchors,4] -> [All,2] 
	box_pred_txty,box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)

	# get the original anchor box from x1y1x2y2 to center xaya and wh
	anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 4))
	anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
	waha = anchors_x2y2 - anchors_x1y1
	xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

	# get the refined box
	# predicted twth is in log
	wbhb = tf.exp(tf.minimum(box_pred_twth, decode_clip)) * waha
	xbyb = box_pred_txty * waha + xaya

	# get the refined box in x1y1x2y2
	x1y1 = xbyb - wbhb*0.5
	x2y2 = xbyb + wbhb*0.5
	out = tf.concat([x1y1,x2y2],axis=-1) # [All,4]
	return tf.reshape(out,tf.shape(anchors))

def resizeImage(im, short_size, max_size):
	h, w = im.shape[:2]
	neww, newh = get_new_hw(h, w, short_size, max_size)
	if (h==newh) and (w==neww):
		return im
	return cv2.resize(im, (neww, newh), interpolation=cv2.INTER_LINEAR)

def get_new_hw(h,w,size,max_size):
	scale = size * 1.0 / min(h, w)
	if h < w:
		newh, neww = size, scale * w
	else:
		newh, neww = scale * h, size
	if max(newh, neww) > max_size:
		scale = max_size * 1.0 / max(newh, neww)
		newh = newh * scale
		neww = neww * scale
	neww = int(neww + 0.5)
	newh = int(newh + 0.5)
	return neww,newh

# given MxM mask, put it to the whole (4) image
# TODO, make it just to box size to save memroy?
def fill_full_mask(box, mask, im_shape):
	# int() is floor
	# box fpcoor=0.0 -> intcoor=0.0
	x0, y0 = list(map(int, box[:2] + 0.5))
	# box fpcoor=h -> intcoor=h-1, inclusive
	x1, y1 = list(map(int, box[2:] - 0.5))	# inclusive
	x1 = max(x0, x1) # require at least 1x1
	y1 = max(y0, y1)

	w = x1 + 1 - x0
	h = y1 + 1 - y0

	# rounding errors could happen here, because masks were not originally computed for this shape.
	# but it's hard to do better, because the network does not know the "original" scale
	mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
	ret = np.zeros(im_shape, dtype='uint8')
	ret[y0:y1 + 1, x0:x1 + 1] = mask
	return ret


# get the diff (t_x,t_y,t_w,t_h) for each target anchor and original anchor
def encode_bbox_target(target_boxes,anchors):
	# target_boxes, [FS,FS,num_anchors,4] # each anchor's nearest assigned gt bounding box ## some may be zero, so some anchor doesn't match to any gt bounding box
	# anchors, [FS,FS,num_anchors,4] # all posible anchor box
	with tf.name_scope("encode_bbox_target"):
		# encode the box to center xy and wh
		# as the Faster-RCNN paper 
		anchors_x1y1x2y2 = tf.reshape(anchors,(-1,4)) # (N_num_anchors,X,Y)
		anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis = 1)
		# [N_num_anchors,2]
		# get the box center x,y and w,h
		waha = anchors_x2y2 - anchors_x1y1
		xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

		target_boxes_x1y1x2y2 = tf.reshape(target_boxes,(-1,4)) # (N_num_anchors,X,Y)
		target_boxes_x1y1, target_boxes_x2y2 = tf.split(target_boxes_x1y1x2y2, 2, axis = 1)
		# [N_num_anchors,2]
		# get the box center x,y and w,h
		wghg = target_boxes_x2y2 - target_boxes_x1y1
		xgyg = (target_boxes_x2y2 + target_boxes_x1y1) * 0.5

		# some box is zero for non-positive anchor
		TxTy = (xgyg - xaya) / waha
		TwTh = tf.log(wghg / waha)
		encoded = tf.concat([TxTy,TwTh],axis =-1)# [N_num_anchors,4]
		return tf.reshape(encoded,tf.shape(target_boxes))


# https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
def focal_loss(logits, labels, alpha=0.25, gamma=2):
	# labels are one-hot encode float type
	# [N, num_classes]
	assert len(logits.shape) == 2
	assert len(labels.shape) == 2

	sigmoid_p = tf.nn.sigmoid(logits)

	zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

	pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

	neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)

	focal_loss = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

	return tf.reduce_sum(focal_loss)

def deformable_conv2d(inputs, offset, ch_out, kernel_size=3, activation=tf.identity, use_bias=True, W_init=None, data_format="NHWC", scope="deformable"):
	with tf.variable_scope(scope):		
		if data_format == "NCHW": # need NHWC
			inputs = tf.transpose(inputs, [0, 2, 3, 1])
			offset = tf.transpose(offset, [0, 2, 3, 1])
		in_shape = inputs.get_shape().as_list()
		in_channel = in_shape[3]
		input_h, input_w = tf.shape(inputs)[1], tf.shape(inputs)[2]

		shape = (kernel_size, kernel_size, in_channel, ch_out)
		kernel_n = shape[0] * shape[1]

		# [3x3, 2] local offset
		# (0, 0), (0, 1) ...
		# (1, 0), (1, 1) ..
		# ..
		initial_offset = tf.stack(
			tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij')
		)
		# [3x3, 2]
		initial_offset = tf.reshape(initial_offset, (-1, 2))
		# [1, 1, 3x3, 2]
		initial_offset = tf.expand_dims(tf.expand_dims(initial_offset, 0), 0) 
		# [h, w, 3x3, 2]
		initial_offset = tf.tile(initial_offset, [input_h, input_w, 1, 1])
		initial_offset = tf.cast(initial_offset, "float32")

		grid = tf.meshgrid(
			tf.range(-int((shape[0] - 1) / 2.0), input_h - int((shape[0] - 1) / 2.0), 1),
			tf.range(-int((shape[0] - 1) / 2.0), input_w - int((shape[0] - 1) / 2.0), 1), indexing='ij'
		)
		#[h, w, 2] # each h, w location?
		grid = tf.stack(grid, axis=-1)
		grid = tf.cast(grid, 'float32')
		grid = tf.expand_dims(grid, 2)
		# [h, w, 3*3, 2]
		grid = tf.tile(grid, [1, 1, kernel_n, 1])
		grid_offset = grid + initial_offset

		# [b, h, w, 3x3, c]
		input_deform = _tf_batch_map_offsets(inputs, offset, grid_offset)

		if W_init is None:
			W_init = tf.variance_scaling_initializer(scale=2.0)
		W = tf.get_variable(name="W", shape=[shape[0], shape[1], shape[-2], shape[-1]], initializer=W_init)

		W = tf.reshape(W, [1, 1, shape[0]*shape[1], shape[-2], shape[-1]])

		output = tf.nn.conv3d(input_deform, W, strides=[1, 2, 2, 1, 1], data_format="NDHWC", padding="VALID", name=None)
		# output is [b, new_h, new_w, 1, c]
		
		output = tf.squeeze(output, axis=3) # [b, h, w, c]
		if data_format == "NCHW":
			output = tf.transpose(output, [0, 3, 1, 2])

		if use_bias:
			b_init = tf.constant_initializer()
			b = tf.get_variable('b', [ch_out], initializer=b_init)
			output = tf.nn.bias_add(output, b, data_format=data_format)

		output = activation(output, name="output")
		return output

















