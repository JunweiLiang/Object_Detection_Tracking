# coding=utf-8
# helper function for deformable conv
import tensorflow as tf

def _to_bc_h_w(x, x_shape):
		"""(b, h, w, c) -> (b*c, h, w)"""
		x = tf.transpose(x, [0, 3, 1, 2])
		x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
		return x

def _to_b_h_w_n_c(x, x_shape):
	"""(b*c, h, w, n) -> (b, h, w, n, c)"""
	x = tf.reshape(x, (-1, x_shape[4], x_shape[1], x_shape[2], x_shape[3]))
	x = tf.transpose(x, [0, 2, 3, 4, 1])
	return x

def tf_flatten(a):
	"""Flatten tensor"""
	return tf.reshape(a, [-1])

def _get_vals_by_coords(inputs, coords, idx, out_shape):
	indices = tf.stack(
		[idx, tf_flatten(coords[:, :, :, :, 0]),
		 tf_flatten(coords[:, :, :, :, 1])], axis=-1
	)
	vals = tf.gather_nd(inputs, indices)
	vals = tf.reshape(vals, out_shape)
	return vals

def _tf_repeat(a, repeats):
	"""Tensorflow version of np.repeat for 1D"""
	# https://github.com/tensorflow/tensorflow/issues/8521

	if len(a.get_shape()) != 1:
		raise AssertionError("This is not a 1D Tensor")

	a = tf.expand_dims(a, -1)
	a = tf.tile(a, [1, repeats])
	a = tf_flatten(a)
	return a

def _tf_batch_map_coordinates(inputs, coords):
	"""Batch version of tf_map_coordinates

	Only supports 2D feature maps

	Parameters
	----------
	inputs : ``tf.Tensor``
		shape = (b*c, h, w)
	coords : ``tf.Tensor``
		shape = (b*c, h, w, n, 2)

	Returns
	-------
	``tf.Tensor``
		A Tensor with the shape as (b*c, h, w, n)

	"""
	input_shape = inputs.get_shape()
	coords_shape = coords.get_shape()
	batch_channel = tf.shape(inputs)[0]
	input_h = tf.shape(inputs)[1]
	input_w = tf.shape(inputs)[2]
	kernel_n = int(coords_shape[3])
	n_coords = input_h * input_w * kernel_n

	coords_lt = tf.cast(tf.floor(coords), 'int32')
	coords_rb = tf.cast(tf.ceil(coords), 'int32')
	coords_lb = tf.stack([coords_lt[:, :, :, :, 0], coords_rb[:, :, :, :, 1]], axis=-1)
	coords_rt = tf.stack([coords_rb[:, :, :, :, 0], coords_lt[:, :, :, :, 1]], axis=-1)

	idx = _tf_repeat(tf.range(batch_channel), n_coords)

	vals_lt = _get_vals_by_coords(inputs, coords_lt, idx, (batch_channel, input_h, input_w, kernel_n))
	vals_rb = _get_vals_by_coords(inputs, coords_rb, idx, (batch_channel, input_h, input_w, kernel_n))
	vals_lb = _get_vals_by_coords(inputs, coords_lb, idx, (batch_channel, input_h, input_w, kernel_n))
	vals_rt = _get_vals_by_coords(inputs, coords_rt, idx, (batch_channel, input_h, input_w, kernel_n))

	coords_offset_lt = coords - tf.cast(coords_lt, 'float32')

	vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, :, :, 0]
	vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, :, :, 0]
	mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, :, :, 1]

	return mapped_vals

def _tf_batch_map_offsets(inputs, offsets, grid_offset):
	"""Batch map offsets into input

	Parameters
	------------
	inputs : ``tf.Tensor``
		shape = (b, h, w, c)
	offsets: ``tf.Tensor``
		shape = (b, h, w, 2*n)
	grid_offset: `tf.Tensor``
		Offset grids shape = (h, w, n, 2)

	Returns
	-------
	``tf.Tensor``
		A Tensor with the shape as (b, h, w, c)

	"""
	input_shape = inputs.get_shape()
	batch_size = tf.shape(inputs)[0]
	kernel_n = int(int(offsets.get_shape()[3]) / 2)
	input_h = tf.shape(inputs)[1]
	input_w = tf.shape(inputs)[2]
	channel = input_shape[3]

	# inputs (b, h, w, c) --> (b*c, h, w)
	inputs = _to_bc_h_w(inputs, tf.shape(inputs))

	# offsets (b, h, w, 2*n) --> (b, h, w, n, 2)
	offsets = tf.reshape(offsets, (batch_size, input_h, input_w, kernel_n, 2))
	# offsets (b, h, w, n, 2) --> (b*c, h, w, n, 2)
	# offsets = tf.tile(offsets, [channel, 1, 1, 1, 1])

	coords = tf.expand_dims(grid_offset, 0)  # grid_offset --> (1, h, w, n, 2)
	coords = tf.tile(coords, [batch_size, 1, 1, 1, 1]) + offsets  # grid_offset --> (b, h, w, n, 2)

	# clip out of bound
	coords = tf.stack(
		[
			tf.clip_by_value(coords[:, :, :, :, 0], 0.0, tf.cast(input_h - 1, 'float32')),
			tf.clip_by_value(coords[:, :, :, :, 1], 0.0, tf.cast(input_w - 1, 'float32'))
		], axis=-1
	)
	coords = tf.tile(coords, [channel, 1, 1, 1, 1])

	mapped_vals = _tf_batch_map_coordinates(inputs, coords)
	# (b*c, h, w, n) --> (b, h, w, n, c)
	mapped_vals = _to_b_h_w_n_c(mapped_vals, [batch_size, input_h, input_w, kernel_n, channel])

	return mapped_vals