# coding=utf-8
# trainer class, given the model (model has the function to get_loss())



import tensorflow as tf
import sys
from models import assign_to_device

def average_gradients(tower_grads,sum_grads=False):
	"""Calculate the average/summed gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
		over the devices. The inner list ranges over the different variables.
	Returns:
			List of pairs of (gradient, variable) where the gradient has been averaged
			across all towers.
	"""
	average_grads = []
	nr_tower = len(tower_grads)
	for grad_and_vars in zip(*tower_grads):

		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = [g for g, _ in grad_and_vars]
		if sum_grads:
			#grad = tf.reduce_sum(grads, 0)
			grad = tf.add_n(grads)
		else:
			grad = tf.multiply(tf.add_n(grads), 1.0 / nr_tower)
			#grad = tf.reduce_mean(grads, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		average_grads.append((grad, v))
	return average_grads

class Trainer():
	def __init__(self,models,config):
		self.config = config
		self.models = models 
		self.global_step = models[0].global_step # 
		
		learning_rate = config.init_lr

		if config.use_lr_decay:
			# always use warmup, set step to zero to disable
			warm_up_start = config.init_lr * 0.33
			# linear increasing from 0.33*lr to lr in warm_up_steps
			warm_up_lr = tf.train.polynomial_decay(
				warm_up_start,
				self.global_step,
				config.warm_up_steps,
				config.init_lr,
				power=1.0, 
			)
						
			if config.use_cosine_schedule:				
				max_steps = int(config.train_num_examples / config.im_batch_size * config.num_epochs)
				schedule_lr = tf.train.cosine_decay(
				 	config.init_lr,
					self.global_step - config.warm_up_steps - config.same_lr_steps,
					max_steps - config.warm_up_steps - config.same_lr_steps,
					alpha=0.0
				)			
			else:
				decay_steps = int(config.train_num_examples / config.im_batch_size * config.num_epoch_per_decay)
				schedule_lr = tf.train.exponential_decay(
				 	config.init_lr,
					self.global_step,
					decay_steps,
					config.learning_rate_decay,
					staircase=True
				)

			boundaries = [config.warm_up_steps, config.warm_up_steps + config.same_lr_steps] # before reaching warm_up steps, use the warm up learning rate.
			values = [warm_up_lr, config.init_lr, schedule_lr]
			learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
			print "learning rate warm up lr from %s to %s in %s steps, then keep for %s steps, then schedule learning rate decay" % (warm_up_start, config.init_lr, config.warm_up_steps, config.same_lr_steps)

			self.learning_rate = learning_rate
		else:
			self.learning_rate = None

		if config.optimizer == 'adadelta':
			self.opt = tf.train.AdadeltaOptimizer(learning_rate)
		elif config.optimizer == "adam":
			self.opt = tf.train.AdamOptimizer(learning_rate)
		elif config.optimizer == "sgd":
			self.opt = tf.train.GradientDescentOptimizer(learning_rate)
		elif config.optimizer == "momentum":
			self.opt = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
		else:
			print "optimizer not implemented"
			sys.exit()

		self.rpn_label_losses = [model.rpn_label_loss for model in models]
		self.rpn_box_losses = [model.rpn_box_loss for model in models]
		self.fastrcnn_label_losses = [model.fastrcnn_label_loss for model in models]
		self.fastrcnn_box_losses = [model.fastrcnn_box_loss for model in models]
		

		if config.wd is not None:
			self.wd = [model.wd for model in models]
		if config.use_small_object_head:
			self.so_label_losses = [model.so_label_loss for model in models]

		if config.add_act:
			self.act_losses = [model.act_losses for model in self.models]

		self.losses = []
		self.grads = []
		for model in self.models:
			gpuid = model.gpuid
			# compute gradients on each gpu devices
			with tf.device(assign_to_device("/GPU:%s"%(gpuid), config.controller)):
				self.losses.append(model.loss)
				grad = self.opt.compute_gradients(model.loss)

				grad = [(g,var) for g, var in grad if g is not None] # we freeze resnet, so there will be none gradient

				# whehter to clip gradient
				if config.clip_gradient_norm is not None:
					grad = [(tf.clip_by_value(g, -1*config.clip_gradient_norm, config.clip_gradient_norm), var) for g, var in grad]
				self.grads.append(grad)
		
		# apply gradient on the controlling device
		with tf.device(config.controller):
			avg_loss = tf.reduce_mean(self.losses)
			avg_grads = average_gradients(self.grads,sum_grads=True)

			self.train_op = self.opt.apply_gradients(avg_grads,global_step=self.global_step)
			self.loss = avg_loss

		

	def step(self,sess,batch,get_summary=False): 
		assert isinstance(sess,tf.Session)
		config = self.config

		# idxs is a tuple (23,123,33..) index for sample
		batchIdx,batch_datas = batch
		#assert len(batch_datas) == len(self.models) # there may be less data in the end
		
		feed_dict = {}
	
		for batch_data, model in zip(batch_datas, self.models): # if batch is smaller so will the input?
			feed_dict.update(model.get_feed_dict(batch_data,is_train=True))

		sess_input = []
		sess_input.append(self.loss)

		for i in xrange(len(self.models)):
			sess_input.append(self.rpn_label_losses[i])
			sess_input.append(self.rpn_box_losses[i])
			sess_input.append(self.fastrcnn_label_losses[i])
			sess_input.append(self.fastrcnn_box_losses[i])
			
			if config.wd is not None:
				sess_input.append(self.wd[i])

			if config.use_small_object_head:
				sess_input.append(self.so_label_losses[i])
			if config.add_act:
				sess_input.append(self.act_losses[i])

		sess_input.append(self.train_op)
		sess_input.append(self.learning_rate)

		outs = sess.run(sess_input,feed_dict=feed_dict)

		loss = outs[0]

		skip = 4 + int(config.add_act) + int(config.use_small_object_head)
		rpn_label_losses = outs[1::skip][:len(self.models)]
		rpn_box_losses = outs[2::skip][:len(self.models)]
		fastrcnn_label_losses = outs[3::skip][:len(self.models)]
		fastrcnn_box_losses = outs[4::skip][:len(self.models)]
		
		now = 4
		wd = [-1 for m in self.models]
		if config.wd is not None:
			now+=1
			wd = outs[now::skip][:len(self.models)]

		so_label_losses = [-1 for m in self.models]
		if config.use_small_object_head:
			now+=1
			so_label_losses = outs[now::skip][:len(self.models)]
		act_losses = [-1 for m in self.models]
		if config.add_act:
			now+=1
			act_losses = outs[now::skip][:len(self.models)]
		

		"""
		if config.add_act:
			out = [self.loss, self.rpn_label_loss, self.rpn_box_loss, self.fastrcnn_label_loss, self.fastrcnn_box_loss, self.train_op]

			act_losses_pl = [model.act_losses for model in self.models]
			out = act_losses_pl + out
			things = sess.run(out,feed_dict=feed_dict)
			act_losses = things[:len(act_losses_pl)]

			loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op = things[len(act_losses_pl):]
		else:
			loss,rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, train_op = sess.run([self.loss,self.rpn_label_loss, self.rpn_box_loss, self.fastrcnn_label_loss, self.fastrcnn_box_loss,self.train_op],feed_dict=feed_dict)
			act_losses = None
		"""
		learning_rate = outs[-1]
		return loss, wd, rpn_label_losses, rpn_box_losses, fastrcnn_label_losses, fastrcnn_box_losses, so_label_losses, act_losses, learning_rate

	


