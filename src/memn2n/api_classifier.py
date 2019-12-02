from __future__ import absolute_import
from __future__ import division

import copy 
import numpy as np
import tensorflow as tf

from datetime import datetime
from memn2n.attention_wrapper import *
from memn2n.beam_decoder import *
from memn2n.dynamic_decoder import *
from memn2n.helper import *
from six.moves import range
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, rnn
from tensorflow.python.util import nest

###################################################################################################
#########                                  Helper Functions                              ##########
###################################################################################################

def zero_nil_slot(t, name=None):
	"""
	Overwrites the nil_slot (first row) of the input Tensor with zeros.

	The nil_slot is a dummy slot and should not be trained and influence
	the training algorithm.
	"""
	with tf.name_scope(name, "zero_nil_slot", [t]) as name:
		t = tf.convert_to_tensor(t, name="t")
		s = tf.shape(t)[1]
		z = tf.zeros(tf.stack([1, s]))
		return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)


###################################################################################################
#########                                     Model Class                                ##########
###################################################################################################

class APIClassifier(object):
	"""End-To-End Memory Network with a generative decoder."""
	def __init__(self, args, glob):

		# Initialize Model Variables
		self._batch_size = args.batch_size
		self._beam = args.beam
		self._simple_beam = args.simple_beam
		self._beam_width = args.beam_width
		self._candidate_sentence_size = glob['candidate_sentence_size']
		self._debug = args.debug
		self._decode_idx = glob['decode_idx']
		self._embedding_size = args.embedding_size
		self._hierarchy = args.hierarchy
		self._hops = args.cl_hops
		self._init = tf.random_normal_initializer(stddev=0.1)
		self._max_grad_norm = args.max_grad_norm
		self._name = 'Classifier'
		self._opt = glob['classifier_optimizer']
		self._p_gen_loss = args.p_gen_loss
		self._p_gen_loss_weight = args.p_gen_loss_weight
		self._rl = args.rl
		self._fixed_length_decode = args.fixed_length_decode
		self._sentence_size = glob['sentence_size']
		self._soft_weight = args.soft_weight
		self._task_id = args.task_id
		self._vocab_size = glob['vocab_size']
		self._constraint_mask = glob['constraint_mask']
		self._state_mask = glob['state_mask']
		self._constraint = args.constraint
		self._rl_mode = args.rl_mode 
		self._split_emb = args.split_emb

		self.phase = args.phase

		# Add unk and eos
		self.UNK = self._decode_idx["UNK"]
		self.EOS = self._decode_idx["EOS"]
		self.GO_SYMBOL = self._decode_idx["GO_SYMBOL"]

		self.START_STATE = 0

		self._decoder_vocab_size = len(self._decode_idx)

		# Add RL specific variables
		if self._rl:
			self._rl_idx = glob['rl_idx']
			self._rl_vocab_size = len(self._rl_idx)
			self._max_api_length = args.max_api_length

			if self._fixed_length_decode:
				self._rl_decode_length_lookup_array = glob['rl_decode_length_lookup_array']
		
		self._build_inputs()
		self._build_vars()

		### API Turn prediction
		encoder_states_var, _, _ = self._encoder_var(self._stories, self._queries, emb=self.A)
		self.api_predict_op = self._decoder_predict_api(encoder_states_var)

		## Training ##
		self.api_loss_op = self._decoder_train_api(encoder_states_var)
		loss, logits, gold, predictions = self.api_loss_op

		# Gradient Pipeline
		api_grads_and_vars = self._opt.compute_gradients(loss)
		api_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in api_grads_and_vars if g != None]
		api_nil_grads_and_vars = [(zero_nil_slot(g), v) if v.name in self._nil_vars else (g, v) for g, v, in api_grads_and_vars]
		self.api_train_op = self._opt.apply_gradients(api_nil_grads_and_vars, name="api_train_op")
			
		init_op = tf.global_variables_initializer()
		self._sess = glob['classifier_session']
		self._sess.run(init_op)

	def _build_inputs(self):
		'''
			Define Input Variables to be given to the model
		'''
		## Encode Ids ##
		self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
		self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
		self._answers = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers")
		
		## Sizes ##
		self._sentence_sizes = tf.placeholder(tf.int32, [None, None], name="sentence_sizes")
		self._query_sizes = tf.placeholder(tf.int32, [None, 1], name="query_sizes")
		self._answer_sizes = tf.placeholder(tf.int32, [None, 1], name="answer_sizes")

		## OOV Helpers ##
		self._oov_ids = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="oov_ids")
		self._oov_sizes = tf.placeholder(tf.int32, [None], name="oov_sizes")
		if self._rl:
			self._rl_oov_ids = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="rl_oov_ids")
			self._rl_oov_sizes = tf.placeholder(tf.int32, [None], name="rl_oov_sizes")
		
		## Train Helpers ###
		self._intersection_mask = tf.placeholder(tf.float32, [None, self._candidate_sentence_size], name="intersection_mask")
		self._answers_emb_lookup = tf.placeholder(tf.int32, [None, self._candidate_sentence_size], name="answers_emb")
		self._keep_prob = tf.placeholder(tf.float32)

		## RL ##
		if self._rl:
			## API Turn Prediction ##
			self._make_api = tf.placeholder(tf.float32, [None], name="make_api")

			self._rl_actions = tf.placeholder(tf.int32, [None, self._max_api_length], name="actions")
			self._rl_actions_emb_lookup = tf.placeholder(tf.int32, [None, self._max_api_length], name="actions_emb")
			self._rl_action_sizes = tf.placeholder(tf.int32, [None, 1], name="action_sizes")
			self._rl_rewards = tf.placeholder(tf.float32, [None, 1], name="rewards")
			if self._fixed_length_decode:
				self._rl_decode_length_classes_count = len(self._rl_decode_length_lookup_array)
				self._rl_decode_length_lookup = tf.constant(np.asarray(self._rl_decode_length_lookup_array, dtype=np.int32))
				self._rl_decode_length_class_ids = tf.placeholder(tf.int32, [None, 1], name="fld_class_ids")

	def _build_vars(self):
		'''
			Define Model specific variables used to train and fit and test the model
		'''
		with tf.variable_scope(self._name):
			nil_word_slot = tf.zeros([1, self._embedding_size])

			# Initialize Embedding for Encoder
			A = tf.concat([nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])], 0)
			self.A = tf.Variable(A, name="A", trainable=False)

			# Hop Context Vector to Output Query 
			self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H", trainable=False)

			with tf.variable_scope("encoder"):
				self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
				self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

		self._nil_vars = set([self.A.name])

	def set_vars(self, A, H, fwd, bwd):
		self.A.assign(A)
		self.H.assign(H)
		self.encoder_fwd = fwd
		self.encoder_bwd = bwd

	###################################################################################################
	#########                                  	  Encoder                                    ##########
	########################s###########################################################################

	def _encoder_var(self, stories, queries, emb):
		'''
			Arguments:
				stories 	-	batch_size x memory_size x sentence_size
				queries 	-	batch_size x sentence_size
			Outputs:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	-	batch_size x memory_size x sentence_size x embedding_size

		'''
		with tf.variable_scope(self._name):
			### Set Variables ###
			self._batch_size = tf.shape(stories)[0]
			self._memory_size = tf.shape(stories)[1]

			### Transform Queries ###
			query = tf.Variable(tf.random_uniform([64, self._embedding_size]))
			# query = tf.constant(tf.random_uniform([64, self._embedding_size]))
			sliced_query = query[:self._batch_size, :]
			u = [sliced_query]
			
			### Transform Stories ###
			# memory_word_emb : batch_size x memory_size x sentence_size x embedding_size
			memory_word_emb = tf.nn.embedding_lookup(emb, stories)
			memory_emb = tf.reshape(memory_word_emb, [-1, self._sentence_size, self._embedding_size])

			sentence_sizes = tf.reshape(self._sentence_sizes, [-1])
			with tf.variable_scope("encoder"):
				(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, memory_emb, sequence_length=sentence_sizes, dtype=tf.float32)
			(f_state, b_state) = output_states
			
			line_memory = tf.concat(axis=1, values=[f_state, b_state])
			# line_memory : batch_size x memory_size x embedding_size
			line_memory = tf.reshape(line_memory, [self._batch_size, self._memory_size, self._embedding_size])
			
			(f_states, b_states) = outputs
			word_memory = tf.concat(axis=2, values=[f_states, b_states]) 
			word_memory = tf.reshape(word_memory, [self._batch_size, self._memory_size, self._sentence_size, self._embedding_size])

			### Implement Hop Network ###
			for hop_index in range(self._hops):
				
				# hack to get around no reduce_dot
				u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
				dotted = tf.reduce_sum(line_memory * u_temp, 2)

				# Calculate probabilities
				probs = tf.nn.softmax(dotted)
				probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
				c_temp = tf.transpose(line_memory, [0, 2, 1])
				o_k = tf.reduce_sum(c_temp * probs_temp, 2)
				u_k = tf.matmul(u[-1], self.H) + o_k
				u.append(u_k)
			
			return u_k, line_memory, word_memory

	###################################################################################################
	#########                                API-Classifier                                  ##########
	###################################################################################################
	def linear(self, args, output_size, bias, bias_start=0.0, scope=None):
		# Now the computation.
		with tf.variable_scope(scope or "Linear"):
			matrix = tf.get_variable("Matrix", [self._embedding_size, output_size])
			res = tf.matmul(args[0], matrix)
			if not bias:
				return res
			bias_term = tf.get_variable(
				"Bias", [output_size], initializer=tf.constant_initializer(bias_start))
		return res + bias_term 

	def _decoder_train_api(self, bag_of_words):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:
				loss 	- 	Total Loss (Sequence Loss + PGen Loss) (Float)
		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('api_classifier', reuse=True):
				## Get logits
				logits = tf.reshape(tf.sigmoid(self.linear([bag_of_words], 1, True)), [-1])
				predictions = tf.round(logits)

				## Calculate Loss
				y_pred = tf.clip_by_value(logits,1e-2,1.0-1e-2)
				loss = -tf.reduce_sum(10 * self._make_api * tf.log(y_pred) + (1 - self._make_api) * tf.log(1 - y_pred))

				return loss, y_pred, self._make_api, predictions

	def _decoder_predict_api(self, bag_of_words):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('api_classifier'):
				## Linear Layer
				predictions = tf.round(tf.reshape(tf.sigmoid(self.linear([bag_of_words], 1, True)), [-1]))
				return predictions


	def _check_shape(self, name, array):
		try:
			shape = array[0].shape
			for i, arr in enumerate(array):
				sh = arr.shape
				if sh != shape:
					print(name, i, shape, sh)
		except:
			print('FAILED' + name); print(array)

	def _print_feed(self, feed_dict, actions_and_rewards, train):
		self._check_shape('Stories: ', feed_dict[self._stories])
		self._check_shape('Story Sizes: ', feed_dict[self._sentence_sizes])
		self._check_shape('Queries: ', feed_dict[self._queries])
		self._check_shape('Queries Sizes: ', feed_dict[self._query_sizes])
		self._check_shape('oov ids: ', feed_dict[self._oov_ids])
		self._check_shape('oov sizes: ', feed_dict[self._oov_sizes])
		self._check_shape('intersection mask: ', feed_dict[self._intersection_mask])
		if self._rl:
			self._check_shape('rl_oov_ids: ', feed_dict[self._rl_oov_ids])
			self._check_shape('rl_oov_sizes: ', feed_dict[self._rl_oov_sizes])
		if train:
			self._check_shape('answers: ', feed_dict[self._answers])
			self._check_shape('answers_emb_lookup: ', feed_dict[self._answers_emb_lookup] )
			self._check_shape('answer_sizes: ', feed_dict[self._answer_sizes])
			if self._rl and actions_and_rewards:
				self._check_shape('rl_actions_emb_lookup: ', feed_dict[self._rl_actions_emb_lookup])
				self._check_shape('rl_actions: ', feed_dict[self._rl_actions] )
				self._check_shape('rl_rewards: ', feed_dict[self._rl_rewards])
				self._check_shape('rl_action_sizes: ', feed_dict[self._rl_action_sizes])

	def _make_feed_dict(self, batch, actions_and_rewards=None, train=True, api=False):
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

		Args:
		  batch: Batch object
		  just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		feed_dict = {}
		feed_dict[self._stories] = np.array(batch.stories)
		feed_dict[self._queries] = np.array(batch.queries)
		feed_dict[self._sentence_sizes] = np.array(batch.story_sizes)
		feed_dict[self._query_sizes] = np.array(batch.query_sizes)
		feed_dict[self._oov_ids] = np.array(batch.oov_ids)
		feed_dict[self._oov_sizes] = np.array(batch.oov_sizes)
		feed_dict[self._intersection_mask] = np.array(batch.intersection_set)
		if self._rl:
			feed_dict[self._rl_oov_ids] = np.array(batch.rl_oov_ids)
			feed_dict[self._rl_oov_sizes] = np.array(batch.rl_oov_sizes)
		if train:
			feed_dict[self._answers] = np.array(batch.answers)
			feed_dict[self._answers_emb_lookup] = np.array(batch.answers_emb_lookup)
			feed_dict[self._answer_sizes] = np.array(batch.answer_sizes)
			feed_dict[self._keep_prob] = 0.5
			if self._rl and actions_and_rewards:
				feed_dict[self._rl_actions_emb_lookup] = np.array(actions_and_rewards.actions_emb_lookup)
				feed_dict[self._rl_actions] = np.array(actions_and_rewards.actions)
				feed_dict[self._rl_rewards] = np.array(actions_and_rewards.rewards)
				feed_dict[self._rl_action_sizes] = np.array(actions_and_rewards.action_sizes)
				if self._fixed_length_decode:
					feed_dict[self._rl_decode_length_class_ids] = np.array(actions_and_rewards.rl_decode_length_class_ids)
		else:
			feed_dict[self._keep_prob] = 1.0
		if api:
			feed_dict[self._make_api] = np.array(batch.make_api)
		if self._debug:
			self._print_feed(feed_dict, actions_and_rewards, train)
		return feed_dict

	def fit_api_call(self, batch):
		"""
			Runs the training algorithm over the passed batch
		  Returns:
			loss: floating-point number, the loss computed for the batch
		"""
		feed_dict = self._make_feed_dict(batch, api=True)
		loss, _= self._sess.run([self.api_loss_op, self.api_train_op], feed_dict=feed_dict)
		return loss

	def predict_api_call(self, batch):
		"""
			Predicts answers as one-hot encoding.
		  Returns:
			answers: Tensor (None, vocab_size)
		"""
		feed_dict = self._make_feed_dict(batch, train=False, api=True)
		return self._sess.run(self.api_predict_op, feed_dict=feed_dict)