from __future__ import absolute_import
from __future__ import division

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

class MemN2NGeneratorDialog(object):
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
		self._hops = args.hops
		self._init = tf.random_normal_initializer(stddev=0.1)
		self._max_grad_norm = args.max_grad_norm
		self._name = 'MemN2N'
		self._opt = glob['optimizer']
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

		## Encoding ##
		encoder_states, line_memory, word_memory = self._encoder(self._stories, self._queries, emb=self.A)

		if self._rl:
			if self._split_emb:
				encoder_states_rl, line_memory, word_memory = self._encoder(self._stories, self._queries, emb=self.B)
			else:
				encoder_states_rl, line_memory, word_memory = self._encoder(self._stories, self._queries, emb=self.A)

		## Predicting ##
		self.predict_op = self._decoder_predict(encoder_states, line_memory, word_memory)

		## Training ##
		self.loss_op = self._decoder_train(encoder_states, line_memory, word_memory)

		# Gradient Pipeline
		grads_and_vars = self._opt.compute_gradients(self.loss_op[0])
		grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars if g != None]
		nil_grads_and_vars = [(zero_nil_slot(g), v) if v.name in self._nil_vars else (g, v) for g, v, in grads_and_vars]
		self.train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

		if self._rl:
			## Predicting ##
			self.rl_predict_op = self._decoder_predict_rl(encoder_states_rl, line_memory, word_memory)

			## Training ##
			self.rl_loss_op, rl_logits, rl_p_gens = self._decoder_train_rl(encoder_states_rl, line_memory, word_memory)

			## Probability Scores ##
			self.prob_op = self._decoder_prob_rl(encoder_states_rl, line_memory, word_memory)

			# Gradient Pipeline
			rl_grads_and_vars = self._opt.compute_gradients(self.rl_loss_op)
			rl_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in rl_grads_and_vars if g != None]
			rl_nil_grads_and_vars = [(zero_nil_slot(g), v) if v.name in self._nil_vars else (g, v) for g, v, in rl_grads_and_vars]
			self.rl_train_op = self._opt.apply_gradients(rl_nil_grads_and_vars, name="rl_train_op")
			
		init_op = tf.global_variables_initializer()
		self._sess = glob['session']
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
			self.A = tf.Variable(A, name="A")

			# Initialize Embedding for RL Encoder
			B = tf.concat([nil_word_slot, self._init([self._vocab_size-1, self._embedding_size])], 0)
			self.B = tf.Variable(A, name="B")
			
			# Initialize Embedding for Response-Decoder
			C = tf.concat([nil_word_slot, self._init([self._decoder_vocab_size-1, self._embedding_size])], 0)
			self.C = tf.Variable(C, name="C")

			# Initialize Embedding for RL-Decoder
			if self._rl:
				R = tf.concat([nil_word_slot, self._init([self._rl_vocab_size-1, self._embedding_size])], 0)
				self.R = tf.Variable(R, name="R")

			# Hop Context Vector to Output Query 
			self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

			# Initialize Embedding for Position Information
			if self._rl:
				P = self._init([self._max_api_length, int(self._embedding_size)])
				self.P = tf.Variable(P, name="P")

			with tf.variable_scope("encoder"):
				self.encoder_fwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)
				self.encoder_bwd = tf.contrib.rnn.GRUCell(self._embedding_size / 2)

			with tf.variable_scope('decoder'):
				self.decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
				self.projection_layer = layers_core.Dense(self._decoder_vocab_size, use_bias=False)

			if self._rl:
				with tf.variable_scope('rl_decoder'):
					self.rl_decoder_cell = tf.contrib.rnn.GRUCell(self._embedding_size)
					self.rl_projection_layer = layers_core.Dense(self._rl_vocab_size, use_bias=False)
					if self._fixed_length_decode:
						self.action_length_ff_layer = tf.Variable(self._init([self._embedding_size, self._rl_decode_length_classes_count]), name="length_ff")

		self._nil_vars = set([self.A.name, self.B.name, self.C.name])
		if self._rl:
			self._nil_vars.add(self.R.name)

	###################################################################################################
	#########                                  	  Encoder                                    ##########
	########################s###########################################################################

	def _encoder(self, stories, queries, emb):
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
			# query_emb : batch_size x sentence_size x embedding_size
			query_emb = tf.nn.embedding_lookup(emb, queries)

			query_sizes = tf.reshape(self._query_sizes, [-1])
			with tf.variable_scope("encoder"):
				(outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_fwd, self.encoder_bwd, query_emb, sequence_length=query_sizes, dtype=tf.float32)
			(f_state, b_state) = output_states
			u_0 = tf.concat(axis=1, values=[f_state, b_state])
			# u_0 : batch_size x embedding_size
			u = [u_0]
			
			### Transform Stories ###
			# memory_word_emb : batch_size x memory_size x sentence_size x embedding_size
			memory_word_emb = tf.nn.embedding_lookup(emb, stories)
			memory_emb = tf.reshape(memory_word_emb, [-1, self._sentence_size, self._embedding_size])

			sentence_sizes = tf.reshape(self._sentence_sizes, [-1])
			with tf.variable_scope("encoder", reuse=True):
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
	#########                                  	 Decoders                                    ##########
	###################################################################################################

	def _get_decoder(self, encoder_states, line_memory, word_memory, helper, batch_size):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
		'''
		# make the shape concrete to prevent ValueError caused by (?, ?, ?)
		reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
		reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
		attention_mechanism = CustomAttention(self._embedding_size, reshaped_line_memory, reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
		decoder_cell_with_attn = AttentionWrapper(self.decoder_cell, attention_mechanism, output_attention=False)			
		wrapped_encoder_states = decoder_cell_with_attn.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)

		init_ids = tf.fill([batch_size], self.GO_SYMBOL)
		state_ids = tf.fill([batch_size], self.START_STATE)
		decoder = BasicDecoder(decoder_cell_with_attn, self.P, helper, wrapped_encoder_states, init_ids, state_ids, output_layer=self.projection_layer)
		return decoder

	def _get_beam_decoder(self, encoder_states, line_memory, word_memory, batch_size):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		# make the shape concrete to prevent ValueError caused by (?, ?, ?)
		reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
		reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
		tiled_reshaped_line_memory = tf.contrib.seq2seq.tile_batch(reshaped_line_memory, multiplier=self._beam_width)
		tiled_reshaped_word_memory = tf.contrib.seq2seq.tile_batch(reshaped_word_memory, multiplier=self._beam_width)
		attention_mechanism = CustomAttention(self._embedding_size, tiled_reshaped_line_memory, tiled_reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
		decoder_cell_with_attn = AttentionWrapper(self.decoder_cell, attention_mechanism, output_attention=False)			
		
		tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_states, multiplier=self._beam_width)
		decoder_initial_state = decoder_cell_with_attn.zero_state(batch_size=(batch_size * self._beam_width), dtype=tf.float32).clone(cell_state=tiled_encoder_final_state)
		
		init_ids = tf.fill([batch_size, self._beam_width], self.GO_SYMBOL)
		state_ids = tf.fill([batch_size, self._beam_width], self.START_STATE)
		decoder = BeamSearchDecoder(decoder_cell_with_attn, self.P, self.C, tf.fill([batch_size], self.GO_SYMBOL), self._decode_idx['EOS'], decoder_initial_state, self._beam_width, self._decoder_vocab_size, init_ids, state_ids, output_layer=self.projection_layer)
		return decoder

	def _get_rl_decoder(self, encoder_states, line_memory, word_memory, helper, batch_size, predict_flag=True):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:
		'''

		# make the shape concrete to prevent ValueError caused by (?, ?, ?)
		reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
		reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
		attention_mechanism = CustomAttention(self._embedding_size, reshaped_line_memory, reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
		decoder_cell_with_attn = AttentionWrapper(self.rl_decoder_cell, attention_mechanism, rl=True, output_attention=False)			
		wrapped_encoder_states = decoder_cell_with_attn.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)

		init_ids = tf.fill([batch_size], self.GO_SYMBOL)
		state_ids = tf.fill([batch_size], self.START_STATE)
		if predict_flag and self._constraint:
			decoder = BasicDecoder(decoder_cell_with_attn, self.P, helper, wrapped_encoder_states, init_ids, state_ids, state_mask=self._state_mask, constraint_mask=self._constraint_mask, output_layer=self.rl_projection_layer)
		else:
			decoder = BasicDecoder(decoder_cell_with_attn, self.P, helper, wrapped_encoder_states, init_ids, state_ids, output_layer=self.rl_projection_layer)
		return decoder

	def _get_rl_beam_decoder(self, encoder_states, line_memory, word_memory, batch_size):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		# make the shape concrete to prevent ValueError caused by (?, ?, ?)
		reshaped_line_memory = tf.reshape(line_memory,[batch_size, -1, self._embedding_size])
		reshaped_word_memory = tf.reshape(word_memory,[batch_size, -1, self._sentence_size, self._embedding_size])
		tiled_reshaped_line_memory = tf.contrib.seq2seq.tile_batch(reshaped_line_memory, multiplier=self._beam_width)
		tiled_reshaped_word_memory = tf.contrib.seq2seq.tile_batch(reshaped_word_memory, multiplier=self._beam_width)
		attention_mechanism = CustomAttention(self._embedding_size, tiled_reshaped_line_memory, tiled_reshaped_word_memory, hierarchy=self._hierarchy, soft_weight=self._soft_weight)
		decoder_cell_with_attn = AttentionWrapper(self.rl_decoder_cell, attention_mechanism, rl=True, output_attention=False)			
		
		tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_states, multiplier=self._beam_width)
		decoder_initial_state = decoder_cell_with_attn.zero_state(batch_size=(batch_size * self._beam_width), dtype=tf.float32).clone(cell_state=tiled_encoder_final_state)
		
		init_ids = tf.fill([batch_size, self._beam_width], self.GO_SYMBOL)
		state_ids = tf.fill([batch_size, self._beam_width], self.START_STATE)
		if self._constraint:
			decoder = BeamSearchDecoder(decoder_cell_with_attn, self.P, self.R, tf.fill([batch_size], self.GO_SYMBOL), self._decode_idx['EOS'], decoder_initial_state, self._beam_width, self._rl_vocab_size, init_ids, state_ids, state_mask=self._state_mask, constraint_mask=self._constraint_mask, output_layer=self.rl_projection_layer)
		else:
			decoder = BeamSearchDecoder(decoder_cell_with_attn, self.P, self.R, tf.fill([batch_size], self.GO_SYMBOL), self._decode_idx['EOS'], decoder_initial_state, self._beam_width, self._rl_vocab_size, init_ids, state_ids, output_layer=self.rl_projection_layer)
		return decoder

	###################################################################################################
	#########                                 Response-Decoder                               ##########
	###################################################################################################
	def _decoder_train(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:
				loss 	- 	Total Loss (Sequence Loss + PGen Loss) (Float)
		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder', reuse=True):

				batch_size = tf.shape(self._stories)[0]

				## Create Training Helper ##
				# decoder_input = batch_size x candidate_sentence_size
				decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._answers_emb_lookup[:, :]],axis=1)
				# decoder_emb_inp = batch_size x candidate_sentence_size x embedding_size
				decoder_emb_inp = tf.nn.embedding_lookup(self.C, decoder_input)
				answer_sizes = tf.reshape(self._answer_sizes,[-1])
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)

				## Run Decoder ##
				decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				outputs,p_gens = dynamic_decode(decoder, batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, impute_finished=False)
				
				## Prepare Loss Helpers ##
				final_dists = outputs.rnn_output
				max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
				ans = self._answers[:, :max_length]
				target_weights = tf.reshape(self._answer_sizes,[-1])
				target_weights = tf.sequence_mask(target_weights, self._candidate_sentence_size, dtype=tf.float32)
				target_weights = target_weights[:, :max_length]

				## Calculate Sequence Loss ##
				max_oov_len = tf.reduce_max(self._oov_sizes, reduction_indices=[0])
				extended_vsize =  self._decoder_vocab_size + max_oov_len
				y_pred = tf.clip_by_value(final_dists,1e-20,1.0)
				y_true = tf.one_hot(ans, extended_vsize)
				seq_loss_comp = -tf.reduce_sum(y_true*tf.log(y_pred))

				## Calculate PGen Loss ##
				intersect_mask = self._intersection_mask[:, :max_length]
				reshaped_p_gens=tf.reshape(tf.squeeze(p_gens), [-1])
				p = tf.reshape(intersect_mask, [-1])
				q = tf.clip_by_value(reshaped_p_gens,1e-20,1.0)
				one_minus_q = tf.clip_by_value(1-reshaped_p_gens,1e-20,1.0)
				p_gen_loss = p*tf.log(q) + (1-p)*tf.log(one_minus_q)
				pgen_loss_comp = -tf.reduce_sum(p_gen_loss * tf.reshape(target_weights, [-1]))
								
				loss = seq_loss_comp + self._p_gen_loss_weight*pgen_loss_comp

				return loss, seq_loss_comp, pgen_loss_comp

	def _decoder_predict(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('decoder'):
				batch_size = tf.shape(self._stories)[0]
				if self._simple_beam:
					tiled_oov_ids = tf.contrib.seq2seq.tile_batch(self._oov_ids, multiplier=self._beam_width)
					decoder = self._get_beam_decoder(encoder_states, line_memory, word_memory, batch_size)				
					outputs,_ = dynamic_decode(decoder, batch_size, self._decoder_vocab_size, self._oov_sizes, tiled_oov_ids, maximum_iterations=2*self._candidate_sentence_size)
					return outputs.parent_ids, outputs.predicted_ids
				else:
					helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
					decoder = self._get_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
					outputs,_ = dynamic_decode(decoder, batch_size, self._decoder_vocab_size, self._oov_sizes, self._oov_ids, maximum_iterations=2*self._candidate_sentence_size)
					return tf.argmax(outputs.rnn_output, axis=-1)

	###################################################################################################
	#########                                  	RL-Decoder                                   ##########
	###################################################################################################
	def _decoder_predict_rl(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			with tf.variable_scope('rl_decoder'):

				batch_size = tf.shape(self._stories)[0]
				
				# predict lengths based on the context
				if self._fixed_length_decode:
					# encoder_states: batch_size x embedding_size
					# self.action_length_ff_layer: embedding_size x rl_length_classes
					# predicted_length_logits: batch_size x rl_length_classes
					predicted_length_logits = tf.matmul(encoder_states, self.action_length_ff_layer)
					# length_dist: batch_size x rl_length_classes
					length_dist = tf.nn.softmax(predicted_length_logits)
					# predicted_length_ids: batch_size x 1
					predicted_length_ids = tf.argmax(length_dist, axis=-1)
					# predicted_lengths: batch_size
					predicted_lengths = tf.gather(self._rl_decode_length_lookup_array,predicted_length_ids,axis=0)
					
					# predicted_lengths_to_return: batch_size x 1
					predicted_lengths_to_return = tf.reshape(predicted_lengths, [batch_size, 1])
				else:
					predicted_lengths = None
					predicted_lengths_to_return = tf.zeros([batch_size, 1], tf.int32)

				if self._beam:
					# TODO: implement FLD with beam search
					tiled_oov_ids = tf.contrib.seq2seq.tile_batch(self._rl_oov_ids, multiplier=self._beam_width)
					decoder = self._get_rl_beam_decoder(encoder_states, line_memory, word_memory, batch_size)
					outputs,_ = dynamic_decode(decoder, batch_size, self._rl_vocab_size, self._rl_oov_sizes, tiled_oov_ids, rl=True, maximum_iterations=self._max_api_length)
					return outputs.parent_ids, outputs.predicted_ids
				else:
					if self._fixed_length_decode:
						helper = FixedLengthGreedyEmbeddingHelper(self.C, tf.fill([batch_size], self.GO_SYMBOL), predicted_lengths)
					else:
						helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.C,tf.fill([batch_size], self.GO_SYMBOL), self.EOS)
					decoder = self._get_rl_decoder(encoder_states, line_memory, word_memory, helper, batch_size, predict_flag=True)
					outputs,_ = dynamic_decode(decoder, batch_size, self._rl_vocab_size, self._rl_oov_sizes, self._rl_oov_ids, rl=True, maximum_iterations=2*self._candidate_sentence_size)
					return outputs.sample_id, predicted_lengths_to_return
					# return tf.argmax(outputs.rnn_output, axis=-1), predicted_lengths_to_return

	def _decoder_train_rl(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			
			batch_size = tf.shape(self._stories)[0]
			# decoder_input = batch_size x candidate_sentence_size
			decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._rl_actions_emb_lookup[:, :]],axis=1)
			# decoder_emb_inp = batch_size x rl_max_action_size x embedding_size
			decoder_emb_inp = tf.nn.embedding_lookup(self.R, decoder_input)
			
			with tf.variable_scope('rl_decoder', reuse=True):
				
				answer_sizes = tf.reshape(self._rl_action_sizes,[-1])
				if self._fixed_length_decode:
					answer_sizes = tf.subtract(answer_sizes, tf.ones_like(answer_sizes))
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)
				decoder = self._get_rl_decoder(encoder_states, line_memory, word_memory, helper, batch_size,predict_flag=False)
				outputs,p_gens = dynamic_decode(decoder, batch_size, self._rl_vocab_size, self._rl_oov_sizes, self._rl_oov_ids, rl=True, impute_finished=False)
				final_dists = outputs.rnn_output
				max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
				ans = self._rl_actions[:, :max_length]

				max_oov_len = tf.reduce_max(self._rl_oov_sizes, reduction_indices=[0])
				extended_vsize =  self._rl_vocab_size + max_oov_len
				y_pred = tf.clip_by_value(final_dists,1e-20,1.0)
				y_true = tf.one_hot(ans, extended_vsize)
				seq_loss_comp = -tf.reduce_sum(y_true*tf.log(y_pred))
				
				if self._fixed_length_decode:
					# encoder_states: batch_size x embedding_size
					# self.action_length_ff_layer: embedding_size x rl_length_classes
					# predicted_length_logits: batch_size x rl_length_classes
					predicted_length_logits = tf.matmul(encoder_states, self.action_length_ff_layer)
					rl_decode_length_class_ids = tf.reshape(self._rl_decode_length_class_ids, [-1])
					length_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rl_decode_length_class_ids, logits=predicted_length_logits)
					reinforce_loss = tf.reduce_sum(seq_loss_comp * self._rl_rewards) + tf.reduce_sum(length_loss * self._rl_rewards)
				else:
					if self._rl_mode == "GT" or self._rl_mode == "GREEDY":
						reinforce_loss = tf.reduce_sum(seq_loss_comp)
					else:
						reinforce_loss = tf.reduce_sum(seq_loss_comp * self._rl_rewards)

				return reinforce_loss, final_dists, p_gens

	def _decoder_prob_rl(self, encoder_states, line_memory, word_memory=None):
		'''
			Arguments:
				encoder_states 	-	batch_size x embedding_size
				line_memory 	-	batch_size x memory_size x embedding_size
				word_memory 	- 	batch_size x memory_size x sentence_size x embedding_size
			Outputs:

		'''
		with tf.variable_scope(self._name):
			
			batch_size = tf.shape(self._stories)[0]
			# decoder_input = batch_size x candidate_sentence_size
			decoder_input = tf.concat([tf.fill([batch_size, 1], self.GO_SYMBOL), self._rl_actions_emb_lookup[:, :]],axis=1)
			# decoder_emb_inp = batch_size x rl_max_action_size x embedding_size
			decoder_emb_inp = tf.nn.embedding_lookup(self.R, decoder_input)

			with tf.variable_scope('rl_decoder', reuse=True):
				
				answer_sizes = tf.reshape(self._rl_action_sizes,[-1])

				# predict lengths based on the context
				if self._fixed_length_decode:
					# encoder_states: batch_size x embedding_size
					# self.action_length_ff_layer: embedding_size x rl_length_classes
					# predicted_length_logits: batch_size x rl_length_classes
					predicted_length_logits = tf.matmul(encoder_states, self.action_length_ff_layer)
					# length_dist: batch_size x rl_length_classes
					length_log_dist = math_ops.log(predicted_length_logits)
					
					# remove eos from the answers
					answer_sizes = tf.subtract(answer_sizes, tf.ones_like(answer_sizes))

				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_sizes)
				decoder = self._get_rl_decoder(encoder_states, line_memory, word_memory, helper, batch_size)
				outputs,p_gens = dynamic_decode(decoder, batch_size, self._rl_vocab_size, self._rl_oov_sizes, self._rl_oov_ids, rl=True, impute_finished=False)
				final_dists = outputs.rnn_output
				max_length = tf.reduce_max(answer_sizes, reduction_indices=[0])
				ans = self._rl_actions[:, :max_length]

				max_oov_len = tf.reduce_max(self._rl_oov_sizes, reduction_indices=[0])
				extended_vsize =  self._rl_vocab_size + max_oov_len
				y_pred = tf.clip_by_value(final_dists,1e-20,1.0)
				y_true = tf.one_hot(ans, extended_vsize)
				
				if self._fixed_length_decode:
					# self._rl_decode_length_class_ids: batch_size x 1
					# rl_decode_length_class_ids: batch_size
					rl_decode_length_class_ids = tf.reshape(self._rl_decode_length_class_ids, [-1])
					# l_true: batch_size x rl_length_classes
					l_true = tf.one_hot(rl_decode_length_class_ids, self._rl_decode_length_classes_count)
					# element wise multiplication
					# length_prob: batch_size
					length_prob = tf.reduce_sum(l_true * length_log_dist, axis=-1)
					probs = tf.reduce_sum(y_true*tf.log(y_pred), [1, 2]) + length_prob
				else:
					probs = tf.reduce_sum(y_true*tf.log(y_pred), [1, 2])
				
				return probs

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

	def _make_feed_dict(self, batch, actions_and_rewards=None, train=True):
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
		if self._debug:
			self._print_feed(feed_dict, actions_and_rewards, train)
		return feed_dict

	def fit(self, batch):
		"""
			Runs the training algorithm over the passed batch
		  Returns:
			loss: floating-point number, the loss computed for the batch
		"""
		feed_dict = self._make_feed_dict(batch)
		loss, _= self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
		return loss

	def predict(self, batch):
		"""
			Predicts answers as one-hot encoding.
		  Returns:
			answers: Tensor (None, vocab_size)
		"""
		feed_dict = self._make_feed_dict(batch, train=False)
		return self._sess.run(self.predict_op, feed_dict=feed_dict)

	def api_predict(self, batch):
		"""
			Runs the RL algorithm over the passed batch
		  Returns:
			actions: API calls to generate responses
		"""
		feed_dict = self._make_feed_dict(batch, train=False)
		return self._sess.run(self.rl_predict_op, feed_dict=feed_dict)

	def api_fit(self, batch, actions_and_rewards):
		"""
			Runs the REINFORCE algorithm over the passed batch
		"""
		feed_dict = self._make_feed_dict(batch, actions_and_rewards=actions_and_rewards)
		loss, _ = self._sess.run([self.rl_loss_op, self.rl_train_op], feed_dict=feed_dict)
		return loss

	def api_prob(self, batch, actions_and_rewards):
		"""
			Runs the REINFORCE algorithm over the passed batch
		"""
		feed_dict = self._make_feed_dict(batch, actions_and_rewards=actions_and_rewards)
		return self._sess.run(self.prob_op, feed_dict=feed_dict)