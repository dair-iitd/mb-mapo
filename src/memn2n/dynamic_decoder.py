# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Seq2seq layer operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import collections
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.layers import base as layers_base
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops.decoder import Decoder


__all__ = ["BasicDecoder", "dynamic_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access

class BasicDecoderOutput(
		collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
	pass


class BasicDecoder(Decoder):
	"""Basic sampling decoder."""

	def __init__(self, cell, pos_embedding,  helper, initial_state, initial_ids, state_ids, state_mask=None, constraint_mask=None, output_layer=None):
		"""Initialize BasicDecoder.
		Args:
			cell: An `RNNCell` instance.
			helper: A `Helper` instance.
			initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
				The initial state of the RNNCell.
			output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
				`tf.layers.Dense`. Optional layer to apply to the RNN output prior
				to storing the result or sampling.
		Raises:
			TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
		"""
		if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
			raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
		# if not isinstance(helper, helper_py.Helper):
		# 	raise TypeError("helper must be a Helper, received: %s" % type(helper))
		if (output_layer is not None
				and not isinstance(output_layer, layers_base.Layer)):
			raise TypeError(
					"output_layer must be a Layer, received: %s" % type(output_layer))
		self._cell = cell
		self._helper = helper
		self._initial_state = initial_state
		self._output_layer = output_layer
		
		if state_mask is not None:
			self._state_mask = tf.convert_to_tensor(state_mask, dtype=tf.float32)
		else:
			self._state_mask = None
		if constraint_mask is not None:
			self._constraint_mask = tf.convert_to_tensor(constraint_mask, dtype=tf.float32)
		else:
			self._constraint_mask = None
		self._initial_ids = initial_ids
		self._state_ids = state_ids

		self._pos_embedding = pos_embedding
		
		'''
		if callable(pos_embedding):
			self._pos_embedding_fn = pos_embedding
		else:
			self._pos_embedding_fn = (lambda ids: embedding_ops.embedding_lookup(pos_embedding, ids))
		'''

	@property
	def batch_size(self):
		return self._helper.batch_size

	def _rnn_output_size(self):
		size = self._cell.output_size
		if self._output_layer is None:
			return size
		else:
			# To use layer's compute_output_shape, we need to convert the
			# RNNCell's output_size entries into shapes with an unknown
			# batch size.  We then pass this through the layer's
			# compute_output_shape and read off all but the first (batch)
			# dimensions to get the output size of the rnn with the layer
			# applied to the top.
			output_shape_with_unknown_batch = nest.map_structure(
					lambda s: tensor_shape.TensorShape([None]).concatenate(s), size)
			layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
					output_shape_with_unknown_batch)
			return nest.map_structure(lambda s: s[1:], layer_output_shape)

	@property
	def output_size(self):
		# Return the cell output and the id
		return BasicDecoderOutput(
				rnn_output=nest.map_structure(lambda s: s[0], self._rnn_output_size()),
				sample_id=tensor_shape.TensorShape([]))

	@property
	def output_dtype(self):
		# Assume the dtype of the cell is the output_size structure
		# containing the input_state's first component's dtype.
		# Return that structure and the sample_ids_dtype from the helper.
		dtype = nest.flatten(self._initial_state)[0].dtype
		return BasicDecoderOutput(
				nest.map_structure(lambda _: dtype, self._rnn_output_size()),
				dtypes.int32)

	@property
	def type(self):
		return "Basic"

	def _calc_final_dist(self, next_outputs, attn_dists, p_gens, oov_ids_n, oov_sizes_n, decoder_vocab_size_n, batch_size, prev_ids, state_ids):
			"""Calculate the final distribution, for the pointer-generator model

			Args:
				vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
				attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

			Returns:
				final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
			"""
			with tf.variable_scope('final_distribution'):
				vocab_dists = tf.nn.softmax(next_outputs.rnn_output)

				next_state_ids = state_ids

				# oov_ids_n = tf.Print(oov_ids_n, [oov_ids_n], 'printing oov_ids', summarize=1000)
				# oov_sizes_n = tf.Print(oov_sizes_n, [oov_sizes_n], 'printing oov_sizes', summarize=1000)
				# decoder_vocab_size_n = tf.Print(decoder_vocab_size_n, [decoder_vocab_size_n], 'printing oov_sizes', summarize=1000)
				if self._constraint_mask is not None:
					# prev_ids = prev_outputs.sample_ids
					# prev_ids = tf.Print(prev_ids, [prev_ids], 'printing sample_ids', summarize=32)
					# indices = tf.concat([state_ids, prev_ids], axis=0)
					indices = tf.stack([state_ids, prev_ids], axis=-1)
					indices = tf.reshape(indices, [batch_size, 2])
					# indices = tf.Print(indices, [indices], 'printing indices', summarize=64)
					# one_hots = tf.one_hot(prev_ids, decoder_vocab_size_n)
					# one_hots = tf.reshape(one_hots, [batch_size, decoder_vocab_size_n])
					# self._constraint_mask = tf.reshape(self._constraint_mask, [decoder_vocab_size_n, decoder_vocab_size_n])
					indices = tf.cast(indices, tf.int32)
					next_state_ids = tf.gather_nd(self._state_mask, indices)
					next_state_ids = tf.cast(next_state_ids, tf.int32)
					# next_state_ids = tf.Print(next_state_ids, [next_state_ids], 'printing next_state_ids', summarize=1000)
					one_hots = tf.one_hot(next_state_ids, tf.shape(self._state_mask)[0])
					mask = tf.matmul(one_hots, self._constraint_mask)
					#attn_dists = tf.Print(attn_dists, [attn_dists[:1]], 'printing Attn', summarize=832)
					# mask = tf.reshape(mask, [batch_size, decoder_vocab_size_n])
					# vocab_dists = tf.reshape(vocab_dists, [batch_size, decoder_vocab_size_n])
					vocab_dists = tf.multiply(vocab_dists, mask)
					vocab_dists = tf.reshape(vocab_dists, [batch_size, decoder_vocab_size_n])
					p_gens = tf.map_fn(one_minus_fn, p_gens)

				# raise ValueError("Expected memory to have fully defined inner dims, "
				#                  "but saw shape: %s" % (vocab_dists.get_shape()))
				else:
					vocab_dists = tf.multiply(vocab_dists, p_gens)
					one_minus_fn = lambda x: 1 - x
					p_gens = tf.map_fn(one_minus_fn, p_gens)
					attn_dists = tf.multiply(p_gens, attn_dists)

				max_oov_len = tf.reduce_max(oov_sizes_n, reduction_indices=[0])
				# max_oov_len = tf.Print(max_oov_len, [max_oov_len], 'printing max_oov_len', summarize=10000)


				# Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
				extended_vsize =  decoder_vocab_size_n + max_oov_len # the maximum (over the batch) size of the extended vocabulary
				extra_zeros = tf.zeros((batch_size, max_oov_len))

				# raise ValueError("Expected memory to have fully defined inner dims, "
				#                  "but saw shape: %s" % (extra_zeros.get_shape()))
				# vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)
				vocab_dists_extended = array_ops.concat([vocab_dists, extra_zeros], axis=1)
				# Project the values in the attention distributions onto the appropriate entries in the final distributions
				# This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
				# This is done for each decoder timestep.
				# This is fiddly; we use tf.scatter_nd to do the projection
				batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
				batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
				attention_ids = tf.reshape(oov_ids_n, [batch_size, -1])
				attn_len = tf.shape(attention_ids)[1] # number of states we attend over

				batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
				indices = tf.stack((batch_nums, attention_ids), axis=2) # shape (batch_size, enc_t, 2)
				
				shape = [batch_size, extended_vsize]
				# attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)
				# attn_dists_projected = array_ops.scatter_nd(indices, attn_dists, shape)
				attn_dists_projected = tf.scatter_nd(indices, attn_dists, shape)
				# Add the vocab distributions and the copy distributions together to get the final distributions
				# final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
				# Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
				# final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
				
				if self._constraint_mask is not None:
					ind = tf.constant([1])
					mask = tf.reshape(mask, [batch_size, decoder_vocab_size_n])
					result = tf.nn.embedding_lookup(tf.transpose(mask), ind)
					unk_mask = tf.transpose(tf.tile(result, [extended_vsize, 1]))
					unk_mask = tf.reshape(unk_mask, [batch_size, extended_vsize])
					# unk_mask = tf.Print(unk_mask, [unk_mask], 'printing unk_mask', summarize=1000)
					attn_dists_projected = tf.multiply(attn_dists_projected, unk_mask)

					first_part = vocab_dists_extended[:,:1]
					attn_dists_projected =  tf.concat([tf.zeros_like(first_part), attn_dists_projected[:,1:]], 1)
					
					# remove UNKs from the final_dists
					unk_removed_vocab_dists_extended = tf.concat([first_part, tf.zeros_like(first_part), vocab_dists_extended[:,2:]], 1)
					#unk_removed_vocab_dists_extended = tf.Print(unk_removed_vocab_dists_extended, [unk_removed_vocab_dists_extended[:1]], 'printing vocab_dist', summarize=1000)
					#vocab_dists_extended =  tf.Print(vocab_dists_extended, [vocab_dists_extended[:1]], 'printing vocab_dists_extended', summarize=1000)
					#attn_dists_projected =  tf.Print(attn_dists_projected, [attn_dists_projected[:1]], 'printing attn_dists_projected', summarize=1000)
					
					final_dists = math_ops.add(unk_removed_vocab_dists_extended, attn_dists_projected)
					#final_dists =  tf.Print(final_dists, [final_dists[:1]], 'printing final_dists', summarize=1000)
					
				else:
					final_dists = math_ops.add(vocab_dists_extended, attn_dists_projected)
				full_sample_ids = tf.argmax(final_dists, axis=-1, output_type=tf.int32)
				vocab_sample_ids = tf.argmax(vocab_dists_extended, axis=-1, output_type=tf.int32)
				# final_dists = tf.Print(final_dists, [final_dists], 'printing final_dists', summarize=max_oov_len)
				# full_sample_ids = tf.Print(full_sample_ids, [full_sample_ids], 'printing full_sample_ids', summarize=32)
				return BasicDecoderOutput(final_dists, full_sample_ids), vocab_sample_ids, next_state_ids

	def initialize(self, name=None):
		"""Initialize the decoder.
		Args:
			name: Name scope for any created operations.
		Returns:
			`(finished, first_inputs, initial_state)`.
		"""
		return self._helper.initialize() + (self._initial_state,) + (self._initial_ids,) + (self._state_ids,)

	def step(self, time, prev_ids, state_ids, inputs, state, oov_ids, oov_sizes, decoder_vocab_size, batch_size, rl, name=None):
		"""Perform a decoding step.
		Args:
			time: scalar `int32` tensor.
			inputs: A (structure of) input tensors.
			state: A (structure of) state tensors and TensorArrays.
			name: Name scope for any created operations.
		Returns:
			`(outputs, next_state, next_inputs, finished)`.
		"""
		with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
			# inputs = tf.Print(inputs, [inputs], 'printing inputs', summarize=1000)
			new_batch_size = ops.convert_to_tensor([batch_size], name="new_batch_size") 

			# if rl:
			# 	pos = ops.convert_to_tensor([time], name="pos") 
			# 	positions = tf.tile(pos, new_batch_size)
			# 	# positions = tf.Print(positions, [positions], '\nprinting positions', summarize=400)
			# 	position_emb = self._pos_embedding_fn(positions)
			# 	# position_emb = tf.Print(position_emb, [position_emb], '\nprinting position_emb', summarize=400)
			# 	inputs = (inputs, position_emb)
			# else:
			# 	inputs = (inputs, tf.constant(False))

			# if rl:
			# 	pos = ops.convert_to_tensor([time], name="pos") 
			# 	positions = tf.tile(pos, new_batch_size)
			# 	position_emb = self._pos_embedding_fn(positions)
			# 	cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))
			# 	inputs = cell_input_fn(inputs, position_emb)
		
			
			#Uncomment this for PE
			if rl:
				pos = ops.convert_to_tensor([time], name="pos") 
				positions = tf.tile(pos, new_batch_size)
				position_emb = tf.nn.embedding_lookup(self._pos_embedding, positions)
				#print_position_emb = tf.Print(position_emb,[tf.shape(state[0]), tf.shape(position_emb)], message="rl-dyn")
				#modified_state = state.clone(cell_state=tf.add(state[0], position_emb))
				modified_state = state.clone()
				#inputs = tf.add(inputs, print_position_emb)
			else:
				modified_state = state.clone()
			
			cell_outputs, cell_state = self._cell(inputs, modified_state)
			(cell_outputs, attention, p_gens) = cell_outputs
			if self._output_layer is not None:
				cell_outputs = self._output_layer(cell_outputs)
			sample_ids = self._helper.sample(
					time=time, outputs=cell_outputs, state=cell_state)
			(finished, next_inputs, next_state) = self._helper.next_inputs(
					time=time,
					outputs=cell_outputs,
					state=cell_state,
					sample_ids=sample_ids)
		outputs = BasicDecoderOutput(cell_outputs, sample_ids)
		outputs, next_ids, next_state_ids = self._calc_final_dist(outputs, attention, p_gens, oov_ids, oov_sizes, decoder_vocab_size, batch_size, prev_ids, state_ids)
		return (outputs, p_gens, next_state, next_inputs, finished, next_ids, next_state_ids)

def _create_zero_outputs(size, dtype, batch_size):
	"""Create a zero outputs Tensor structure."""
	def _t(s):
		return (s if isinstance(s, ops.Tensor) else constant_op.constant(
				tensor_shape.TensorShape(s).as_list(),
				dtype=dtypes.int32,
				name="zero_suffix_shape"))

	def _create(s, d):
		return array_ops.zeros(
				array_ops.concat(
						([batch_size], _t(s)), axis=0), dtype=d)

	return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
				   batch_size,
				   decoder_vocab_size,
				   oov_sizes,
				   oov_ids,
				   rl=False,
				   output_time_major=False,
				   impute_finished=False,
				   maximum_iterations=None,
				   parallel_iterations=32,
				   swap_memory=False,
				   scope=None):
	"""Perform dynamic decoding with `decoder`.
	Calls initialize() once and step() repeatedly on the Decoder object.
	Args:
		decoder: A `Decoder` instance.
		output_time_major: Python boolean.  Default: `False` (batch major).  If
			`True`, outputs are returned as time major tensors (this mode is faster).
			Otherwise, outputs are returned as batch major tensors (this adds extra
			time to the computation).
		impute_finished: Python boolean.  If `True`, then states for batch
			entries which are marked as finished get copied through and the
			corresponding outputs get zeroed out.  This causes some slowdown at
			each time step, but ensures that the final state and outputs have
			the correct values and that backprop ignores time steps that were
			marked as finished.
		maximum_iterations: `int32` scalar, maximum allowed number of decoding
			 steps.  Default is `None` (decode until the decoder is fully done).
		parallel_iterations: Argument passed to `tf.while_loop`.
		swap_memory: Argument passed to `tf.while_loop`.
		scope: Optional variable scope to use.
	Returns:
		`(final_outputs, final_state, final_sequence_lengths)`.
	Raises:
		TypeError: if `decoder` is not an instance of `Decoder`.
		ValueError: if `maximum_iterations` is provided but is not a scalar.
	"""
	if not isinstance(decoder, Decoder):
		raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
										type(decoder))

	with variable_scope.variable_scope(scope, "decoder") as varscope:
		# Properly cache variable values inside the while_loop
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		if maximum_iterations is not None:
			maximum_iterations = ops.convert_to_tensor(
					maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
			if maximum_iterations.get_shape().ndims != 0:
				raise ValueError("maximum_iterations must be a scalar")

		initial_finished, initial_inputs, initial_state, initial_ids, initial_state_ids = decoder.initialize()

		zero_outputs = _create_zero_outputs(decoder.output_size,
																				decoder.output_dtype,
																				decoder.batch_size)

		if maximum_iterations is not None:
			initial_finished = math_ops.logical_or(
					initial_finished, 0 >= maximum_iterations)
		initial_sequence_lengths = array_ops.zeros_like(
				initial_finished, dtype=dtypes.int32)
		initial_time = constant_op.constant(0, dtype=dtypes.int32)

		def _shape(batch_size, from_shape):
			if not isinstance(from_shape, tensor_shape.TensorShape):
				return tensor_shape.TensorShape(None)
			else:
				batch_size = tensor_util.constant_value(
						ops.convert_to_tensor(
								batch_size, name="batch_size"))
				return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

		def _create_ta(s, d):
			return tensor_array_ops.TensorArray(
					dtype=d,
					size=0,
					dynamic_size=True,
					clear_after_read=False,
					element_shape=_shape(decoder.batch_size, s))

		initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
																						decoder.output_dtype)
		initial_attention = nest.map_structure(_create_ta, tensor_shape.TensorShape(None),
																							dtypes.float32)
		initial_line_alignment = nest.map_structure(_create_ta, tensor_shape.TensorShape(None),
																							dtypes.float32)
		initial_word_alignment = nest.map_structure(_create_ta, tensor_shape.TensorShape(None),
																							dtypes.float32)
		initial_p_gens = nest.map_structure(_create_ta, tensor_shape.TensorShape(1),
																							dtypes.float32)

		def condition(unused_time, unused_ids, unused_state_ids, unused_outputs_ta, unused_state, unused_inputs,
									finished, unused_sequence_lengths, p_gens):
			return math_ops.logical_not(math_ops.reduce_all(finished))

		def body(time, prev_ids, state_ids, outputs_ta, state, inputs, finished, sequence_lengths, p_gens):
			"""Internal while_loop body.
			Args:
				time: scalar int32 tensor.
				outputs_ta: structure of TensorArray.
				state: (structure of) state tensors and TensorArrays.
				inputs: (structure of) input tensors.
				finished: bool tensor (keeping track of what's finished).
				sequence_lengths: int32 tensor (keeping track of time of finish).
			Returns:
				`(time + 1, outputs_ta, next_state, next_inputs, next_finished,
					next_sequence_lengths)`.
				```
			"""
			# oov_ids = tf.Print(oov_ids, [oov_ids], 'printing oov_ids', summarize=1000)
			# oov_sizes = tf.Print(oov_sizes, [oov_sizes], 'printing oov_sizes', summarize=1000)
			# inputs = tf.Print(inputs, [inputs], 'printing inputs', summarize=1000)

			(next_outputs, next_p_gens, decoder_state, next_inputs, decoder_finished, next_ids, next_state_ids) = \
					decoder.step(time, prev_ids, state_ids, inputs, state, oov_ids, oov_sizes, decoder_vocab_size, batch_size, rl)
			
			# next_p_gens = tf.Print(next_p_gens, [next_p_gens], 'printing next_p_gens', summarize=1000)
			# next_inputs = tf.Print(next_inputs, [next_inputs], 'printing next_inputs', summarize=1000)
			next_ids = tf.Print(next_ids, [next_ids], 'printing next_ids', summarize=1000)

			next_finished = math_ops.logical_or(decoder_finished, finished)
			if maximum_iterations is not None:
				next_finished = math_ops.logical_or(
						next_finished, time + 1 >= maximum_iterations)
			next_sequence_lengths = array_ops.where(
					math_ops.logical_and(math_ops.logical_not(finished), next_finished),
					array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
					sequence_lengths)

			nest.assert_same_structure(state, decoder_state)
			nest.assert_same_structure(prev_ids, next_ids)
			nest.assert_same_structure(state_ids, next_state_ids)
			nest.assert_same_structure(outputs_ta, next_outputs)
			nest.assert_same_structure(inputs, next_inputs)
			nest.assert_same_structure(p_gens, next_p_gens)

			# Zero out output values past finish
			if impute_finished:
				emit = nest.map_structure(
						lambda out, zero: array_ops.where(finished, zero, out),
						next_outputs,
						zero_outputs)
			else:
				emit = next_outputs

			# Copy through states past finish
			def _maybe_copy_state(new, cur):
				# TensorArrays and scalar states get passed through.
				if isinstance(cur, tensor_array_ops.TensorArray):
					pass_through = True
				else:
					new.set_shape(cur.shape)
					pass_through = (new.shape.ndims == 0)
				return new if pass_through else array_ops.where(finished, cur, new)

			if impute_finished:
				next_state = nest.map_structure(
						_maybe_copy_state, decoder_state, state)
			else:
				next_state = decoder_state

			outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
																			outputs_ta, emit)
			p_gens = nest.map_structure(lambda ta, out: ta.write(time, out),
																			 p_gens, next_p_gens)

			return (time + 1, next_ids, next_state_ids, outputs_ta, next_state, next_inputs, next_finished,
							next_sequence_lengths, p_gens)

		res = control_flow_ops.while_loop(
				condition,
				body,
				loop_vars=[
						initial_time, initial_ids, initial_state_ids, initial_outputs_ta, initial_state, initial_inputs,
						initial_finished, initial_sequence_lengths, initial_p_gens
				],
				parallel_iterations=parallel_iterations,
				swap_memory=swap_memory)

		final_outputs_ta = res[3]
		final_p_gens = res[8]

		final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
		final_p_gens = nest.map_structure(lambda ta: ta.stack(), final_p_gens)

		if not output_time_major:
			final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)
			final_p_gens = nest.map_structure(_transpose_batch_time, final_p_gens)

	return final_outputs, final_p_gens

