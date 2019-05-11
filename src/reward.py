import sys
import copy
import random
import db_engine
import numpy as np
from data import Batch

## Constants ##
INVALID = 0
NO_RES = 1
RECALL_WEIGHT = 1000

MAPO_ALPHA = 0.6

class ActionsAndRewards(object):

	Cache = {}

	def __init__(self):
		self._actions = []
		self._actions_emb_lookup = []
		self._action_sizes = []
		self._rewards = []
		self._rl_decode_length_class_ids = None
	
	@property
	def actions(self):
		return self._actions
	
	@property
	def actions_emb_lookup(self):
		return self._actions_emb_lookup
	
	@property
	def action_sizes(self):
		return self._action_sizes

	@property
	def rewards(self):
		return self._rewards
	
	@property
	def rl_decode_length_class_ids(self):
		return self._rl_decode_length_class_ids

	def add_entry(self, action, action_emb_lookup, action_sizes, reward):
		self._actions.append(action)
		self._actions_emb_lookup.append(action_emb_lookup)
		self._action_sizes.append(action_sizes)
		self._rewards.append(reward)

	def get(self, index):
		return self._actions[index], self._actions_emb_lookup[index], self._action_sizes[index], self._rewards[index]
	
	def get_reward(self, index):
		return self._rewards[index]

	def get_range(self, start, end, fixed_length_decode=False, glob=None):
		range_actions_and_rewards = ActionsAndRewards()
		for index in range(start, end):
			range_actions_and_rewards.add_entry(self._actions[index], self._actions_emb_lookup[index], self._action_sizes[index], self._rewards[index])
		if fixed_length_decode:
			range_actions_and_rewards.populate_rl_decode_length_class_id(glob['rl_decode_length_vs_index'])
		return range_actions_and_rewards

	def populate_rl_decode_length_class_id(self, rl_decode_length_vs_index):
		rl_decode_length_class_ids = []
		for action_size in self._action_sizes:
			# ignore last token (EOS) for FLD
			class_id = rl_decode_length_vs_index[action_size[0]-1]
			rl_decode_length_class_ids.append([class_id])
		self._rl_decode_length_class_ids = np.array(rl_decode_length_class_ids)

def get_reward_and_results(db_engine, query, is_valid_query, next_entities_in_dialog):

	if is_valid_query:
		select_fields, db_results, result_entities_set = db_engine.execute(query)
		if len(db_results)==0:
			reward = NO_RES
		else:
			match = 0
			for next_entity in next_entities_in_dialog:
				if next_entity in result_entities_set:
					match += 1
			if len(next_entities_in_dialog) == 0:
				recall = 0.0
			else:
				recall = float(match/len(next_entities_in_dialog))
			precision = float(match/len(result_entities_set))
			reward = recall + RECALL_WEIGHT*precision
	else:
		select_fields = []
		db_results = []
		reward = INVALID

	return reward, select_fields, db_results

def get_reward_for_query(query, cache_key_prefix, db_engine, next_entities_in_dialog):

	cache_key = cache_key_prefix+query
	if cache_key in ActionsAndRewards.Cache:
		reward = ActionsAndRewards.Cache[cache_key]
	else:
		reward, _, _ = get_reward_and_results(db_engine, query, True, next_entities_in_dialog)
		ActionsAndRewards.Cache[cache_key] = reward
	
	return reward
	
def get_action_from_query(query, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog):
	
	words = query.split(' ')
	action = []
	action_emb = []
	
	for word in words:
		if word not in rl_idx:
			if word in rl_oov_word_list:
				index = rl_oov_word_list.index(word)
				action.append(total_rl_words+index)
			else:
				action.append(rl_idx['UNK'])
			action_emb.append(rl_idx['UNK'])
		else:
			action.append(rl_idx[word])
			action_emb.append(rl_idx[word])

	#action.append(rl_idx['EOS'])
	#action_emb.append(rl_idx['EOS'])

	action_size = len(action)

	if action_size > max_api_length:
		action = action[:max_api_length]
	pad = max(0, max_api_length - action_size)
	action = action + [rl_idx['PAD']]*pad

	if len(action_emb) > max_api_length:
		action_emb = action_emb[:max_api_length]
	pad = max(0, max_api_length - len(action_emb))
	action_emb = action_emb + [rl_idx['PAD']]*pad

	reward = get_reward_for_query(query, cache_key_prefix, db_engine, next_entities_in_dialog)

	return action, action_emb, [action_size], [reward]

# input: SQL queries (high reward queries)
# output: ActionsAndRewards objects
def queries_to_actions_and_rewards(queries, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog):
	
	buffer_actions_and_rewards = ActionsAndRewards()
	
	for query in queries:
		action, action_emb, action_size, reward = get_action_from_query(query, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)
		buffer_actions_and_rewards.add_entry(np.array(action), np.array(action_emb), np.array([action_size[0]]), np.array([float(reward[0])]))

	return buffer_actions_and_rewards

def get_best_action_with_reward(queries, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog):
	
	best_index = 0
	best_reward = 0

	# best query with best reward
	for index, query in enumerate(queries):
		reward = get_reward_for_query(query, cache_key_prefix, db_engine, next_entities_in_dialog)
		if reward > best_reward:
			best_index = index
			best_reward = reward

	return get_action_from_query(queries[best_index], rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)
	
def get_gt_action_with_results(api_call, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog):
	
	# the logical form and api_call are the same
	query = api_call
	_, select_fields, db_results = get_reward_and_results(db_engine, query, True, next_entities_in_dialog)
	action, action_emb, action_size, reward = get_action_from_query(query, rl_idx, max_api_length, rl_oov_word_list, total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)

	return action, action_emb, action_size, reward, select_fields, db_results 

def is_perfect_match(db_engine, api_call, sql_query):
	# the logical form and api_call are the same
	sql_query_formatted = sql_query
	if api_call == sql_query_formatted:
		return 1
	else:
		return 0

def sample_k_from_prob_dist(prob_dist, k):
	samples = []
	for i in range(k):
		sample = sample_from_prob_dist(prob_dist, samples)
		samples.append(sample)
	return samples

def sample_from_prob_dist(prob_dist, already_sampled_list):
	rand_value = random.uniform(0.0, 1.0)
	running_sum=0

	temp_prob_dist = copy.deepcopy(prob_dist)
	for index in already_sampled_list:
		temp_prob_dist[index] = 0
	
	temp_prob_dist_normalized = [float(i)/sum(temp_prob_dist) for i in temp_prob_dist]

	for i,prob in enumerate(temp_prob_dist_normalized):
		if rand_value > running_sum and rand_value <= running_sum+prob:
			return i
		running_sum+=prob
	return (len(prob_dist)-1)

def process_action_beam(action_beam, action_length, args, glob, rl_oov_words, batch_index, total_rl_words, max_api_length, db_engine, next_entities_in_dialog):

	# process_action_beam(action_beams[batch_index][0], pred_action_lengths[batch_index][0], args, glob, rl_oov_words, batch_index, total_rl_words, max_api_length, db_engine, next_entities_in_dialog)
	high_probable_action = action_beam
	# unused
	if args.fixed_length_decode:
		high_probable_action_length = action_length

	action_emb_lookup = []
	action_surface_form = ""
	action_size = 0

	for word_id in high_probable_action:
		action_size+=1
		if word_id not in glob['idx_rl']:
			action_emb_lookup.append(glob['rl_idx']['UNK'])
			'''
			if (word_id - total_rl_words) < len(rl_oov_words[batch_index]):
				action_surface_form += " " + rl_oov_words[batch_index][word_id - total_rl_words]
			else:
				action_surface_form += " UNK"
			'''
			action_surface_form += " " + rl_oov_words[batch_index][word_id - total_rl_words]
			if args.fixed_length_decode and action_size == high_probable_action_length:
				# as action_size is reduced by 1 
				action_size+=1
				break
		else:
			word_form = glob['idx_rl'][word_id]
			action_emb_lookup.append(word_id)
			if args.fixed_length_decode:
				if action_size == high_probable_action_length:
					action_surface_form += " " + word_form
					# as action_size is reduced by 1 
					action_size+=1
					break
			else:
				if word_form == "EOS":
					break
			action_surface_form += " " + word_form
	
	action_surface_form = action_surface_form.strip()
	is_valid_query = db_engine.is_query_valid(action_surface_form)
	
	if len(high_probable_action) < max_api_length: pad = max_api_length - len(high_probable_action)
	else: pad = 0
	if len(action_emb_lookup) < max_api_length: pad_e = max_api_length - len(action_emb_lookup)
	else: pad_e = 0

	reward, select_fields, db_results = get_reward_and_results(db_engine, action_surface_form, is_valid_query, next_entities_in_dialog)
	formatted_results = db_engine.get_formatted_results(select_fields, db_results)
	
	return high_probable_action, action_surface_form, action_emb_lookup, action_size, reward, formatted_results, pad, pad_e, is_valid_query

# Possible modes: 
# GT : No decoder, the ground truth is returned
# GREEDY : the decoder is trained with the action with highest reward in the buffer
# MAPO : the decoder is trained using MAPOs
def calculate_reward(glob, action_beams, pred_action_lengths, batch, rlData, db_engine, model, args, data, output=False, mode="GT"):
	'''
		Input: An 3D array of actions [batch_size x beam_size x action]
			- For each action/API convert from idx to rl vocab (use glob['idx_rl'])
			- Get back responses for each API 
			- Calculate Reward for each response set
		Output:
			(III) DB-Results: The list of best results in word format [batch_size x results(#results*(#fields-1)) x response_string(3)]
							 - the results from the highest probable action from the beam decoder
			(II) batched_actions_and_rewards:
				- rewards: Calculated reward for each action [batch_size x beam_size]
				- actions: the input actions [batch_size x beam_size x action]
				## Use glob['rl_idx'] to convert high recall queries to indices
	'''

	dialog_ids = batch.dialog_ids
	turn_ids = batch.turn_ids
	
	rl_oov_words = batch.rl_oov_words
	rl_word_idx  = glob['idx_rl']
	total_rl_words = len(rl_word_idx)

	max_api_length = args.max_api_length
	beam_width = args.beam_width

	# convert to surface forms
	batched_db_results = []
	batched_actions_and_rewards = []

	total_high_probable_rewards = 0
	total_repeated_rewards = 0
	total_high_recall_actions_rewards = 0
	
	if mode == "MAPO":
		total_width = beam_width
	else:
		total_width = 1

	for i in range(total_width):
		batched_actions_and_rewards.append(ActionsAndRewards())

	total_entries = 0
	valid_entries = 0
	perfect_match_entries = 0

	if output: 
		file = open('logs/api.txt', 'a')

	batch_size = len(batch.stories)

	for batch_index in range(batch_size):
		turn_id = turn_ids[batch_index]
		dialog_id = dialog_ids[batch_index]
		cache_key_prefix = str(dialog_id) + "_" + str(turn_id) + "_"
		next_entities_in_dialog = rlData[dialog_id][turn_id]['next_entities']

		high_recall_queries = rlData[dialog_ids[batch_index]][turn_ids[batch_index]]['high_recall_queries']
		
		mapo_pi_b = 0
		queries_added = 0

		# there is atleast one query in the buffer
		if len(high_recall_queries) > 0:

			if mode == "MAPO":
				
				# compute prob for all high recall queries in the Buffer
				high_recall_actions_and_rewards = queries_to_actions_and_rewards(high_recall_queries, glob['rl_idx'], max_api_length, rl_oov_words[batch_index], total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)
				
				buffer_log_probs = []
				loop_count = len(high_recall_queries)//batch_size
				last_batch_count = len(high_recall_queries)%batch_size
				
				if loop_count > 0:
					api_prob_batch = Batch(data, None, args, glob, repeatCopyMode=True, batchToCopy=batch, indexToCopy=batch_index, noOfCopies=batch_size)
					ranges = [i*batch_size for i in range(loop_count)]
					for start_idx in ranges:
						probs = model.api_prob(api_prob_batch, high_recall_actions_and_rewards.get_range(start_idx, start_idx + batch_size, fixed_length_decode=args.fixed_length_decode, glob=glob))
						buffer_log_probs += list(probs)
				
				if last_batch_count > 0:
					api_prob_batch = Batch(data, None, args, glob, repeatCopyMode=True, batchToCopy=batch, indexToCopy=batch_index, noOfCopies=last_batch_count)
					probs = model.api_prob(api_prob_batch, high_recall_actions_and_rewards.get_range(loop_count*batch_size, len(high_recall_queries), fixed_length_decode=args.fixed_length_decode, glob=glob))
					buffer_log_probs += list(probs)
				
				# prob of all queries in the buffer
				buffer_probs = np.exp(buffer_log_probs)
				unclipped_mapo_pi_b = np.sum(buffer_probs)
				
				
				reward_weight = 1
				if unclipped_mapo_pi_b < MAPO_ALPHA:
					'''
					if unclipped_mapo_pi_b == 0:
						equal_prob = MAPO_ALPHA/len(buffer_log_probs)
						for i in range(len(buffer_probs)):
							buffer_probs[i] = equal_prob
						unclipped_mapo_pi_b = MAPO_ALPHA
					else:
						reward_weight = MAPO_ALPHA/unclipped_mapo_pi_b
					'''
					reward_weight = MAPO_ALPHA/unclipped_mapo_pi_b

				mapo_pi_b = max(unclipped_mapo_pi_b, MAPO_ALPHA)
				
				# print top 10 high recall queries
				if batch_index == 0:
					#print("buffer_log_probs", buffer_log_probs)
					#print("buffer_probs", buffer_probs)
					#print("unclipped_mapo_pi_b", unclipped_mapo_pi_b)

					#normalized_buffer_prob = buffer_probs/unclipped_mapo_pi_b
					normalized_buffer_prob = buffer_probs
					sortedRes = sorted(zip(normalized_buffer_prob, high_recall_queries), key=lambda x: x[0], reverse=True)
					print("-------------------------")
					print("reward_weight", reward_weight)
					for i, res in enumerate(sortedRes):
						action, action_emb_lookup, action_size, reward = high_recall_actions_and_rewards.get(i)
						if i == 10:
							break
						print(res, reward[0])
					print("-------------------------")

				# push all the high recall queries into batched_actions_and_rewards

				for i in range(len(high_recall_queries)):
					action, action_emb_lookup, action_size, reward = high_recall_actions_and_rewards.get(i)
					total_high_recall_actions_rewards+= reward[0]
					
					reward[0] *= reward_weight*buffer_probs[i]
					
					batched_actions_and_rewards[i].add_entry(
						copy.deepcopy(action), copy.deepcopy(action_emb_lookup), copy.deepcopy(action_size), copy.deepcopy(reward))
					queries_added+=1
			elif mode == "GREEDY":
				action, action_emb_lookup, action_size, reward = get_best_action_with_reward(high_recall_queries, glob['rl_idx'], max_api_length, rl_oov_words[batch_index], total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)
				total_high_probable_rewards+= reward[0]
				batched_actions_and_rewards[0].add_entry(
					copy.deepcopy(np.array(action)), copy.deepcopy(np.array(action_emb_lookup)), copy.deepcopy(np.array(action_size)), copy.deepcopy(np.array(reward)))
				total_entries += 1
			elif mode == "GT":
				action, action_emb_lookup, action_size, reward, select_fields, db_results = get_gt_action_with_results(rlData[dialog_id][turn_id]['api_call'], glob['rl_idx'], max_api_length, rl_oov_words[batch_index], total_rl_words, cache_key_prefix, db_engine, next_entities_in_dialog)
				batched_db_results.append(db_engine.get_formatted_results(select_fields, db_results))
				total_high_probable_rewards+= reward[0]
				total_entries += 1
		
		# MAPO: rejection sampling outside the buffer
		#	number of samples to be added = (beam_width - queries_added)
		# Greedy, if no high recall action, just sample from policy
		# GT, leave it blank
		
		if mode == "GT":
			valid_entries += 1
			perfect_match_entries += 1
			if len(high_recall_queries) == 0:
				batched_db_results.append([])
		elif mode == "GREEDY":
			# if there are no queries in buffer, then use the high probable one for greedy mode
			pred_action_length = 0
			if args.fixed_length_decode:
				pred_action_length = pred_action_lengths[batch_index][0]
			high_probable_action, action_surface_form, action_emb_lookup, action_size, reward, formatted_results, pad, pad_e, is_valid_query = process_action_beam(action_beams[batch_index][0], pred_action_length, args, glob, rl_oov_words, batch_index, total_rl_words, max_api_length, db_engine, next_entities_in_dialog)
			if len(high_recall_queries) == 0:
				batched_actions_and_rewards[0].add_entry(
					(high_probable_action[:max_api_length] + [0]*pad), np.array(action_emb_lookup[:max_api_length] + [0]*pad_e), np.array([action_size]), np.array([reward]))
				total_repeated_rewards += reward
		else:
			# MAPO

			# number of samples outside the buffer
			K = total_width - queries_added
			one_minus_mapo_pi_b_by_K = (1-mapo_pi_b)/K

			for beam_index in range(beam_width):

				if queries_added == total_width:
					break
				
				pred_action_length = 0
				if args.fixed_length_decode:
					pred_action_length = pred_action_lengths[batch_index][beam_index]

				high_probable_action, action_surface_form, action_emb_lookup, action_size, reward, formatted_results, pad, pad_e, is_valid_query = process_action_beam(action_beams[batch_index][beam_index], pred_action_length, args, glob, rl_oov_words, batch_index, total_rl_words, max_api_length, db_engine, next_entities_in_dialog)
				
				if beam_index == 0:
					batched_db_results.append(formatted_results)
					
				if action_surface_form in high_recall_queries:
					continue
				
				total_entries += 1
				if beam_index == 0 and is_valid_query:
					if len(formatted_results) > 0:
						valid_entries += 1
					perfect_match_entries += is_perfect_match(db_engine, rlData[dialog_id][turn_id]['api_call'], action_surface_form)
				
				if batch_index == 0:
					if beam_index == 0:
						print("---------------------------------")
						print("GT:", rlData[dialog_id][turn_id]['api_call'])
						print("PREDICT:",action_surface_form)
						print("---------------------------------")
					print(action_surface_form, float(reward))
					
				# add on policy samples for mapo: outside the buffer
				batched_actions_and_rewards[queries_added].add_entry(
						np.array(high_probable_action[:max_api_length] + [0]*pad), np.array(action_emb_lookup[:max_api_length] + [0]*pad_e), np.array([action_size]), np.array([float(reward)*one_minus_mapo_pi_b_by_K]))
				total_high_probable_rewards += float(reward)*one_minus_mapo_pi_b_by_K
				#batched_actions_and_rewards[queries_added].add_entry(
				#		np.array(high_probable_action[:max_api_length] + [0]*pad), np.array(action_emb_lookup[:max_api_length] + [0]*pad_e), np.array([action_size]), np.array([float(reward)]))
				#total_high_probable_rewards += float(reward)

				queries_added += 1

	# batched_actions_and_rewards is not used in GT mode
	'''
	if mode != "GT" and mode != "GREEDY":
		# average all rewards to compute the baseline
		# subtract baseline from each reward
		total_rewards = total_high_probable_rewards + total_high_recall_actions_rewards + total_repeated_rewards
		print("total_rewards", total_rewards)
		if total_rewards > 0:
			# average_rewards
			baseline = (total_rewards)/float(total_width*batch_size)
			print("baseline", (total_rewards)/float(total_width*batch_size))
			for i in range(total_width):
					for j in range(len(batched_actions_and_rewards[i].rewards)):
						if i==0:
							print(batched_actions_and_rewards[i].rewards[j], batched_actions_and_rewards[i].rewards[j] - baseline)
						batched_actions_and_rewards[i].rewards[j] -= baseline
	'''		
						
	if args.fixed_length_decode:
		for actions_and_rewards in batched_actions_and_rewards:
			actions_and_rewards.populate_rl_decode_length_class_id(glob['rl_decode_length_vs_index'])
	
	return batched_db_results, batched_actions_and_rewards, total_high_probable_rewards, total_entries, valid_entries, perfect_match_entries