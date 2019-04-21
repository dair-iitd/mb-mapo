from __future__ import absolute_import
from __future__ import print_function

import json
import logging
import numpy as np
import os
import pdb
import sys
import tensorflow as tf
from collections import deque
from data import Data, Batch
from data_utils import *
from db_engine import DbEngine, QueryGenerator
from evaluation import evaluate
from itertools import chain
from memn2n.memn2n_dialog_generator import MemN2NGeneratorDialog
from operator import itemgetter
from params import get_params, print_params
from reward import calculate_reward
from six.moves import range, reduce
from sklearn import metrics
from tqdm import tqdm

args = get_params()
glob = {}

class chatBot(object):

	def __init__(self):
		# Create Model Store Directory
		self.model_dir = (args.model_dir + "task" + str(args.task_id) + "_" + args.data_dir.split('/')[-2] + "_lr-" + str(args.learning_rate) 
										+ "_hops-" + str(args.hops) + "_emb-size-" + str(args.embedding_size) + "_sw-" + str(args.soft_weight) 
										+ "_wd-" + str(args.word_drop_prob) + "_pw-" + str(args.p_gen_loss_weight) 
										+ "_rlmode-" + str(args.rl_mode) + "_model/")
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
			
		''' Three Vocabularies
		1) Decoder Vocab [decode_idx, idx_decode] 	# Used to Encode Response by Response-Decoder
		2) Context Vocab [word_idx, idx_word] 		# Used to Encode Context Input by Encoder
		3) RL Vocab [rl_idx, idx_rl] 				# Used to Encode API Call by RL-Decoder
		'''

		# 1) Load Response-Decoder Vocabulary
		glob['decode_idx'], glob['idx_decode'], glob['candidate_sentence_size'] = get_decoder_vocab(args.data_dir, args.task_id)
		print("Decoder Vocab Size : {}".format(len(glob['decode_idx'])))
		print("candidate_sentence_size : {}".format(glob['candidate_sentence_size'])); sys.stdout.flush()
		# Retreive Task Data
		self.trainData, self.testData, self.valData, self.testOOVData = load_dialog_task(args.data_dir, args.task_id, args.rl, args.sort)


		# 2) Build RL Vocab for RL-Decoder
		if args.rl:
			self.db_engine = DbEngine(args.kb_file, "R_name")
			self.RLtrainData, self.RLtestData, self.RLvalData, self.RLtestOOVData = load_RL_data(args.data_dir, args.task_id)
			if args.fixed_length_decode:
				glob['rl_decode_length_vs_index'], glob['rl_decode_length_lookup_array'] = get_rl_decode_length_vs_index(self.RLtrainData, self.RLvalData)
			glob['rl_idx'], glob['idx_rl'], glob['fields'], glob['rl_vocab_size'], glob['constraint_mask'], glob['state_mask'] = get_rl_vocab(self.db_engine)
			print("RL Vocab Size : {}".format(glob['rl_vocab_size'])); sys.stdout.flush()
		else:
			self.RLtrainData = self.RLtestData = self.RLvalData = self.RLtestOOVData = None

		# 3) Build the Context Vocabulary
		self.build_vocab(self.trainData)

		# Define MemN2N + Generator Model
		glob['optimizer'] = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon)
		glob['session'] = tf.Session()
		self.model = MemN2NGeneratorDialog(args, glob)
		self.saver = tf.train.Saver(max_to_keep=4)

	def build_vocab(self, data):
		'''
			Get vocabulary from the Train data
		'''
		vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a, _, _, _ in data))
		vocab = sorted(vocab)
		glob['word_idx'] = dict((c, i + 2) for i, c in enumerate(vocab))
		glob['word_idx']['']=0
		glob['word_idx']['UNK']=1
		if args.rl:
			glob['word_idx']['$db'] = len(glob['word_idx'])
			for field in glob['fields']:
				glob['word_idx'][field] = len(glob['word_idx'])
		glob['vocab_size'] = len(glob['word_idx']) + 1  # +1 for nil word
		glob['idx_word'] = {v: k for k, v in glob['word_idx'].items()}
		print("Context Vocab Size : {}".format(glob['vocab_size'])); sys.stdout.flush()

		sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _, _ in data)))
		query_size = max(map(len, (q for _, q, _, _, _, _ in data)))
		glob['sentence_size'] = max(query_size, sentence_size)

	def train(self):
		'''
			Train the model
		'''
		print("------------------------")
		Data_train = Data(self.trainData, args, glob)
		n_train = len(Data_train.stories)
		print("Training Size", n_train)

		# Data_val = Data(self.valData, args, glob)
		# n_val = len(Data_val.stories)
		# print("Validation Size", n_val)

		# Data_test = Data(self.testData, args, glob)
		# n_test = len(Data_test.stories)
		# print("Test Size", n_test)

		# if args.task_id < 6:
		# 	Data_test_OOV = Data(self.testOOVData, args, glob)
		# 	n_oov = len(Data_test_OOV.stories)
		# 	print("Test OOV Size", n_oov)
		# sys.stdout.flush()

		# Create Batches
		batches_train = create_batches(Data_train, args.batch_size, self.RLtrainData)
		# batches_val = create_batches(Data_val, args.batch_size, self.RLvalData)
		# batches_test = create_batches(Data_test, args.batch_size, self.RLtestData)
		# if args.task_id < 6:
		# 	batches_oov = create_batches(Data_test_OOV, args.batch_size, self.RLtestOOVData)

		# Look for previously saved checkpoint
		if args.save:
			ckpt = tf.train.get_checkpoint_state(self.model_dir)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
			else:
				print("...no checkpoint found...")
			print('*Predict Validation*'); sys.stdout.flush()
			val_accuracies = self.batch_predict(Data_val, batches_val, self.RLvalData)
			best_validation_accuracy = val_accuracies['comp']
		else:
			best_validation_accuracy = 0

		# Train Model in Batch Mode
		loss_buffer = deque()
		for epoch in range(1, args.epochs + 1):			
			print('************************')
			print('Epoch {}'.format(epoch)); sys.stdout.flush()
			
			if self.model.phase >= 10:
				total_cost_pre = self.batch_train(Data_train, batches_train[0])					
				print('Total Cost Pre: {}'.format(total_cost_pre)); sys.stdout.flush()
				total_loss = total_cost_pre
			
			if args.rl and self.model.phase >= 2:
				Data_train.reset_responses()
				total_cost_api, total_reward, api_metric = self.batch_train_api(Data_train, batches_train[1], self.RLtrainData)		
				print('Total Cost API: {}'.format(total_cost_api)); sys.stdout.flush()
				print('Total Reward: {}'.format(total_reward)); sys.stdout.flush()
				print('Valid Queries: {}'.format(api_metric)); sys.stdout.flush()
			
			if (args.rl and self.model.phase >= 30):
				total_cost_post = self.batch_train(Data_train, batches_train[0] + batches_train[2] , Data_train.responses)		
				print('Total Cost Post: {}'.format(total_cost_post)); sys.stdout.flush()
				total_loss = total_cost_post 
			
			
			# # Evaluate Model	
			# if epoch % args.evaluation_interval == 0:
			# 	print('*Predict Train*'); sys.stdout.flush()
			# 	train_accuracies = self.batch_predict(Data_train, batches_train, self.RLtrainData)
			# 	print('*Predict Validation*'); sys.stdout.flush()
			# 	val_accuracies = self.batch_predict(Data_val, batches_val, self.RLvalData)
			# 	print('-----------------------')
			# 	print('SUMMARY')
			# 	print('PHASE {}'.format(self.model.phase))
			# 	print('Epoch {}'.format(epoch))
			# 	print('Loss: {}'.format(total_loss))
			# 	if args.bleu_score:
			# 		print('{0:30} : {1:6f}'.format("Train BLEU", train_accuracies['bleu']))
			# 	print('{0:30} : {1:6f}'.format("Train Accuracy", train_accuracies['acc']))
			# 	print('{0:30} : {1:6f}'.format("Train Dialog", train_accuracies['dialog']))
			# 	print('{0:30} : {1:6f}'.format("Train F1", train_accuracies['f1']))
			# 	print('{0:30} : {1:6f}'.format("Train API Match", train_accuracies['api']))
			# 	print('------------')
			# 	if args.bleu_score:
			# 		print('{0:30} : {1:6f}'.format("Validation BLEU", val_accuracies['bleu']))
			# 	print('{0:30} : {1:6f}'.format("Validation Accuracy", val_accuracies['acc']))
			# 	print('{0:30} : {1:6f}'.format("Validation Dialog", val_accuracies['dialog']))
			# 	print('{0:30} : {1:6f}'.format("Validation F1", val_accuracies['f1']))
			# 	print('{0:30} : {1:6f}'.format("Validation API Match", val_accuracies['api']))

			# 	print('------------')
			# 	sys.stdout.flush()
				
			# 	if self.model.phase == 2:
			# 		loss_metric = 1.0 - api_metric
			# 	else:
			# 		loss_metric = total_loss
			# 	loss_buffer.append(loss_metric)
			# 	if (len(loss_buffer) == 6 and args.rl) or loss_metric == 0.0:
			# 		val = loss_buffer.popleft()
			# 		if val < loss_metric or loss_metric == 0.0:
			# 			if (self.model.phase == 2 and glob['valid_query']) or self.model.phase == 1:
			# 				self.model.phase += 1
			# 				print("PHASE change to {}".format(self.model.phase))
			# 				self.saver.save(glob['session'], self.model_dir + 'model.ckpt', global_step=epoch)
			# 				print('MODEL SAVED')
			# 				test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData, output=True)
			# 				best_validation_accuracy = 0.0
			# 				loss_buffer = deque()
			# 				if self.model.phase == 2:
			# 					glob['valid_query'] = False
			# 				continue

			# 	# Save best model
			# 	val_to_compare = val_accuracies['comp']
			# 	if val_to_compare > best_validation_accuracy:
			# 		best_validation_accuracy = val_to_compare
			# 		self.saver.save(glob['session'], self.model_dir + 'model.ckpt', global_step=epoch)
			# 		print('MODEL SAVED')
			# 		# test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData, output=True)
				
			# 		print('Predict Test'); sys.stdout.flush()
			# 		test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData)
			# 		if args.task_id < 6:
			# 			print('\nPredict OOV'); sys.stdout.flush()
			# 			test_oov_accuracies = self.batch_predict(Data_test_OOV, batches_oov, self.RLtestOOVData)
					
			# 		print('-----------------------')
			# 		print('SUMMARY')
			# 		if args.bleu_score:
			# 			print('{0:30} : {1:6f}'.format("Test BLEU", test_accuracies['bleu']))
			# 		print('{0:30} : {1:6f}'.format("Test Accuracy", test_accuracies['acc']))
			# 		print('{0:30} : {1:6f}'.format("Test Dialog", test_accuracies['dialog']))
			# 		print('{0:30} : {1:6f}'.format("Test F1", test_accuracies['f1']))
			# 		print('{0:30} : {1:6f}'.format("Test API Match", test_accuracies['api']))
			# 		if args.task_id < 6:
			# 			print('------------')
			# 			if args.bleu_score:
			# 				print('{0:30} : {1:6f}'.format("Test OOV BLEU", test_oov_accuracies['bleu']))
			# 			print('{0:30} : {1:6f}'.format("Test OOV Accuracy", test_oov_accuracies['acc']))
			# 			print('{0:30} : {1:6f}'.format("Test OOV Dialog", test_oov_accuracies['dialog']))
			# 			print('{0:30} : {1:6f}'.format("Test OOV F1", test_oov_accuracies['f1']))
			# 			print('{0:30} : {1:6f}'.format("Test OOV API Match", test_oov_accuracies['api']))
			# 		print('-----------------------')
			# 		sys.stdout.flush()
			
	def test(self):
		'''
			Test the model
		'''
		# Look for previously saved checkpoint
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
		else:
			print("...no checkpoint found...")

		if args.OOV:
			Data_test = Data(self.testOOVData, args, glob)
			n_test = len(Data_test_OOV.stories)
			print("Test OOV Size", n_test)
		else:
			Data_test = Data(self.testData, args, glob)
			n_test = len(Data_test.stories)
			print("Test Size", n_test)
		sys.stdout.flush()

		print('*Predict Test*'); sys.stdout.flush()
		if args.OOV:
			batches_test = create_batches(Data_test, args.batch_size, self.RLtestOOVData)
			test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestOOVData, output=True)
		else:
			batches_test = create_batches(Data_test, args.batch_size, self.RLtestData)
			test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData, output=True)

		print('-----------------------')
		print('SUMMARY')
		if args.bleu_score:
			print('{0:30} : {1:6f}'.format("Test BLEU", test_accuracies['bleu']))
		print('{0:30} : {1:6f}'.format("Test Accuracy", test_accuracies['acc']))
		print('{0:30} : {1:6f}'.format("Test Dialog", test_accuracies['dialog']))
		print('{0:30} : {1:6f}'.format("Test F1", test_accuracies['f1']))
		print('{0:30} : {1:6f}'.format("Test API Match", test_accuracies['api']))
		print("------------------------")
		sys.stdout.flush()

	def batch_train(self, data, train_batches, responses=None):
		'''
			Train Model for a Batch of Input Data
		'''
		idxs = range(len(train_batches))
		batches = train_batches.copy()
		np.random.shuffle(batches)
		total_cost = 0.0	# Total Loss
		total_seq = 0.0		# Sequence Loss
		total_pgen = 0.0	# Pgen Loss

		pbar = tqdm(enumerate(batches),total=len(train_batches))
		for i, indecies in pbar:
			idx = indecies[0]
			indecies = indecies[1:]
			if idx == 2:
				batch_entry = Batch(data, indecies, args, glob, responses, train=True)
			else:
				batch_entry = Batch(data, indecies, args, glob, None, train=True)
			cost_t, seq_loss, pgen_loss = self.model.fit(batch_entry)
			total_seq += seq_loss
			total_pgen += pgen_loss
			total_cost += cost_t
			pbar.set_description('TL:{:.2f}, SL:{:.2f}, PL:{:.2f}'.format(total_cost/(i+1),total_seq/(i+1),total_pgen/(i+1)))

		return total_cost

	def batch_predict(self, data, batches, rl_data, output=False):
		'''
			Get Predictions for Input Data batchwise
		'''

		batches_pre = batches[0]
		batches_api = batches[1]
		batches_post = batches[2]

		predictions = []

		dialog_ids = []
		entities = []
		oov_words = []
		golds = []

		post_index = len(batches_pre)
		batches = batches_pre + batches_post

		if args.rl and self.model.phase >= 2:
			_, _, matched_query_ratio = self.batch_train_api(data, batches_api, rl_data, train=False, output=output)

		pbar = tqdm(enumerate(batches),total=len(batches))
		for i, indecies in pbar:
			idx = indecies[0]
			indecies = indecies[1:]
			# Get predictions
			if i < post_index: 	data_batch = Batch(data, indecies, args, glob, None)
			elif self.model.phase < 3: break
			else: 				data_batch = Batch(data, indecies, args, glob, data.responses)

			if args.beam:
				parent_ids, predict_ids = self.model.predict(data_batch)
			else:
				preds = self.model.predict(data_batch)

			# Store prediction outputs
			if args.beam:
				actions = calculate_beam_result(parent_ids, predict_ids, glob['candidate_sentence_size'])
				for action in actions: predictions.append(action[0]) 
			else:
				predictions += pad_to_answer_size(list(preds), glob['candidate_sentence_size'])
			dialog_ids += data_batch.dialog_ids
			entities += data_batch.entities
			oov_words += data_batch.oov_words
			golds += data_batch.answers

		# Evaluate metrics
		acc = evaluate(args, glob, predictions, golds, entities, dialog_ids, oov_words, out=output)
		acc['api'] = (matched_query_ratio if args.rl and self.model.phase >= 2 else 0.0)
		return acc

	def batch_train_api(self, data, batches, rl_data, train=True, output=False):
		'''
			Train Model for a Batch of Input Data
		'''
		if output and args.rl: open('logs/api.txt', 'w+')

		if train: np.random.shuffle(batches)
		total_cost = 0.0
		total_reward = 0.0
		total_entries_sum = 0.0
		valid_entries_sum = 0.0
		perfect_match_entries_sum = 0.0

		pbar = tqdm(enumerate(batches),total=len(batches))
		for i, indecies in pbar:
			idx = indecies[0]
			indecies = indecies[1:]
			batch_entry = Batch(data, indecies, args, glob)

			# dont run api_predict if we have to just append Ground Truth
			if args.rl_mode == "GT":
				actions = None
				pred_action_lengths = None
			else:
				if args.beam:
					parent_ids, predict_ids = self.model.api_predict(batch_entry)
					actions = calculate_beam_result(parent_ids, predict_ids, args.max_api_length)
					pred_action_lengths = None
				else:
					preds, pred_action_lengths = self.model.api_predict(batch_entry)
					actions = pad_to_answer_size(list(preds), args.max_api_length, True)

			responses, batched_actions_and_rewards, high_probable_rewards, total_entries, valid_entries, perfect_match_entries = \
					calculate_reward(glob, actions, pred_action_lengths, batch_entry, rl_data, self.db_engine, self.model, args, data, output=output, mode=args.rl_mode)
			total_entries_sum += total_entries
			valid_entries_sum += valid_entries
			perfect_match_entries_sum += perfect_match_entries
			matched_query_ratio = float(valid_entries_sum)/float(total_entries_sum)
			perfect_query_ratio = float(perfect_match_entries_sum)/float(total_entries_sum)
			glob['valid_query'] = True if matched_query_ratio > 0.5 else False
			total_reward += high_probable_rewards
			
			# dont run api_fit if are in Ground Truth mode
			if train and args.rl_mode != "GT":
				for actions_and_rewards in batched_actions_and_rewards:
					total_cost += self.model.api_fit(batch_entry, actions_and_rewards)
			
			for id, response in zip(batch_entry.dialog_ids, responses):
				if len(response) > 0:
					data.responses[id] = response
			pbar.set_description('RW:{:.2f} AV:{:.2f} AP:{:.2f}'.format(total_reward/(i+1), matched_query_ratio, perfect_query_ratio))

		return total_cost, total_reward, matched_query_ratio

	def close_session(self):
		glob['session'].close()

''' Main Function '''
if __name__ == '__main__': 

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
	
	print_params(logging, args)
	chatbot = chatBot()
	print("CHATBOT READY"); sys.stdout.flush();
	
	chatbot.train() if args.train else chatbot.test()

	chatbot.close_session()