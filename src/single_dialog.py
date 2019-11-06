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
#from tqdm import tqdm

args = get_params()
glob = {}

class chatBot(object):

	def __init__(self):
		# Create Model Store Directory
		self.run_id = ("task" + str(args.task_id) + "_" + args.data_dir.split('/')[-2] 
						+ "_lr-" + str(args.learning_rate) 
						+ "_hops-" + str(args.hops) 
						+ "_emb-size-" + str(args.embedding_size) 
						+ "_sw-" + str(args.soft_weight) 
						+ "_wd-" + str(args.word_drop_prob) 
						+ "_pw-" + str(args.p_gen_loss_weight) 
						+ "_rlmode-" + str(args.rl_mode)
						+ "_idx-" + str(args.model_index))
		self.model_dir = (args.model_dir +  self.run_id + "_model/")
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
		glob['session'] = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
		self.model = MemN2NGeneratorDialog(args, glob)
		self.saver = tf.train.Saver(max_to_keep=4)

	def build_vocab(self, data):
		'''
			Get vocabulary from the Train data
		'''
		vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + list(chain.from_iterable(d))) for s, q, a, _, _, d in data))
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
		Data_train = Data(self.trainData, args, glob, self.RLtrainData)
		n_train = len(Data_train.stories)
		print("Training Size", n_train)

		Data_val = Data(self.valData, args, glob, self.RLvalData)
		n_val = len(Data_val.stories)
		print("Validation Size", n_val)

		Data_test = Data(self.testData, args, glob, self.RLtestData)
		n_test = len(Data_test.stories)
		print("Test Size", n_test)

		if args.task_id < 6:
			Data_test_OOV = Data(self.testOOVData, args, glob, self.RLtestOOVData)
			n_oov = len(Data_test_OOV.stories)
			print("Test OOV Size", n_oov)
		sys.stdout.flush()

		if args.rl:
			glob['valid_query'] = False
			glob['best_validation_rewards'] = 0.0

		# Create Batches
		batches_train = create_batches(Data_train, args.batch_size, self.RLtrainData)
		batches_val = create_batches(Data_val, args.batch_size, self.RLvalData)
		batches_test = create_batches(Data_test, args.batch_size, self.RLtestData)
		if args.task_id < 6:
			batches_oov = create_batches(Data_test_OOV, args.batch_size, self.RLtestOOVData)

		# Look for previously saved checkpoint
		
		#writer = tf.summary.FileWriter("output", glob['session'].graph)
		#writer.close()
		if args.save:
			ckpt = tf.train.get_checkpoint_state(self.model_dir)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
			else:
				print("...no checkpoint found...")
			#print('*Predict Validation*'); 
			sys.stdout.flush()
			val_accuracies = self.batch_predict(Data_val, batches_val, self.RLvalData)
			best_validation_accuracy = val_accuracies['comp']
		else:
			best_validation_accuracy = 0

		# Train Model in Batch Mode
		loss_buffer = deque()
		for epoch in range(1, args.epochs + 1):			
			print('************************')
			print('\nEpoch {}'.format(epoch), 'Phase {}'.format(self.model.phase)); sys.stdout.flush()
				
			if args.rl and self.model.phase >= 1:
				total_reward, perfect_query_ratio, valid_query_ratio, _ = self.batch_train_api(Data_train, batches_train[1], self.RLtrainData)

				total_rewards, perfect_query_ratio, valid_query_ratio, train_db_results_map = self.batch_train_api(Data_train, batches_train[1], self.RLtrainData, train=False, output=True, epoch_str=str(epoch)+"-trn")
				print("\nTrain Rewards: {0:6.4f} Valid-Ratio:{1:6.4f} Perfect-Ratio:{2:6.4f}".format(total_rewards, valid_query_ratio, perfect_query_ratio))
						
			if (args.rl and self.model.phase >= 2 and epoch > args.rl_warmp_up):
				total_cost_post = self.batch_train(Data_train, batches_train[0] + batches_train[2], Data_train.responses)		
				total_loss = total_cost_post 
			
			if epoch > 200:
				glob['valid_query'] = True

			# Evaluate Model	
			if epoch % args.evaluation_interval == 0:
				
				total_rewards, perfect_query_ratio, valid_query_ratio, valid_db_results_map = self.batch_train_api(Data_val, batches_val[1], self.RLvalData, train=False, epoch_str=str(epoch)+"-val")
				
				if total_rewards > glob['best_validation_rewards']:
					
					print("\nValidation Rewards:{0:6.4f} Best-So-far:{1:6.4f}".format(total_rewards, glob['best_validation_rewards']))
					print("\nValidation Valid-Ratio:{0:6.4f} Perfect-Ratio:{1:6.4f}".format(valid_query_ratio, perfect_query_ratio))
					glob['best_validation_rewards'] = total_rewards 
					
					for id, db_results in train_db_results_map.items():
						Data_train.responses[id] = db_results

					for id, db_results in valid_db_results_map.items():
						Data_val.responses[id] = db_results

					test_rewards, perfect_query_ratio, valid_query_ratio, db_results_map = self.batch_train_api(Data_test, batches_test[1], self.RLtestData, train=False, output=True, epoch_str="tst-"+str(epoch))
					print("Test       Rewards:{0:6.4f}".format(test_rewards))
					print("Test       Valid-Ratio:{0:6.4f} Perfect-Ratio:{1:6.4f}".format(valid_query_ratio, perfect_query_ratio))
					for id, db_results in db_results_map.items():
						Data_test.responses[id] = db_results
							
					if args.task_id < 6:
						test_oov_rewards, perfect_query_ratio, valid_query_ratio, db_results_map = self.batch_train_api(Data_test_OOV, batches_oov[1], self.RLtestOOVData, train=False, output=True, epoch_str='tst-OOV-'+str(epoch))
						print("Test-OOV   Rewards:{0:6.4f}".format(test_oov_rewards))
						print("Test-OOV   Valid-Ratio:{0:6.4f} Perfect-Ratio:{1:6.4f}".format(valid_query_ratio, perfect_query_ratio))
						for id, db_results in db_results_map.items():
							Data_test_OOV.responses[id] = db_results
					
					if valid_query_ratio > 0.5:
						glob['valid_query'] = True

				
				if glob['valid_query'] and epoch > args.rl_warmp_up:
					train_accuracies = self.batch_predict(Data_train, batches_train, self.RLtrainData)
					print('')
					if args.bleu_score:
						print('{0:30} : {1:6f}'.format("Train BLEU", train_accuracies['bleu']))
					print('{0:30} : {1:6f}'.format("Train Accuracy", train_accuracies['acc']))
					print('{0:30} : {1:6f}'.format("Train Dialog", train_accuracies['dialog']))
					print('{0:30} : {1:6f}'.format("Train F1", train_accuracies['f1']))
					#print('{0:30} : {1:6f}'.format("Train API Match", train_accuracies['api']))
					print('')
					
					val_accuracies = self.batch_predict(Data_val, batches_val, self.RLvalData)
					if args.bleu_score:
						print('{0:30} : {1:6f}'.format("Validation BLEU", val_accuracies['bleu']))
					print('{0:30} : {1:6f}'.format("Validation Accuracy", val_accuracies['acc']))
					print('{0:30} : {1:6f}'.format("Validation Dialog", val_accuracies['dialog']))
					print('{0:30} : {1:6f}'.format("Validation F1", val_accuracies['f1']))
					#print('{0:30} : {1:6f}'.format("Validation API Match", val_accuracies['api']))
					
					print('')
					sys.stdout.flush()

					# Save best model
					val_to_compare = val_accuracies['comp']
					if val_to_compare > best_validation_accuracy and self.model.phase == 2:
						best_validation_accuracy = val_to_compare
						self.saver.save(glob['session'], self.model_dir + 'model.ckpt', global_step=epoch)
						print('MODEL SAVED')
						test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData, output=True, epoch_str=str(epoch))
					
						#print('Predict Test'); 
						sys.stdout.flush()
						#test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData)
						if args.task_id < 6:
							#print('\nPredict OOV'); sys.stdout.flush()
							test_oov_accuracies = self.batch_predict(Data_test_OOV, batches_oov, self.RLtestOOVData, output=True, epoch_str=str(epoch)+'-OOV')
						
						print('')
						#print('SUMMARY')
						if args.bleu_score:
							print('{0:30} : {1:6f}'.format("Test BLEU", test_accuracies['bleu']))
						print('{0:30} : {1:6f}'.format("Test Accuracy", test_accuracies['acc']))
						print('{0:30} : {1:6f}'.format("Test Dialog", test_accuracies['dialog']))
						print('{0:30} : {1:6f}'.format("Test F1", test_accuracies['f1']))
						#print('{0:30} : {1:6f}'.format("Test API Match", test_accuracies['api']))
						
						if args.task_id < 6:
							print('')
							if args.bleu_score:
								print('{0:30} : {1:6f}'.format("Test OOV BLEU", test_oov_accuracies['bleu']))
							print('{0:30} : {1:6f}'.format("Test OOV Accuracy", test_oov_accuracies['acc']))
							print('{0:30} : {1:6f}'.format("Test OOV Dialog", test_oov_accuracies['dialog']))
							print('{0:30} : {1:6f}'.format("Test OOV F1", test_oov_accuracies['f1']))
							#print('{0:30} : {1:6f}'.format("Test OOV API Match", test_oov_accuracies['api']))
							
						print('')
						sys.stdout.flush()

				if (self.model.phase == 1 and glob['valid_query']):
					self.model.phase += 1
					print("PHASE change to {}".format(self.model.phase))
					print('')

	# this function doesnt work due to api prediction code
	def test(self):
		'''
			Test the model
		'''
		self.model.phase = 2
		# Look for previously saved checkpoint
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(glob['session'], ckpt.model_checkpoint_path)
		else:
			print("...no checkpoint found...")

		if args.OOV:
			Data_test = Data(self.testOOVData, args, glob,  self.RLtestOOVData)
			n_test = len(Data_test_OOV.stories)
			print("Test OOV Size", n_test)
			prefix = "Test OOV"
		else:
			Data_test = Data(self.testData, args, glob,  self.RLtestData)
			n_test = len(Data_test.stories)
			print("Test Size", n_test)
			prefix = "Test"
		sys.stdout.flush()

		print('*Predict Test*'); sys.stdout.flush()
		if args.OOV:
			batches_test = create_batches(Data_test, args.batch_size, self.RLtestOOVData)
			test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestOOVData, output=True, epoch_str="test-OOV")
		else:
			batches_test = create_batches(Data_test, args.batch_size, self.RLtestData)
			test_accuracies = self.batch_predict(Data_test, batches_test, self.RLtestData, output=True, epoch_str="test")

		print('-----------------------')
		print('SUMMARY')
		if args.bleu_score:
			print('{0} {1} : {2:6f}'.format(prefix, "BLEU", test_accuracies['bleu']))
		print('{0} {1} : {2:6f}'.format(prefix, "Accuracy", test_accuracies['acc']))
		print('{0} {1} : {2:6f}'.format(prefix, "Dialog", test_accuracies['dialog']))
		print('{0} {1} : {2:6f}'.format(prefix, "F1", test_accuracies['f1']))
		print('{0} {1} : {2:6f}'.format(prefix, "API Match", test_accuracies['api']))
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

		#pbar = tqdm(enumerate(batches),total=len(train_batches))
		#for i, indecies in pbar:
		
		for i, indecies in enumerate(batches):
			idx = indecies[0]
			indecies = indecies[1:]	
			if len(indecies) == 0:
				continue
			if idx == 2:
				batch_entry = Batch(data, indecies, args, glob, responses, train=True)
			else:
				batch_entry = Batch(data, indecies, args, glob, None, train=True)
			cost_t, seq_loss, pgen_loss = self.model.fit(batch_entry)
			total_seq += seq_loss
			total_pgen += pgen_loss
			total_cost += cost_t
			#pbar.set_description('TL:{:.2f}, SL:{:.2f}, PL:{:.2f}'.format(total_cost/(i+1),total_seq/(i+1),total_pgen/(i+1)))

		print('\nTotal L:{:.2f}, Sequence L:{:.2f}, P-Gen L:{:.2f}'.format(total_cost,total_seq,total_pgen))
		return total_cost

	def batch_predict(self, data, batches, rl_data, output=False, epoch_str=""):
		'''
			Get Predictions for Input Data batchwise
		'''

		batches_pre = batches[0]
		batches_api = batches[1]
		batches_post = batches[2]

		#if args.rl and self.model.phase >= 1:
		#	total_reward, perfect_query_ratio, valid_query_ratio = self.batch_train_api(data, batches_api, rl_data, train=False, output=output, epoch_str=epoch_str)
			
		if args.rl and self.model.phase >= 2:
			predictions = []

			dialog_ids = []
			entities = []
			oov_words = []
			golds = []

			post_index = len(batches_pre)
			batches = batches_pre + batches_post

			
			#pbar = tqdm(enumerate(batches),total=len(batches))
			#for i, indecies in pbar:
			for i, indecies in enumerate(batches):
				idx = indecies[0]
				indecies = indecies[1:]
				if len(indecies) == 0:
					continue
				# Get predictions
				if i < post_index: 	data_batch = Batch(data, indecies, args, glob, None)
				elif self.model.phase < 2: break
				else: 				data_batch = Batch(data, indecies, args, glob, data.responses)
				
				if args.simple_beam:
					parent_ids, predict_ids = self.model.predict(data_batch)
				else:
					preds = self.model.predict(data_batch)

				# Store prediction outputs
				if args.simple_beam:
					actions = calculate_beam_result(parent_ids, predict_ids, glob['candidate_sentence_size'])
					for action in actions: predictions.append(action[0]) 
				else:
					predictions += pad_to_answer_size(list(preds), glob['candidate_sentence_size'])
				dialog_ids += data_batch.dialog_ids
				entities += data_batch.entities
				oov_words += data_batch.oov_words
				golds += data_batch.answers

			# Evaluate metrics
			acc = evaluate(args, glob, predictions, golds, entities, dialog_ids, oov_words, out=output, run_id=self.run_id, epoch_str=epoch_str)
		else:
			acc = {}
			acc['bleu'] = 0.0
			acc['acc'] = 0.0
			acc['dialog'] = 0.0
			acc['f1'] = 0.0
			acc['comp'] = 0.0

		# not-used 
		# acc['api'] = (perfect_query_ratio if args.rl and self.model.phase >= 1 else 0.0)
		return acc
	
	def surface_form(self, batch, parent_ids, predict_ids, actions, batch_index):
		rl_oov_words = batch.rl_oov_words
		rl_word_idx  = glob['idx_rl']
		total_rl_words = len(rl_word_idx)
		action_set = []
		hit = False
		for i, action in enumerate(actions):
			action_surface_form = ""
			for word_id in action:
				if word_id not in glob['idx_rl']:
					action_surface_form += " " + rl_oov_words[batch_index][word_id - total_rl_words]
				else:
					word_form = glob['idx_rl'][word_id]
					action_surface_form += " " + word_form
			# print(action)
			# print(action_surface_form)
			if action_surface_form not in action_set:
				action_set.append(action_surface_form)
			else:
				#print(action_surface_form)
				action_set.append(action_surface_form)
				hit = True
		# if hit:			
		# print(action_set)	
		# print(parent_ids)
		# print(predict_ids)

	def batch_train_api(self, data, batches, rl_data, train=True, output=False, epoch_str=""):
		'''
			Train Model for a Batch of Input Data
		'''
		
		file=None
		if output and args.rl and args.rl_mode != 'GT':
			dirName = 'logs/api/'+self.run_id
			if not os.path.exists(dirName):
				os.mkdir(dirName)
			file = open(dirName + '/'+ epoch_str +'.log', 'w+')

		if train: np.random.shuffle(batches)
		total_cost = 0.0
		total_reward = 0.0
		total_entries_sum = 0.0
		valid_entries_sum = 0.0
		perfect_match_entries_sum = 0.0
		
		db_results_map = {}
		
		#pbar = tqdm(enumerate(batches),total=len(batches))
		#for i, indecies in pbar:

		for i, indecies in enumerate(batches):
			idx = indecies[0]
			indecies = indecies[1:]
			batch_entry = Batch(data, indecies, args, glob,train=train)
			
			# dont run api_predict if we have to just append Ground Truth
			if args.rl_mode == "GT":
				actions = None
				pred_action_lengths = None
			else:
				if args.beam:
					parent_ids, predict_ids = self.model.api_predict(batch_entry)
					# print(parent_ids)
					# print(predict_ids)
					actions = calculate_beam_result(parent_ids, predict_ids, args.max_api_length)
					pred_action_lengths = None
					#for batch_index, action_set in enumerate(actions):
						# print()
						# self.surface_form(batch_entry, parent_ids[batch_index], predict_ids[batch_index], action_set, batch_index)
					# sys.exit()
				else:
					preds, pred_action_lengths = self.model.api_predict(batch_entry)
					actions = pad_to_answer_size(list(preds), args.max_api_length, True)
					#print(actions)

			db_results, batched_actions_and_rewards, high_probable_rewards, total_entries, valid_entries, perfect_match_entries = \
					calculate_reward(glob, actions, pred_action_lengths, batch_entry, rl_data, self.db_engine, self.model, args, data, out_file=file, mode=args.rl_mode, epoch_str=epoch_str)
			total_entries_sum += total_entries
			valid_entries_sum += valid_entries
			perfect_match_entries_sum += perfect_match_entries
			total_reward += high_probable_rewards
			
			# dont run api_fit if are in Ground Truth mode
			if train and args.rl_mode != "GT":
				for actions_and_rewards in batched_actions_and_rewards:
					total_cost += self.model.api_fit(batch_entry, actions_and_rewards)
			
			for id, response in zip(batch_entry.dialog_ids, db_results):
				db_results_map[id] = response
				
			
		valid_query_ratio = float(valid_entries_sum)/float(total_entries_sum)
		perfect_query_ratio = float(perfect_match_entries_sum)/float(total_entries_sum)	
		
		if output and args.rl and args.rl_mode != 'GT':
			file.close()

		return total_reward, perfect_query_ratio, valid_query_ratio, db_results_map

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