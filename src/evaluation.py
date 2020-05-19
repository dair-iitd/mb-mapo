import os
import json
import numpy as np 
from measures import moses_multi_bleu
from sklearn.metrics import f1_score
import pickle as pkl
from collections import defaultdict
import re
import sys
import random
import db_engine

stop_words=set(["a","an","the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

def process(preds, golds):
	for i, (pred, gold) in enumerate(zip(preds, golds)):
		preds[i] = [x for x in pred if x != EOS_INDEX and x != PAD_INDEX]
		golds[i] = [x for x in gold if x != EOS_INDEX and x != PAD_INDEX]
	return preds, golds

def get_surface_form(index_list, word_map, oov_words, context=False):
	size = len(word_map)
	maxs = size + len(oov_words)
	if context:
		surfaces = []
		for story_line in index_list:
			surfaces.append([word_map[i] if i in word_map else oov_words[i - size] for i in story_line])
		return surfaces
	else:
		lst = []
		for i in index_list:
			if i in word_map:	lst.append(i)
			elif i < maxs: 		lst.append(oov_words[i - size])
		return lst

def surface(index_list, word_map, oov_words, context=False):
	surfaces = []
	for i, lst in enumerate(index_list):
		surfaces.append(get_surface_form(lst, word_map, oov_words[i], context))
	return surfaces

def accuracy(preds, golds, dialog_ids):
	total_score = 0
	dialog_dict = {}

	for i, (pred, gold) in enumerate(zip(preds, golds)):
		if pred==gold:
			total_score += 1
			if dialog_ids[i] not in dialog_dict:
				dialog_dict[dialog_ids[i]] = 1
		else:
			dialog_dict[dialog_ids[i]] = 0
	
	# Calculate Response Accuracy
	size = len(preds)
	response_accuracy = "{:.2f}".format(float(total_score) * 100.0 / float(size))		

	# Calculate Dialog Accuracy
	dialog_size = len(dialog_dict)
	correct_dialogs = list(dialog_dict.values()).count(1)
	dialog_accuracy = "{:.2f}".format(float(correct_dialogs) * 100.0 / float(dialog_size))

	return response_accuracy, dialog_accuracy 

def compute_macro_f1(pr_list, re_list):

	#average of F1s
	#arithmetic mean over harmonic mean
	total_f1 = 0
	total_pr = 0
	total_re = 0
	for pr, re in zip(pr_list, re_list):
		if pr+re > 0:
			f1=(2*pr*re)/(pr+re)
			total_f1 += f1
		total_pr += pr
		total_re += re

	return total_pr/float(len(pr_list)), total_re/float(len(pr_list)), total_f1/float(len(pr_list))
		
def f1(preds, golds, stories, entities, dids, answers, ordered_oovs, word_map, encoder_word_map, db_engine):
	
	punc = ['.', ',', '!', '\'', '\"', '-', '?']

	'''
	punc = ['.', ',', '!', '\'', '\"', '-', '?']
	for entity_idxs, gold in zip(entities, golds):
		processed_gold = [x for x in gold if x != EOS_INDEX and x != PAD_INDEX]
		print("gold", gold)
		print("processed_gold", processed_gold)
		print("entity_idxs", entity_idxs)
		for e in entity_idxs:
			if e < len(processed_gold):
				e_surface_form = word_map[processed_gold[e]]
				if e_surface_form not in punc:
					print("\t", e_surface_form)
		print("----------------------")
	'''

	pr_list = []
	re_list = []

	pr_infor_list = []
	re_infor_list = []

	pr_req_list = []
	re_req_list = []


	for i, (pred, gold, entity, did, answer, story) in enumerate(zip(preds, golds, entities, dids, answers, stories)):

		entities_in_pred = set([])
		entities_in_gold = set([])

		informable_entities_in_pred = set([])
		informable_entities_in_gold = set([])

		'''
		print("did", did)
		print("gold", gold, get_tokenized_response_from_padded_vector(gold, word_map, ordered_oovs[did]))
		print("pred", pred, get_tokenized_response_from_padded_vector(pred, word_map, ordered_oovs[did]))
		print("entity", entity)
		for e in entity:
			e_surface =  word_map[gold[e]]
			if e_surface not in punc:
				entities_in_gold.add(e_surface)
				if db_engine.is_informable_field_value(e_surface):
					informable_entities_in_gold.add(e_surface)
		'''
		
		story_entities = set([])
		for story_line in story:
			story_line_str = " ".join(story_line)
			story_line_str = story_line_str.strip()
			if story_line_str.strip() != "" and "$db" not in story_line_str:
				line_entities, _ = db_engine.get_entities_in_utt(story_line_str)
				for e in line_entities:
					story_entities.add(e)

		entities_in_gold, _ = db_engine.get_entities_in_utt(answer)
		story_entities_in_gold = entities_in_gold.intersection(story_entities)

		if len(entities_in_gold) == 0:
			continue

		entities_in_pred, _ = db_engine.get_entities_in_utt(get_tokenized_response_from_padded_vector(pred, word_map, ordered_oovs[did]))
		story_entities_in_pred = entities_in_pred.intersection(story_entities)

		common = float(len(entities_in_gold.intersection(entities_in_pred)))
		re_list.append(common/len(entities_in_gold))
		if len(entities_in_pred) == 0:
			pr_list.append(0)
		else:
			pr_list.append(common/len(entities_in_pred))
		
		if len(story_entities_in_gold) > 0:
			common = float(len(story_entities_in_gold.intersection(story_entities_in_pred)))
			if len(story_entities_in_pred) == 0:
				pr_infor_list.append(0)
			else:
				pr_infor_list.append(common/len(story_entities_in_pred))
			re_infor_list.append(common/len(story_entities_in_gold))
		#else:
		#	pr_infor_list.append(0)
		#	re_infor_list.append(0)

		req_in_gold = entities_in_gold.difference(story_entities_in_gold)
		if len(req_in_gold) > 0:
			req_in_pred = entities_in_pred.difference(story_entities_in_pred)
			common = float(len(req_in_gold.intersection(req_in_pred)))
			if len(req_in_pred) == 0:
				pr_req_list.append(0)
			else:
				pr_req_list.append(common/len(req_in_pred))
			re_req_list.append(common/len(req_in_gold))
		#else:
		#	pr_req_list.append(0)
		#	re_req_list.append(0)

	macro_pr, macro_re, macro_f1 = compute_macro_f1(pr_list, re_list)

	macro_infor_pr, macro_infor_re, macro_infor_f1 = compute_macro_f1(pr_infor_list, re_infor_list)
	
	macro_req_pr, macro_req_re, macro_req_f1 = compute_macro_f1(pr_req_list, re_req_list)

	return ((macro_pr, macro_re, macro_f1), (macro_infor_pr, macro_infor_re, macro_infor_f1), (macro_req_pr, macro_req_re, macro_req_f1))

def get_tokenized_response_from_padded_vector(vector, word_map, oov):
	final = []
	maxs = len(oov) + len(word_map)
	for x in vector:
		if x in word_map: 	final.append(word_map[x])
		elif x < maxs: 		final.append(oov[x - len(word_map)])
		else:				final.append('UNK')
	return ' '.join(final)

def BLEU(preds, golds, word_map, dids, tids, answers, oovs, output=False, run_id="", epoch_str=""):
	tokenized_preds = []
	tokenized_golds = []
	
	if output:
		dirName = 'logs/preds/'+run_id
		if not os.path.exists(dirName):
			os.mkdir(dirName)
		file = open(dirName+'/'+epoch_str+'.log', 'w+')
	for i, (pred, gold, did, tid, answer) in enumerate(zip(preds, golds, dids, tids, answers)):
		#print(epoch_str, "-------")
		#print("did", str(did), "tid", str(tid))
		#print("oovs", oovs[dids[i]])
		pred = get_tokenized_response_from_padded_vector(pred, word_map, oovs[dids[i]])
		#print("pred", pred)
		#gold = get_tokenized_response_from_padded_vector(gold, word_map, oovs[dids[i]])
		#print("gold", gold)
		#print("answer", answer)

		tokenized_preds.append(pred)
		tokenized_golds.append(answer)
		if output:
			file.write('did : ' + str(did) + '\n' + 'tid : ' + str(tid) + '\n' + 'gold : ' + answer + '\n'+'pred : ' + pred + '\n\n')
			file.flush()
	if output:
		file.close()
			
	return "{:.2f}".format(moses_multi_bleu(tokenized_preds, tokenized_golds, True))

def tokenize(vals, dids):
	tokens = []
	punc = ['.', ',', '!', '\'', '\"', '-', '?']
	for i, val in enumerate(vals):
		sval = [x.strip() for x in re.split('(\W+)?', val) if x.strip()]
		idxs = []
		did = dids[i] + 1
		oov_word = ordered_oovs[did]
		sval = [x for x in sval if '$$$$' not in x]
		for i, token in enumerate(sval):
			if token in index_map:
				idx = index_map[token]
			elif token in oov_word:
				idx = len(index_map) + oov_word.index(token)
			else:
				idx = UNK_INDEX
			if token not in punc or i+1 < len(sval) :
				idxs.append(idx)
		tokens.append(idxs)
	return tokens

def merge(ordered, gold_out=True):
	preds = []
	golds = []
	entities = []
	dids = []
	tids = []
	answers = []
	stories = []
	for i in range(1, len(ordered)+1):
		val = ordered[i]
		for dct in val:
			preds.append(dct['preds'])
			golds.append(dct['golds'])
			entities.append(dct['entity'])
			tids.append(dct['turn'])
			answers.append(dct['answer'])
			stories.append(dct['story'])
			dids.append(i)
	return preds, golds, entities, dids, tids, answers, stories

def evaluate(args, glob, preds, golds, stories, entities, dialog_ids, turn_ids, readable_answers, oov_words, db_engine, out=False, run_id="", epoch_str=""):
	word_map = glob['idx_decode'] 
	index_map = glob['decode_idx']
	encode_word_map = glob['idx_word']

	preds, golds = process(preds, golds)

	# process converts indices to surface forms
	# loop over to print surface forms

	ordered_oovs = {}
	for num, words in zip(dialog_ids, oov_words):
		if num not in ordered_oovs:
			ordered_oovs[num] = list(words)
		else:
			if len(list(words)) > len(ordered_oovs[num]):
				ordered_oovs[num] = list(words)

	ordered_orig = defaultdict(list)

	orginal = zip(preds, golds, entities, turn_ids, readable_answers, stories)

	for num, org in zip(dialog_ids, orginal):
		p, g, e, t, a, s = org
		element_dict = defaultdict(list)
		element_dict['preds'] = p
		element_dict['golds'] = g
		element_dict['entity'] = e
		element_dict['turn'] = t
		element_dict['answer'] = a
		element_dict['story'] = s
		ordered_orig[num].append(element_dict)

	preds, golds, ordered_entities, dids, tids, answers, ord_stories = merge(ordered_orig, True)

	output = {}
	output['bleu'] = float(BLEU(preds, golds, word_map, dids, tids, answers, ordered_oovs, output=out, run_id=run_id, epoch_str=epoch_str))
	acc, dial = accuracy(preds, golds, dids)
	output['acc'] = float(acc)
	output['dialog'] = float(dial)
	output['f1'] = f1(preds, golds, ord_stories, ordered_entities, dids, answers, ordered_oovs, word_map, encode_word_map, db_engine)
	if args.bleu_score:
		output['comp'] = output['bleu'] 
	else:
		output['comp'] = output['acc'] 
	return output