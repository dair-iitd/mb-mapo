import os
import sys
import pprint
import math
import copy
from tabulate import tabulate

task=""

def is_terminated(file):
	
	numbers_map = {}
	lines = [line.strip() for line in open(file)]

	for line in lines:
		if "TERM_MEMLIMIT: job killed after reaching LSF memory usage limit." in line:
			return True
		
	return False

def terminate_at_last_test_eval(lines):

	filtered_lines = []
	reversed_range = list(reversed(range(0,len(lines))))
	last_index = len(lines)
	for i in reversed_range:
		if "Test OOV F1  " in lines[i]:
			last_index = i+1
			break
		if "Test F1" in lines[i]:
			last_index = i+1
			break
	
	return lines[:last_index]

def process_file(file):
	
	print("Processing", file)
	numbers_map = {}
	lines = [line.strip() for line in open(file)]
	lines = terminate_at_last_test_eval(lines)

	validation_rewards_line = ""
	test_rewards_line = ""
	test_ratios_line = ""

	test_bleu = ""
	test_acc = ""
	test_dial_acc = ""
	test_f1 = ""

	test_oov_acc = ""
	test_oov_dial_acc = ""
	test_oov_f1 = ""

	value_map = {}
	for idx, line in enumerate(lines):
		if "Validation Rewards:" in line:
			validation_rewards_line = line
		if "Test       Rewards:" in line:
			test_rewards_line = lines[idx]
			test_ratios_line = lines[idx+1]
		if task == "task7":
			if "Test BLEU" in line:
				value_map['validation_rewards_line'] = validation_rewards_line
				value_map['test_rewards_line'] = test_rewards_line
				value_map['test_ratios_line'] = test_ratios_line
				test_bleu = line
				test_f1 = lines[idx+3]
		elif task == "task3" or task=="task5" or task=="task6":
			if "Test Accuracy" in line:
				#print(idx, line)
				value_map['validation_rewards_line'] = validation_rewards_line
				value_map['test_rewards_line'] = test_rewards_line
				value_map['test_ratios_line'] = test_ratios_line
				test_acc = line
				test_dial_acc = lines[idx+1]
				test_f1 = lines[idx+2]
				if task!="task6":
					test_oov_acc = lines[idx+4]
					test_oov_dial_acc = lines[idx+5]
					test_oov_f1 = lines[idx+6]
		

	if 'validation_rewards_line' not in value_map:
		print("skipping", file)
		return numbers_map
	
	validation_rewards_line = value_map['validation_rewards_line'] 
	test_rewards_line = value_map['test_rewards_line']
	test_ratios_line = value_map['test_ratios_line']

	if validation_rewards_line == "":
		print("skipping", file)
		return numbers_map

	if validation_rewards_line != "":
		validation_rewards = float((validation_rewards_line.replace("Validation Rewards:", "")).split(' ')[0])
		numbers_map['validation_rewards'] = validation_rewards

	if test_rewards_line != "":
		test_rewards_line = float((test_rewards_line.replace("Test       Rewards:", "")).split(' ')[0])
		numbers_map['test_rewards'] = test_rewards_line

	if test_ratios_line != "":
		test_ratios_line = test_ratios_line.replace("Test", "").strip()
		test_ratios_line_split = test_ratios_line.split(" ")
		valid_ratio = float(test_ratios_line_split[0].replace("Valid-Ratio:", ""))
		perfect_ratio = float(test_ratios_line_split[1].replace("Perfect-Ratio:", ""))
		numbers_map['valid_ratio'] = valid_ratio
		numbers_map['perfect_ratio'] = perfect_ratio

	if task == "task7":
		if test_bleu.strip() != "":
			test_bleu = float(test_bleu.replace("Test BLEU                      : ", "").strip())
			test_f1 = float(test_f1.replace("Test F1                        : ", "").strip())
			numbers_map['test_bleu'] = test_bleu
			numbers_map['test_f1'] = test_f1
	else:

		if test_acc.strip() != "":
			test_acc = float(test_acc.replace("Test Accuracy                  : ", "").strip())
			test_dial_acc = float(test_dial_acc.replace("Test Dialog                    : ", "").strip())
			test_f1 = float(test_f1.replace("Test F1                        : ", "").strip())
			numbers_map['test_acc'] = test_acc
			numbers_map['test_dial_acc'] = test_dial_acc
			numbers_map['test_f1'] = test_f1

		if task!="task6":
			if test_oov_acc.strip() != "":
				test_oov_acc = float(test_oov_acc.replace("Test OOV Accuracy              : ", "").strip())
				test_oov_dial_acc = float(test_oov_dial_acc.replace("Test OOV Dialog                : ", "").strip())
				test_oov_f1 = float(test_oov_f1.replace("Test OOV F1                    : ", "").strip())
				numbers_map['test_oov_acc'] = test_oov_acc
				numbers_map['test_oov_dial_acc'] = test_oov_dial_acc
				numbers_map['test_oov_f1'] = test_oov_f1

	return numbers_map
	
if __name__ == "__main__":

	task = sys.argv[1]
	folder = sys.argv[1]
	
	if task == "task7":
		metrics = ["validation_rewards","test_rewards","valid_ratio","perfect_ratio","test_f1","test_bleu"]
	elif task == "task6":
		metrics = ["validation_rewards","test_rewards","valid_ratio","perfect_ratio",'test_acc', 'test_dial_acc', "test_f1"]
	else:
		metrics = ["validation_rewards","test_rewards","valid_ratio","perfect_ratio",'test_acc', 'test_dial_acc', "test_f1", 'test_oov_acc', 'test_oov_dial_acc', "test_oov_f1"]

	modes = ["GT", "GREEDY", "MAPO", "HYBRID"]
	modes_map = {}
	modes_max_map = {}
	for mode in modes:

		acc_numbers_map = {}

		files = [f for f in os.listdir(folder) if mode in f and '.out' in f]
		for file in files:
			if is_terminated(folder + "/" + file):
				continue
			numbers_map = process_file(folder + "/" + file)
			for key, value in numbers_map.items():
				if key not in acc_numbers_map:
					acc_numbers_map[key] = []
				acc_numbers_map[key].append(value)
		
		metric_map = {}
		max_map = {}
		for key, value in acc_numbers_map.items():
			sum = 0.0
			max = 0.0
			for v in value:
				sum += v
				if v > max:
					max = v
			mean = sum/len(value)
			
			variance = 0
			for v in value:
				variance += math.pow(v-mean, 2)
			variance = variance/len(value)
			sd = math.sqrt(variance)

			metric_map[key] = "{0:0.2f}".format(mean)+"-"+"{0:0.2f}".format(sd)
			max_map[key] = "{0:0.2f}".format(max)
		modes_map[mode] = copy.deepcopy(metric_map)
		modes_max_map[mode] = copy.deepcopy(max_map)

	tabulate_list = []
	for metric in metrics:
		tabulate_entry = []
		tabulate_entry.append(metric)
		for mode in modes:
			if task == "task7":
				if metric in modes_map[mode]:
					tabulate_entry.append(modes_map[mode][metric])
				else:
					tabulate_entry.append(0)
			else:
				if metric in modes_max_map[mode]:
					tabulate_entry.append(modes_max_map[mode][metric])
				else:
					tabulate_entry.append(0)
		tabulate_list.append(tabulate_entry)
	print(tabulate(tabulate_list, headers=["", "GT", "RANDOM", "MAPO", "MIL"]))