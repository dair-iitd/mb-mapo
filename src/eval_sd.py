
import os
import sys
import pprint
import math
import copy
import pprint
from tabulate import tabulate
from ast import literal_eval as make_tuple

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
		if "Test OOV F1" in lines[i]:
			last_index = i+1
			break
		if "Test F1" in lines[i]:
			last_index = i+1
			break
	
	return lines[:last_index]

def process_file(file):
	
	#print("Processing", file)
	numbers_map = {}
	lines = [line.strip() for line in open(file)]
	lines = terminate_at_last_test_eval(lines)
	
	validation_rewards_line = ""
	test_rewards_line = ""
	test_ratios_line = ""
	epoch_line = ""

	test_bleu = ""
	test_f1 = ""

	value_map = {}
	for idx, line in enumerate(lines):
		if "Validation Rewards:" in line:
			validation_rewards_line = line
			epoch_line = lines[idx-4]
		if "Test       Rewards:" in line:
			test_rewards_line = lines[idx]
			test_ratios_line = lines[idx+1]
		if "Test BLEU" in line:
			test_bleu = line
			test_f1 = lines[idx+3]

			test_oov_bleu = lines[idx+5]
			test_oov_f1 = lines[idx+8]

	if validation_rewards_line != "":
		validation_rewards = float((validation_rewards_line.replace("Validation Rewards:", "")).split(' ')[0])
		numbers_map['validation_rewards'] = validation_rewards
	#else:
	#	numbers_map['validation_rewards'] = 0

	if test_rewards_line != "":
		test_rewards_line = float((test_rewards_line.replace("Test       Rewards:", "")).split(' ')[0])
		numbers_map['test_rewards'] = test_rewards_line
	#else:
	#	numbers_map['test_rewards'] = 0

	if test_ratios_line != "":
		test_ratios_line = test_ratios_line.replace("Test", "").strip()
		test_ratios_line_split = test_ratios_line.split(" ")
		valid_ratio = float(test_ratios_line_split[0].replace("Valid-Ratio:", ""))
		perfect_ratio = float(test_ratios_line_split[1].replace("Perfect-Ratio:", ""))
		numbers_map['valid_ratio'] = valid_ratio
		numbers_map['perfect_ratio'] = perfect_ratio
		numbers_map['best_test_epoch'] = int(epoch_line.split(" ")[1])
	#else:
	#	numbers_map['valid_ratio'] = 0
	#	numbers_map['perfect_ratio'] = 0

	if test_bleu != "":
		
		test_bleu = float(test_bleu.replace("Test BLEU                      : ", "").strip())
		test_f1_metrics = make_tuple(test_f1.replace("Test F1", "").strip())
		numbers_map['test_bleu'] = test_bleu

		test_oov_bleu = float(test_oov_bleu.replace("Test OOV BLEU                  : ", "").strip())
		test_oov_f1_metrics = make_tuple(test_oov_f1.replace("Test OOV F1 ", "").strip())
		numbers_map['test_oov_bleu'] = test_oov_bleu
		
		#((macro_pr, macro_re, macro_f1), 
		# (macro_context_pr, macro_context_re, macro_context_f1), 
		# (macro_kb_pr, macro_kb_re, macro_kb_f1))
		numbers_map['test_pr'] = test_f1_metrics[0][0]
		numbers_map['test_re'] = test_f1_metrics[0][1]
		numbers_map['test_f1'] = test_f1_metrics[0][2]
		#numbers_map['test_f1_1'] = (2*test_f1_metrics[0][0]*test_f1_metrics[0][1])/(test_f1_metrics[0][0]+test_f1_metrics[0][1]) if test_f1_metrics[0][1]+test_f1_metrics[0][0] > 0 else 0

		numbers_map['test_context_pr'] = test_f1_metrics[1][0]
		numbers_map['testre'] = test_f1_metrics[1][1]
		numbers_map['test_context_f1'] = test_f1_metrics[1][2]
		#numbers_map['test_context_f1_1'] = (2*test_f1_metrics[1][0]*test_f1_metrics[1][1])/(test_f1_metrics[1][0]+test_f1_metrics[1][1]) if test_f1_metrics[1][1]+test_f1_metrics[1][0] > 0 else 0

		numbers_map['test_kb_pr'] = test_f1_metrics[2][0]
		numbers_map['test_kb_re'] = test_f1_metrics[2][1]
		numbers_map['test_kb_f1'] = test_f1_metrics[2][2]
		#numbers_map['test_kb_f1_1'] = (2*test_f1_metrics[2][0]*test_f1_metrics[2][1])/(test_f1_metrics[2][0]+test_f1_metrics[2][1]) if test_f1_metrics[2][1]+test_f1_metrics[2][0] > 0 else 0

		numbers_map['test_oov_pr'] = test_oov_f1_metrics[0][0]
		numbers_map['test_oov_re'] = test_oov_f1_metrics[0][1]
		numbers_map['test_oov_f1'] = test_oov_f1_metrics[0][2]
		#numbers_map['test_oov_f1_1'] = (2*test_oov_f1_metrics[0][0]*test_oov_f1_metrics[0][1])/(test_oov_f1_metrics[0][0]+test_oov_f1_metrics[0][1]) if test_oov_f1_metrics[0][1]+test_oov_f1_metrics[0][0] > 0 else 0


		numbers_map['test_oov_context_pr'] = test_oov_f1_metrics[1][0]
		numbers_map['test_oov_context_re'] = test_oov_f1_metrics[1][1]
		numbers_map['test_oov_context_f1'] = test_oov_f1_metrics[1][2]
		#numbers_map['test_oov_context_f1_1'] = (2*test_oov_f1_metrics[1][0]*test_oov_f1_metrics[1][1])/(test_oov_f1_metrics[1][0]+test_oov_f1_metrics[1][1]) if test_oov_f1_metrics[1][1]+test_oov_f1_metrics[1][0] > 0 else 0

		numbers_map['test_oov_kb_pr'] = test_oov_f1_metrics[2][0]
		numbers_map['test_oov_kb_re'] = test_oov_f1_metrics[2][1]
		numbers_map['test_oov_kb_f1'] = test_oov_f1_metrics[2][2]
		#numbers_map['test_oov_kb_f1_1'] = (2*test_oov_f1_metrics[2][0]*test_oov_f1_metrics[2][1])/(test_oov_f1_metrics[2][0]+test_oov_f1_metrics[2][1]) if test_oov_f1_metrics[2][1]+test_oov_f1_metrics[2][0] > 0 else 0


	return numbers_map
	
if __name__ == "__main__":
	
	folder = sys.argv[1]
	print_count = False
	print_perfect_ratios = False
	api_folder_prefix = ""
	api_folder_suffix = ""
	remote_machine = ""
	if (len(sys.argv) == 3 and sys.argv[2] == "c"):
		print_count = True
	if (len(sys.argv) > 3):
		logs_path = sys.argv[2]
		for i in range(3,len(sys.argv)):
			if sys.argv[i] == "p":
				print_perfect_ratios = True
				remote_machine = "scp diraghu1@dccxl012.pok.ibm.com:"
				if "7" in folder:
					api_folder_prefix = "/dccstor/dineshwcs/" + logs_path+ "/logs/api/task7_dialog-bAbI-tasks_lr-0.0005_hops-6_emb-size-256_sw-1_wd-0.1_pw-1.0_rlmode-" 
					api_folder_suffix = "_pi_b-0.6"
				if "6" in folder:
					api_folder_prefix = "/dccstor/dineshwcs/" + logs_path+ "/logs/api/task6_dialog-bAbI-tasks_lr-0.0025_hops-6_emb-size-256_sw-1_wd-0.1_pw-1.0_rlmode-" 
					api_folder_suffix = "_pi_b-0.6"
			if sys.argv[i] == "c":
				print_count = True

	#metrics = ["validation_rewards","test_rewards","valid_ratio","perfect_ratio", "test_f1", "test_f1_1", "test_context_f1", "test_context_f1_1", "test_kb_f1", "test_kb_f1_1","test_bleu", "test_oov_f1", "test_oov_f1_1", "test_oov_context_f1", "test_oov_context_f1_1", "test_oov_kb_f1", "test_oov_kb_f1_1","test_oov_bleu"]
	metrics = ["validation_rewards","test_rewards","valid_ratio","perfect_ratio", "test_f1", "test_context_f1", "test_kb_f1","test_bleu",  "test_oov_f1", "test_oov_context_f1", "test_oov_kb_f1","test_oov_bleu"]
	modes = ["GT", "SL", "RL","MAPO", "HYBRID"]
	#modes = ["HYBRIDCA", "HYBRIDUR", "HYBRID"]
	
	modes_map = {}
	ssh_commands = "\n"
	for mode in modes:

		acc_numbers_map = {}

		files = [f for f in os.listdir(folder) if "train."+mode+".lr" in f and '.out' in f]
		for file in files:
			if is_terminated(folder + "/" + file):
				print("terminated", file)
				continue
			numbers_map = process_file(folder + "/" + file)
			for key, value in numbers_map.items():
				if print_perfect_ratios and key == 'perfect_ratio':
					print(file, "\t", value, "\tEpoch",numbers_map['best_test_epoch'])
					epoch = numbers_map['best_test_epoch']

					idx_index = file.index(".i.")
					next_dot = file.index(".",idx_index+3)
					idx = file[idx_index+3:next_dot]
					api_folder_middle = mode+"_idx-"+idx
					api_folder = api_folder_prefix + api_folder_middle + api_folder_suffix
					ssh_commands_for_file = ""
					ssh_commands_for_file += file + "\n"
					ssh_commands_for_file += "------------"+ "\n"
					ssh_commands_for_file += remote_machine + "\""
					ssh_commands_for_file += api_folder + "/" +str(epoch) +"-trn.log "
					ssh_commands_for_file += api_folder + "/" +str(epoch) +"-val.log "
					ssh_commands_for_file += api_folder + "/tst-" +str(epoch) +".log "
					ssh_commands_for_file += api_folder + "/tst-OOV-" +str(epoch) +".log\" ."+"\n"
					ssh_commands_for_file += "\n"
					ssh_commands += ssh_commands_for_file
				if key == "best_test_epoch":
					continue
				if key not in acc_numbers_map:
					acc_numbers_map[key] = []
				acc_numbers_map[key].append(value)
		
		if print_perfect_ratios:
			continue

		metric_map = {}
		for key, value in acc_numbers_map.items():
			sum = 0.0
			for v in value:
				sum += v
			mean = sum/len(value)
			
			variance = 0
			for v in value:
				variance += math.pow(v-mean, 2)
			variance = variance/len(value)
			sd = math.sqrt(variance)
			
			if print_count:
				metric_map[key] = "{0:0.2f}".format(mean)+"-"+"{0:0.2f}".format(sd)+" (" + str(len(value)) + ")"
			else:
				metric_map[key] = "{0:0.2f}".format(mean)+"-"+"{0:0.2f}".format(sd)
		modes_map[mode] = copy.deepcopy(metric_map)

	if print_perfect_ratios:
		print(ssh_commands)
		exit(0)

	tabulate_list = []
	for metric in metrics:
		tabulate_entry = []
		tabulate_entry.append(metric)
		for mode in modes:
			if metric in modes_map[mode]:
				tabulate_entry.append(modes_map[mode][metric])
			else:
				if print_count:
					tabulate_entry.append("0 (0)")
				else:
					tabulate_entry.append("0")
		tabulate_list.append(tabulate_entry)
	print(tabulate(tabulate_list, headers=[""] + modes))