import sys
sys.path.append("..")

import re
import copy
import json
from db_engine import DbEngine, QueryGenerator

entities=set([])
max_total_high_recall_queries = 0

## Camrest - has too many inconsistent entity annotations
#            after the convertion process "-modified" files will be created
#            Please use these "-modified" files and delete the others

## babi3 - does not have api_calls in between
#          after the convertion process "-with-api" files will be created
#          Please use these "-with-api" files and delete the others

## dstc2

# set to babi or camrest
dataset="dstc2"
# for babi, set the task
task=3

find_replace={}

def populate_find_replace():
	
	# price range
	find_replace['moderately'] ='moderate'
	find_replace['expensively']='expensive'

	# cuisine
	find_replace['asian oriental']='asian_oriental'
	find_replace['north american']='north_american'

	# phone
	find_replace['017-335-3355']='01733_553355'
	find_replace['01223 301030']='01223_307030'
	find_replace['01223323178'] ='01223_323178'
	find_replace['01223315232'] ='01223_362456 '
	find_replace['01223413000'] ='01223_413000'

	# location
	find_replace['eastern']='east'
	find_replace['southern']='south'
	find_replace['center']='centre'

	# restaurants
	find_replace['efes_restaurants']='efes_restaurant'
	
	# address
	find_replace['59 hills road city']='59_hills_road_city_centre'
	find_replace['108 regent_street_city_centre']='108_regent_street_city_centre'
	find_replace['139 huntingdon_road_city_centre']='cambridge_lodge_hotel_139_huntingdon_road_city_centre'
	find_replace['21-24 northampton road']='21_-_24_northampton_road'
	find_replace['205 victoria road , chesterton']='205_victoria_road_chesterton'
	find_replace['21-24 northampton street']='21_-_24_northampton_street'
	find_replace['grafton hotel 619 newmarket_road_fen_ditton']='grafton_hotel_619_newmarket_road_fen_ditton'
	find_replace['14 - 16 bridge street']='14_-16_bridge_street'
	find_replace['cambridge_leisure_park_clifton_way cherry hinton']='cambridge_leisure_park_clifton_way_cherry_hinton'
	find_replace['619 newmarket_road_fen_ditton']='grafton_hotel_619_newmarket_road_fen_ditton'
	find_replace['20 milton road , chesterton']='20_milton_road_chesterton'
	find_replace['290 mill_road_city_centre']='290_mill_road_city_centre'
	find_replace['196 mill_road_city_centre']='196_mill_road_city_centre'
	find_replace['100 mill_road_city_centre']='100_mill_road_city_centre'
	find_replace['32 bridge_street_city_centre']='32_bridge_street_city_centre'
	find_replace['169 high street chesterton']='169_high_street_chesterton_chesterton'
	find_replace['84 regent_street_city_centre']='84_regent_street_city_centre'
	find_replace['290 mill road city']='290_mill_road_city_centre'
	find_replace['290 mill_road_city_centre']='290_mill_road_city_centre'
	find_replace['3-34 saint andrews street']='33-34_saint_andrews_street'
	find_replace['22 chesterton road']='22_chesterton_road_chesterton'
	find_replace['280 mill_road_city_centre']='290_mill_road_city_centre'
	find_replace['108 regent street']='108_regent_street_city_centre'
	find_replace['hotel 139 huntingdon_road_city_centre']='cambridge_lodge_hotel_139_huntingdon_road_city_centre'
	find_replace['35 sanit andrews street city centre']='35_saint_andrews_street_city_centre'
	find_replace['24 green street city']='24_green_street_city_centre'
	find_replace['52 mill_road_city_centre']='52_mill_road_city_centre'
	find_replace['152-154 hills road']='152_-_154_hills_road'
	find_replace['24 green street city centr']='24_green_street_city_centre'

	# pincodes
	find_replace["c.b 5 , 8 p a"]="cb58pa"
	find_replace["c.b 2 , 1.d p"]="cb21dp"
	find_replace["c b 25 , 9 a q"]="cb259aq"
	find_replace["c.b.1 7 d.y"]="cb17dy"
	find_replace["c.b . 1 7 d.y"]="cb17dy"
	find_replace["c.b 4 , 3.h l"]="cb43hl"
	find_replace["c.b 4 , 1.uy"]="cb41uy"
	find_replace["c.b 1,2 q.a"]="cb12qa"
	find_replace["c.b2 , 1 u.f"]="cb21uf"

	# mix
	find_replace['cheap european']='cheap modern_european'

# Module to clean up utterances CamRest dataset
def pre_process(utt):
	for key in find_replace.keys():
		if key in utt:
			utt = utt.replace(key, find_replace[key])
	return utt

def load_kb_entities(kb_file):
	print("Reading KB File...")
	with open(kb_file) as f:
		for line in f:
			line = line.strip()
			line_split = re.split("[ \t]+", line)

			entity_1=line_split[1]
			entity_2=line_split[3]
			
			entities.add(entity_1)
			entities.add(entity_2)

	# Missing enitities in CamRest
	if dataset == "camrest":
		entities.add("indonesian")
		entities.add("catalan")
		entities.add("scandinavian")
		entities.add("panasian")
		entities.add("scottish")
		entities.add("australasian")
		entities.add("brazilian")
		entities.add("austrian")
		entities.add("swedish")
		entities.add("belgian")
		entities.add("danish")
		entities.add("english")
		entities.add("moroccan")
		entities.add("australian")
		entities.add("venetian")
		entities.add("malaysian")
		entities.add("hungarian")
		entities.add("vegetarian")
		entities.add("persian")
		entities.add("romanian")
		entities.add("russian")
		entities.add("irish")
		entities.add("polish")
		entities.add("cantonese")
		entities.add("traditional")
		entities.add("world")
		entities.add("swiss")
		entities.add("caribbean")
		entities.add("american")
		entities.add("corsica")
		entities.add("welsh")
		entities.add("kosher")
		entities.add("cuban")
		entities.add("jamaican")
		entities.add("bistro")
		entities.add("halal")
		entities.add("singaporean")
		entities.add("christmas")
		entities.add("canapes")
		entities.add("eritrean")
		entities.add("tuscan")
		entities.add("greek")
		entities.add("unusual")
		entities.add("fusion")
		entities.add("crossover")
		entities.add("afghan")
		entities.add("german")
		entities.add("barbeque")
		entities.add("inexpensive")
		
def get_entities(str):
	words = str.split(" ")
	str_entities=[]
	for word in words:
		if word in entities:
			str_entities.append(word)
	return str_entities

# After a complete dialog is read, annotate:
#  1. entities_so_far
#  2. next_new_kb_entity_distance
#  3. next_entities
		
def update_kb_links(dialog):
	
	turns = dialog['turns']
	
	entities_in_user=[]
	entities_in_agent=[]
	entities_so_far=set([])
	next_kb_entities_map={}
 
	for i in range(len(turns)):
		turn = turns[i]
		user_entities=turn['user'].get('kb_entities',[])
		entities_in_user.append(user_entities)
		entities_so_far |= set(user_entities)
		
		agent_entities=turn['agent'].get('kb_entities',[])
		new_agent_entities=[]
		for agent_entity in agent_entities:
			if agent_entity  not in entities_so_far:
				new_agent_entities.append(agent_entity)
		
		if len(new_agent_entities)>0:
			next_kb_entities_map[i]=new_agent_entities

		entities_in_agent.append(agent_entities)
		entities_so_far |= set(agent_entities)

	entities_so_far=set([])
	for i in range(len(entities_in_user)):
		entities_so_far |= set(turns[i]['user'].get('kb_entities',[]))
		turns[i]['entities_so_far'] = copy.deepcopy(list(entities_so_far))
		entities_so_far |= set(turns[i]['agent'].get('kb_entities',[]))
		
		entity_distance_logged=False
		next_remaining_entities=[]

		for j in range(i,len(entities_in_agent)):
			if not entity_distance_logged and len(entities_in_agent[j]) > 0:
				turns[i]['next_new_kb_entity_distance']=j-i
				entity_distance_logged=True
			if not entity_distance_logged and j == len(entities_in_agent)-1:
				turns[i]['next_new_kb_entity_distance']=-1
				entity_distance_logged=True
			
			if j in next_kb_entities_map:
					next_remaining_entities+=next_kb_entities_map[j]

		turns[i]['next_entities']=next_remaining_entities
	
	'''
	for i in range(len(turns)):
		turn = turns[i]
		turn['user']=turn['user']['utt']
		turn['agent']=turn['agent']['utt']
	'''
	
	return dialog

def convert_file(input_file, output_file, queryGenerator):
	
	global max_total_high_recall_queries

	corpus=[]
	dialog_id=1

	turns=[]
	turn_id=1
	dialog={}
	dialog['dialog_id']=dialog_id

	prev_user_utt=""
	api_call_flag=False
	api_call_str=""

	modified_file_contents = ""
	
	with open(input_file) as f:
		lines = f.readlines()
		for i in range(0, len(lines)):
			line=lines[i].strip()
			if line:
				line_no, line = line.split(' ', 1)
				if '\t' in line:
					u, r = line.split('\t')
					if dataset == "camrest" or dataset == "dstc2":
						u = pre_process(u)
						r = pre_process(r)

						modified_file_contents += line_no + " " + u + "\t" + r + "\n"
					else:
						modified_file_contents += line_no + " " + line + "\n"
						
					if r.startswith("api_call"):
						api_call_flag=True
						# Comment this if dontcare is slot specific
						'''
						api_call_words = r.split()
						api_call_str = ""
						for idx, api_call_word in enumerate(api_call_words):
							if api_call_word == 'dontcare':
								api_call_str = api_call_str + " dontcare" + str(idx)
							else:
								api_call_str = api_call_str + " " + api_call_word
						api_call_str = api_call_str.strip()
						'''

						# in dstc2 the field name is specified instead of dontcare
						# this snippet changes them to dontcare
						api_call_words = r.split()
						api_call_str = ""
						for idx, api_call_word in enumerate(api_call_words):
							if "R_" in api_call_word:
								api_call_str = api_call_str + " dontcare"
							else:
								api_call_str = api_call_str + " " + api_call_word
						api_call_str = api_call_str.strip()

						#api_call_str = r.strip()
						
						prev_user_utt=u
						continue
					turn = {}
					
					turn['user'] = {}
					if api_call_flag and (dataset == "camrest" or dataset == "dstc2"):
						u = prev_user_utt
					turn['user']['utt']=u
					kb_entities=get_entities(u)
					if len(kb_entities)>0:
						turn['user']['kb_entities']=kb_entities
					
					turn['agent']={}
					turn['agent']['utt']=r
					if api_call_flag==True:
						if dataset == "camrest" or dataset == "dstc2" or (dataset == "babi" and  u == "<SILENCE>"):
							turn['make_api_call']=True
							turn['api_call']=api_call_str
						else:
							turn['make_api_call']=False
							turn['api_call']=''
						api_call_flag=False
					else:
						turn['make_api_call']=False
						turn['api_call']=''

					kb_entities=get_entities(r)
					if len(kb_entities)>0:
						turn['agent']['kb_entities']=kb_entities
						
					turn['turn_id']=turn_id
					turn_id=turn_id+1
					turns.append(turn)
				else:
					modified_file_contents += line_no + " " + line + "\n"
			else:
				
				dialog['turns']=turns
				dialog=update_kb_links(dialog)
				corpus.append(copy.deepcopy(dialog))
				dialog_id=dialog_id+1
				turns=[]
				turn_id=1
				dialog={}
				dialog['dialog_id']=dialog_id
				modified_file_contents += "\n"
		
		if len(turns)>0:
			dialog['turns']=turns
			dialog=update_kb_links(dialog)
			corpus.append(copy.deepcopy(dialog))
		
	# for each dialog, generate high recall actions
	count = 1
	total = len(corpus)
	for dialog in corpus:
		print(count,"/",total)
		count+=1
		turns = dialog['turns']
		for turn in turns:
			if turn['make_api_call']==True:
				input_entities = turn['entities_so_far']
				output_entities = turn['next_entities']
				high_recall_queries = queryGenerator.get_high_recall_queries(input_entities, output_entities)
				if len(high_recall_queries) > max_total_high_recall_queries:
					max_total_high_recall_queries = len(high_recall_queries)
					print('input_entities', input_entities)
					print('output_entities', output_entities)
					for high_recall_query in high_recall_queries:
						print("\t", len(high_recall_queries), high_recall_query)
				turn['high_recall_queries']=high_recall_queries
			else:
				turn['high_recall_queries']=[]

	with open(output_file, 'w') as outfile:
		json.dump(corpus, outfile)
	
	modified_file = open(input_file.replace(".txt", "-modified.txt"), "w")
	modified_file.write(modified_file_contents)
	modified_file.close()

def insert_api_calls(input_file, output_file):
	
	outfile = open(output_file, "w")
	api_conditions={}
	db_results = []
	api_added_flag = False
	lineNo=1
	
	with open(input_file) as f:
		lines = f.readlines()
		for i in range(0, len(lines)):
			line=lines[i].strip()
			if line:
				_, line = line.split(' ', 1)
				if '\t' in line:
					if not api_added_flag and 'resto_' in line:
						# api_call vietnamese seoul eight expensive
						api_call = "api_call " + api_conditions['R_cuisine'] + " " + api_conditions['R_location'] + " " +api_conditions['R_number'] + " " + api_conditions['R_price']
						outfile.write(str(lineNo) + " " + "<SILENCE>\t" + api_call + "\n")
						lineNo += 1
						for db_result in db_results:
							outfile.write(str(lineNo) + " " + db_result + "\n")
							lineNo += 1
						api_added_flag = True
					outfile.write(str(lineNo) + " " + line + "\n")
					lineNo += 1
						
				else:
					db_results.append(line)
					value = line.split(' ')[2]
					if "R_cuisine" in line:
						api_conditions['R_cuisine'] = value
					elif "R_location" in line:
						api_conditions['R_location'] = value
					elif "R_number" in line:
						api_conditions['R_number'] = value
					elif "R_price" in line:
						api_conditions['R_price'] = value	
			else:
				api_conditions={}
				db_results = []
				api_added_flag = False
				lineNo=1
				outfile.write("\n")

	outfile.close()

def remove_dstc2_dialogs(input_file, output_file):
	
	outfile = open(output_file, "w")
	
	dialogue = ""
	total = 0
	with open(input_file) as f:
		lines = f.readlines()
		for i in range(0, len(lines)):
			line=lines[i].strip()
			if line:
				dialogue += line + "\n"	
			else:
				count = dialogue.count("api_call")
				if count <= 1:
					outfile.write(dialogue)
					outfile.write("\n")
					total += 1
				dialogue = ""
	
	if dialogue != "":
		count = dialogue.count("api_call")
		if count <= 1:
			outfile.write(dialogue)
			outfile.write("\n")
			total += 1

	print(total, input_file)
	outfile.close()

if __name__ == "__main__":

	input_folder = "../../data/dialog-bAbI-tasks/"
	files = ['tst.txt', 'trn.txt', 'dev.txt']
	
	if dataset=='babi':
		files.append('tst-OOV.txt')
		if task == 5:
			input_prefix = 'dialog-babi-task5-full-dialogs-'
			output_folder = input_folder + "task5/"
			kb_file = input_folder+'dialog-babi-kb-all.txt'
		else:
			input_prefix_no_api = 'dialog-babi-task3-options-'
			input_prefix =  'dialog-babi-task3-options-with-api-'
			output_folder = input_folder + "task3/"
			for file in files:
				insert_api_calls(input_folder+input_prefix_no_api+file, input_folder+input_prefix+file)
			kb_file = input_folder+'dialog-babi-kb-task3.txt'
		output_prefix = 'dialog-babi-'
		load_kb_entities(kb_file)
	elif dataset=="dstc2":
		populate_find_replace()
		input_prefix_unfiltered = 'dialog-babi-task6-dstc2-'
		input_prefix = 'dialog-babi-task6-dstc2-filtered-'
		for file in files:
			remove_dstc2_dialogs(input_folder+input_prefix_unfiltered+file, input_folder+input_prefix+file)
		output_prefix = 'dialog-babi-task6-dstc2-'
		output_folder = input_folder + "task6/"
		kb_file = input_folder+'dialog-dstc2-kb-all.txt'
		load_kb_entities(kb_file)
	else:
		populate_find_replace()
		input_prefix = 'dialog-babi-task7-camrest676-'
		output_prefix = 'dialog-camrest-'
		output_folder = input_folder + "task7/"
		kb_file = input_folder+'dialog-camrest-kb-all.txt'
		load_kb_entities(kb_file)
	
	dbEngine = DbEngine(kb_file, "R_name")
	queryGenerator = QueryGenerator(dbEngine, useOrderBy=False)

	for file in files:
		print("Processing", file)
		convert_file(input_folder+input_prefix+file, output_folder+output_prefix+file.replace(".txt",".json"), queryGenerator)
	
	print("max high recall queries", max_total_high_recall_queries)