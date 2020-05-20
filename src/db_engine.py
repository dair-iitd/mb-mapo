import re
import sys
import copy
import itertools

from operator import itemgetter

class DbEngine(object):

	KEY_INDEX = 0

	valid_conditions = set([])
	invalid_conditions = set([])

	valid_queries = {}

	def __init__(self, kb_file, key_field_name):
		
		self.kb_file = kb_file
		self.key_field_name = key_field_name
		
		if 'babi' in kb_file:
			self._data_set = "babi"
			self._informable_field_names = ["R_cuisine","R_location","R_price","R_number"]
		elif 'camrest' in kb_file:
			self._data_set = "camrest"
			self._informable_field_names = ['R_area', 'R_food','R_pricerange']
		elif 'dstc2' in kb_file:
			self._data_set = "dstc2"
			self._informable_field_names = ['R_cuisine', 'R_location','R_price']
		else:
			print("ERROR: unknown kb file")
			sys.exit()
		
		self._informable_field_values = set([])
		self._all_field_values = set([])
		self._populate_field_name_to_index_map()
		self._read_db()
		self._build_index()

	@property
	def fields(self):
		return self.field_names

	@property
	def informable_field_names(self):
		return self._informable_field_names

	@property
	def entities(self):
		return self._entities
	
	@property
	def requestable_to_informable_slots(self):
		return self._requestable_to_informable_slots

	def _populate_field_name_to_index_map(self):

		field_name_set=set([])
		with open(self.kb_file) as f:
			for line in f:
				line = line.strip()
				line_split = re.split("[ \t]+", line)
				field_name = line_split[2]
				field_name_set.add(field_name)
		
		self.field_names = [""]*(len(field_name_set)+1)
		self._field_name_to_index_map = {}
		self._field_name_to_index_map[self.key_field_name]=DbEngine.KEY_INDEX
		self.field_names[DbEngine.KEY_INDEX] = self.key_field_name
		index=1
		for field_name in field_name_set:
			self._field_name_to_index_map[field_name]=index
			self.field_names[index] = field_name
			index+=1
		#print(self.field_names)
		#sys.exit(0)

	def _read_db(self):
	
		self._db = []
		key_field_to_all_fields_map = {}
		

		total_fields = len(self._field_name_to_index_map)
		entities_set = set([])
		with open(self.kb_file) as f:
			record=[""]*total_fields
			prev_key=""
			for line in f:
				line = line.strip()
				line_split = re.split("[ \t]+", line)
				
				key = line_split[1]
				rel = line_split[2]
				value = line_split[3]
				if key not in key_field_to_all_fields_map:
					key_field_to_all_fields_map[key] = []
				key_field_to_all_fields_map[key].append((rel,value))

				self._all_field_values.add(key)
				self._all_field_values.add(value)
				if rel in self._informable_field_names:
					self._informable_field_values.add(value)
				else:
					self._all_field_values.add(key[::-1])
					self._all_field_values.add(value[::-1])
					
				entities_set.add(value)
				if key != prev_key or record[self._field_name_to_index_map[rel]] != "":
					if record[0] != '':
						self._db.append(copy.deepcopy(record))
						record=[""]*total_fields
					record[0]=key
				record[self._field_name_to_index_map[rel]] = value

				prev_key=key
			self._db.append(copy.deepcopy(record))

		self._entities = []
		for entity in entities_set:
			self._entities.append(entity)

		self._index_to_field_name_map={}
		for key,value in self._field_name_to_index_map.items():
			self._index_to_field_name_map[value]=key

		self._requestable_to_informable_slots = {}
		for key in key_field_to_all_fields_map:
			self._requestable_to_informable_slots[key] = []
			for rel, value in key_field_to_all_fields_map[key]:
				if rel in self._informable_field_names:
					self._requestable_to_informable_slots[key].append(value)
			for rel, value in key_field_to_all_fields_map[key]:
				if rel not in self._informable_field_names:
					self._requestable_to_informable_slots[value] = copy.deepcopy(self._requestable_to_informable_slots[key])

	def _build_index(self):
		
		self._inverted_index = {}
		for value in self._index_to_field_name_map.values():
			self._inverted_index[value]={}

		for db_id in range(len(self._db)):
			record = self._db[db_id]
			for field_id in range(len(record)):
				value = record[field_id]
				if value in self._inverted_index[self._index_to_field_name_map[field_id]]:
					self._inverted_index[self._index_to_field_name_map[field_id]][value].add(db_id)
				else:
					self._inverted_index[self._index_to_field_name_map[field_id]][value] = set([db_id])

	def is_informable_field_value(self, entity):
		if entity in self._informable_field_values:
			return True
		return False
	
	def get_entities_in_utt(self,utt):
		words = utt.split()
		entities = set([])
		informable_entities = set([])
		for word in words:
			if word == "ask":
				continue
			if word in self._all_field_values:
				entities.add(word)
				if word in self._informable_field_values:
					informable_entities.add(word)
		
		return entities, informable_entities

	def is_query_valid(self, query):
		
		if not query.startswith("api_call "):
			return False
		
		if query in DbEngine.valid_queries:
			return DbEngine.valid_queries[query]

		words = query.strip().split()
		
		if (self._data_set == "babi" and len(words) == 5) or (self._data_set == "camrest" and len(words) == 4) or (self._data_set == "dstc2" and len(words) == 4):
			'''
			_, results, _ = self.execute(query)
			if len(results) > 0:
				DbEngine.valid_queries[query] = True
				return True
			'''

			where_clauses = self.get_where_clauses_from_api_call(query)
			for where_clause in where_clauses:
				where_clause_split = where_clause.strip().split(" ")
				field_name = where_clause_split[0].strip()
				value = where_clause_split[1].strip()
				if value not in self._inverted_index[field_name]:
					DbEngine.valid_queries[query] = False
					return False

			DbEngine.valid_queries[query] = True
			return True

		DbEngine.valid_queries[query] = False
		return False
		
	def execute(self, query):

		select_fields = []
		results = []
		result_entities_set = set([])

		try:
			
			record_set = None
			where_clauses = self.get_where_clauses_from_api_call(query)
			for where_clause in where_clauses:
				where_clause_split = where_clause.strip().split(" ")
				field_name = where_clause_split[0].strip()
				value = where_clause_split[1].strip()

				if field_name not in self._inverted_index:
					return results, result_entities_set

				if record_set == None:
					if value in self._inverted_index[field_name]:
						record_set = set(self._inverted_index[field_name][value])
					else:
						record_set = set([])
				else:
					if value in self._inverted_index[field_name]:
						record_set = record_set.intersection(self._inverted_index[field_name][value])
					else:
						record_set = record_set.intersection(set([]))
						
			if record_set != None:
				for index in record_set:
					results.append(self._db[index])

			# the entities are computed after applying the select clause	
			for result in results:
				result_entities_set = result_entities_set.union(set(result))

		except:
			e = sys.exc_info()[0]
			print( "ERROR: ", e )
		finally:
			return select_fields, results, result_entities_set 

	def get_formatted_results(self, select_fields, results):
		formatted_results = []
		if len(select_fields) == 0:
			select_field_indices = [i for i in range(len(self.field_names))]
		else:
			select_field_indices = [self._field_name_to_index_map[field] for field in select_fields]
		for result in results:
			key_value = result[DbEngine.KEY_INDEX]
			for i, field in zip(select_field_indices, result):
				if i != DbEngine.KEY_INDEX:
					formatted_results.append([key_value.lower(), self.field_names[i], field.lower()])
		return formatted_results[:150]

	def modify_non_informable_slots_results(self, results):
		modified_results = []
		if len(results) == 0:
			return results
		else:
			for result in results:
				modified_result = []
				modified_result.append(result[0][::-1])
				modified_result.append(result[1])
				if self.is_informable_field_value(result[2]):
					modified_result.append(result[2])
				else:
					modified_result.append(result[2][::-1])
				modified_results.append(copy.deepcopy(modified_result))
		return modified_results

	def _is_invalid_condition(self, condition):

		if condition in DbEngine.valid_conditions:
			return False
		elif condition in DbEngine.invalid_conditions:
			return True
		else:
			condition_split = condition.strip().split(" ")
			field_name = condition_split[0].strip()
			value = condition_split[1].strip()

			results=[]
			if value in self._inverted_index[field_name]:
				results = self._inverted_index[field_name][value]
			if len(results) > 0:
				DbEngine.valid_conditions.add(condition)
				return False
			else:
				DbEngine.invalid_conditions.add(condition)
				return True

	def get_api_call_from_where_clauses(self, where_clauses):

		where_map = {}
		for where_clause in where_clauses:
			where_clause_split = where_clause.strip().split(" ")
			field_name = where_clause_split[0].strip()
			value = where_clause_split[1].strip()
			where_map[field_name] = value

		if "camrest" in self.kb_file:
			
			# api_call dontcare east expensive
			# index the dontcares for slot specific dontcares
			food  = where_map["R_food"] if "R_food" in where_map.keys() else "dontcare"
			area = where_map["R_area"] if "R_area" in where_map.keys() else "dontcare"
			pricerange  = where_map["R_pricerange"] if "R_pricerange" in where_map.keys() else "dontcare"

			api_call = "api_call " + food + " " + area + " " + pricerange 
		
		elif "babi" in self.kb_file:
			
			# api_call japanese bangkok eight moderate
			cuisine = where_map["R_cuisine"] if "R_cuisine" in where_map.keys() else "dontcare"
			location = where_map["R_location"] if "R_location" in where_map.keys() else "dontcare"
			number = where_map["R_number"] if "R_number" in where_map.keys() else "dontcare"
			price = where_map["R_price"] if "R_price" in where_map.keys() else "dontcare"
			
			api_call = "api_call " + cuisine + " " + location + " " + number + " " + price
		
		elif "dstc2" in self.kb_file:
			
			# api_call vietnamese north cheap
			cuisine = where_map["R_cuisine"] if "R_cuisine" in where_map.keys() else "dontcare"
			location = where_map["R_location"] if "R_location" in where_map.keys() else "dontcare"
			price = where_map["R_price"] if "R_price" in where_map.keys() else "dontcare"
			
			api_call = "api_call " + cuisine + " " + location + " " + price
				
		else:
			print("ERROR: Unknown KB File in DbEngine")
			sys.exit(1)
		
		return api_call

	def get_where_clauses_from_api_call(self, api_call):

		where_clauses = []
		api_call_arr = api_call.strip().split(" ") 
			
		if "camrest" in self.kb_file:
			'''
			if api_call_arr[1] != "dontcare1": where_clauses.append('R_food ' + api_call_arr[1])
			if api_call_arr[2] != "dontcare2": where_clauses.append('R_area ' + api_call_arr[2])
			if api_call_arr[3] != "dontcare3": where_clauses.append('R_pricerange ' + api_call_arr[3])
			'''
			if api_call_arr[1] != "dontcare": where_clauses.append('R_food ' + api_call_arr[1])
			if api_call_arr[2] != "dontcare": where_clauses.append('R_area ' + api_call_arr[2])
			if api_call_arr[3] != "dontcare": where_clauses.append('R_pricerange ' + api_call_arr[3])
			
		elif "babi" in self.kb_file:
			
			'''
			if api_call_arr[1] != "dontcare1": where_clauses.append('R_cuisine ' + api_call_arr[1])
			if api_call_arr[2] != "dontcare2": where_clauses.append('R_location ' + api_call_arr[2])
			if api_call_arr[3] != "dontcare3": where_clauses.append('R_number ' + api_call_arr[3])
			if api_call_arr[4] != "dontcare4": where_clauses.append('R_price ' + api_call_arr[4])
			'''
			if api_call_arr[1] != "dontcare": where_clauses.append('R_cuisine ' + api_call_arr[1])
			if api_call_arr[2] != "dontcare": where_clauses.append('R_location ' + api_call_arr[2])
			if api_call_arr[3] != "dontcare": where_clauses.append('R_number ' + api_call_arr[3])
			if api_call_arr[4] != "dontcare": where_clauses.append('R_price ' + api_call_arr[4])
		
		elif "dstc2" in self.kb_file:
			if api_call_arr[1] != "dontcare" and api_call_arr[1] != "R_cuisine": where_clauses.append('R_cuisine ' + api_call_arr[1])
			if api_call_arr[2] != "dontcare" and api_call_arr[2] != "R_location": where_clauses.append('R_location ' + api_call_arr[2])
			if api_call_arr[3] != "dontcare" and api_call_arr[3] != "R_price": where_clauses.append('R_price ' + api_call_arr[3])
		else:
			
			print("ERROR: Unknown KB File in DbEngine")
			sys.exit(1)
		
		return where_clauses

class QueryGenerator(object):

	Cache = {}

	def __init__(self, dbObject, useOrderBy=False):
		self._dbObject = dbObject
		self._use_order_by = useOrderBy

	def _get_permuatations_till_length_k(self, conditions, k):
	
		permutations_till_k = []
		for r in range(1, k + 1):
			permutations_till_k += copy.deepcopy(list(itertools.permutations(conditions, r)))
		return permutations_till_k

	def _get_combinations_till_length_k(self, conditions, k):
	
		combinations_till_k = []
		for r in range(1, k + 1):
			combinations_till_k += copy.deepcopy(list(itertools.combinations(conditions, r)))
		return combinations_till_k

	def _get_conditions(self,entities):
	
		all_conditions = []
		field_names = self._dbObject.informable_field_names
		field_names.sort()
		
		# should be set to max if anything other than AND conditions are used 
		max_length = min(len(field_names), len(entities))
		
		for k in range(1, max_length+1):
			
			entity_permutations = list(itertools.permutations(entities, k))
			
			# change permutation to combination to avoid repeatitions 
			# (only one of (x and y), (y and x) is added in combinations)
			#field_name_combinations = list(itertools.permutations(field_names, k))
			field_name_combinations = list(itertools.combinations(field_names, k))
			
			total_length = len(entity_permutations)
			#print("\t",total_length)
			for idx, entity_permutation in enumerate(entity_permutations):
				#print("\t\t", idx, total_length)
				for field_name_combination in field_name_combinations:
					assert len(entity_permutation) == len(field_name_combination)
					conditions = []
					invalid_flag = False
					for i in range(len(entity_permutation)):
						condition = (field_name_combination[i] + ' ' + entity_permutation[i])
						if self._dbObject._is_invalid_condition(condition):
							invalid_flag = True
							break
						else:
							conditions.append(condition)

					if not invalid_flag:
						#print(conditions)
						all_conditions.append(conditions)
		
		return all_conditions

	def get_high_recall_queries(self, input_entities, output_entities, train_entities):
		
		high_recall_queries = []
		output_entities_set = set(output_entities)

		
		modified_input_entities = set([])
		for input_entity in input_entities:
			modified_input_entities.add(input_entity)

		for output_entity in output_entities:
			if output_entity in self._dbObject.requestable_to_informable_slots:
				related_informable_entities = self._dbObject.requestable_to_informable_slots[output_entity]
				for related_informable_entity in related_informable_entities:
					if related_informable_entity in train_entities:
						modified_input_entities.add(related_informable_entity)

		#print("input_entities", input_entities)
		#print("output_entities", output_entities)
		#print("modified_input_entities", modified_input_entities)

		where_clauses = self._get_conditions(modified_input_entities)
		alreadyChecked = set([])
		for where_clause in where_clauses:
			query = self._dbObject.get_api_call_from_where_clauses(where_clause)
			if query in alreadyChecked:
				continue
			alreadyChecked.add(query)
			if query in QueryGenerator.Cache:
				result_set = QueryGenerator.Cache[query]
			else:
				_, _, result_set = self._dbObject.execute(query)
				QueryGenerator.Cache[query] = result_set
			if len(result_set)>0:
				# full recall fscore rewards - first line
				# fscore rewards - second line
				if result_set.intersection(output_entities_set) == output_entities_set:
				#if len(result_set.intersection(output_entities_set)) > 0: 
					if result_set.intersection(output_entities_set) != output_entities_set:
						print(query)
						print(result_set.intersection(output_entities_set))
						print(output_entities_set)
						print("")
					high_recall_queries.append(query)

		return high_recall_queries

if __name__ == "__main__":

	
	#kb_file="../data/dialog-bAbI-tasks/dialog-dstc2-kb-all.txt"
	kb_file="../data/dialog-bAbI-tasks/dialog-camrest-kb-all.txt"
	babi_db = DbEngine(kb_file, "R_name")

	query = 'api_call dontcare1 south expensive'
	
	print("Is Query Valid: ", babi_db.is_query_valid(query))
	
	if babi_db.is_query_valid(query):
		select_fields, results, result_entities_set = babi_db.execute(query)
		results = babi_db.get_formatted_results(select_fields, results)
		print("")
		for result in results:
			print(result)
		print("\n-----------------------\n")

	babi_query_generator = QueryGenerator(babi_db, useOrderBy=False)
	input_entities=['mill_road_city_centre', 'bridge_street_city_centre', 'expensive', 'centre', 'afghan', 'turkish', 'king_street_city_centre']
	output_entities=['anatolia', 'moderate', '30_bridge_street_city_centre']
	high_recall_queries = babi_query_generator.get_high_recall_queries(input_entities, output_entities, [])

	for high_recall_query in high_recall_queries:
		print(high_recall_query)
	
	'''
	query = 'api_call dontcare west moderate'
	
	print("Is Query Valid: ", babi_db.is_query_valid(query))
	
	if babi_db.is_query_valid(query):
		select_fields, results, result_entities_set = babi_db.execute(query)
		results = babi_db.get_formatted_results(select_fields, results)
		modified_results = babi_db.modify_non_informable_slots_results(results)
		print("")
		for result in modified_results:
			print(result)
		print("\n-----------------------\n")
	'''
	
	#entities, informable_entities = babi_db.get_entities_in_utt("the dojo_noodle_bar serves asian_oriental food . they are located at 40210_millers_yard_city_centre and their phone number is 01223_363471 .")
	#print(entities)
	#print(informable_entities)

	'''
	babi_query_generator = QueryGenerator(babi_db, useOrderBy=False)
	input_entities=['mill_road_city_centre', 'bridge_street_city_centre', 'italian', 'moderate', 'west', 'turkish', 'king_street_city_centre']
	output_entities=['saint_johns_chop_house_post_code', 'prezzo_address']
	high_recall_queries = babi_query_generator.get_high_recall_queries(input_entities, output_entities, train_entities)

	for high_recall_query in high_recall_queries:
		print(high_recall_query)	
	'''