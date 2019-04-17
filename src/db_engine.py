import re
import sys
import copy
import itertools

from operator import itemgetter

####
# Currently supports:
#  1) SELECT - > * and combination of field names
#  2) WHERE -> equality comparision and combining conditions with boolean AND
#  3) ORDER BY -> only supports single variable
#
# To extend support 
#   1) update the regex pattern (self._pattern) that checks if a query is valid 
#   2) update the execute() method
#   3) update the QueryGenerator
####

TABLE_NAME = "table"
SELECT_ALL_CLAUSE = "SELECT * FROM " + TABLE_NAME + " "

class DbEngine(object):

	KEY_INDEX = 0

	valid_conditions = set([])
	invalid_conditions = set([])

	def __init__(self, kb_file, key_field_name):
		
		self.kb_file = kb_file
		self.key_field_name = key_field_name
		
		self._populate_field_name_to_index_map()
		self._read_db()
		self._build_index()

	@property
	def fields(self):
		return self.field_names

	@property
	def entities(self):
		return self._entities

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

	def _read_db(self):
	
		self._db = []
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
				entities_set.add(value)
				#if key != prev_key or record[self._field_name_to_index_map[rel]] != "":
				if key != prev_key:
					if record[0] != '':
						self._db.append(copy.deepcopy(record))
						#record=[""]*total_fields
					record[0]=key
				record[self._field_name_to_index_map[rel]] = value

				prev_key=key
			self._db.append(copy.deepcopy(record))

		self._entities = []
		for entity in entities_set:
			self._entities.append(entity)

		field_names = ""
		self._index_to_field_name_map={}
		for key,value in self._field_name_to_index_map.items():
			self._index_to_field_name_map[value]=key
			if field_names == "":
				field_names = "("+key
			else:
				field_names += "|"+key
		field_names += ")"

		select_clause_regex = "SELECT (\*|((" + field_names + ")( , " + field_names + ")*)) FROM " + TABLE_NAME
		
		where_condition_regex = field_names + ' [^( |(AND))]*'
		where_clause_regex = " WHERE " + where_condition_regex +  "( AND " + where_condition_regex + ")*"
		
		order_by_condition_regex = field_names + ' (ASC|DESC)'
		order_by_clause_regex = " ORDER BY " + order_by_condition_regex
		
		self._pattern = re.compile(
								select_clause_regex
								+ where_clause_regex
								+ "(" + order_by_clause_regex + ")*" # ORDER BY is optional
							)
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

	def is_query_valid(self, query):
		result = self._pattern.search(query)
		if result != None and result.end() - result.start() == len(query):
			return True
		else:
			return False
		
	def execute(self, query):

		select_fields = []
		results = []
		result_entities_set = set([])

		try:
			select_clause = query[(query.index("SELECT ") + len("SELECT ")):query.index(" FROM")]
			where_index = query.index(" WHERE ")
			if "ORDER BY" in query:
				order_by_index = query.index(" ORDER BY ")
				where_clauses = query[where_index + len(" WHERE "):order_by_index].split("AND")
			else:
				where_clauses = query[where_index + len(" WHERE "):].split("AND")

			record_set = None			
			for where_clause in where_clauses:
				#where_clause = where_clause.strip().replace("\"","")
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

			'''
			# sorted it based on the ORDER BY clause
			# for now only one variable condition is allowed
			if "ORDER BY" in query:
				order_by_clause = query[order_by_index+ len(" ORDER BY "):]
				order_by_clause_split = order_by_clause.split(" ")
				sort_index = self._field_name_to_index_map[order_by_clause_split[0]]
				reverse_flag = False
				if len(order_by_clause_split) == 2 and order_by_clause_split[1] == "DESC":
					reverse_flag = True
				results = sorted(results, key=itemgetter(sort_index),reverse=reverse_flag)
			
			if select_clause != "*":
				select_applied_results = []
				select_fields = select_clause.split(" , ")
				select_field_indices = [self._field_name_to_index_map[field] for field in select_fields]
				for result in results:
					select_applied_results.append([result[i] for i in select_field_indices])
				results = select_applied_results
			'''

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
		return formatted_results

	def _is_invalid_condition(self, condition):

		if condition in DbEngine.valid_conditions:
			return False
		elif condition in DbEngine.invalid_conditions:
			return True
		else:
			_, results, _ = self.execute(SELECT_ALL_CLAUSE+ "WHERE " + condition)
			if len(results) > 0:
				DbEngine.valid_conditions.add(condition)
				return False
			else:
				DbEngine.invalid_conditions.add(condition)
				return True
	
	def get_query_in_api_format(self, query):
		
		if not self.is_query_valid(query):
			return ''
		
		where_index = query.index(" WHERE ")
		if "ORDER BY" in query:
			order_by_index = query.index(" ORDER BY ")
			where_clauses = query[where_index + len(" WHERE "):order_by_index].split("AND")
		else:
			where_clauses = query[where_index + len(" WHERE "):].split("AND")
		
		where_map = {}
		for where_clause in where_clauses:
			#where_clause = where_clause.strip().replace("\"","")
			where_clause_split = where_clause.strip().split(" ")
			field_name = where_clause_split[0].strip()
			value = where_clause_split[1].strip()
			where_map[field_name] = value
		
		if self.kb_file.endswith("camrest-kb-all.txt"):
			
			# api_call dontcare east expensive
			area = where_map["R_area"] if "R_area" in where_map.keys() else "dontcare"
			food  = where_map["R_food"] if "R_food" in where_map.keys() else "dontcare"
			pricerange  = where_map["R_pricerange"] if "R_pricerange" in where_map.keys() else "dontcare"

			api_call = "api_call " + food + " " + area + " " + pricerange 
		
		elif (self.kb_file.endswith("babi-kb-all.txt") or self.kb_file.endswith("babi-kb-task3.txt") or self.kb_file.endswith("babi-kb-task3-fabricated.txt")):
			
			if "R_cuisine" not in where_map or "R_location" not in where_map or "R_number" not in where_map or "R_price" not in where_map:
				api_call = ''
			else:
				# api_call japanese bangkok eight moderate
				cuisine = where_map["R_cuisine"]
				location = where_map["R_location"] 
				number = where_map["R_number"]
				price = where_map["R_price"]
				
				api_call = "api_call " + cuisine + " " + location + " " + number + " " + price
				
		else:
			print("ERROR: Unknown KB File in DbEngine")
			sys.exit(1)
		
		return api_call

	def get_api_call_in_sql_query_format(self, api_call):

		query = SELECT_ALL_CLAUSE + "WHERE "
		where_clauses = []

		if self.kb_file.endswith("camrest-kb-all.txt"):
			
			api_call_arr = api_call.strip().split(" ") 
			food = api_call_arr[1]
			area = api_call_arr[2]
			pricerange = api_call_arr[3]
			if food != "dontcare": where_clauses.append('R_food ' + food)
			if area != "dontcare": where_clauses.append('R_area ' + area)
			if pricerange != "dontcare": where_clauses.append('R_pricerange ' + pricerange)
		
		elif (self.kb_file.endswith("babi-kb-all.txt") or self.kb_file.endswith("babi-kb-task3.txt") or self.kb_file.endswith("babi-kb-task3-fabricated.txt")):
			
			api_call_arr = api_call.strip().split(" ")
			where_clauses.append('R_cuisine ' + api_call_arr[1])
			where_clauses.append('R_location ' + api_call_arr[2])
			where_clauses.append('R_number ' + api_call_arr[3])
			where_clauses.append('R_price ' + api_call_arr[4])
		
		else:
			
			print("ERROR: Unknown KB File in DbEngine")
			sys.exit(1)
		
		query += " AND ".join(where_clauses)

		#print("api_call:", api_call)
		#print("query   :", query)
		
		return query

class QueryGenerator(object):

	Cache = {}

	def __init__(self, dbObject, useOrderBy=False):
		self._dbObject = dbObject
		self._select_clauses = self._get_select_clauses()
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

	def _get_select_clauses(self):

		select_clauses = []
		select_clauses.append(SELECT_ALL_CLAUSE)
		return select_clauses
		'''

		field_names = self._dbObject.field_names
		field_names.sort()
		all_select_clauses_fields = self._get_combinations_till_length_k(field_names, len(field_names))
		for select_clauses_fields in all_select_clauses_fields:
			if "R_name" in select_clauses_fields:
				select_clauses.append("SELECT " + " , ".join(select_clauses_fields) + " FROM " + TABLE_NAME + " ")
		
		return select_clauses
		'''
	def _get_conditions(self,entities):
	
		all_conditions = []
		field_names = self._dbObject.field_names
		field_names.sort()
		
		# should be set to max if anything other than AND conditions are used 
		max_length = min(len(field_names), len(entities))
		
		for k in range(1, max_length+1):
			
			entity_permutations = list(itertools.permutations(entities, k))
			
			# change permutation to combination to avoid repeatitions 
			# (only one of (x and y), (y and x) is added in combinations)
			#field_name_combinations = list(itertools.permutations(field_names, k))
			field_name_combinations = list(itertools.combinations(field_names, k))
			
			for entity_permutation in entity_permutations:
				for field_name_combination in field_name_combinations:
					assert len(entity_permutation) == len(field_name_combination)
					conditions = ""
					invalid_flag = False
					for i in range(len(entity_permutation)):
						if i > 0:
							conditions += " AND "
						#condition = (field_name_combination[i] + ' = " ' + entity_permutation[i] + ' "')
						condition = (field_name_combination[i] + ' ' + entity_permutation[i])
						if self._dbObject._is_invalid_condition(condition):
							invalid_flag = True
							break
						else:
							conditions += condition

					if not invalid_flag:
						all_conditions.append(conditions)
		
		# include order by statements
		if self._use_order_by:
			all_conditions_with_order_by = []
			for i in range(len(all_conditions)):
				conditions = all_conditions[i]
				all_conditions_with_order_by.append(conditions)
				for field_name in self._dbObject.field_names:
					all_conditions_with_order_by.append(conditions + " ORDER  BY " + field_name + " ASC")
					all_conditions_with_order_by.append(conditions + " ORDER  BY " + field_name + " DESC")
			return all_conditions_with_order_by
		else:
			return all_conditions

	def get_high_recall_queries(self, input_entities, output_entities):
		
		high_recall_queries = []
		output_entities_set = set(output_entities)
		select_clauses = self._get_select_clauses()

		where_clauses = self._get_conditions(input_entities)
		for where_clause in where_clauses:
			for select_clause  in select_clauses:
				query = select_clause + "WHERE " + where_clause
				if query in QueryGenerator.Cache:
					result_set = QueryGenerator.Cache[query]
				else:
					_, _, result_set = self._dbObject.execute(query)
					QueryGenerator.Cache[query] = result_set
				if len(result_set)>0:
					if result_set.intersection(output_entities) == output_entities_set:
						high_recall_queries.append(query)
		return high_recall_queries

if __name__ == "__main__":

	kb_file="../data/dialog-bAbI-tasks/dialog-babi-kb-task3-fabricated.txt"
	babi_db = DbEngine(kb_file, "R_name")

	query = 'SELECT * FROM table WHERE R_cuisine british AND R_location british AND R_price expensive'
	
	print("Is Query Valid: ", babi_db.is_query_valid(query))
	if babi_db.is_query_valid(query):
		select_fields, results, result_entities_set = babi_db.execute(query)
		results = babi_db.get_formatted_results(select_fields, results)
		print("")
		for result in results:
			print(result)
		print("\n-----------------------\n")
	
	babi_query_generator = QueryGenerator(babi_db, useOrderBy=False)
	input_entities=["rome", "italian", "moderate", "four"]
	#output_entities=["resto_paris_moderate_french_6stars", "resto_paris_moderate_french_6stars_address"]
	output_entities=[]
	high_recall_queries = babi_query_generator.get_high_recall_queries(input_entities, output_entities)

	for high_recall_query in high_recall_queries:
		print(high_recall_query)