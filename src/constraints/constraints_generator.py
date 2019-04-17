import json
import random

class ConstraintsGenerator:
	
	TRANSITIONS_FILEPATH = "transitions.tsv"
	MASKS_FILEPATH = "masks.json"

	def construct_grammer(self, fields, fixed_length_decode=False, serialize=False):
		
		fields.sort()
		
		# construct masks for select clause
		masks = {}
		
		masks[1] = ["SELECT"]
		#masks[2] = ["*"] + fields
		masks[2] = ["*"]
		masks[3] = ["FROM"]
		for i in range(len(fields)-1):
			masks[len(masks)+1] = [",","FROM"]
			masks[len(masks)+1] = fields[i+1:]
		table_mask = len(masks)+1
		masks[len(masks)+1] = ["table"]
		where_mask = len(masks)+1
		masks[len(masks)+1] = ["WHERE"]
		
		for i in range(len(fields)):
			masks[len(masks)+1] = fields[i:]
			#masks[len(masks)+1] = ["="]
			#masks[len(masks)+1] = ["\""]
			masks[len(masks)+1] = ["UNK"]
			#masks[len(masks)+1] = ["\""]
			if i == len(fields)-1:
				if not fixed_length_decode:
					masks[len(masks)+1] = ["EOS"]
			else:
				if fixed_length_decode:
					masks[len(masks)+1] = ["AND"]
				else:
					masks[len(masks)+1] = ["EOS", "AND"]

		self._state_to_surface_forms = masks
		
		self._transition_rules = {}
		for i in range(len(masks)):
			self._transition_rules[i]={}
		
		self._transition_rules[1]["SELECT"] = 2
		self._transition_rules[2]["*"] = 3
		for i, field in enumerate(fields):
			if i == len(fields)-1:
				self._transition_rules[2][field] = 3
				self._transition_rules[3]["FROM"] = table_mask
			else:
				self._transition_rules[2][field] = 4+i*2
				self._transition_rules[4+i*2]["FROM"] = table_mask
				self._transition_rules[4+i*2][","] = 4+(i*2)+1
				for j, transition_field in enumerate(masks[4+(i*2)+1]):
					if j == len(masks[4+(i*2)+1])-1:
						self._transition_rules[4+(i*2)+1][transition_field] = 3
					else:
						self._transition_rules[4+(i*2)+1][transition_field] = 4+(i*2)+1+(j*2)+1
		
		self._transition_rules[table_mask]["table"] = where_mask

		self._transition_rules[where_mask]["WHERE"] = where_mask+1
		
		for i in range(len(fields)):
			'''
			for j in range(i, len(fields)):
				self._transition_rules[where_mask+(6*i+1)][fields[j]] = where_mask+(6*j+2)
			self._transition_rules[where_mask+(6*i+2)]["="] = where_mask+(6*i+3)
			self._transition_rules[where_mask+(6*i+3)]["\""] = where_mask+(6*i+4)
			self._transition_rules[where_mask+(6*i+4)]["UNK"] = where_mask+(6*i+5)
			if i < len(fields)-1:
				self._transition_rules[where_mask+(6*i+5)]["\""] = where_mask+(6*i+6)
				self._transition_rules[where_mask+(6*i+6)]["AND"] = where_mask+(6*i+7)
			else:
				if not fixed_length_decode:
					self._transition_rules[where_mask+(6*i+5)]["\""] = where_mask+(6*i+6)
			'''
			for j in range(i, len(fields)):
				self._transition_rules[where_mask+(3*i+1)][fields[j]] = where_mask+(3*j+2)
			self._transition_rules[where_mask+(3*i+2)]["UNK"] = where_mask+(3*i+3)
			if i < len(fields)-1:
				self._transition_rules[where_mask+(3*i+3)]["AND"] = where_mask+(3*i+4)
		if serialize:
			self.serialize()
		
		return self._state_to_surface_forms, self._transition_rules

	def serialize(self):
		with open(ConstraintsGenerator.MASKS_FILEPATH, 'w') as mask_file:
			json.dump(self._state_to_surface_forms, mask_file)

		with open(ConstraintsGenerator.TRANSITIONS_FILEPATH, 'w') as transitions_file:
			for state, vocab_map in self._transition_rules.items():
				for word, next_state in vocab_map.items():
					transitions_file.write(str(state) + "\t" + word + "\t" + str(next_state) + "\n")

	def read_files(self):
		
		with open(ConstraintsGenerator.MASKS_FILEPATH) as mask_file:  
			self._state_to_surface_forms = json.load(mask_file)

		self._transition_rules={}
		with open(ConstraintsGenerator.TRANSITIONS_FILEPATH) as transitions_file:
			for line in transitions_file:
				arr = line.strip().split()
				key = arr[0]
				if key not in self._transition_rules:
					self._transition_rules[key] = {}
				self._transition_rules[key][arr[1]]=arr[2]

	def generate_valid_query(self):
		state = 1
		query = ""
		surface_forms = self._state_to_surface_forms[state]
		word = random.choice(surface_forms)
		while(word != 'EOS'):
			query += " " + word
			if state not in self._transition_rules:
				break
			state = self._transition_rules[state][word]
			surface_forms = self._state_to_surface_forms[state]
			word = random.choice(surface_forms)
		return query.strip()

if __name__ == "__main__":
	
	fields = ["R_cuisine","R_location","R_price","R_rating","R_phone","R_address","R_number"]

	babi = ConstraintsGenerator()
	babi.construct_grammer(fields, fixed_length_decode=False,serialize=True)

	for i in range(100):
		print(babi.generate_valid_query())