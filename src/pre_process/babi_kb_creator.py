def create_task3_kb():
	input_folder = "../../data/dialog-bAbI-tasks/"
	output_file_path  = input_folder + "dialog-babi-kb-task3-new.txt"

	input_file_prefix = 'dialog-babi-task3-options-with-api-'
	input_file_paths = ['trn.txt', 'dev.txt', 'tst.txt', 'tst-OOV.txt']

	data = []
	for input_file_path in input_file_paths:
		input_file = input_folder + input_file_prefix + input_file_path
		with open(input_file) as f:
			for line in f:
				if "R_" in line:
					line = line.strip()
					line = line[line.index(' ')+1:]
					data.append(line)

	kb = set([])
	prev_key = None
	kb_record = ""
	count = 0

	missing_set = set([])

	for line in data:
		line_split = line.split(" ")
		key = line_split[0]
		count+=1
		if key != prev_key and prev_key != None:
			if count <= 7:
				kb.add(kb_record.strip())
				if count < 7:
					missing_set.add((prev_key + " is missing " + str(7-count) + " field"))
			# gretaer than 7 is usually 14
			# it happens when the same restaurant occurs consecutively
			# we can ignore them

			kb_record = ""
			count=0

		kb_record += line+"\n"
		prev_key=key

	output_file = open(output_file_path,"w") 
	output_file.write("######################## \n")
	output_file.write("## Missing fields info:  \n")
	index = 1
	for missing_item in missing_set:
		output_file.write("# " + str(index) + ". " + missing_item + "\n")
		index+=1
	output_file.write("######################## \n")

	for record in kb:
		lines = record.split("\n")
		for line in lines:
			line_split = line.split(" ")
			line = line_split[0] + " " + line_split[1] + "\t" + line_split[2]
			output_file.write("1 " + line.strip() + "\n")
	output_file.close()
		
def create_artificial_task3_kb():
	input_folder = "../../data/dialog-bAbI-tasks/"
	output_file_path  = input_folder + "dialog-babi-kb-task3-fabricated.txt"
	output_file = open(output_file_path,"w") 

	input_file_prefix = 'dialog-babi-task3-options-with-api-'
	input_file_paths = ['trn.txt', 'dev.txt', 'tst.txt', 'tst-OOV.txt']

	data = []

	field_value_map = {}
	for input_file_path in input_file_paths:
		input_file = input_folder + input_file_prefix + input_file_path
		with open(input_file) as f:
			for line in f:
				if "R_" in line:
					line = line.strip()
					line = line[line.index(' ')+1:]
					split_line = line.split()
					field = split_line[1]
					value = split_line[2]

					if field not in field_value_map:
						field_value_map[field] = set([])
					field_value_map[field].add(value)
	
	for location in field_value_map['R_location']:
		for price in field_value_map['R_price']:
			for cuisine in field_value_map['R_cuisine']:
				for rating in field_value_map['R_rating']:
					for number in field_value_map['R_number']:
						rest_name = "resto_" + location + "_" + price + "_" + cuisine + "_" + rating + "stars"
						db_entry = 		"1 " + rest_name + " R_phone\t" + rest_name + "_phone\n"
						db_entry +=		"1 " + rest_name + " R_cuisine\t" + cuisine + "\n"
						db_entry +=		"1 " + rest_name + " R_address\t" + rest_name + "_address\n"
						db_entry +=		"1 " + rest_name + " R_location\t" + location + "\n"
						db_entry +=		"1 " + rest_name + " R_number\t" + number + "\n"
						db_entry +=		"1 " + rest_name + " R_price\t" + price + "\n"
						db_entry +=		"1 " + rest_name + " R_rating\t" + rating + "\n"
						output_file.write(db_entry)
	output_file.close()

create_task3_kb()