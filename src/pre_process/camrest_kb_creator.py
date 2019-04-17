####
# Please check the output file and manually add the missing fields for restaurants printed
#
# 1. city_stop_restaurant
# 2. meze_bar_restaurant
# 3. ugly_duckling
# 4. the_slug_and_lettuce
####

input_folder = "../../data/dialog-bAbI-tasks/"
output_file_path  = input_folder + "dialog-camrest-kb-all_please_insert_missing_fields.txt"

input_file_prefix = 'dialog-babi-task7-camrest676-'
input_file_paths = ['trn.txt', 'dev.txt', 'tst.txt']

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
				missing_set.add((prev_key + " is missiong " + str(7-count) + " field"))
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
		
