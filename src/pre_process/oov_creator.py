import sys
sys.path.append("..")

import copy
import json

from db_engine import DbEngine

def get_dialogs(test_file):

    dialogs = []
    with open(test_file) as f:
        lines = f.readlines()
        dialog = []
        for i in range(0, len(lines)):
            line=lines[i].strip()
            if line:
                _, line = line.split(' ', 1)
                if '\t' in line:
                    u, r = line.split('\t')
                    dialog.append([u,r])
                else:
                    dialog.append([line])
            else:
                dialogs.append(dialog)
                dialog = []
        dialogs.append(dialog)
    return dialogs

def printable_dialog(dialog):
    dialog_str = ""
    index = 1
    for turn in dialog:
        if len(turn) == 1:
            dialog_str += (str(index) + " " + turn[0] + "\n")
        else:
            dialog_str += (str(index) + " " + turn[0] + "\t" + turn[1] + "\n")
        index+=1
    dialog_str += ("\n")
    return dialog_str

def find_replace(utt):
    entities, informable_entities = dbEngine.get_entities_in_utt(utt)
    modified_utt = copy.deepcopy(utt)
    for entity in entities:
        if entity not in informable_entities:
            modified_utt = modified_utt.replace(entity, entity[::-1])
    return modified_utt

def find_replace_in_entities_list(entity_list):
    modified_entity_list = []
    for entity in entity_list:
        if dbEngine.is_informable_field_value(entity):
            modified_entity_list.append(entity)
        else:
            modified_entity_list.append(entity[::-1])
    return modified_entity_list

def modify_dialog(dialog):
    modified_dialog = []
    for turn in dialog:
        if len(turn) == 1:
            modified_dialog.append([find_replace(turn[0])])
        else:
            u = find_replace(turn[0])
            a = find_replace(turn[1])
            modified_dialog.append([u,a])

    return modified_dialog

def modify_dialog_json(dialog):
    modified_dialog = copy.deepcopy(dialog)
    modified_turns = []
    for turn in dialog['turns']:
        print(dialog['dialog_id'], turn['turn_id'])
        modified_turn = copy.deepcopy(turn)
        modified_turn['user']['utt'] = find_replace(turn['user']['utt'])
        modified_turn['agent']['utt'] = find_replace(turn['agent']['utt'])
        if 'kb_entities' in turn['user']:
            modified_turn['user']['kb_entities'] = find_replace_in_entities_list(turn['user']['kb_entities'])
        if 'kb_entities' in turn['agent']:
            modified_turn['agent']['kb_entities'] = find_replace_in_entities_list(turn['agent']['kb_entities'])
        modified_turn['entities_so_far'] = find_replace_in_entities_list(turn['entities_so_far'])
        modified_turn['next_entities'] = find_replace_in_entities_list(turn['next_entities'])
        modified_turns.append(modified_turn)
    modified_dialog['turns'] = modified_turns
    return modified_dialog

if __name__ == "__main__":
    
    taskids = [7,6]
    folder = "data"
    #folder = "data-heuristic-predicted"
    
    for taskid in taskids:
        
        if taskid == 7:
            input_folder = "../../"+folder+ "/dialog-bAbI-tasks/"
            tst_files = input_folder+'dialog-babi-task7-camrest676-tst-modified.txt'
            if folder == "data":
                json_file = input_folder+'task7/dialog-camrest-tst.json'
            else:
                json_file = input_folder+'task7/dialog-camrest-tst-modified.json'

            kb_file = input_folder+'dialog-camrest-kb-all.txt'
        elif taskid == 6:
            input_folder = "../../"+folder+ "/dialog-bAbI-tasks/"
            tst_files = input_folder+'dialog-babi-task6-dstc2-filtered-tst-modified.txt'
            if folder == "data":
                json_file = input_folder+'task6/dialog-babi-task6-dstc2-tst.json'
            else:
                json_file = input_folder+'task6/dialog-babi-task6-dstc2-filtered-tst-modified.json'
            kb_file = input_folder+'dialog-dstc2-kb-all.txt'

        dbEngine = DbEngine(kb_file, "R_name")

        dialogs = get_dialogs(tst_files)

        modified_dialogs = []
        for dialog in dialogs:
            modified_dialogs.append(modify_dialog(dialog))

        out_file = open(tst_files.replace("tst","tst-oov"),"w") 
        for dialog in modified_dialogs:
            out_file.write(printable_dialog(dialog))
        out_file.close() 

        with open(json_file) as f:
            dialogs = json.load(f)
        modified_dialogs = []
        for dialog in dialogs:
            modified_dialogs.append(modify_dialog_json(dialog))

        with open(json_file.replace("tst","tst-oov"), 'w') as json_file:
            json.dump(modified_dialogs, json_file)



