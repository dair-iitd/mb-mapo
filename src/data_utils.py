from __future__ import absolute_import

import os
import re
import json
import sys
from constraints.constraints_generator import *
from collections import defaultdict
from measures import moses_multi_bleu
import numpy as np
import tensorflow as tf
from string import punctuation
from sklearn.metrics import f1_score

__all__ =  ["get_decoder_vocab", 
            "load_dialog_task", 
            "load_rl_data",
            "load_rl_test_data",
            "get_api_turns",
            "get_rl_vocab",
            "pad_to_answer_size", 
            "create_batches",
            "calculate_beam_result",
            "get_rl_decode_length_vs_index"]

np.set_printoptions(threshold=np.inf)

###################################################################################################
#########                                  Global Variables                              ##########
###################################################################################################

stop_words=set(["a","an","the"])
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

###################################################################################################
#########                                 Dialog Manipulators                            ##########
###################################################################################################

def get_decoder_vocab(data_dir, task_id):
    ''' 
        Load Vocabulary Space for Response-Decoder 
    '''

    def get_responses(f):
        '''
            Parse dialogs provided in the babi tasks format
        '''
        responses=[]
        with open(f) as f:
            for line in f.readlines():
                line=line.strip()
                if line and '\t' in line:
                    u, r = line.split('\t')
                    responses.append(r)
        return responses

    assert task_id > 0 and task_id < 9
    decoder_vocab_to_index={}
    decoder_vocab_to_index['PAD']=PAD_INDEX             # Pad Symbol
    decoder_vocab_to_index['UNK']=UNK_INDEX             # Unknown Symbol
    decoder_vocab_to_index['GO_SYMBOL']=GO_SYMBOL_INDEX # Start Symbol
    decoder_vocab_to_index['EOS']=EOS_INDEX             # End Symbol

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    candidate_sentence_size = 0
    responses = get_responses(train_file)
    for response in responses:
        line=tokenize(response.strip())
        candidate_sentence_size = max(len(line), candidate_sentence_size)
        for word in line:
            if word not in decoder_vocab_to_index:
                index = len(decoder_vocab_to_index)
                decoder_vocab_to_index[word]=index
    decoder_index_to_vocab = {v: k for k, v in decoder_vocab_to_index.items()}
    return decoder_vocab_to_index, decoder_index_to_vocab, candidate_sentence_size+1 #(EOS)

def load_dialog_task(data_dir, task_id, rl, sort):
    ''' 
        Load Train, Test, Validation Dialogs 
    '''
    assert task_id > 0 and task_id < 9
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    test_file = [f for f in files if s in f and 'tst' in f and 'OOV' not in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = parse_dialogs(train_file, rl, sort)
    test_data = parse_dialogs(test_file, rl, sort)
    val_data = parse_dialogs(val_file, rl, sort)
    if task_id < 6:
        oov_file = [f for f in files if s in f and 'tst-OOV' in f][0]
        oov_data = parse_dialogs(oov_file, rl, sort)
    else:
        oov_data = None        
    return train_data, test_data, val_data, oov_data

def tokenize(sent):
    '''
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>': return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def parse_dialogs(file, rl, sort):
    '''
        Given a file name, read the file, retrieve the dialogs, 
        Parse dialogs provided in the babi tasks format
    '''

    def append_db(database, sort):
        final = []
        if sort: database = sort_db(database)
        for i, data in enumerate(database):
            data.extend(['$db', '#{}'.format(i+1)])
            final.append(data)
        return final

    def sort_db(database):
        entities = []; curr = []; rating = []
        entity = None
        for data in database:
            if entity and data[0] != entity: 
                entities.append(curr)
                curr = []
                curr.append(data)
            else:
                curr.append(data)
            if 'rating' in data[1]: rating.append(int(data[2]))
            entity = data[0]

        if curr != []: entities.append(curr)

        sort = sorted(zip(entities, rating), key=lambda x: x[1], reverse=True)
        
        final = []
        for (entity, _) in sort:
            final.extend(entity)
        return final

    data=[]; context=[]; database=[]
    dialog_id=1; turn_id=1
    db_start = False
    trap_next = False
    for line in open(file).readlines():
        line=line.strip()
        if line:
            _, line = line.split(' ', 1)
            if '\t' in line:
                if db_start and not rl: 
                    db_start = False
                    context.extend(append_db(database, sort))
                    database = []
                u, r = map(tokenize, line.split('\t'))
                if rl and trap_next:
                    trap_next = False
                    u = trap_u
                if rl and 'api_call' in r:
                    if u != '<silence>':
                        trap_next = True
                        trap_u = u[:]
                    continue
                data.append((context[:]+u[:], u[:], r[:], dialog_id, turn_id, database))
                u.extend(['$u', '#{}'.format(turn_id)])
                r.extend(['$r', '#{}'.format(turn_id)])
                context.append(u); context.append(r)
                turn_id += 1
            else:
                if not db_start: db_start = True
                r = tokenize(line)
                database.append(r)
        else:
            # clear context / start of new dialog
            dialog_id+=1; turn_id=1
            context=[]; database=[]
    return data

###################################################################################################
#########                                RL Helper Functions                             ##########
###################################################################################################

def parse_json(datafile):
    '''
        Create a 2-level dictionary to access preprocessed data
        key1 = dialog_id ; key2 = turn_id 
    '''
    json_object = json.load(open(datafile))
    parsed = dict()
    for obj in json_object:
        api_call_turns = []
        parsed[obj["dialog_id"]] = dict()
        for turn in obj["turns"]:
            parsed[obj["dialog_id"]][turn["turn_id"]] = turn
            if turn['make_api_call'] == True:
                api_call_turns.append(turn["turn_id"])
        parsed[obj["dialog_id"]]["api_call_turns"] = api_call_turns
    return parsed

def load_rl_data(data_dir, task_id):
    ''' 
        Load Train, Validation RL Preprocessed Data
    '''
    assert task_id > 0 and task_id < 9
    rL_folder = data_dir + "task{}".format(task_id)
    files = os.listdir(rL_folder)
    files = [os.path.join(rL_folder, f) for f in files]
    train_file = [f for f in files if 'trn' in f][0]
    test_file = [f for f in files if 'tst' in f and 'OOV' not in f][0]
    val_file = [f for f in files if 'dev' in f][0]
    train_data = parse_json(train_file)
    test_data = parse_json(test_file)
    val_data = parse_json(val_file)
    return train_data, val_data

def load_rl_test_data(data_dir, task_id, oov=False):
    ''' 
        Load Test RL Preprocessed Data
    '''
    assert task_id > 0 and task_id < 9
    rL_folder = data_dir + "task{}".format(task_id)
    files = os.listdir(rL_folder)
    files = [os.path.join(rL_folder, f) for f in files]
    if task_id < 6 and oov:
        oov_file = [f for f in files if 'tst-OOV' in f][0]
        test_data = parse_json(oov_file)
    else:
        test_file = [f for f in files if 'tst' in f and 'OOV' not in f][0]
        test_data = parse_json(test_file)
    return test_data

def get_api_turns(data):
    turns = {}
    for dialog in data:
        turns[dialog] = data[dialog]["api_call_turns"]
    return turns

def get_rl_vocab(db_engine):
    '''
        Get RL vocabulary from the Processes Train data
    '''
    fields = db_engine.fields
    #entities = db_engine.entities

    #vocab = ['PAD', 'UNK', 'GO_SYMBOL', 'EOS', 'api_call', 'dontcare1', 'dontcare2', 'dontcare3', 'dontcare4']
    vocab = ['PAD', 'UNK', 'GO_SYMBOL', 'EOS', 'api_call', 'dontcare']
    #rl_word_idx = dict((c, i + len(vocab)) for i, c in enumerate(fields + entities))
    rl_word_idx = {}
    for i, val in enumerate(vocab):
        rl_word_idx[val] = i
    rl_idx_word = {v: k for k, v in rl_word_idx.items()}
    rl_vocab_size = len(rl_word_idx)

    '''
    # MASK_FILE = 'masks.json'
    # TSV_FILE = 'transitions.tsv'

    obj = ConstraintsGenerator()
    constraints_map, transistions = obj.construct_grammer(fields)

    # print(constraints_map)
    # print(transistions)

    # print(fields)

    # with open(MASK_FILE, 'r') as file:
    #     constraints_map = json.load(file)

    constraint_size = len(constraints_map)
    constraint_mask = np.zeros((constraint_size, rl_vocab_size))

    for key in constraints_map:
        constraints = constraints_map[key]
        for constraint in constraints:
            c_id = rl_word_idx[constraint]
            constraint_mask[key-1][c_id] = 1

    # print(constraint_mask)
    # sys.exit()

    state_mask = np.zeros((constraint_size, rl_vocab_size))
    # with open(TSV_FILE, 'r') as file:
    #     for line in file.readlines():
    #         vals = line.strip().split('\t')
    #         start_id = int(vals[0]) - 1
    #         end_id = int(vals[2]) - 1
    #         input_id = rl_word_idx[vals[1]]
    #         state_mask[start_id][input_id] = end_id

    for key in transistions:
        for word in transistions[key]:
            input_id = rl_word_idx[word]
            # if word == 'UNK': print('UNK', key, transistions[key][word])
            if word == '"':
                idx = transistions[key][word]
                # if constraints_map[idx] == ['UNK']:
                #     print('state = ', idx)
                #     print(constraint_mask[idx-1])
                # print(constraints_map[idx])
            state_mask[key-1][input_id] = transistions[key][word] - 1

    # print(state_mask)

    # print(constraint_mask[:, 1])
    # print(state_mask[:, 1])

    # print(rl_word_idx)
    '''
    constraint_mask = {}
    state_mask = {}
    
    return rl_word_idx, rl_idx_word, fields, rl_vocab_size, constraint_mask, state_mask

###################################################################################################
#########                           Evaluation Metrics & Helpers                         ##########
###################################################################################################

def create_split_batches(data, batch_size, RLdata=None):
    '''
        Helps to partition the dialog into three groups
        1) Dialogs occuring before an API call
        2) Dialogs that have an API call
        3) Dialogs that occur after and API call
    '''

    def chunk(set, batch_size, id):
        output = []; lst = [id]
        for i, index in enumerate(set):
            lst.append(index)
            if (i+1) % batch_size == 0: 
                output.append(lst); lst = [id]
        if len(lst) > 1:
            output.append(lst)
        return output

    api_map = {}
    pre_set = set()
    api_set = set()
    post_set = set()

    if RLdata:
        for dialog in RLdata.keys():
            api_map[dialog] = -1
            for turn in RLdata[dialog].keys():
                if turn == "api_call_turns": continue
                if RLdata[dialog][turn]['make_api_call']: api_map[dialog] = turn
    for i, (d, t) in enumerate(zip(data.dialog_ids, data.turn_ids)):
        if d not in api_map:    pre_set.add(i)
        elif t < api_map[d]:    pre_set.add(i)
        elif t > api_map[d]:    post_set.add(i)
        else:                   api_set.add(i); post_set.add(i) 
   
    return chunk(pre_set, batch_size, 0), chunk(api_set, batch_size, 1), chunk(post_set, batch_size, 2)

def create_batches(data, batch_size):
    '''
        Helps to partition the dialog into three groups
        1) Dialogs occuring before an API call
        2) Dialogs that have an API call
        3) Dialogs that occur after and API call
    '''

    output = []; lst = []
    for i, index in enumerate(range(len(data.stories))):
        lst.append(index)
        if (i+1) % batch_size == 0: 
            output.append(lst); lst = []
    if len(lst) > 1:
        output.append(lst)
    return output

def pad_to_answer_size(pred, size, action=False):

    def form(array, action):
        if action:  return[array.tolist()]
        else:       return array
    for i, list in enumerate(pred):
        sz = len(list)
        if sz >= size:  pred[i] = form(list[:size], action)
        else:           pred[i] = form(np.append(list, np.array([PAD_INDEX] * (size - sz))), action)
    return pred

def calculate_beam_result(parents, predictions, size):
    beam_results = []
    beam_width = parents.shape[2]
    length = parents.shape[1]
    for parent, pred in zip(list(parents), list(predictions)):

        # actions = []
        # action = defaultdict(list)
        # for i in range(0, length-1):
        #     for j in range(0, beam_width):
        #         action[j].append(pred[i][parent[i+1][j]])
        # for j in range(0, beam_width):
        #     action[j].append(pred[length-1][j])
        #     actions.append(action[j])
        # beam_results.append(actions)
        actions = []
        for i in range(beam_width):
            action = []
            j = i
            k = length-1
            while k >= 0:
                # print(j, k)
                action.append(pred[k][j]) 
                j = parent[k][j]
                k -= 1;
            actions.append(action[::-1])
        beam_results.append(actions)
        # print(actions)
    # print(beam_results)
    return beam_results

def get_rl_decode_length_vs_index(RLtrainData, RLvalData):
    rl_decode_length_vs_index = {}
    index_vs_rl_decode_length = {}
    
    rl_decode_length_vs_index, index_vs_rl_decode_length = get_rl_decode_length_vs_index_per_rl_data(RLtrainData, rl_decode_length_vs_index, index_vs_rl_decode_length)
    rl_decode_length_vs_index, index_vs_rl_decode_length = get_rl_decode_length_vs_index_per_rl_data(RLvalData, rl_decode_length_vs_index, index_vs_rl_decode_length)
    
    rl_decode_length_lookup_array = []
    for idx in range(len(index_vs_rl_decode_length)):
        rl_decode_length_lookup_array.append(index_vs_rl_decode_length[idx])

    return rl_decode_length_vs_index, rl_decode_length_lookup_array

def get_rl_decode_length_vs_index_per_rl_data(RLdata, rl_decode_length_vs_index, index_vs_rl_decode_length):
    if RLdata:
        for dialog in RLdata.keys():
            for turn in RLdata[dialog].keys():
                if RLdata[dialog][turn]['make_api_call']:
                    for query in RLdata[dialog][turn]['high_recall_queries']:
                        length = len(query.split(" "))
                        if length not in rl_decode_length_vs_index:
                            idx = len(rl_decode_length_vs_index)
                            rl_decode_length_vs_index[length] = idx
                            index_vs_rl_decode_length[idx] = length
    return rl_decode_length_vs_index, index_vs_rl_decode_length
