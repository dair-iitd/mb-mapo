import numpy as np
import random
import sys
from itertools import chain

## Special Indecies
PAD_INDEX = 0
UNK_INDEX = 1
GO_SYMBOL_INDEX = 2
EOS_INDEX = 3

class Data(object):

    def __init__(self, data, args, glob, rldata):
        self._db_vocab_id = glob['word_idx'].get('$db', -1)
        self._decode_vocab_size = len(glob['decode_idx'])
        self._rl = args.rl
        self._rldata = rldata

        ## Sort Dialogs based on turn_id
        self._extract_data_items(data, args.rl)
        
        ## Process Stories
        self._vectorize_stories(self._stories_ext, self._queries_ext, args, glob, self._dialog_ids)

        ## Process Queries
        self._vectorize_queries(self._queries_ext, glob)
        
        ## Process Answers
        self._vectorize_answers(self._answers_ext, glob)
        
        ## Create DB word mappings to Vocab
        self._populate_entity_set(self._stories_ext, self._answers_ext, self._db, glob)
        
        ## Get indicies where copying must take place
        self._intersection_set_mask(self._answers, self._entity_ids, glob)
        
        ## Get entities at response level
        self.get_entity_indecies(self._read_answers, self._entity_set)

    ## Dialogs ##
    @property
    def stories(self):
        return self._stories

    @property
    def queries(self):
        return self._queries

    @property
    def answers(self):
        return self._answers

    @property
    def rldata(self):
        return self._rldata

    ## Sizes ##
    @property
    def story_lengths(self):
        return self._story_lengths

    @property
    def story_sizes(self):
        return self._story_sizes

    @property
    def query_sizes(self):
        return self._query_sizes

    @property
    def answer_sizes(self):
        return self._answer_sizes

    ## Read Dialogs ##
    @property
    def readable_stories(self):
        return self._read_stories

    @property
    def readable_queries(self):
        return self._read_queries

    @property
    def readable_answers(self):
        return self._read_answers

    ## OOV ##
    @property
    def oov_ids(self):
        return self._oov_ids

    @property
    def oov_sizes(self):
        return self._oov_sizes

    @property
    def oov_words(self):
        return self._oov_words

    ## RL OOV ##
    @property
    def rl_oov_ids(self):
        return self._rl_oov_ids

    @property
    def rl_oov_sizes(self):
        return self._rl_oov_sizes

    @property
    def rl_oov_words(self):
        return self._rl_oov_words
    
    ## Dialog Info ##
    @property
    def dialog_ids(self):
        return self._dialog_ids

    @property
    def turn_ids(self):
        return self._turn_ids

    @property
    def db_vocab_id(self):
        return self._db_vocab_id

    ## Decode Variables ##
    @property
    def answers_emb_lookup(self):
        return self._answers_emb_lookup

    ## DB(entity) Words and Vocab Maps ##
    @property
    def db(self):
        return self._db
    
    @property
    def entity_set(self):
        return self._entity_set

    @property
    def entity_ids(self):
        return self._entity_ids
    
    @property
    def encoder_entity_ids(self):
        return self._encoder_entity_ids

    @property
    def entities(self):
        return self._entities

    @property
    def responses(self):
        return self._responses
    
    ## PGen Mask
    @property
    def intersection_set(self):
        return self._intersection_set

    def reset_responses(self):
        for key in range(0, len(self._responses)):
            self._responses[key] = []

    def _extract_data_items(self, data, rl):
        '''
            Sorts the dialogs and seperates into respective lists
        '''
        if rl: data.sort(key=lambda x: x[4])        # Sort based on turn_id
        else: data.sort(key=lambda x: len(x[0]), reverse=True)    # Sort based on dialog size
        self._stories_ext, self._queries_ext, self._answers_ext, self._dialog_ids, self._turn_ids, self._db = zip(*data)

    def _vectorize_stories(self, stories, queries, args, glob, ids):     
        '''
            Maps each story into word and character tokens and assigns them ids
        '''   
        self._stories = []              # Encoded Stories (using word_idx)
        self._story_lengths = []        # Story Lengths
        self._story_sizes = []          # Story sentence sizes
        self._read_stories = []         # Readable Stories
        self._oov_ids = []              # The index of words for copy in Response-Decoder
        self._oov_sizes = []            # The size of OOV words set in Response-Decoder
        self._oov_words = []            # The OOV words in the Stories
        self._rl_oov_ids = []           # The index of words for copy in RL-Decoder
        self._rl_oov_sizes = []         # The size of OOV words set in RL-Decoder
        self._rl_oov_words = []         # The OOV words in the Stories for RL vocab
        self._responses = []

        for i, (story, query) in enumerate(zip(stories, queries)):
            story_sentences = []    # Encoded Sentences of Single Story
            sentence_sizes = []     # List of lengths of each sentence of a Single Story
            story_string = []       # Readable Sentences of a Single Story
            oov_ids = []            # The ids of words in OOV index for copy
            oov_words = []          # The OOV words in a Single Story
            rl_oov_ids = []         # The ids of words in RL_OOV index for copy
            rl_oov_words = []       # The OOV words in a Single Story for RL

            turn = int(len(story) / 2) + 1
            story.append(query + ['$u', '#{}'.format(turn)])
            self._responses.append([])
            for sentence in story:
                sentence=sentence[:glob['sentence_size']]
                pad = max(0, glob['sentence_size'] - len(sentence))
                story_sentences.append([glob['word_idx'][w] if w in glob['word_idx'] else UNK_INDEX for w in sentence] + [0] * pad)
                sentence_sizes.append(len(sentence))
                story_string.append([str(x) for x in sentence] + [''] * pad)

                oov_sentence_ids = []
                rl_oov_sentence_ids = [] 
                for w in sentence:
                    if w not in glob['decode_idx']:
                        if w not in oov_words:
                            oov_sentence_ids.append(self._decode_vocab_size + len(oov_words))
                            oov_words.append(w)
                        else:
                            oov_sentence_ids.append(self._decode_vocab_size + oov_words.index(w))
                    else:
                        oov_sentence_ids.append(glob['decode_idx'][w])
                    
                    if args.rl:
                        if w not in glob['rl_idx']:
                            if w not in rl_oov_words:
                                rl_oov_sentence_ids.append(len(glob['rl_idx']) + len(rl_oov_words))
                                rl_oov_words.append(w)
                            else:
                                rl_oov_sentence_ids.append(len(glob['rl_idx']) + rl_oov_words.index(w)) 
                        else:
                            rl_oov_sentence_ids.append(glob['rl_idx'][w])

                oov_sentence_ids = oov_sentence_ids + [PAD_INDEX] * pad
                oov_ids.append(oov_sentence_ids)
                if args.rl: 
                    rl_oov_sentence_ids = rl_oov_sentence_ids + [PAD_INDEX] * pad
                    rl_oov_ids.append(rl_oov_sentence_ids)

            # take only the most recent sentences that fit in memory
            if len(story_sentences) > args.memory_size:
                story_sentences = story_sentences[::-1][:args.memory_size][::-1]
                sentence_sizes = sentence_sizes[::-1][:args.memory_size][::-1]
                story_string = story_string[::-1][:args.memory_size][::-1]
                oov_ids = oov_ids[::-1][:args.memory_size][::-1]
                if args.rl: rl_oov_ids = rl_oov_ids[::-1][:args.memory_size][::-1]
            else: # pad to memory_size
                mem_pad = max(0, args.memory_size - len(story_sentences))
                for _ in range(mem_pad):
                    story_sentences.append([0] * glob['sentence_size'])
                    sentence_sizes.append(0)
                    story_string.append([''] * glob['sentence_size'])
                    oov_ids.append([0] * glob['sentence_size'])
                    if args.rl: rl_oov_ids.append([0] * glob['sentence_size'])

            self._stories.append(np.array(story_sentences))
            self._story_lengths.append(len(story))
            self._story_sizes.append(np.array(sentence_sizes))
            self._read_stories.append(np.array(story_string))
            self._oov_ids.append(np.array(oov_ids))
            self._oov_sizes.append(np.array(len(oov_words)))
            self._oov_words.append(oov_words)
            if args.rl: 
                self._rl_oov_ids.append(np.array(rl_oov_ids))
                self._rl_oov_sizes.append(np.array(len(rl_oov_words)))
                self._rl_oov_words.append(rl_oov_words)

    def _vectorize_queries(self, queries, glob):
        '''
            Maps each query into word and character tokens and assigns them ids
        '''  
        self._queries = [] 
        self._query_sizes = []
        self._read_queries = []

        for i, query in enumerate(queries):
            pad = max(0, glob['sentence_size'] - len(query))
            query_sentence = [glob['word_idx'][w] if w in glob['word_idx'] else UNK_INDEX for w in query] + [0] * pad

            self._queries.append(np.array(query_sentence))
            self._query_sizes.append(np.array([len(query)]))
            self._read_queries.append(' '.join([str(x) for x in query]))

    def _vectorize_answers(self, answers, glob):
        '''
            Maps each story into word tokens and assigns them ids
        '''   
        self._answers = []
        self._answer_sizes = []
        self._read_answers = []
        self._answers_emb_lookup = []

        for i, answer in enumerate(answers):
            pad = max(0, glob['candidate_sentence_size'] - len(answer) - 1)
            answer_sentence = []
            a_emb_lookup = []
            for w in answer:
                if w in glob['decode_idx']:
                    answer_sentence.append(glob['decode_idx'][w])
                    a_emb_lookup.append(glob['decode_idx'][w])
                #elif w in self._oov_words[i]:
                #    answer_sentence.append(self._decode_vocab_size + self._oov_words[i].index(w))
                #    a_emb_lookup.append(UNK_INDEX)
                else:
                    answer_sentence.append(UNK_INDEX)
                    a_emb_lookup.append(UNK_INDEX)
            answer_sentence = answer_sentence + [EOS_INDEX] + [PAD_INDEX] * pad
            a_emb_lookup = a_emb_lookup + [EOS_INDEX] + [PAD_INDEX] * pad
            self._answers.append(np.array(answer_sentence))
            self._answer_sizes.append(np.array([len(answer)+1]))
            self._read_answers.append(' '.join([str(x) for x in answer]))
            self._answers_emb_lookup.append(np.array(a_emb_lookup))

    def _populate_entity_set(self, stories, answers, database, glob):
        '''
            Create a set of all entity words seen
        '''
        self._entity_set = set()                  # Maintain a set of entities seen
        
        for db in database:
            for entry in db:
                for w in entry:
                    if w not in self._entity_set and not w.startswith('r_'):
                        self._entity_set.add(w)
        
        self._entity_ids = set([glob['decode_idx'][x] for x in self._entity_set if x in glob['decode_idx']])
        self._encoder_entity_ids = set([glob['word_idx'][x] for x in self._entity_set if x in glob['word_idx']])
        
    def _intersection_set_mask(self, answers, entity_ids, glob):
        '''
            Create a mask which tracks the postions to copy a DB word
        '''
        self._intersection_set = []
        for i, answer in enumerate(answers):
            vocab = set(answer).intersection(entity_ids)
            dialog_mask = [0.0 if (x in vocab or x not in glob['idx_decode']) else 1.0 for x in answer]
            self._intersection_set.append(np.array(dialog_mask))

    def get_entity_indecies(self, read_answers, entity_set):
        '''
            Get list of entity indecies in each Dialog Response
        '''
        self._entities = [np.array([i for i, word in enumerate(ans.split()) if word in entity_set ]) for ans in read_answers]

class Batch(Data):

    def __init__(self, data, indecies, args, glob, results=None, train=False, repeatCopyMode=False, batchToCopy=None, indexToCopy=None, noOfCopies=None):
        
        self._rl = args.rl
        self._response_buffer_size = 150

        if repeatCopyMode:

            ## Dialogs ##
            self._stories = self._repeat_copy(batchToCopy.stories, indexToCopy, noOfCopies)
            self._queries = self._repeat_copy(batchToCopy.queries, indexToCopy, noOfCopies)
            self._answers = self._repeat_copy(batchToCopy.answers, indexToCopy, noOfCopies)
            self._rldata = data.rldata

            ## Sizes ##
            self._story_lengths = self._repeat_copy(batchToCopy.story_lengths, indexToCopy, noOfCopies)
            self._story_sizes   = self._repeat_copy(batchToCopy.story_sizes, indexToCopy, noOfCopies)
            self._query_sizes   = self._repeat_copy(batchToCopy.query_sizes, indexToCopy, noOfCopies)
            self._answer_sizes  = self._repeat_copy(batchToCopy.answer_sizes, indexToCopy, noOfCopies)

             ## Read Dialogs ##
            self._read_stories = self._repeat_copy(batchToCopy.readable_stories, indexToCopy, noOfCopies)
            self._read_queries = self._repeat_copy(batchToCopy.readable_queries, indexToCopy, noOfCopies)
            self._read_answers = self._repeat_copy(batchToCopy.readable_answers, indexToCopy, noOfCopies)

            ## OOV ##
            self._oov_ids   = self._repeat_copy(batchToCopy.oov_ids, indexToCopy, noOfCopies)
            self._oov_sizes = self._repeat_copy(batchToCopy.oov_sizes, indexToCopy, noOfCopies)
            self._oov_words = self._repeat_copy(batchToCopy.oov_words, indexToCopy, noOfCopies)

            ## RL OOV ##
            if args.rl: 
                self._rl_oov_ids   = self._repeat_copy(batchToCopy.rl_oov_ids, indexToCopy, noOfCopies)
                self._rl_oov_sizes = self._repeat_copy(batchToCopy.rl_oov_sizes, indexToCopy, noOfCopies)
                self._rl_oov_words = self._repeat_copy(batchToCopy.rl_oov_words, indexToCopy, noOfCopies)

            ## Dialog Info ##
            self._dialog_ids  = self._repeat_copy(batchToCopy.dialog_ids, indexToCopy, noOfCopies)
            self._turn_ids    = self._repeat_copy(batchToCopy.turn_ids, indexToCopy, noOfCopies)
            self._db_vocab_id = batchToCopy.db_vocab_id

            ## Decode Variables ##
            self._answers_emb_lookup = self._repeat_copy(batchToCopy.answers_emb_lookup, indexToCopy, noOfCopies)

            ## DB(entity) Words and Vocab Maps ##
            self._db = batchToCopy.db
            self._entity_set = batchToCopy.entity_set
            self._entity_ids = batchToCopy.entity_ids
            self._encoder_entity_ids = batchToCopy.encoder_entity_ids
            self._entities = self._repeat_copy(batchToCopy.entities, indexToCopy, noOfCopies)
            self._intersection_set = self._repeat_copy(batchToCopy.intersection_set, indexToCopy, noOfCopies)
            
        else:
            
            ## Dialogs ##
            self._stories = [data.stories[idx] for idx in indecies]
            self._queries = [data.queries[idx] for idx in indecies]
            self._answers = [data.answers[idx] for idx in indecies]
            self._rldata = data.rldata

            ## Sizes ##
            self._story_lengths = [data.story_lengths[idx] for idx in indecies]
            self._story_sizes   = [data.story_sizes[idx] for idx in indecies]
            self._query_sizes   = [data.query_sizes[idx] for idx in indecies]
            self._answer_sizes  = [data.answer_sizes[idx] for idx in indecies]

            ## Read Dialogs ##
            self._read_stories = [data.readable_stories[idx] for idx in indecies]
            self._read_queries = [data.readable_queries[idx] for idx in indecies]
            self._read_answers = [data.readable_answers[idx] for idx in indecies]

            ## OOV ##
            self._oov_ids   = [data.oov_ids[idx] for idx in indecies]
            self._oov_sizes = [data.oov_sizes[idx] for idx in indecies]
            self._oov_words = [data.oov_words[idx] for idx in indecies]

            ## RL OOV ##
            if args.rl: 
                self._rl_oov_ids   = [data.rl_oov_ids[idx] for idx in indecies]
                self._rl_oov_sizes = [data.rl_oov_sizes[idx] for idx in indecies]
                self._rl_oov_words = [data.rl_oov_words[idx] for idx in indecies]

            ## Dialog Info ##
            self._dialog_ids  = [data.dialog_ids[idx] for idx in indecies]
            self._turn_ids    = [data.turn_ids[idx] for idx in indecies]
            self._db_vocab_id = data.db_vocab_id

            ## Decode Variables ##
            self._answers_emb_lookup = [data.answers_emb_lookup[idx] for idx in indecies]

            if results:
                # self._append_results(args, glob, list(itemgetter(*indecies)(results)))
                self._append_results(args, glob, results)
            else:
                self._append_results(args, glob) 

            ## DB(entity) Words and Vocab Maps ##
            self._db = data.db
            self._entity_set = data.entity_set
            self._entity_ids = data.entity_ids
            self._encoder_entity_ids = data.encoder_entity_ids

            if args.rl:
                self.get_entity_indecies(self._read_answers, self._entity_set)
            else:
                self._entities = [data.entities[idx] for idx in indecies]

            self._intersection_set = [data.intersection_set[idx] for idx in indecies]

            if args.word_drop and train:
                self._stories = self._all_db_to_unk(self._stories, data.db_vocab_id, args.word_drop_prob)
    
    def _repeat_copy(self, arrayToCopy, indexToCopy, noOfCopies):
        return [arrayToCopy[indexToCopy]]*noOfCopies

    def check_res_length(self, results):
        for res in results:
            if len(res) > 0:
                return True
        return False

    def _append_results(self, args, glob, results=None):
        if results:
            selected = [results[id] for id in self._dialog_ids]
        if results and self.check_res_length(selected):
            for i, res in enumerate(selected):
                if len(res) > self._response_buffer_size:
                    selected[i] = res[:self._response_buffer_size]
            self._story_lengths = list(map(lambda x, y: x + len(y), self._story_lengths, selected))
        memory_size = max(self._story_lengths)
        for i, story in enumerate(self._stories):
            if results: response = selected[i]
            else:       response = []
            story_sentences = []    
            sentence_sizes = []     
            story_string = []       
            oov_ids = []            
            
            api_call_turns = self._rldata[self._dialog_ids[i]]["api_call_turns"]
            api_call_turn_id = -1
            if len(api_call_turns) > 0:
                api_call_turn_id = api_call_turns[0]
            position_to_insert = -1
            if "#"+str(api_call_turn_id) in self._oov_words[i]:
                position_to_insert = self._oov_words[i].index("#"+str(api_call_turn_id))
                prefix_oov_words = self._oov_words[i][:position_to_insert+1]
                suffix_oov_words = self._oov_words[i][position_to_insert+1:]
            else:
                prefix_oov_words = self._oov_words[i]
                suffix_oov_words = []

            if args.rl: 
                rl_oov_ids = []         
                rl_oov_words = self._rl_oov_words[i]
            for j, sentence_c in enumerate(response):
                sentence = sentence_c.copy()
                sentence.extend(['$db', '#{}'.format(j+1)])
                pad = max(0, glob['sentence_size'] - len(sentence))
                story_sentences.append([glob['word_idx'][w] if w in glob['word_idx'] else UNK_INDEX for w in sentence] + [0] * pad)
                sentence_sizes.append(len(sentence))
                story_string.append([str(x) for x in sentence] + [''] * pad)

                oov_sentence_ids = []
                rl_oov_sentence_ids = []

                for w in sentence:
                    if w not in glob['decode_idx']:
                        if w not in prefix_oov_words:
                            oov_sentence_ids.append(len(glob['decode_idx']) + len(prefix_oov_words))
                            prefix_oov_words.append(w)
                        else:
                            oov_sentence_ids.append(len(glob['decode_idx']) + prefix_oov_words.index(w))
                    else:
                        oov_sentence_ids.append(glob['decode_idx'][w])
                    
                    if args.rl:
                        if w not in glob['rl_idx']:
                            if w not in rl_oov_words:
                                rl_oov_sentence_ids.append(len(glob['rl_idx']) + len(rl_oov_words))
                                rl_oov_words.append(w)
                            else:
                                rl_oov_sentence_ids.append(len(glob['rl_idx']) + rl_oov_words.index(w)) 
                        else:
                            rl_oov_sentence_ids.append(glob['rl_idx'][w])

                oov_sentence_ids = oov_sentence_ids + [PAD_INDEX] * pad
                oov_ids.append(oov_sentence_ids)
                if args.rl: 
                    rl_oov_sentence_ids = rl_oov_sentence_ids + [PAD_INDEX] * pad
                    rl_oov_ids.append(rl_oov_sentence_ids)
            for word in suffix_oov_words:
                if word not in prefix_oov_words:
                    prefix_oov_words.append(word)

            original_mem_size = memory_size-len(response)
            maxl = max(self._story_lengths)

            if len(response) == 0:
                self._stories[i] = story[:original_mem_size]
                self._story_sizes[i] = self._story_sizes[i][:original_mem_size]
                self._read_stories[i] = self._read_stories[i][:original_mem_size]
                self._oov_ids[i] = self._oov_ids[i][:original_mem_size]
                if args.rl: self._rl_oov_ids[i] = self._rl_oov_ids[i][:original_mem_size]
            else:
                '''
                for line in story:
                    surface_line = ""
                    for w in line:
                        surface_line = surface_line + " " + glob['idx_word'][w]
                    print(surface_line.strip())
                sys.stdout.flush()
                '''
                self._stories[i] = np.concatenate((story[:original_mem_size], np.array(story_sentences)))
                self._story_sizes[i] = np.concatenate((self._story_sizes[i][:original_mem_size], np.array(sentence_sizes)))
                self._read_stories[i] = np.concatenate((self._read_stories[i][:original_mem_size], np.array(story_string)))
                self._oov_words[i] = prefix_oov_words
                self._oov_ids[i] = np.concatenate((self._oov_ids[i][:original_mem_size], np.array(oov_ids)))
                self._oov_sizes[i] = np.array(len(prefix_oov_words))
                if args.rl: 
                    self._rl_oov_ids[i] = np.concatenate((self._rl_oov_ids[i][:original_mem_size], np.array(rl_oov_ids)))
                    self._rl_oov_sizes[i] = np.array(len(rl_oov_words))
                    self._rl_oov_words[i] = rl_oov_words
        if results:
            self._recreate_answer_embeddings(glob, selected)

    def _recreate_answer_embeddings(self, glob, results):
        for i, answer in enumerate(self._answers):
            for j, val in enumerate(answer):
                if val == UNK_INDEX:
                    read = self._read_answers[i].split()
                    word = read[j]
                    if word in self._oov_words[i]:
                        self._answers[i][j] = len(glob['decode_idx']) + self._oov_words[i].index(word)
                    #else:
                    #    print("WARNING: Missing the oov word -", word)
            sys.stdout.flush()
        
    def _all_db_to_unk(self, stories, db_vocab_id, word_drop_prob):
        '''
            Perform Entity-Dropout on stories and story tokens
        '''
        new_stories = []
        
        for k, story in enumerate(stories):
            new_story = story.copy()
            for i in range(new_story.shape[0]):
                if db_vocab_id not in new_story[i]:
                    for j in range(new_story.shape[1]):
                        if new_story[i][j] in self._encoder_entity_ids:
                            sample = random.uniform(0,1)
                            if sample < word_drop_prob:
                                new_story[i][j] = UNK_INDEX
            new_stories.append(new_story)
        return new_stories
