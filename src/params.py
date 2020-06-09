import tensorflow as tf

flags = tf.app.flags

# Model Constants
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
flags.DEFINE_integer("memory_size", 200, "Maximum size of memory.")
flags.DEFINE_integer("epochs", 600, "Number of epochs to train for.")

# Model Params
flags.DEFINE_float("learning_rate", 0.0025, "Learning rate for Adam Optimizer.")
flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")
flags.DEFINE_integer("embedding_size", 32, "Embedding size for embedding matrices.")
flags.DEFINE_integer("soft_weight", 1, "Weight given to softmax function")
flags.DEFINE_integer("beam_width", 6, "Width of Beam for BeamSearchDecoder")
flags.DEFINE_integer("phase", 1, "Start Phase for RL training")
flags.DEFINE_integer("model_index", 1, "integer id when multiple runs are launched for the same param settings")

# Entity Word Drop
flags.DEFINE_float("word_drop_prob", 0.1, "value to set, if word_drop is set to True")
flags.DEFINE_boolean('word_drop', True, 'if True, drop db words in story')

# PGen Loss
flags.DEFINE_float("p_gen_loss_weight", 1.0, 'relative weight to p_gen loss, > 1 gives more weight to p_gen loss')
flags.DEFINE_boolean("p_gen_loss", True, 'if True, uses additional p_gen loss during training')

# Model Type
flags.DEFINE_boolean("hierarchy", True, "if True, uses hierarchy pointer attention")
flags.DEFINE_boolean("beam", True, "if True, uses beam search decoder")
flags.DEFINE_boolean("simple_beam", False, "if True, uses beam search decoder for non rl decoding")
flags.DEFINE_boolean("sort", False, "if True, sort db results on rating")
flags.DEFINE_boolean("constraint", False, "if True, perform constraint decoding")

# RL Params
flags.DEFINE_boolean("rl", True, 'if True, uses RL decoder')
flags.DEFINE_string("rl_mode", "MAPO", 'takes the following values: GT, GREEDY, MAPO, HYBRID')
flags.DEFINE_boolean("fixed_length_decode", False, 'sample length of action before decoding the action')
flags.DEFINE_integer("max_api_length", 4, "Set the value based on DBEngine and QueryGenerator")
flags.DEFINE_boolean("split_emb", True, "Use separate embedding for RL encoder")
flags.DEFINE_integer("rl_warmp_up", 40, "Set the number of epochs for which RL should run before SL")
flags.DEFINE_boolean("filtering", False, "Filter valid queries during inference")
flags.DEFINE_float("pi_b", 0.6, 'memory weight in HYBRID')
flags.DEFINE_boolean("load_api_from_file", True, "loads api_calls from file - used for e2e tod run")

# Output and Evaluation Specifications
flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
flags.DEFINE_boolean("bleu_score", False, 'if True, uses BLUE word score to compute best model')
flags.DEFINE_boolean("save", False, "if True, trains using previously saved model")
flags.DEFINE_boolean("debug", False, 'if True, enables debug mode (Verbose Errors, but slower)')

# Task Type
flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 8")
flags.DEFINE_boolean('train', True, 'if True, begin to train')
flags.DEFINE_boolean('oov', False, 'if True, use OOV test set')

# File Locations
flags.DEFINE_string("data_dir", "../data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
flags.DEFINE_string("logs_dir", "logs/", "Directory containing bAbI tasks")
flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
#flags.DEFINE_string("kb_file", "../data/dialog-bAbI-tasks/dialog-babi-kb-all.txt", "kb file for this task")
#flags.DEFINE_string("kb_file", "../data/dialog-bAbI-tasks/dialog-babi-kb-task3.txt", "kb file for this task")
#flags.DEFINE_string("kb_file", "../data/dialog-bAbI-tasks/dialog-camrest-kb-all.txt", "kb file for this task")
flags.DEFINE_string("kb_file", "../data/dialog-bAbI-tasks/dialog-dstc2-kb-all.txt", "kb file for this task")
flags.DEFINE_string("vocab_ext", "trn", "Data Set used to build the decode vocabulary")

def get_params():
	return flags.FLAGS

def print_params(logging, args):

	if args.beam == False:
		args.beam_width = 1
	
	args.constraint == False

	if "babi" in args.kb_file:
		args.max_api_length = 5
		args.rl_warmp_up = 20
		if args.rl_mode == 'MAPO':
			args.beam == True
			args.beam_width = 32
		elif args.rl_mode == 'HYBRID':
			args.beam == True
			args.beam_width = 4
		elif args.rl_mode == 'SL':
			args.beam == True
			args.beam_width = 1
		elif args.rl_mode == 'RL':
			args.beam == True
			args.beam_width = 8
		args.bleu_score = False
	if "camrest" in args.kb_file:
		args.rl_warmp_up = 100
		args.max_api_length = 4
		if args.rl_mode == 'MAPO':
			args.beam == True
			args.beam_width = 16
		elif args.rl_mode == 'HYBRID':
			args.beam == True
			args.beam_width = 4
		elif args.rl_mode == 'SL':
			args.beam == True
			args.beam_width = 1
		elif args.rl_mode == 'RL':
			args.beam == True
			args.beam_width = 8
			args.rl_warmp_up = 0
		args.bleu_score = True
	if "dstc2" in args.kb_file:
		args.max_api_length = 4
		args.rl_warmp_up =100
		if args.rl_mode == 'MAPO':
			args.beam == True
			args.beam_width = 16
		elif args.rl_mode == 'HYBRID':
			args.beam == True
			args.beam_width = 8
		elif args.rl_mode == 'SL':
			args.beam == True
			args.beam_width = 1
		elif args.rl_mode == 'RL':
			args.beam == True
			args.beam_width = 1
			args.rl_warmp_up = 0
		args.bleu_score = True
	
	if args.load_api_from_file:
		args.rl_warmp_up = 0
	
	'''
	Print important model parameters
	'''
	print('\n# {}'.format('Model Params'))
	logging.info('[{}] : {}'.format('learning_rate', args.learning_rate))
	logging.info('[{}] : {}'.format('batch_size', args.batch_size))
	logging.info('[{}] : {}'.format('hops', args.hops))
	logging.info('[{}] : {}'.format('embedding_size', args.embedding_size))
	logging.info('[{}] : {}'.format('soft_weight', args.soft_weight))
	logging.info('[{}] : {}'.format('beam_width', args.beam_width))
	logging.info('[{}] : {}'.format('memory_size', args.memory_size))
	logging.info('[{}] : {}'.format('phase', args.phase))
	logging.info('[{}] : {}'.format('model_index', args.model_index))

	print('\n# {}'.format('Word Drop'))
	logging.info('[{}] : {}'.format('word_drop', args.word_drop))
	logging.info('[{}] : {}'.format('word_drop_prob', args.word_drop_prob))

	print('\n# {}'.format('PGen Loss'))
	logging.info('[{}] : {}'.format('p_gen_loss', args.p_gen_loss))
	logging.info('[{}] : {}'.format('p_gen_loss_weight', args.p_gen_loss_weight))

	print('\n# {}'.format('RL Params'))
	logging.info('[{}] : {}'.format('rl', args.rl))
	logging.info('[{}] : {}'.format('rl_mode', args.rl_mode))
	logging.info('[{}] : {}'.format('max_api_length', args.max_api_length))
	logging.info('[{}] : {}'.format('fixed_length_decode', args.fixed_length_decode))
	logging.info('[{}] : {}'.format('rl_warmp_up', args.rl_warmp_up))
	logging.info('[{}] : {}'.format('filtering', args.filtering))
	logging.info('[{}] : {}'.format('load_api_from_file', args.load_api_from_file))

	print('\n# {}'.format('Model Type'))
	logging.info('[{}] : {}'.format('hierarchy', args.hierarchy))
	logging.info('[{}] : {}'.format('beam', args.beam))
	logging.info('[{}] : {}'.format('sort', args.sort))
	logging.info('[{}] : {}'.format('constraint', args.constraint))

	print('\n# {}'.format('Evaluation Type'))
	logging.info('[{}] : {}'.format('evaluation_interval', args.evaluation_interval))
	logging.info('[{}] : {}'.format('bleu_score', args.bleu_score))
	logging.info('[{}] : {}'.format('save', args.save))
	logging.info('[{}] : {}'.format('debug', args.debug))

	print('\n# {}'.format('Task Type'))
	logging.info('[{}] : {}'.format('task_id', args.task_id))
	logging.info('[{}] : {}'.format('train', args.train))
	logging.info('[{}] : {}'.format('OOV', args.oov))

	print('\n# {}'.format('File Locations'))
	logging.info('[{}] : {}'.format('model_dir', args.model_dir))
	logging.info('[{}] : {}'.format('data_dir', args.data_dir.split('/')[-2]))
	logging.info('[{}] : {}'.format('kb_file', args.kb_file))

