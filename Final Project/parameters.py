import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'
sourceTestFileName = 'en_bg_data/test.en'
targetTestFileName = 'en_bg_data/test.bg'

corpusOriginalFileName = 'corpusOriginal'
corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

pathsFileName = 'pathsData'

device = torch.device("cuda:0")
# device = torch.device("cpu")

n_head = 4

d_model = 512
d_keys = d_model // n_head
d_values = d_model // n_head
d_ff = 4 * d_model

num_layers = 5

norm_first = True

dropout = 0.1
encoder_dropout, decoder_dropout = 0.1, 0.2
source_dropout, target_dropout = 0.05, 0.05

learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.9

batchSize = 16

warmup_steps = 6000

bpe_dropout = 0.1
byte_pairs_eng = 2500
byte_pairs_bg = 2500

temperature = 1

maxEpochs = 1
log_every = 50
test_every = 250

max_patience = 5
max_trials = 5
