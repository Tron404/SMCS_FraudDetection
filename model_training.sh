# neighbourhood = how many neighbouring nodes to sample in the ith hop (e.g. [10, 5] --> 10 nodes in the first hop and 5 in the second hop)
# model = which model architecture to use (choices: sage_attn, gcn, gat, sage)
# dataset = which dataset to train on (choices: elliptic++, dgraphfin)
# device = on which device to train/load data (choices: cpu, cuda)
# epochs = for how many epochs to train a model

python training_script.py --neighbourhood "10,5" --epochs 2 --model sage_attn --dataset "elliptic++" --device cuda:0