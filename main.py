import numpy as np
import torch, os, sys
import matplotlib.pyplot as plt
import visdom
from PIL import Image
from argparse import ArgumentParser
from utils import *
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F


###ALL HYPER-PARAMETERS AND GLOBAL VARS DEFINITION###

NUM_CHANNELS = 0
VAL_PERCENT = 0.1
NUM_CLASSES = 0
basename = 'DOC_SEG_'
modelname = 'fix_consloss_equalcoeff'
weights_folder = './Weights/' + modelname
plot_folder = './Plots/' + modelname

## Training Hyperparameters
lr = 1e-3
momentum = 0.6
weight_decay = 0

resume = False
if resume:
	checkpoint_path = './Weights/checkpoint.pth.tar'

### Object Definitions
vis = visdom.Visdom()



###Train Function###
def train(args, model):

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])

	print('''
	Starting training:
		Epochs: {}
		Batch size: {}
		Training size: {}
		Validation size: {}
		CUDA: {}
		Model Parameters: {}
	''')

	train = train_data
	val = val_data

	model.train()

	
	###LOSS DEFINITION###
	if args.cuda:
		criterion = torch.nn.CrossEntropyLoss(weight.cuda())
	else:
		criterion = torch.nn.CrossEntropyLoss(weight)
	###LOSS DEFINITION###

	###OPTIMIZER DEFINITION###
	optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
	###OPTIMIZER DEFINITION###

	if resume == False:
		train_loss = []
		train_loss_epoch = []
		seg_loss_all = []
		recon_loss_all = []
		cons_loss_all = []
		precision_train = []
		recall_train = []
		f1_scores_train = []	
		avg_precision_train = []
		avg_recall_train = []
		avg_f1_scores_train = []

	best_prec1 = 0
	start_epoch = 1
	early_stop_counter = 0

	if resume:
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint['epoch'] + 1
		best_prec1 = checkpoint['best_prec1']

	for epoch in range(start_epoch, args.num_epochs+1):

		if resume:
			train_loss = []
			train_loss_epoch = []
			seg_loss_all = []
			recon_loss_all = []
			cons_loss_all = []
			precision_train = []
			recall_train = []
			f1_scores_train = []	
			avg_precision_train = []
			avg_recall_train = []
			avg_f1_scores_train = []

		epoch_loss = []
		epoch_seg_loss = []
		epoch_recon_loss = []
		epoch_cons_loss = []
		epoch_acc = []
		step = 1
		true_positives = np.zeros(NUM_CLASSES)
		false_negatives = np.zeros(NUM_CLASSES)
		false_positives = np.zeros(NUM_CLASSES)		
		for i, b in enumerate(batch(train, args.batch_size)):
			X = [z[0] for z in b]
			y = [z[1] for z in b]
			X = np.asarray(X)
			y = np.asarray(y)
			X = torch.FloatTensor(X)
			y = torch.LongTensor(y)


			if args.cuda:
				X = Variable(X).cuda()
				y = Variable(y).cuda()
			else:
				X = Variable(X)
				y = Variable(y)

			outputs = model(X)
			optimizer.zero_grad()

			## Get argmax from outputs
			argmax = outputs.cpu().max(1)[1].data.numpy()
			
			## Loss Computation and Backprop
			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()

			## Acc calculation
			tp,fp,fn = calc_metrics(y.cpu().data.numpy().flatten(), argmax.flatten())
			true_positives += tp
			false_positives += fp
			false_negatives += fn

			epoch_loss.append(loss.data.cpu().numpy()[0])

			if args.steps_loss > 0 and step % args.steps_loss == 0:
				print("Epoch: %d, Batch: %d/%d, Losses - Total:%.6f"
					%(epoch,i,len(iddataset['train'])/args.batch_size,epoch_loss))

			step+=1

		### METRICS COMPUTATION AND PLOTTING ###
		precision_train.append(true_positives / (true_positives+false_positives))
		recall_train.append(true_positives / (true_positives+false_negatives))
		f1 = calc_f1(precision_train[-1],recall_train[-1])
		f1_scores_train.append(f1)
		train_loss.extend(epoch_loss)
		train_loss_epoch.append(sum(epoch_loss)/len(epoch_loss))
		avg_precision_train.append(precision_train[-1].mean())
		avg_recall_train.append(recall_train[-1].mean())
		avg_f1_scores_train.append(f1_scores_train[-1].mean())

		## Plots
		plot_visdom(epoch, train_loss, 'loss_each_batch', legend=['train'])
		plot_visdom(epoch, train_loss_epoch, 'mean_loss_per_epoch', legend=['train'])
		
		plot_visdom(epoch, precision_train, win='precision_train', legend=labels)
		plot_visdom(epoch, recall_train, win='recall_train', legend=labels)
		plot_visdom(epoch, f1_scores_train, win='f1_scores_train', legend=labels)

		plot_visdom(epoch, avg_precision_train, win='avg_precision_train', legend=['train'])
		plot_visdom(epoch, avg_recall_train, win='avg_recall_train', legend=['train'])
		plot_visdom(epoch, avg_f1_scores_train, win='avg_f1_scores_train', legend=['train'])

		## Save plots incase visdom gets restarted
		vis.save(envs=[basename+modelname])

		## Saving weights
		prec1 = avg_f1_scores_train[-1]
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		### Saving weights
		save_checkpoint({
			'epoch': epoch,
			'arch': args.model,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, is_best)
		### Train Module Ends ###

def evaluate(args, model):
	model.eval()

def main(args):
	print('Selected Model: {}'.format(args.model))
	model = Net(NUM_CLASSES)

	if args.cuda:
		model = model.cuda()
	if args.resume:
		resume_or_not
	if args.mode == 'eval':
		evaluate(args, model)
	if args.mode == 'train':
		train(args, model)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--cuda', action='store_true', default='True')
	parser.add_argument('--model', default='unet')
	parser.add_argument('--state')

	### Subparsers
	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_eval = subparsers.add_parser('eval')
	parser_eval.add_argument('image')
	parser_eval.add_argument('label')

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--port', type=int, default=8097)
	parser_train.add_argument('--num-epochs', type=int, default=32)
	parser_train.add_argument('--num-workers', type=int, default=4)
	parser_train.add_argument('--batch-size', type=int, default=1)
	parser_train.add_argument('--steps-loss', type=int, default=100)
	parser_train.add_argument('--steps-plot', type=int, default=100)
	parser_train.add_argument('--steps-save', type=int, default=500)
	parser_train.add_argument('--resume', default='None')
	parser_train.add_argument('--load_pickles', type=int, default=0)

	make_dir(weights_folder)
	make_dir(plot_folder)

	main(parser.parse_args())
