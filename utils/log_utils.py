import matplotlib.pyplot as plt


def plot_visdom(epoch,y,win,legend=[]):
	if resume:
		x = np.asarray(range(epoch*len(y), (epoch+1)*len(y)))
		y = np.asarray(y)
		vis.line(Y=y, X=x, win=win, env=basename + modelname, opts=dict(title=win,legend=legend), update='append')
	else:
		x = np.arange(len(y))
		y = np.asarray(y)
		vis.line(Y=y, X=x, win=win, env=basename + modelname, opts=dict(title=win,legend=legend))
		plt.plot(x,y)
		plt.savefig(plot_folder + win + '.jpg')
		plt.clf()

def make_dir(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def calc_metrics(gt, op):

	gt,op = np.asarray(gt), np.asarray(op)

	tp = np.zeros(NUM_CLASSES)
	fn = np.zeros(NUM_CLASSES)
	fp = np.zeros(NUM_CLASSES)

	cm = confusion_matrix(gt,op,labels)
	for i in range(NUM_CLASSES):
		tp[i] = cm[i,i]
		fn[i] = cm[i,:].sum() - tp[i]
		fp[i] = cm[:,i].sum() - tp[i]

	return tp, fp, fn

def calc_f1(prec, rec):

	return 2.*prec*rec / (prec + rec)
