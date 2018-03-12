import numpy as np
import torch, os
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import misc
from skimage.io import imsave

plot_folder = './Plots/tesing_plots/'

def predict(model, imgfile):
	im, image_height, image_width = np.asarray(resize_and_crop(Image.open(imgfile).convert('RGB'))).transpose((2,0,1))
	Image.fromarray(img.transpose(1,2,0)).save(plot_folder + 'X.png')
	img_normalized = img/255.0
	img_normalized = Variable(torch.FloatTensor(img_normalized).unsqueeze(0))
	outputs, recon = model(img_normalized)
	outputs = F.softmax(outputs, 1).data.squeeze().cpu().numpy()
	for i in range(outputs.shape[0]):
		imsave(plot_folder + 'output_total_%d.jpg'%i, outputs[i])

if __name__ == '__main__':
	
	make_dir(plot_folder)
	model = Net()
	model.load_state_dict(torch.load('Weights')['state_dict'])
	model.eval()

	predict_for_img(model, 'file.png')