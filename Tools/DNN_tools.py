# utils
from torch.utils.data import Dataset,  DataLoader
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import matplotlib

torch.set_default_dtype(torch.float64)

#-----------------------------------------------------------------#
# auto-batching tools for NF model
class MyDatasetX(Dataset):
	def __init__(self, X):
		super(MyDatasetX, self).__init__()        
		self.X = X

	# number of samples to be batched
	def __len__(self):
		return self.X.shape[0] 
	   
	# get samples
	def __getitem__(self, index):
		return self.X[index]
#------------------------------------------------------------------#


#-----------------------------------------------------------#
# auto-batching tools
class MyDatasetXY(Dataset):
	def __init__(self, X, Y):
		super(MyDatasetXY, self).__init__()
		
		# sample size checker, the first dimension is always the batch size
		assert X.shape[0] == Y.shape[0]
		
		self.X = X
		self.Y = Y

	# number of samples to be batched
	def __len__(self):
		return self.X.shape[0] 
	   
	# get samples
	def __getitem__(self, index):
		return self.X[index], self.Y[index]
#-----------------------------------------------------------#

#-----------------------------------------------------------#
# Nonlinear MLP function, can be generalized for various purposes
# inputs:         
		# NI: input size
		# NO: ouput size
		# NN: hidden size
		# NL: num of hidden layers
		# act: type of nonlinear activations, default: relu
# output:
#       sequential of layers

def MLP_nonlinear(NI,NO,NN,NL,act='relu'):

	# select act functions
	if act == "relu":
		actF = nn.ReLU()
	elif act == "tanh":
		actF = nn.Tanh()
	elif act == "sigmoid":
		actF = nn.Sigmoid()
	elif act == 'leaky':
		actF = nn.LeakyReLU(0.1)
	elif act == 'identity':
		actF = nn.Identity()
	elif act == 'silu':
		actF = nn.SiLU()
	elif act == 'exlu':
		actF = nn.ELU()
	elif act == 'gelu':
		actF = nn.GELU()

	#----------------construct layers----------------#
	MLP_layer = []

	# Input layer
	MLP_layer.append( nn.Linear(NI, NN) )
	MLP_layer.append(actF)
	
	# Hidden layer, if NL < 2 then no hidden layers
	for ly in range(NL-2):
		MLP_layer.append(nn.Linear(NN, NN))
		MLP_layer.append(actF)
   
	# Output layer
	MLP_layer.append(nn.Linear(NN, NO))
	
	# seq
	return nn.Sequential(*MLP_layer)
#-----------------------------------------------------------#



#-----------------------------------------------------------#
# general function to plot training and testing curves
# Inputs:
#        path: where to save the data
#        training: training losses
#        testing:  testing losses
#        args: plotting args

def TT_plot(PATH, training, testing, ylabel, yscale = 'normal' ):

	# plotting specs
	fs = 24
	plt.rc('font',  family='serif')
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	plt.rc('text',  usetex=True)

	# plot loss curves
	fig1 = plt.figure(figsize=(10,8))

	# If apply axis scaling
	if yscale == 'semilogy':
		plt.semilogy(training, '-b', linewidth=2, label = 'Training');
		plt.semilogy(testing, '-r', linewidth=2, label = 'Testing');
	else:
		plt.plot(training, '-b', linewidth=2, label = 'Training');
		plt.plot(testing, '-r', linewidth=2, label = 'Testing');
	

	matplotlib.rc('font', size=fs+2)
	plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
	plt.ylabel(ylabel,fontsize=fs)
	plt.tick_params(labelsize=fs+2)
	plt.legend(fontsize=fs-3)
	   
	# save the fig   
	fig_name = PATH + '/'+ ylabel +'.png'
	plt.savefig(fig_name)


	# save the data
	train_name   = PATH + '/' + ylabel + '-train.csv'
	test_name    = PATH + '/' + ylabel + '-test.csv'
	
	np.savetxt(train_name, training,   delimiter = ',')
	np.savetxt(test_name, testing,   delimiter = ',')
			
	return 0
#----------------------------------------------------------------#