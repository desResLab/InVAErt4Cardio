# InVAErt network training for the CVSim-6 model for the EHR data inference
import time
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from Tools.DNN_tools import *
from Tools.Model import *
from Tools.Training_tools import *

plt.rcParams.update({'figure.max_open_warning': 0})

torch.set_default_dtype(torch.float64)

# Basic setup
#--------------------------------------------------------------------------------------#
# determine if to use GPU, GPU is faster with large mini-batch
device = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# looping tho different noise levels
for noise_level in [0.25, 0.5, 1.0, 2.0]:

	print('Current noise level is:' + str(noise_level))

	# calculate standard deviation of each output components
	noise_std   = noise_level * np.array([  3.0,      # hr
											1.5,      # Pa_sys
											1.5,      # Pa_dia
											1.0,      # Pr_sys
											1.0,      # Pr_dia
											1.0,      # Ppa_sys
											1.0,      # Ppa_dia
											1.0,      # Pr_edp
											1.0,      # Pw
											0.5,      # P_cvp
											10.0,     # V_lsys
											20.0,     # V_ldia
											2./100,   # LVEF    # need to divided by 100 (percentage to fraction)
											0.2,      # CO
											50.0/80,  # SVR     # need to divided by 80 (cgs to HRU)
											5.0/80    # PVR     # need to divided by 80 (cgs to HRU)
											])   

	# path for saving the trained neural network model 
	folder_name ='Model/CVSIM6/EHR/noise-level='+str(noise_level) + '/'

	os.makedirs(folder_name, exist_ok = True)
	#--------------------------------------------------------------------------------------#

	# Data Loading and scaling (always do scaling)
	# ------------------------------------------------------------------------------------- #
	X = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-input.csv', delimiter=',')
	Y = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-output.csv', delimiter=',')

	# exclude the validation set
	Sample_size = 50000
	X,Y         = X[:Sample_size], Y[:Sample_size]

	# ------------------------------ scaling ----------------------------------------- #
	# call standard scaler for the input
	scalerX = StandardScaler()
	X       = scalerX.fit_transform(X)
	# save scaling constants
	np.savetxt(folder_name+'X-mu.csv',  scalerX.mean_, delimiter=',')
	np.savetxt(folder_name+'X-std.csv', scalerX.scale_, delimiter=',')

	# For output, need to consider noise
	ymu    = Y.mean(axis = 0) # data mean
	yscale = np.sqrt( Y.std(axis = 0) ** 2 + noise_std ** 2 )
	# save scaling constants
	np.savetxt(folder_name+'Y-mu.csv',  ymu, delimiter    = ',')
	np.savetxt(folder_name+'Y-std.csv', yscale, delimiter = ',')
	# forward scaling for y, noise will be added during training
	Y = (Y - ymu)/yscale
	# ------------------------------------------------------------------------------------- #


	# Define inVAErt class
	# ----------------------------------------------------------------------------------------- #
	dimV        = X.shape[1]   # dimension of the input
	dimY        = Y.shape[1]   # dimension of the output
	print('Input dimension:' + str(dimV) + ', output dimension:' + str(dimY))
	para_dim    = dimV         # in this case, no aux data
	latent_plus = 12           # how many additional dimension added to the latent space

	# Define inVAErt class
	InVAErtClass     = inVAErt(dimV, dimY, para_dim, device, latent_plus)
	# ----------------------------------------------------------------------------------------- #

	# ------------------  Define hyper-parameters of the InVAErt modules  -------------------------------------- #
	# ========================================================================================================== #
	# Note: emulator is not trained in the noisy case, and we just need to load the trained noise-free emulator
	em_para      = [80,8,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for the emulator
	# ========================================================================================================== #

	nf_para      = [18,4,10,True]        # num_of_neuron, num_of_layer_per_block, num_of_block, if_use_BN
	vae_para     = [32,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for VAE encoder 
	decoder_para = [64,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for decoder 
	num_epochs   = [None, 1000, 2000]    # number of epoches for each module
	nB           = [None, 512, 512]      # minibatch size for each module
	lr           = [None, 4e-3, 1e-3]    # initial learning rate for each module
	steps        = [None, 200, 100]      # steps in LR scheduler for each module
	decay        = [None, 0.5, 0.97]     # lr decay rate in LR scheduler for each module
	l2_decay     = [None, 2e-4, 2e-4]    # penalty for l2 regularization for each module
	penalty      = [1, 1000, 10]         # penalty for KL div and decoder reconstruction loss and emulator reconst
	# ------------------------------------------------------------------------------------------------------------#


	# --------------- save hyperparameter info----------------#
	# write parameter file to model saving folder
	with open(folder_name +  'Hyper-parameter-inv.txt', 'w') as f:
		f.write('Emulator_para:' + str(em_para))
		f.write('\n')
		f.write('NF_para:' + str(nf_para))
		f.write('\n')
		f.write('VAE_para:' + str(vae_para))
		f.write('\n')
		f.write('Decoder_para:' + str(decoder_para))
		f.write('\n')
		f.write('Mini-Batch_size:' + str(nB))
		f.write('\n')
		f.write('penalty:' + str(penalty))
		f.write('\n')
		f.write('init lr:' + str(lr))
		f.write('\n')
		f.write('lr decay:' + str(decay))
		f.write('\n')
		f.write('lr decay step:' + str(steps))
		f.write('\n')
		f.write('l2 weight decay:' + str(l2_decay))
		f.write('\n')
		f.write('Latent_plus:' + str(latent_plus))
		f.write('\n') 
		f.write('Num_epochs:' + str(num_epochs))
	#--------------------------------------------------------#


	# ---------------------- Define inVAErt components (models are of double precision) -------------------------- #
	Emulator_model, NF_model, Inverse_model = InVAErtClass.Define_Models(em_para, nf_para, vae_para, decoder_para)

	# Print out model stats
	#print(NF_model)
	#print(Inverse_model)
	# ------------------------------------------------------------------------------------------------------------ #

	# ----------------------------------------- Train each component of inVAErt ----------------------------------------- #
	# decide if add noise to NF and inv
	noise_NF  = True 
	noise_inv = True
	yscale    = torch.from_numpy(yscale).to(device)
	noise_std = torch.from_numpy(noise_std).to(device)

	# -------------------------------------------------------------------------------------------------------------------------- #
	# Train Real-NVP sampler
	t_nf = time.time()
	InVAErtClass.NF_train_test(folder_name, X, Y, NF_model, num_epochs[1], nB[1], lr[1], steps[1], decay[1], l2_decay=l2_decay[1],\
																		noise = noise_NF, yscale = yscale, noise_std = noise_std)
	print('Training real-nvp: total time in hours is: ' + str( (time.time() - t_nf)/3600.  ) + ' Hrs')
	# -------------------------------------------------------------------------------------------------------------------------- #

	#  -------------------------------- Load previously pre-trained emulator ----------------------------------- #
	emulator_path = 'Model/CVSIM6/EHR/NoiseFreeEmulator/Emulator_model.pth'     # the path of pre-trained emulator
	if os.path.exists(emulator_path):
		print('Pre-trained emulator exists!')
	Emulator_model.load_state_dict(torch.load(emulator_path, map_location=device)) # load learned weights
	#----------------------------------------------------------------------------------------------------------- #


	# ------------------------------------------------------------------------------------------------------------------------------ #
	# Train the inverse model
	t_inv = time.time()
	InVAErtClass.Inverse_train_test(folder_name, X, Y, Inverse_model, num_epochs[2], nB[2], lr[2], steps[2], decay[2],\
			penalty, l2_decay = l2_decay[2], noise = noise_inv, yscale = yscale, noise_std = noise_std, EN = Emulator_model)
	print('Training inverse problem: total time in hours is: ' + str( (time.time() - t_inv)/3600.  ) + ' Hrs')
	# ------------------------------------------------------------------------------------------------------------------------------ #