# Noise-free emulator training for the CVSim-6 model used in EHR data inference
import time
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from Tools.DNN_tools import *
from Tools.Model import *
from Tools.Training_tools import *

torch.set_default_dtype(torch.float64)

plt.rcParams.update({'figure.max_open_warning': 0})

# Basic setup
#--------------------------------------------------------------------------------------#
# determine if to use GPU, GPU is faster with large mini-batch
device = 'cpu'
folder_name ='Model/CVSIM6/EHR/NoiseFreeEmulator/'
os.makedirs(folder_name, exist_ok = True)
#--------------------------------------------------------------------------------------#

# Data Loading and scaling (always do scaling)
# ------------------------------------------------------------------------------------- #
X = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-input.csv', delimiter=',')
Y = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-output.csv', delimiter=',')

# exclude the validation set, validation set is for offline
Sample_size = 50000
X,Y         = X[:Sample_size], Y[:Sample_size]

# -------------------------------- scaling ------------------------------------------- #
# call standard scaler for the input
scalerX = StandardScaler()
X       = scalerX.fit_transform(X)
# save scaling constants
np.savetxt(folder_name+'X-mu.csv',  scalerX.mean_, delimiter=',')
np.savetxt(folder_name+'X-std.csv', scalerX.scale_, delimiter=',')

# call standard scaler for the output
scalerY = StandardScaler()
Y       = scalerY.fit_transform(Y)
# save scaling constants
np.savetxt(folder_name+'Y-mu.csv',  scalerY.mean_, delimiter=',')
np.savetxt(folder_name+'Y-std.csv', scalerY.scale_, delimiter=',')
# ------------------------------------------------------------------------------------- #


# Define inVAErt class
# ----------------------------------------------------------------------------------------- #
dimV        = X.shape[1]   # dimension of the input
dimY        = Y.shape[1]   # dimension of the output
print('Input dimension:' + str(dimV) + ', output dimension:' + str(dimY))
para_dim    = dimV         # in this case, no aux data

# -------------this is not used here, but needed for code consistency---------------- #
latent_plus = 12           # how many additional dimension added to the latent space
# ------------------------------------------------------------------------------------ #

# Define inVAErt class
InVAErtClass     = inVAErt(dimV, dimY, para_dim, device, latent_plus)
# ----------------------------------------------------------------------------------------- #

# ------------------  Define hyper-parameters of the InVAErt modules  -------------------------------------- #
em_para      = [80,8,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for the emulator

# ------------- Not used, needed for code consistency -------------------------------------------------- #
nf_para      = [32,4,6,False]        # num_of_neuron, num_of_layer_per_block, num_of_block, if_use_BN
vae_para     = [32,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for VAE encoder 
decoder_para = [64,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for decoder 
# ------------------------------------------------------------------------------------------------------- #

num_epochs   = [25000, None, None]  # number of epoches for each module
nB           = [256, None, None]    # minibatch size for each module
lr           = [3e-3, None, None]   # initial learning rate for each module
steps        = [100, None, None]    # steps in LR scheduler for each module
decay        = [0.98, None, None]  # lr decay rate in LR scheduler for each module
l2_decay     = [0., None, None]		# penalty for l2 regularization for each module
# ------------------------------------------------------------------------------------------------------------#

# --------------- save hyperparameter info----------------#
# write parameter file to model saving folder
with open(folder_name +  'Hyper-parameter.txt', 'w') as f:
	f.write('Emulator_para:' + str(em_para))
	f.write('\n')
	f.write('Mini-Batch_size:' + str(nB))
	f.write('\n')
	f.write('init lr:' + str(lr))
	f.write('\n')
	f.write('lr decay:' + str(decay))
	f.write('\n')
	f.write('lr decay step:' + str(steps))
	f.write('\n')
	f.write('l2 weight decay:' + str(l2_decay))
	f.write('\n')
	f.write('Num_epochs:' + str(num_epochs))
#--------------------------------------------------------#

# ---------------------- Define inVAErt components (models are of double precision) -------------------------- #
# Skipping NF and VAE models here
Emulator_model, _, _ = InVAErtClass.Define_Models(em_para, nf_para, vae_para, decoder_para)
# ------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------------------------------------------------------------------------------- #
# Train emulator
t_em = time.time()
InVAErtClass.Emulator_train_test(folder_name, X, Y, Emulator_model, num_epochs[0], nB[0], \
																lr[0], steps[0], decay[0], l2_decay = l2_decay[0])
print('Training emulator: total time in hours is: ' + str( (time.time() - t_em)/3600.  ) + ' Hrs')
# ---------------------------------------------------------------------------------------------------------------- #

