# InVAErt network training for the CVSim-6 model without noise
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
device = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

# path for saving the trained neural network model 
folder_name ='Model/CVSIM6/Structural_id_study/'

os.makedirs(folder_name, exist_ok = True)
#--------------------------------------------------------------------------------------#

# Data Loading and scaling (always do scaling)
# ------------------------------------------------------------------------------------- #
X = np.genfromtxt('Solver/CVSIM6-Training-Data/Structural_Study/cvsim6-input.csv', delimiter=',')
Y = np.genfromtxt('Solver/CVSIM6-Training-Data/Structural_Study/cvsim6-output.csv', delimiter=',')

# exclude the validation set
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
latent_plus = 12           # how many additional dimension added to the latent space

# Define inVAErt class
InVAErtClass     = inVAErt(dimV, dimY, para_dim, device, latent_plus)
# ----------------------------------------------------------------------------------------- #

# ------------------  Define hyper-parameters of the InVAErt modules  -------------------------------------- #
em_para      = [60,8,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for the emulator
nf_para      = [24,4,12,False]       # num_of_neuron, num_of_layer_per_block, num_of_block, if_use_BN
vae_para     = [32,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for VAE encoder 
decoder_para = [64,6,'silu']         # num_of_neuron, num_of_layer, type_of_act fun for decoder 
num_epochs   = [20000, 1000, 20000]  # number of epoches for each module
nB           = [256, 400, 256]       # minibatch size for each module
lr           = [1e-3, 2e-3, 1e-3]    # initial learning rate for each module
steps        = [100, 200, 100]       # steps in LR scheduler for each module
decay        = [0.98, 0.8, 0.985]    # lr decay rate in LR scheduler for each module
l2_decay     = [0., 0., 0.]		     # penalty for l2 regularization for each module
penalty      = [1, 2000, 20]         # penalty for KL div and decoder reconstruction loss and emulator reconst
# ------------------------------------------------------------------------------------------------------------#

# --------------- save hyperparameter info----------------#
# write parameter file to model saving folder
with open(folder_name +  'Hyper-parameter.txt', 'w') as f:
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
print(Emulator_model)
print(NF_model)
print(Inverse_model)
# ------------------------------------------------------------------------------------------------------------ #

# ----------------------------------------- Train each component of inVAErt --------------------------------------- #

# ----------------------------------------------------------------------------------------------------------------- #
# Train emulator
t_em = time.time()
InVAErtClass.Emulator_train_test(folder_name, X, Y, Emulator_model, num_epochs[0], nB[0], \
																lr[0], steps[0], decay[0], l2_decay = l2_decay[0])
print('Training emulator: total time in hours is: ' + str( (time.time() - t_em)/3600.  ) + ' Hrs')
# ----------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------- #
# Train Real-NVP sampler
t_nf = time.time()
InVAErtClass.NF_train_test(folder_name, X, Y, NF_model, num_epochs[1], nB[1], lr[1], steps[1], decay[1], \
																						l2_decay=l2_decay[1], clip = True)
print('Training real-nvp: total time in hours is: ' + str( (time.time() - t_nf)/3600.  ) + ' Hrs')
# -------------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------------------ #
# Train the inverse model
t_inv = time.time()
InVAErtClass.Inverse_train_test(folder_name, X, Y, Inverse_model, num_epochs[2], nB[2], lr[2], steps[2], decay[2],\
																	penalty, l2_decay = l2_decay[2], EN = Emulator_model)
print('Training inverse problem: total time in hours is: ' + str( (time.time() - t_inv)/3600.  ) + ' Hrs')
# ------------------------------------------------------------------------------------------------------------------------------ #