# InVAErt network validation for the CVSim-6 model without noise
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from Tools.DNN_tools import *
from Tools.Model import *
from Tools.Training_tools import *
from Tools.cvsim6_scripts import *
from Tools.plotter import *

torch.set_default_dtype(torch.float64)

plt.rcParams.update({'figure.max_open_warning': 0})

# Basic setup
#--------------------------------------------------------------------------------------#
# determine if to use GPU, GPU is faster with large mini-batch
device = 'cpu'

# path for saving the trained neural network model 
folder_name ='Model/CVSIM6/Structural_id_study/'
# -------------------------------------------------------------------------------------- #

# ----------------------------------------------------------- #
# load scaling constants (always use scaling)
x_mu  = np.genfromtxt(folder_name+'X-mu.csv',  delimiter=',')
x_std = np.genfromtxt(folder_name+'X-std.csv', delimiter=',')
y_mu  = np.genfromtxt(folder_name+'Y-mu.csv',  delimiter=',')
y_std = np.genfromtxt(folder_name+'Y-std.csv', delimiter=',')
# ----------------------------------------------------------- #

# ------------------------------------------------------------------------------------ #
# Define inVAErt class
dimV        = 23           # dimension of the input
dimY        = 16           # dimension of the output
para_dim    = dimV         # in this case, no aux data
latent_plus = 12           # how many additional dimension added to the latent space

# Define inVAErt class
InVAErtClass     = inVAErt(dimV, dimY, para_dim, device, latent_plus)
# -------------------------------------------------------------------------------------#

# Define and load the trained models
#------------------------------------------------------------------------------------------------------------ #
em_para      = [60,8,'silu']       # num_of_neuron, num_of_layer, type_of_act fun for the emulator 
nf_para      = [24,4,16,False]     # num_of_neuron, num_of_layer_per_block, num_of_block, if_use_BN
vae_para     = [32,6,'silu']       # num_of_neuron, num_of_layer, type_of_act fun for VAE encoder 
decoder_para = [64,6,'silu']       # num_of_neuron, num_of_layer, type_of_act fun for decoder 

# Define inVAErt components
Emulator_model, NF_model, Inverse_model = InVAErtClass.Define_Models(em_para, nf_para, vae_para, decoder_para)

# load trained weights (For emulator, always load the noise-less one)
Emulator_model.load_state_dict(     torch.load(folder_name+'Emulator_model.pth', map_location=device))
NF_model.load_state_dict(           torch.load(folder_name+'NF_model.pth',       map_location=device)) 
Inverse_model.load_state_dict(      torch.load(folder_name+'Inverse_model.pth',  map_location=device)) 
# ------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
# load the dataset for validation
X = np.genfromtxt('Solver/CVSIM6-Training-Data/Structural_Study/cvsim6-input.csv', delimiter=',')
Y = np.genfromtxt('Solver/CVSIM6-Training-Data/Structural_Study/cvsim6-output.csv', delimiter=',')

# exclude the training set and use the validation set for checking the performance
Sample_size     = 50000
X_val,Y_val     = X[Sample_size:], Y[Sample_size:]
# ------------------------------------------------------------------------------------------------- #

# ========================== Emulator check  ==========================  #
# ---****Tested****---- 05/01/2024
Emulator_model.eval()
print('\n'+'---------------- Emulator check --------------------')

# normalizing the validation input set
X_val = torch.from_numpy(  ( X_val - x_mu ) / x_std  )

# forward the emulator set (scaled)
y_emu_pred = Emulator_model(X_val).detach().numpy()

# scale it back 
y_emu_pred = y_emu_pred * y_std + y_mu

# Save/print out the results
fig_path = folder_name + 'fig/emulator/'
rel_error_output(fig_path, y_emu_pred, Y_val, hist = True)
# ====================================================================== # 


# ====================================== NF check ================================== #
# ---****Tested****---- 05/11/2024
NF_model.eval()
print('\n'+'---------------- NF check --------------------')

# sample z from N(0,1) and transform back to y
Y_hat_samples = NF_model.sampling(len(Y), NF_model, seed_control = 89153) 

# scale it back 
Y_hat_samples = Y_hat_samples * y_std + y_mu

# plot the histograms for the marginals 
fig_path = folder_name + 'fig/realnvp/'
NF_check_hist(fig_path, Y, Y_hat_samples)    
# ==================================================================================== #


# ========================= Inverse Reconstruction check ================================= #
# ---****Tested****---- 05/03/2024
print('\n'+'---------------- Inverse reconstruction check --------------------')
inverse_check_reconstruction(Inverse_model, Y_val, [y_mu, y_std], [x_mu, x_std])
# ======================================================================================== #


# ================================== Inverse: checking pressure/volume curves ============================================== #
# ---****Tested****---- 05/23/2024
id_path   = folder_name + 'fig/inverse/identifiability/'
W4error      = 100       # number of samples needed for computing reconstruction error and plot input variability
W4PCA        = 5000      # number of samples needed for performing PCA and correlation study
test_seed    = 432751    # selected seed for the output y   
obs, t_pred, p_pred, _, v_pred = inverse_given_output(id_path, W4PCA, NF_model, \
											Inverse_model, [y_mu, y_std], [x_mu, x_std], y_seed = test_seed, w4error=W4error)
# # ----- plot solution curves -----
ps      = 20             # plotting size
t_range = [7.55, 9.87]   # plotting range (remember to change this)
alpha   = 0.12

# Pa-Pasys-Padia
cvsim6_curve_plotter(id_path, 'Pa-Curves', t_pred[:ps], p_pred[:ps, 1], t_range, '$P_a$ (mmHg)', hl=[obs[0,1],obs[0,2]], \
								hl_legend=['$P_{a,sys}$','$P_{a,dia}$'], alpha = alpha)

# Pr-Prsys-Prdia
cvsim6_curve_plotter(id_path, 'Pr-Curves', t_pred[:ps], p_pred[:ps, 3], t_range, '$P_r$ (mmHg)', hl=[obs[0,3],obs[0,4], obs[0,7]], \
								hl_legend=['$P_{r,sys}$','$P_{r,dia}$','$P_{r,edp}$'], alpha = alpha)

# Ppa-Ppasys-Ppadia
cvsim6_curve_plotter(id_path, 'Ppa-Curves', t_pred[:ps], p_pred[:ps, 4], t_range, '$P_{pa}$ (mmHg)', hl=[obs[0,5],obs[0,6]], \
								hl_legend=['$P_{pa,sys}$','$P_{pa,dia}$'], alpha = alpha)

# Vl-Vlsys-Vldia
cvsim6_curve_plotter(id_path, 'Vl-Curves', t_pred[:ps], v_pred[:ps, 0], t_range, '$V_l$ (mL)', hl=[obs[0,10],obs[0,11]], \
								hl_legend=['$V_{l,sys}$','$V_{l,dia}$'], alpha = alpha)
# ============================================================================================================================= #


# ===================================== Inverse: synthetic missing data analysis =================================== #
# # ---****Tested****---- 05/13/2024
missing_data_path   = folder_name + 'fig/inverse/MissingData/'
missing_set         = {2,4,6,8,10,14,15}   # the components in the output corresponding to these indices are missing
complement_index    = list(set(range(dimY)) - missing_set) # this set is not missing
alpha               = 0.25

# use synthetic data, block the components that are assumed missing
obs[:,list(missing_set)] = None

# how many groups will be considered, each group corresponds to a possible output
group  = 4

# for each selected possible output (group), how many latent variables to be sampled
w_each = 5

# start missing data inference. fm: from missing
V_pred_fm = missing_data_inference(obs, w_each, NF_model, Inverse_model, complement_index, \
									[y_mu, y_std], [x_mu, x_std], group = group, num_samples = int(1e6))

# solve the cvsim-6 system by the inverse predictions
cvoutput_fm, t_fm, p_fm, _, v_fm = solve_by_invPred(V_pred_fm)

# Pa-Pasys
cvsim6_curve_plotter(missing_data_path, 'Pa-Curves',  t_fm, p_fm[:,1], t_range, '$P_a$ (mmHg)',    hl=[obs[0,1]], \
													group = group,	hl_legend=['$P_{a,sys}$'], alpha = alpha)
# Pr-Prsys-Predp
cvsim6_curve_plotter(missing_data_path, 'Pr-Curves',  t_fm, p_fm[:,3], t_range, '$P_r$ (mmHg)',    hl=[obs[0,3],obs[0,7]], \
													group = group,	hl_legend=['$P_{r,sys}$','$P_{r,edp}$'], alpha = alpha)
# Ppa-Ppasys
cvsim6_curve_plotter(missing_data_path, 'Ppa-Curves', t_fm, p_fm[:,4], t_range, '$P_{pa}$ (mmHg)', hl=[obs[0,5]], \
													group = group,	hl_legend=['$P_{pa,sys}$'], alpha = alpha)
# Vl-Vldia
cvsim6_curve_plotter(missing_data_path, 'Vl-Curves',  t_fm, v_fm[:,0], t_range, '$V_l$ (mL)',      hl=[obs[0,11]], \
													group = group,	hl_legend=['$V_{l,dia}$'], alpha = alpha)


# Calculate reconstruction error based on the components that are not missing
# Note: no need to change units for SVR/PVR here, since they are assumed missing and not considered here
obs_avi     = obs[:,complement_index]         # this is the truth
NN_pred     = cvoutput_fm[:,complement_index] # this is the prediction
avi_error   = np.linalg.norm(obs_avi - NN_pred, axis=1, ord=2)/np.linalg.norm(obs_avi, axis=1, ord=2) * 100
print('Worst-case rel error of the prediction is: ' + str(avi_error.max()) + '(%)')
# ======================================================================================================================== #
