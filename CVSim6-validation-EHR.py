# InVAErt network validation for the CVSim-6 model with the EHR dataset
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from Tools.DNN_tools import *
from Tools.Model import *
from Tools.Training_tools import *
from Tools.cvsim6_scripts import *
from Tools.plotter import *
from Tools.EHR_tools import *

torch.set_default_dtype(torch.float64)

plt.rcParams.update({'figure.max_open_warning': 0})

# Basic setup
#--------------------------------------------------------------------------------------#
# determine if to use GPU, GPU is faster with large mini-batch
device        = 'cpu'
# -------------------------------------------------------------------------------------- #

# ======================================================================================= #
# ********* Tested: 06/20/2024 ***********
# Check noise-free emulator first
print('\n'+'---------------------- Noise Free Emulator check ----------------------------')
# load the dataset for validation
X = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-input.csv', delimiter =',')
Y = np.genfromtxt('Solver/CVSIM6-Training-Data/EHR/cvsim6-output.csv', delimiter=',')

# exclude the training set and use the validation dataset
Sample_size     = 50000
X_val,Y_val     = X[Sample_size:], Y[Sample_size:]

# use the trained, noise-free emulator to test the validation set
emulator_path = 'Model/CVSIM6/EHR/NoiseFreeEmulator/'
em_para       = [80,8,'silu']   # num_of_neuron, num_of_layer, type_of_act fun for the emulator 
Y_pred        = NoiseFreeEmulatorPrediction(emulator_path, em_para, X_val)

# Save/print out the results
em_path_fig = emulator_path + 'fig/emulator/'
rel_error_output(em_path_fig, Y_pred, Y_val, hist = True)
# =======================================================================================  #



# Test noisy models of different levels at the same time
# =============================================================================================================== #
# Define noisy models
noise_level_set = [0.25, 0.5, 1.0, 2.0]

for noise_level in noise_level_set:
	
	print('\n-----------------------------------------')
	print('Current noise level is: ' + str(noise_level) + '\n')

	# compute the amount of noise used during training
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
	
	# folder name of the model of certain noise
	folder_name ='Model/CVSIM6/EHR/noise-level='+str(noise_level) + '/'

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

	# Define and load the trained NF and decoder models
	# Note (1): No noisy emulator exists in this folder, but the code requires its definition, we just skip it 
	#       when loading the trained weights.
	# Note (2): Different noisy levels suppose to have different hyperparameters for the best performance, here we
	#       just fix one set of hyperparameters for comparison
	#------------------------------------------------------------------------------------------------------------ #
	nf_para      = [18, 4, 10, True]   # num_of_neuron, num_of_layer_per_block, num_of_block, if_use_BN
	vae_para     = [32,6,'silu']       # num_of_neuron, num_of_layer, type_of_act fun for VAE encoder 
	decoder_para = [64,6,'silu']       # num_of_neuron, num_of_layer, type_of_act fun for decoder 


	# Define inVAErt components (skipping emulator)
	_, NF_model, Inverse_model = InVAErtClass.Define_Models(em_para, nf_para, vae_para, decoder_para)

	# load trained weights (skipping emulator)
	NF_model.load_state_dict(           torch.load(folder_name+'NF_model.pth',       map_location=device)) 
	Inverse_model.load_state_dict(      torch.load(folder_name+'Inverse_model.pth',  map_location=device)) 
	# ------------------------------------------------------------------------------------------------------------ #

	# # ====================================== NF check ================================== #
	# # ********* Tested: 05/12/2024 ***********
	# # make sure the distribution is learned
	# NF_model.eval()
	# print('\n'+'------------------------------- NF check --------------------------------')

	# # sample z from N(0,1) and transform back to y
	# Y_hat_samples = NF_model.sampling(len(Y), NF_model)

	# # scale it back 
	# Y_hat_samples = Y_hat_samples * y_std + y_mu

	# # add the same amount of noise to the true labels for comparison
	# Y_noise       = Y + np.random.randn(Y.shape[0], Y.shape[1]) * noise_std

	# # plot the histograms for the marginals 
	# fig_path_nf = folder_name + 'fig/realnvp/'
	# NF_check_hist(fig_path_nf, Y_noise, Y_hat_samples)    
	# # ==================================================================================== #



	# ======================================= Inverse check ======================================= #
	# --------- Missing data inference with the EHR dataset ------------
	fig_path_ehr = folder_name + 'fig/inverse/'
	patient_ID   = [] # placeholder for patient IDs

	# see if interested patient measurement fall into the range of the output
	for i in range(1,85): # Note: patient id starts with 1 instead of 0

		# extract the missing indices and available measurements per patient
		missing_set, y_patient = EHR_extraction(i)
		complement_index       = list(set(range(dimY)) - missing_set) # this set is not missing
		y_patient              = np.array(y_patient).reshape(1,-1)
		w_size                 = 100  # how many latent variable samples needed for each inference

		# ----------- skip patient if the number of measurement is less than a certain number -------------- #
		if len(complement_index) > 10:

			# keep the patient ID if selected
			patient_ID.append(i)

			# unit change to make it consistent with the current cvsim-6 model
			if y_patient[0,12]  is not None:
				y_patient[0,12] /= 100.     # for LVEF, change unit from percentage to fraction 
			if y_patient[0, 14] is not None:
				y_patient[0,14] /= 80.      # for SVR change unit from cgs to HRU
			if y_patient[0, 15] is not None:
				y_patient[0,15] /= 80.      # for PVR change unit from cgs to HRU	

			# Missing data inference, get the MOST POSSIBLE point
			V_ehr = missing_data_inference(y_patient, w_size, NF_model, Inverse_model, complement_index, \
							[y_mu, y_std], [x_mu, x_std], group = 1, num_samples = int(5.1e6), verbose = False)

			# forward the pretrained noise-less emulator with the inverse predictions
			yehr_EmuPred = NoiseFreeEmulatorPrediction(emulator_path, em_para, V_ehr)

			# -------------------- save V_ehr, yehr_Emu for post processing --------------------- #
			V_path = fig_path_ehr + 'InputPred_per_patient/'
			os.makedirs(V_path, exist_ok = True)
			np.savetxt( V_path + 'patient_id=' + str(i) + '.csv', V_ehr, delimiter = ',')

			Yemu_path = fig_path_ehr + 'EmuPred_per_patient/'
			os.makedirs(Yemu_path, exist_ok = True)
			np.savetxt( Yemu_path + 'patient_id=' + str(i) + '.csv', yehr_EmuPred, delimiter = ',')
			# ------------------------------------------------------------------------------------ #

			# save patient-specific errors
			error_per_patient(i, fig_path_ehr, yehr_EmuPred, y_patient, complement_index)
			# ---------------------------------------------------------------------------------------------- #

	# save patient IDs
	patient_ID = np.array(patient_ID,dtype=int)
	np.savetxt(fig_path_ehr + 'Selected_patient_IDs.csv', patient_ID, delimiter = ',')
	# ============================================================================================= #



