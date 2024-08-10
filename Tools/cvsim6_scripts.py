import numpy as np
import os
from Tools.plotter import *
from matplotlib import pyplot as plt
import copy
from Solver.cvsim6_simulator import *
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, PercentFormatter
from scipy import stats
import scipy as sp
from sklearn.preprocessing import StandardScaler
import paxplot
import torch
torch.set_default_dtype(torch.float64)

DPI = 300

# unit conversion
mmHg2Barye = 1333.22
HRU2cgs    = 80.

# Define input IDs
# ---****Checked****---- 05/01/2024
# --------------------------------------------------------------------------------------------------------------------------------- #
input_id =  ['$Hr$ (bpm)', '$P_{th}$ (mmHg)','$r_{sys}$ (-)', '$C_{l,dia}$ (mL/Barye)',
			'$C_{l,sys}$ (mL/Barye)', '$C_{a}$ (mL/Barye)', '$C_{v}$ (mL/Barye)',
			'$C_{r,dia}$ (mL/Barye)', '$C_{r,sys}$ (mL/Barye)', '$C_{pa}$ (mL/Barye)',
			'$C_{pv}$ (mL/Barye)', '$R_{l,in}$ (Barye$\cdot$s/mL)', '$R_{l,out}$ (Barye$\cdot$s/mL)', '$R_{a}$ (Barye$\cdot$s/mL)',
			'$R_{r,in}$ (Barye$\cdot$s/mL)', '$R_{r,out}$ (Barye$\cdot$s/mL)', '$R_{pv}$ (Barye$\cdot$s/mL)', '$V_{l}^{0}$ (mL)',
			'$V_{a}^{0}$ (mL)', '$V_{v}^{0}$ (mL)', '$V_{r}^{0}$ (mL)', '$V_{pa}^{0}$ (mL)', '$V_{pv}^{0}$ (mL)']


input_id_no_unit =  ['$Hr$', '$P_{th}$','$r_{sys}$', '$C_{l,dia}$',
					'$C_{l,sys}$', '$C_{a}$', '$C_{v}$',
					'$C_{r,dia}$', '$C_{r,sys}$', '$C_{pa}$',
					'$C_{pv}$', '$R_{l,in}$', '$R_{l,out}$', '$R_{a}$',
					'$R_{r,in}$', '$R_{r,out}$', '$R_{pv}$', '$V_{l}^{0}$',
					'$V_{a}^{0}$', '$V_{v}^{0}$', '$V_{r}^{0}$', '$V_{pa}^{0}$', '$V_{pv}^{0}$']


# define reference parameter (Davis 1991)
input_refs = np.array([72. , -4., 1./3, \
							10., 0.4, 1.6, 100., 20., 1.2, 4.3, 8.4, \
									0.01, 0.006, 1., 0.05, 0.003, 0.08, \
											15., 715., 2500., 15., 90., 490. ])
# --------------------------------------------------------------------------------------------------------------------------------- #


# Define output IDs, consistent with the paper
# ---****Checked****---- 05/01/2024
# ----------------------------------------------------------------------------------------------------------------- #
output_id = ['$Hr$ (bpm)', '$P_{a,sys}$ (mmHg)','$P_{a,dia}$ (mmHg)', '$P_{r,sys}$ (mmHg)', '$P_{r,dia}$ (mmHg)',
			'$P_{pa,sys}$ (mmHg)','$P_{pa,dia}$ (mmHg)','$P_{r,edp}$  (mmHg)', '$P_{w}$ (mmHg)',
			'$P_{cvp}$ (mmHg)','$V_{l,sys}$ (mL)','$V_{l,dia}$ (mL)','LVEF (-)','CO (L/min)',\
			'SVR (dyn$\cdot$s/cm$^5$)','PVR (dyn$\cdot$s/cm$^5$)']


output_id_no_unit = ['$Hr$', '$P_{a,sys}$','$P_{a,dia}$', '$P_{r,sys}$', '$P_{r,dia}$',
					'$P_{pa,sys}$','$P_{pa,dia}$','$P_{r,edp}$', '$P_{w}$',
					'$P_{cvp}$','$V_{l,sys}$','$V_{l,dia}$','LVEF','CO','SVR','PVR']
# ----------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
# ---****Tested****---- 05/01/2024
# make cvsim6 outputs units consistent to the paper
# SVR/PVR: HRU --> cgs
# Note: A deepcopy is created to avoid the input getting modified
# inputs:
#       y: num_sample x num_of_feature_in_output
def unit_change_output(y):
	assert y.shape[1] == 16, "dimension error in output!"
	y_copy = copy.deepcopy(y)
	y_copy[:,-2] *= HRU2cgs # for SVR
	y_copy[:,-1] *= HRU2cgs # for PVR
	return y_copy
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
# ---****Tested****---- 05/01/2024
# make cvsim6 inputs units consistent to the paper
# C: mL/mmHg --> mL/Barye, R: mmHg\cdot s/mL --> Barye$\cdot$s/mL
# Note: A deepcopy is created to avoid the input getting modified
# inputs:
#       v: num_sample x num_of_feature_in_input
def unit_change_input(v):
	assert v.shape[1] == 23, "dimension error in input!"
	v_copy = copy.deepcopy(v)
	v_copy[:,3:11]  /= mmHg2Barye # for capacitances
	v_copy[:,11:17] *= mmHg2Barye # for resistances
	return v_copy
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/01/2024
# solve the cvsim6 system using inverse prediction (or a given sample)
# inputs:
#        v_input:  inverse predictions (or given inputs): num_sample x dim(v)
# outputs:
#        truncated time, pressure/flow/volume curves and cvsim6 system output
def solve_by_invPred(v_input):

	# ------------ solve for the curves -------------- #
	cvsim6     = simulator_cvsim6()
	t, P, Q, V = [], [], [], [] # solution placeholder
	CV_outputs = []
	# ------------------------------------------------ #
	
	# ------------------------ loop tho the number of inverse predictions ----------------------------- #
	for j in range(len(v_input)): # solve for all samples
		
		cv_output,t_sol,P_sol,Q_sol,V_sol = cvsim6.solve( v_input[j], save_traj = True ) 

		t.append(t_sol)
		P.append(P_sol)
		Q.append(Q_sol)
		V.append(V_sol)
		CV_outputs.append(cv_output)
	# ------------------------------------------------------------------------------------------------- #

	# --------------------- concatenate heart rate --------------------- #
	CV_outputs = np.array(CV_outputs)
	CV_outputs = np.concatenate((v_input[:,[0]],  CV_outputs ), axis = 1)
	# ------------------------------------------------------------------- #
	
	return CV_outputs, np.array(t), np.array(P), np.array(Q), np.array(V)
# ------------------------------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/13/2024
# Plotting pressure/volume curves to verify the results
# Inputs:
#       pth: where to save
#       name: name of the picture to be saved
#        t: time array : num_sample x time steps
#        y: solution array : num_sample x time steps
#       t_range: selected time range for plotting
#       y_label: label for the y-axis
#       group: how many groups to be plotted, each group means a different fixed y
#       hl: how many hlines to be super-imposed, these are usually the fixed values
#       hl_legend: legends for the hlines to be plotted
def cvsim6_curve_plotter(pth, name, t, y, t_range, ylabel, group = 1, hl = None, hl_legend = None, \
							alpha = 0.5, fs = 16):

	# find how many latent variables sampled for each group
	wsize = int(t.shape[0]/group)
	print('\nPlotting ' + name + ' :' + str(t.shape[0]) +' total curves of ' + str(group) + ' groups, each of size ' + str(wsize))
	assert t.shape[0] % group == 0, "number of total samples must be divisible by the group number!"

	# define linestyles of the superimposed hlines
	ls = ['--', '-.', ':']

	# define line colors, each color for one group
	c  = ['b', 'r', 'orange', 'green'] 

	# create folder to save figures if not exist 
	os.makedirs(pth, exist_ok = True)

	# start plotting
	fig, ax = plt.subplots(figsize=(8, 2.5))
	
	# plot solution curves one sample by another
	for j in range(t.shape[0]):
		ax.plot(t[j], y[j], alpha = alpha, color = c[j//wsize], linewidth=0.8) # use quotient to decide the group color

	# plot superimposed hlines
	if type(hl) is list:
		for k in range(len(hl)):
			ax.axhline(y = hl[k], color='k', linestyle=ls[k], label=hl_legend[k], linewidth = 1)
		ax.legend(fontsize = fs-2, shadow=False) # show legend
	
	# general setup
	ax.grid(True,color='0.9') # grid skeleton to be lighter
	ax.tick_params(axis='both', which='major', labelsize=fs-2)
	ax.set_xlabel('$t$ (s)',  fontsize=fs+4)
	ax.set_ylabel(ylabel, fontsize=fs+4)
	ax.set_xlim(t_range)     
	fig.savefig(pth+name+'.png', bbox_inches='tight', pad_inches = 0.02, dpi = DPI)
	return None
# -------------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/11/2024
# Calculate the relative l2 error btw NN prediction and
# the true labels, print out component abs diff and save the hist
# Inputs:
#       pth: where to save the figure
#       ypred: NN prediction: num_sample x num_of_feature (units not changed) 
#       ytruth: labels to be compared: num_sample x num_of_feature (units not changed) 
#       hist: if true, save the error histogram
def rel_error_output(pth, ypred, ytruth, hist = False, fs = 16):

	# change the units for resistances (SVR/PVR)
	pred  = unit_change_output(ypred)
	truth = unit_change_output(ytruth)

	# -------------------------------- compute global l2 relative error ----------------------------------- #
	Rel_error = np.linalg.norm(pred - truth, axis=1, ord=2)/np.linalg.norm(truth, axis=1, ord=2) * 100

	print('Worst global error in y across ' + str(len(pred)) + ' samples is: ' + str(Rel_error.max()) + '(%)')
	
	# save the rel error histogram if needed
	if hist == True:
		hist_plot(pth, 'Emulator-Global_rel_L2-hist', Rel_error, \
			'Relative error (\%)' , Cr='b', bins=20, fs = fs, resize=[9,2])
	# ------------------------------------------------------------------------------------------------------ #

	# -------------------------- Print out component abs error ------------------------#
	for j in range(pred.shape[1]):
		abs_diff = abs(pred[:,j] - truth[:,j])
		print('For {}, average: {:.2e}, max: {:.2e}, std: {:.2e}'.format(output_id[j], \
					 abs_diff.mean(), abs_diff.max(), abs_diff.std())  )
	# ---------------------------------------------------------------------------------#

	return None
# ------------------------------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# ---****Tested****---- 05/11/2024
# Checking marginal output distributions via histgrams
# Inputs:
#        pth: where to save
#        ytruth: truth (data distribution), (unit not changed)
#        ypred:  NF prediction, (unit not changed)
def NF_check_hist(pth, ytruth, ypred, fs = 16):
	# change the units for resistances
	pred  = unit_change_output(ypred)
	truth = unit_change_output(ytruth)
	# generate histograms one by one
	for j in range(ytruth.shape[1]):
		# picture name
		pic_name = 'id = '+ str(j)
		# plot histogram
		hist_plot(pth, pic_name, pred[:,j], output_id[j], y=truth[:,j], fs = fs+20, \
				bins = 30, legend_off = True, y_off = True, alpha = 0.33)	
	return 0
# ------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/03/2024
# Given each y in the validation set, cat ONE w, invert and use cvsim6 to forward it back
# Inputs:
#        model: trained inverse model
#        ytruth: the whole validation set (units not changed) 
#        y_scale: scaling constants of Y, mu and std
#        v_scale: scaling constants of V, mu and std
#        w_seed: seed control of latent variable
def inverse_check_reconstruction(model, ytruth, y_scale, v_scale, w_seed = 0): 

	assert len(ytruth)>1, " test least two samples at a time! "

	# ------ decouple scaling constants ------- #
	y_mu, y_std = y_scale[0], y_scale[1]
	v_mu, v_std = v_scale[0], v_scale[1]
	# ----------------------------------------- #

	# init exact cvsim6 solver
	cvsim6 = simulator_cvsim6()

	print('\n'+'------------------ Reconstruction Test on y --------------------------')
	model.eval()

	# scale forward to the validation labels
	ytruth = (ytruth - y_mu)/y_std

	# Inversion one by one in the validation set
	Y_from_inv = []
	for j in range(len(ytruth)): # looping tho all samples in the validation set
		# status monitoring
		if j % 100 == 0:
			print(str(j+1) + '/' + str(len(ytruth)))
		# give different seed (for w) for each y in the validation set to retain randomness
		inv_V = model.inversion_sampling(model, 1, seed_control = j + w_seed, \
											Task = 'FixY', y_given = ytruth[j], denoise = 2) 
		# scale the inverse prediction back
		inv_V = inv_V * v_std + v_mu
		# solve the cvsim6 system with the inverse prediction
		ypred,_,_,_,_ = solve_by_invPred(inv_V)
		# forward the exact cvsim6 solver with the inverse prediction
		Y_from_inv.append( ypred )

	# numpy it
	Y_from_inv   = np.array(Y_from_inv).squeeze() # squeeze the trivial dimension
	# scale the truth back
	ytruth = ytruth * y_std + y_mu
	# print out the componentwise rel error
	rel_error_output(None, Y_from_inv, ytruth)
	return None
# ----------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------ #
# ---****Tested****---- 05/23/2024
# Sample from the latent space for a particular y and solve the inverse problem via decoder
# Inputs:
#        pth: where to save the images
#        wsize: how many latent variables sampled (used in PCA and correlation study)
#        NF_model: trained NF model
#        Inverse_model: trained VAE and decoder
#        y_scale: scaling constants for the outputs
#        v_scale: scaling constants for the inputs
#        y_seed: random seed for y-sampler
#        w_seed: random seed for w-sampler
#        w4error: number of samples used to compute reconstruction error (Using all would be too expensive)
# Outputs:
#        Yfix: interested observation
#        t_sol: time solution
#        p,q,v_sol: pressure/flow and volume solutions
def inverse_given_output(pth, wsize, NF_model, Inverse_model, yscale, vscale, \
											y_seed = 0, w_seed = 0, w4error = 100):

	# ----------- eval mode ------------- #
	NF_model.eval()
	Inverse_model.eval()
	# ----------------------------------- #

	print('\n'+'--------------  Fix y, sample w from the latent space ------------------')

	# ------ decouple scaling constants ------- #
	y_mu, y_std = yscale[0], yscale[1]
	v_mu, v_std = vscale[0], vscale[1]
	# ----------------------------------------- #

	# sample an output from the trained nf density estimator
	Yfix   = NF_model.sampling(1, NF_model, seed_control= y_seed) 

	# inverse problem with fixed observation
	V_samples     = Inverse_model.inversion_sampling(Inverse_model, wsize, \
					seed_control = w_seed , Task = 'FixY',  y_given = Yfix, denoise = 2)

	# study of the 23-dimensional non-identifiable manifold
	manifold_study(pth, V_samples, vscale)

	# scale the quantities back
	# select the first "w4error" samples to compute the error, using all will be too expensive
	V_samples = (V_samples  * v_std + v_mu)[:w4error]
	Yfix      = Yfix        * y_std + y_mu

	# print-out the fixed components for recording
	Yfix_unit_changed = unit_change_output(Yfix) # change the units for SVR and PVR
	for j in range(Yfix.shape[1]):
		print('Fixed ' + str(output_id[j]) + ' is: ' + str(Yfix_unit_changed[0, j]))

	# Solving the cvsim6 system using the inverse preds
	cvoutput, t_sol, p_sol, q_sol, v_sol = solve_by_invPred(V_samples)

	# compute reconst error and show the worst one
	print('\nReconstruction error based on all ' + str(w4error) + ' inverse predictions:')
	rel_error_output(None, cvoutput, Yfix)

	# plot variability of each componnet on the non-identifiable manifold
	plot_var_fixY(pth, V_samples)

	return Yfix, t_sol, p_sol, q_sol, v_sol
# ------------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/22/2024
# see how much input variability the input components can have under a fixed y
# Inputs:
#       pth: where to save the pictures
#       Vpred  : predicted input samples
def plot_var_fixY(pth, Vpred, markersize = 4, alpha = 0.2, fs = 12):

	# create path to save figures
	os.makedirs(pth, exist_ok = True)

	# ----------------- change the units from mmHg to cgs ----------------------- #
	V     = unit_change_input(Vpred)                         # predicted samples
	V_ref = unit_change_input(input_refs.reshape(1,-1))      # reference point 
	# --------------------------------------------------------------------------- #

	# ------------------- ylim scale factor ---------------------- #
	# Note: For the study with structural non-identifiability,
	#       We assume +/- 50% for the C's and +/- 30% for the rest
	# Note: we can have some predictions outside of the 
	#       prior bounds, so we increase +/-5% here
	Cbounds = [0.45, 1.55] # bounds for the capacitances
	bounds  = [0.65, 1.35] # bounds for the rest of parameters
	# ------------------------------------------------------------ #

	# --------------------------------- plot the first half ------------------------------------ #
	fig, axes = plt.subplots(1, 11, figsize=(12, 2.5))
	for i, ax in enumerate(axes):
		
		if i > 2: # for capacitances, semilog the y-axis
			ax.semilogy(np.zeros(V.shape[0]), V[:,i], 'b.', markersize=markersize, alpha=alpha)
			ax.set_ylim(V_ref[0,i]*Cbounds[0], V_ref[0,i]*Cbounds[1])
			ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
			ax.yaxis.set_minor_locator(plt.NullLocator()) # removing minor ticks
		else:     # for hr and rsys, normal plotting
			ax.plot(np.zeros(V.shape[0]), V[:,i], 'b.', markersize=markersize, alpha=alpha)
			ax.set_ylim(V_ref[0,i]*bounds[0], V_ref[0,i]*bounds[1])
			ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
		if i == 1: # for Pth, flip the bounds since it is negative
			ax.set_ylim(V_ref[0,i]*bounds[1], V_ref[0,i]*bounds[0])
		# for Pth and r_sys, save more digits
		if i == 1 or i == 2:	
			ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

		ax.set_xticks([0]) 
		ax.set_xticklabels([input_id_no_unit[i]], rotation=45, fontsize=fs+3)
		ax.set_xlim(-0.05, 0.05)  # Tight x-axis range
		ax.tick_params(axis='y', labelsize=fs-2)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

	plt.subplots_adjust(wspace=2.5, bottom=0.3)
	plt.savefig(pth+'var-first-half.pdf', bbox_inches='tight', pad_inches = 0)
	# -------------------------------------------------------------------------------------------- #

	# ---------------------------------- plot the second half ------------------------------------- #
	fig, axes = plt.subplots(1, 12, figsize=(12, 2.5))
	for i, ax in enumerate(axes):
		ax.plot(np.zeros(V.shape[0]), V[:,i+11], 'b.', markersize=markersize, alpha=alpha)
		ax.set_ylim(V_ref[0,i+11]*bounds[0], V_ref[0,i+11]*bounds[1])
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
		ax.set_xticks([0]) 
		ax.set_xticklabels([input_id_no_unit[i+11]], rotation=45, fontsize=fs+3)
		ax.set_xlim(-0.05, 0.05)  # Tight x-axis range
		ax.tick_params(axis='y', labelsize=fs-2)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

	plt.subplots_adjust(wspace=2.5, bottom=0.3)
	plt.savefig(pth+'var-second-half.pdf', bbox_inches='tight', pad_inches = 0)
	# -------------------------------------------------------------------------------------------- #

	return None
# ------------------------------------------------------------------------------------------------------------ #



# ----------------------------------------------------------------------------------------------------- #
# ---****Tested****---- 05/13/2024
# Inverse problem with missing data in the observation, replace missing data from NF sample component
# Inputs:
#       obs: observation with missing components
#       w_size: number of latent variables to be sampled for each replaced observation
#       NF_model: trained NF model
#       Inverse_model: trained inverse model
#       idx: indices that are not missing (complement set)
#       yscale: scaling constants of y
#       vscale: scaling constants of v
#       group : how many different combinations to be tested, for EHR, always take the most likely one
#       num_samples: number of samples drawn from NNf (Note: if memory problem occurs, reduce sample size)
#       verbose: if true, print out the related information
# Outputs:
#        V_samples: inverse prediction based on the possible combinations
def missing_data_inference(obs, w_size, NF_model, Inverse_model, idx, yscale, vscale, group = 1,\
								num_samples = int(1e6), verbose = True):
	# eval mode for the trained models
	NF_model.eval()
	Inverse_model.eval()

	# for more groups, define more colors in the ```cvsim6_curve_plotter``` function above
	assert group <= 4,"No more than 4 colors defined!"

	# make sure the obs is the correct shape
	assert obs.shape[1] == 16, "wrong shape of the observation!"

	if verbose == True:
		print('\n'+'------------------ Missing data analysis --------------------------')

	# ------ decouple scaling constants ------- #
	y_mu, y_std = yscale[0], yscale[1]
	v_mu, v_std = vscale[0], vscale[1]
	# ----------------------------------------- #

	# -------------------- prepare as much NF samples as possible -------------------- #
	# sample as much 
	Y_samples  = NF_model.sampling(num_samples, NF_model, seed_control = 56386)
	
	# backward scaling (to original scale)
	Y_samples = Y_samples * y_std + y_mu
	# -------------------------------------------------------------------------------- #

	# --------- Insert the known (not missing) components from the observation ----------#
	# recall idx is the index set of components that are not missing
	Y_samples[:,idx] = obs[0, idx]
	Y_samples 		 = (Y_samples - y_mu)/y_std # scale forward for the inverse problem
	# ------------------------------------------------------------------------------------#

	# ------------------------------- Likelihood ranking --------------------------------- #
	_,ll = NF_model.forward(torch.from_numpy(Y_samples)) # get log-likelihood (log-PDF)

	# rank likelihood via descending order
	ll_sorted, sorted_indices = torch.sort(ll, dim = 0, descending=True)

	if verbose == True:
		print('Top '+str(group)+ ' samples log-PDFs are:\n' + str(ll_sorted[:group]))
	# ------------------------------------------------------------------------------------ #

	# -------- re-order the samples by the log-PDF ranking and solve the inverse problem --------#
	Y_samples_sorted = Y_samples[sorted_indices].squeeze() # squeeze trivial dimension

	# init V_samples
	V_samples = []
	for j in range(group): # each time fix a different y (the selected y's having the highest likelihood)
		V_samples.append( Inverse_model.inversion_sampling(Inverse_model, w_size, seed_control = j, \
									Task = 'FixY', y_given = Y_samples_sorted[j], denoise = 2) )

	# flatten the first two dimensions to make it compatible with the plotting function
	V_samples = np.array(V_samples).reshape(-1,23)
	# ---------------------------------------------------------------------------------------------- #
	
	return V_samples * v_std + v_mu
# -------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------------------- #
# study the 23-dim non-identifiable manifold
# Inputs:
#        pth: where to save
#        V_samples: predicted input samples
#        vscale: scaling constants for the input v
#        ps: plotting size for the parallel chart
def manifold_study(pth, V_samples, vscale, fs = 20, ps = 100):

	print('\n'+'--------------  Fix one y, manifold study! ------------------')

	# =================== predicted samples pre-processing ================= #
	# create a deepcopy incase inplace opt is used
	V = copy.deepcopy(V_samples)
	
	# scaling constants for v
	v_mu, v_std = vscale[0], vscale[1]

	# scale it back and change the units
	V     = unit_change_input(V * v_std + v_mu)
	# change the units for the reference input
	V_ref = unit_change_input(input_refs.reshape(1,-1)).flatten()
	# ======================================================================= #

	# ==== SVD for linear correlation and dim-reduction ==== #
	SVD_manifold(pth, V) # for SVD, using all the samples
	# ====================================================== #

	# =========================== Parallel chart ================================ #
	# Credit: https://kravitsjacob.github.io/paxplot/
	#         https://github.com/kravitsjacob/paxplot
	V = V[:ps] # for parallel chart, use part of it to avoid the figure is packed
	# define bounds for each component
	Cbounds = [0.45, 1.55] # bounds for the capacitances
	bounds  = [0.65, 1.35] # bounds for the rest of parameters

	# --------------------------------------- First Half -------------------------------------------- #
	paxfig = paxplot.pax_parallel(n_axes=12)
	paxfig.plot(V[:,:12], line_kwargs = {'alpha': 0.06, 'color': 'blue'} )
	paxfig.set_labels(input_id_no_unit[:12])

	for j in range(12):
		if j == 1: # for Pth, since it is negative
			selected_ticks = np.linspace(bounds[1]*V_ref[j], bounds[0]*V_ref[j], 4)
			paxfig.set_ticks(
				ax_idx = j,
				ticks  = selected_ticks,
				labels = [f'{tick:.2f}' for tick in selected_ticks])
		if j > 2 and j < 11: # for all compliances
			selected_ticks = np.linspace(Cbounds[0]*V_ref[j], Cbounds[1]*V_ref[j], 4)
			paxfig.set_ticks(
			ax_idx = j,
			ticks  = selected_ticks,
			labels = [f'{tick:.1e}' for tick in selected_ticks])
		else:
			selected_ticks = np.linspace(bounds[0]*V_ref[j], bounds[1]*V_ref[j], 4)
			paxfig.set_ticks(
				ax_idx = j,
				ticks  = selected_ticks,
				labels = [f'{tick:.2f}' for tick in selected_ticks])

	paxfig.set_size_inches(18, 3)
	label_size = 18
	for ax in paxfig.axes:
	    ax.tick_params(axis="x", labelsize=label_size)
	for ax in paxfig.axes:
	    ax.tick_params(axis="y", labelsize=label_size-4)
	paxfig.savefig(pth + 'parallel-chart-first_half.jpg', dpi = 400, bbox_inches='tight', pad_inches = 0.02)
	# ---------------------------------------------------------------------------------------------------- #

	# --------------------------------------- Second Half -------------------------------------------- #
	paxfig = paxplot.pax_parallel(n_axes=12)
	paxfig.plot(V[:,11:], line_kwargs = {'alpha': 0.06, 'color': 'blue'} ) # the last one from the first row is repeated
	paxfig.set_labels(input_id_no_unit[11:])
	for j in range(12):
		selected_ticks = np.linspace(bounds[0]*V_ref[j+11], bounds[1]*V_ref[j+11], 4)
		paxfig.set_ticks(
			ax_idx = j,
			ticks  = selected_ticks,
			labels = [f'{tick:.2f}' for tick in selected_ticks])
	paxfig.set_size_inches(18, 3)
	label_size = 18
	for ax in paxfig.axes:
	    ax.tick_params(axis="x", labelsize=label_size)
	for ax in paxfig.axes:
	    ax.tick_params(axis="y", labelsize=label_size-4)
	paxfig.savefig(pth + 'parallel-chart-second_half.jpg', dpi = 400, bbox_inches='tight', pad_inches = 0.02)
	# ---------------------------------------------------------------------------------------------------- #


	return None
# ---------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------ #
# ---****Tested****---- 07/20/2024
# SVD for dimensional reduction analysis
# Inputs:
#        pth: where to save the pictures
#        V4SVD: inverse prediction samples 
#        mark_nth: mark the nth singular value and the associated cumulative energy (starting from 0)
def SVD_manifold(pth, V4SVD, mark_nth = 11, fs = 16):

	# create path to save figures if not exist already
	os.makedirs(pth, exist_ok = True)
	V = copy.deepcopy(V4SVD) # create a deepcopy in case in-place modified

	# --------------------------------------------- #
	# scale the data to reduce magnitude effect
	scaler         = StandardScaler()
	V              = scaler.fit_transform(V)
	# --------------------------------------------- #

	# ------------------------------- #
	# Singular value decomposition
	_, S, _ = np.linalg.svd(V)
	# ------------------------------- #

	# ------------------------------ #
	# calculate cumulative energy
	cE = np.cumsum(S**2)/np.sum(S**2)
	# ------------------------------ #

	# ------------------------ plot the singular value spectrum --------------------------- #
	fig, ax = plt.subplots(figsize=(6, 2))	
	plt.semilogy(np.arange(len(S)), S, 'k*', markersize = 8, alpha = 0.8)
	if mark_nth != False:
		plt.semilogy(mark_nth, S[mark_nth], 'r*', markersize = 8, alpha = 0.8)
	ax.grid(True,color='0.8') # grid skeleton to be lighter
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.set_xlabel('Number',  fontsize=fs)
	ax.set_ylabel('Singular Value', fontsize=fs)
	plt.savefig(pth+'singular_value.png', bbox_inches='tight', pad_inches = 0.02, dpi = 300)
	# ------------------------------------------------------------------------------------- #

	# ------------------------- plot the cumulative energy ------------------------------- #
	fig, ax = plt.subplots(figsize=(6, 2))	
	plt.plot(np.arange(len(S)), cE, 'k*', markersize = 8, alpha = 0.8)
	if mark_nth != False:
		plt.plot(mark_nth, cE[mark_nth], 'r*', markersize = 8, alpha = 0.8)
		plt.annotate(f'CE={cE[mark_nth]*100:.2f}\%', \
					xy=(mark_nth+0.2, cE[mark_nth]-0.15), fontsize=fs,color='red')
	ax.grid(True,color='0.8') # grid skeleton to be lighter
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.set_xlabel('Number',  fontsize=fs)
	ax.set_ylabel('CE (\%)', fontsize=fs)
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	plt.savefig(pth+'sum_energy.png', bbox_inches='tight', pad_inches = 0.02, dpi = 300)
	# ------------------------------------------------------------------------------------- #
	
	return None
# ------------------------------------------------------------------------------------------------- #
