# Missing data inference of the EHR dataset: post-processing

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

import numpy as np
from matplotlib import pyplot as plt
from Tools.EHR_tools import *
import os
import matplotlib
import torch

torch.set_default_dtype(torch.float64)

plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')  # Load amsmath for math features
plt.rcParams.update({'figure.max_open_warning': 0})
# ----------------------------------------- Noise in the literature -------------------------------------- #
noise_literature = 	np.array([  3.0,      # hr
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
								50.0/80,  # SVR     # need to divided by 80 (HRU to cgs)
								5.0/80    # PVR     # need to divided by 80 (HRU to cgs)
								])   
# -------------------------------------------------------------------------------------------------------- #


# list of different noise level tested
noise_level_set = [0.25, 0.5, 1.0, 2.0]

# load selected patient IDs (same for all noise levels)
patient_IDs = np.genfromtxt('Model/CVSIM6/EHR/noise-level=0.25/fig/inverse/Selected_patient_IDs.csv', delimiter=',').astype(int)

print('A total of ' + str(len(patient_IDs)) + ' patients are considered..')


# =========================== Task 1: Use global accuracy to decide which noise level to use ============================= #
# ---****Checked****---- 05/16/2024
# noise level loop
for nl in noise_level_set:

	print('\n Current noise level tested is: ' + str(nl))

	# initialize error 
	global_error    = np.zeros((16,))

	# component loop
	# compute componenet error average over patients, if the selected patient has a certain measurement
	for k in range(16):
		pk_counter = 0
		# patient loop
		for q in patient_IDs:

			# Loading previously saved errors
			E_file = 'Model/CVSIM6/EHR/noise-level='+str(nl)+'/fig/inverse/Error_per_patient/patient_id='+str(q)+'.csv'
			
			# take the k-th component
			E_k      = np.genfromtxt(E_file, delimiter=',')[:,k] 

			# if the current component is not missing for this patient
			if not np.isnan(E_k[0]): 
				# take the absolute value and compute the average over all latent variable w
				E_k_avg_over_w  = abs(E_k).mean()
				global_error[k] += E_k_avg_over_w
				pk_counter      += 1

		if pk_counter != 0:
			global_error[k] /= pk_counter # compute per-patient error
			print(f"{pk_counter}/{len(patient_IDs)} patients have {output_id_no_unit[k]}, the error is {global_error[k]:.3f}")
		else: # if the component is missing for all patients considered
			global_error[k] = np.nan
			print('0 patient has measurement ' + str( output_id_no_unit[k]  ) )
# =========================================================================================================================== #



# ========================================== Task 2: Show patient-specific plots ============================================== #
# # ---****Checked****---- 05/16/2024
w_cut = 50  # select part of sameples to make the figure not too packed
for nl_test in noise_level_set: 
	# create folder
	fig_path_k = 'Model/CVSIM6/EHR/noise-level='+str(nl_test)+'/fig/inverse/Patient-Sepc_pics/'
	os.makedirs(fig_path_k, exist_ok = True)
	
	# component loop
	for k in range(16):	
		plt.figure(figsize=(7,1.8))  # one picture per component

		# patient loop
		for q in patient_IDs:
			
			# ---------------------------------------------------- Load data ------------------------------------------------- #		
			# load emulator predictions
			Emu_file = 'Model/CVSIM6/EHR/noise-level='+str(nl_test)+'/fig/inverse/EmuPred_per_patient/patient_id='+str(q)+'.csv'
			Emu_k    = np.genfromtxt(Emu_file, delimiter=',')[:w_cut,k] 
			
			# load ehr data 
			_, y_patient = EHR_extraction(q)
			# ---------------------------------------------------------------------------------------------------------------- #

			# If the current patient has the k-th component
			if y_patient[k] is not None:

				if k not in [12,14,15]: # if not LVEF, SVR and PVR
					# plot Nw emulator predictions
					plt.plot(np.zeros( len(Emu_k)) + q , Emu_k, 'b.', alpha = 0.05, markersize = 4)
					# plot the single EHR data
					plt.scatter(q, y_patient[k], s = 40, marker = '*', color='r', alpha = 0.8)
					# shade the area of uncertainty
					plt.fill_betweenx(
							[y_patient[k] - 3*noise_literature[k], y_patient[k] + 3*noise_literature[k] ], 
							q - 0.37, q + 0.37, color = 'gray', alpha = 0.5)

				if k == 12: # For LVEF, need to convert precentage to fraction
					# plot Nw emulator predictions
					plt.plot(np.zeros( len(Emu_k)) + q , Emu_k, 'b.', alpha = 0.05, markersize = 4)
					# plot the single EHR data
					plt.scatter(q, y_patient[k]/100, s = 40, marker = '*', color='r', alpha = 0.8)
					# shade the area of uncertainty
					plt.fill_betweenx(
							[y_patient[k]/100 - 3*noise_literature[k], y_patient[k]/100 + 3*noise_literature[k] ], 
							q - 0.37, q + 0.37, color = 'gray', alpha = 0.5)

				if k == 14 or k == 15: # For PVR/SVR, need to convert HRU to cgs
					# plot Nw emulator predictions
					plt.plot(np.zeros( len(Emu_k)) + q , Emu_k*80., 'b.', alpha = 0.05, markersize = 4)
					# plot the single EHR data
					plt.scatter(q, y_patient[k], s = 40, marker = '*', color='r', alpha = 0.8)
					# shade the area of uncertainty
					plt.fill_betweenx(
							[y_patient[k] - 3*noise_literature[k]*80, y_patient[k] + 3*noise_literature[k]*80 ], 
							q - 0.37, q + 0.37, color = 'gray', alpha = 0.5)

					#plt.xlabel('Patient ID', fontsize = 18)
		plt.ylabel(output_id[k], fontsize = 13)
		plt.tick_params(labelsize=11)
		plt.gca().set_axisbelow(True)
		plt.grid(color='0.9')
		plt.xlim([0,85])
		plt.savefig(fig_path_k + 'k=' + str(k) + '.png',bbox_inches='tight',pad_inches = 0.02, dpi = 300)
# ======================================================================================================================= #