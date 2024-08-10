# sort and extract the EHR dataset to make it compatible with the cvsim-6 model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from Tools.Model import *

import torch
torch.set_default_dtype(torch.float64)

# ============================================================================
# Note: the EHR dataset has 84 patients, none of the patient has complete data
# ============================================================================

# load the EHR dataset
EHRdata = pd.read_csv('Solver/ExternalData/EHR_dataset.csv',header=None)	

# ======================================================================================================= #
# Note: EHRdata has 26 rows and 85 columns, the 0-th column is the varible name, the 1-84 columns are the 
#		measurements for 84 patients. The rows are different measurements.
# Note: Two patients have zero measurement, so removed
# ======================================================================================================= #

# ========================================================================= #
# ********* Tested: 05/01/2024 ***********
# select certain EHR measurements (index is consistent with the EHR dataset)
i_hr      = 0  # Heart rate (bpm)
i_sys_bp  = 1  # P_a,sys, arterial systolic  pressure (mmHg)
i_dia_bp  = 2  # P_a,dia, arterial diastolic pressure (mmHg)
i_co      = 3  # Cardiac output (L/min)
i_svr     = 4  # system vascular resistance (dynes.s.cm^-5)
i_pvr     = 5  # pulmonary vascular resistance (dynes.s.cm^-5)
i_cvp     = 6  # central venous pressure (mmHg)
i_rv_dia  = 7  # right ventricular diastolic pressure (mmHg)
i_rv_sys  = 8  # right ventricular systolic pressure (mmHg)
i_rv_edp  = 9  # right ventricular end diastolic presure (mmHg)
i_lvV_sys = 19 # left ventricular systolic stressed volume (mL)
i_lvV_dia = 20 # left ventricular diastolic stressed volume (mL)
# NOTE: this number is in percentage of the EHR dataset, cvsim6 solver produces the fraction instead, i.e./100
i_LVEF    = 21 # left ventricular ejection fraction (-) 
i_pa_dia  = 23 # pulmonary arterial diastolic pressure (mmHg)
i_pa_sys  = 24 # pulmonary arterial systolic pressure (mmHg)
i_wedge   = 25 # wedge pressure (mmHg)

# Taking interested entries from the EHR dataset and re-order to the cvsim6 solver in the paper
EHR_selected  = np.array([  i_hr    ,  # Hr      --> 0
							i_sys_bp,  # Pa_sys  --> 1
							i_dia_bp,  # Pa_dia  --> 2
							i_rv_sys,  # Pr_sys  --> 3
							i_rv_dia,  # Pr_dia  --> 4
							i_pa_sys,  # Ppa_sys4--> 5
							i_pa_dia,  # Ppa_dia --> 6 
							i_rv_edp,  # Pr_edp  --> 7
							i_wedge,   # Pw      --> 8
							i_cvp,     # Pcvp    --> 9
							i_lvV_sys, # Vl_sys  --> 10
							i_lvV_dia, # Vl_dia  --> 11
							i_LVEF   , # LVEF    --> 12
							i_co     , # CO      --> 13
							i_svr    , # SVR     --> 14
							i_pvr      # PVR     --> 15
						 							], dtype = int)
# ========================================================================= #


# ------------------------------------------------------------------------------ #
# ********* Tested: 05/01/2024 ***********
# Extract patient data from the EHR dataset
# Inputs:
# 		PN: patient number 1~84
# Outputs:
#       missing set: for a patient, which var is missing, this is a 'set'
#       value: re-order EHR measurement for a patient, insert None, if a value is missing
def EHR_extraction(pN):
	assert pN > 0, "start from 1!"
	# locate specific patient from the EHR dataset and extract interested output
	patient_measurement = np.array(EHRdata.iloc[EHR_selected, pN].tolist())

	# float the strings and return the missing index list
	missing_set = []
	value       = []
	for j in range(len(patient_measurement)): # looping tho features
		if patient_measurement[j] == 'none':
			missing_set.append(j)
			value.append(None)
		else:
			value.append(float(patient_measurement[j]))
	return set(missing_set), value
# ------------------------------------------------------------------------------- #



# ------------------------------------------------------------------------------------------------------ #
# ********* Tested: 05/12/2024 ***********
# Use the pre-trained noise-less emulator to foward input predictions from the EHR missing data analysis 
# or just check the accuracy of the emulator with some given input
# Inputs:
#        pth: path of the pretrained emulator
#        para: emulator parameter (hidden units, hidden layer and activation function)
#         V  : ehr prediction (in original scale), or a give set of input parameter
# Outputs:
#        pretrained emulator prediction (in original scale)
def NoiseFreeEmulatorPrediction(pth, para, V):

	# ------------- loading scaling constants -------------------- #
	x_mu  = np.genfromtxt(pth+'X-mu.csv',  delimiter=',')
	x_std = np.genfromtxt(pth+'X-std.csv', delimiter=',')
	y_mu  = np.genfromtxt(pth+'Y-mu.csv',  delimiter=',')
	y_std = np.genfromtxt(pth+'Y-std.csv', delimiter=',')
	# ----------------------------------------------------------- #

	# define and load the weights
	pre_trained_emulator      = Emulator(23, 16, para).double()
	pre_trained_emulator.load_state_dict( torch.load(pth+'/Emulator_model.pth'))
	pre_trained_emulator.eval()
	# normalizing the input set
	V = torch.from_numpy(  ( V - x_mu ) / x_std  )
	# forward the emulator set (scaled)
	Yp_from_em = pre_trained_emulator(V).detach().numpy()
	return Yp_from_em * y_std + y_mu
# ---------------------------------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------- #
# ********* Tested: 05/15/2024 ***********
# Compute error norm between predicted sol and the EHR measurement per patient
# Inputs:
#        patient_id: id of the patient 
#        pth: where to save the data
#        ypred: predicted output by emulator or the exact cvsim-6 model
#        y_patient: patient measurement from the EHR dataset
#        av_id: id of the component that is not missing (complement set)
def error_per_patient(patient_id, pth, ypred, y_patient, av_id):

	assert ypred.shape[1] == y_patient.shape[1], "dimension mismatching!"

	# how many latent variables w sampled
	wsize = ypred.shape[0]
	
	# ----- change PVR and SVR's units back from HRU to cgs to compute error----- #
	# ypred has everything, so use y_patient to tell
	if y_patient[0, 14] is not None:
		y_patient[0, 14] *= 80.
		ypred[:,14]      *= 80.

	if y_patient[0, 15] is not None:
		y_patient[0, 15] *= 80.
		ypred[:,15]      *= 80.
	# ---------------------------------------------------------------------------- #

	# create path if not exist
	new_path = pth + 'Error_per_patient/'
	os.makedirs(new_path, exist_ok = True)

	# initialize error matrix 
	error = np.zeros((wsize,16))

	# loop tho the components that are not missing and compute the componentwise error 
	for j in range(16):
		if j in av_id: # if the current measurement is not missing	
			error[:,j]  = ypred[:,j] - y_patient[0,j]
		else:
			error[:,j]  = None # this component is missing

	# print info for monitoring
	print('patient-' + str(patient_id) + ', ' + str(len(av_id)) + ' measurements are available!')
	np.savetxt(new_path + 'patient_id=' + str(patient_id) + '.csv', error, delimiter = ',')
	return 0
# ---------------------------------------------------------------------------------------------- #