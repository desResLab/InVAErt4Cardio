# Using amortized NPE to solve for posterior distribution of the cvsim-6 system given an output observation
# Ref: https://sbi-dev.github.io/sbi/	
# Ref: https://github.com/sbi-dev/sbi/blob/main/tutorials/01_gaussian_amortized.ipynb
# REf: https://sbi-dev.github.io/sbi/tutorial/02_flexible_interface/


import torch
import numpy as np
from cvsim6_simulator import *
import pickle
import os

# loading sbi essentials
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils

# create path to save the data, if not exist
os.makedirs('SBI_models/',exist_ok = True)



# ------------  Reference values (copied from the data_generator.py) ------------  #

# ****************** Physiological quantities ******************************** #
# 1: Heart rate
Hr_ref = np.array([72.]) # (bpm)
# 2: Transthoracic pressure 
Pth_ref = np.array([-4.]) # (mmHg)
# 3: systolic ratio per hear cycle
rsys_ref = np.array([1./3]) 
# **************************************************************************** #


# ****************** Capacitances (compliances) (ml/mmHg)********************** #
# 1: left ventricular diastolic capacitance 
# 2: left ventricular systolic capacitance 
# 3: arterial capacitance 
# 4: venous capacitance
# 5: right ventricular diastolic capacitance
# 6: right ventricular systolic capacitance
# 7: pulmonary arterial capacitance
# 8: pulmonary venous capacitance

C_ref  = np.array(	  [ 10,     # C_{l,dia}
						0.4,    # C_{l,sys}
						1.6,    # C_a
						100.,   # C_v
						20.,    # C_{r,dia}
						1.2,    # C_{r,sys}
						4.3,    # C_{pa}
						8.4  ]) # C_{pv} 
# **************************************************************************** #


# ************************** Resistances (mmHg.s/ml) ************************* #
# Note: Inflow resistance for each compartment is identical to the 
#		outflow resistances for the preceding adjacent compartment (Davis 1991) 
# 1: Left ventricular inflow resistance
# 2: Left ventricular outflow resistance
# 3: Arterial resistance 
# 4: Right ventricular inflow resistance
# 5: Right ventricular outflow resistance
# 6: Pulmonary venous resistance

R_ref = np.array( [ 0.01,   # R_{li}           
					0.006,  # R_{lo}
					1.,     # R_a
					0.05,   # R_{ri}
					0.003,  # R_{ro}
					0.08] ) # R_{pv}
# **************************************************************************** #


# ************************* Volumes (ml) ************************************* #
# Note: the volumes are defined as the zero-pressure filling volumes
# 1: Unstressed left ventricular volume
# 2: Unstressed arterial volume
# 3: Unstressed venous volume
# 4: Unstressed right ventricular volume
# 5: Unstressed pulmonary arterial volume
# 6: Unstressed pulmonary venous volume

V_ref = np.array([ 	15.,   # V_{0,lv}
					715.,  # V_{0,a}
					2500., # V_{0,v}
					15.,   # V_{0,rv}
					90.,   # V_{0,pa}
					490.]) # V){0,pv}						
# **************************************************************************** #

# ----------------------------------------------------------------------------------- #
# Prepare for bounds of the uniform distributions

# define lower and upper bounds, ref: +/- 30\% of its original magnitude
lp  = 0.7
hp  = 1.3

# bounds for the uniform distributions
# Note: Pth ref is negative, so bounds need to be flipped (actually it does not matter)
lower_bounds = np.concatenate( (Hr_ref * lp, Pth_ref * hp, rsys_ref  * lp,\
								 C_ref * lp, R_ref   * lp,  V_ref    * lp) ) 

upper_bounds = np.concatenate( (Hr_ref * hp, Pth_ref * lp, rsys_ref * hp, \
								 C_ref * hp, R_ref   * hp, V_ref    * hp) )
# ------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------- #
# define prior distribution
prior     = utils.BoxUniform(low=torch.from_numpy(lower_bounds), high=torch.from_numpy(upper_bounds))


# define simulator
cvsim6           = simulator_cvsim6()
simulator, prior = prepare_for_sbi(cvsim6.solve, prior)

# define posterior
inference = SNPE(prior=prior)

# perform simulation based inference
num_sim  = 1000
theta, x          = simulate_for_sbi(simulator, proposal = prior, num_simulations = num_sim, num_workers=12)
density_estimator = inference.append_simulations(theta, x).train()
posterior         = inference.build_posterior(density_estimator)


with open("SBI_models/sbi-cvsim6.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
# ------------------------------------------------------------------------------------------- #

