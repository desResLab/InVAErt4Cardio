# Reference: Teaching physiology through interactive simulation of hemodynamics, T.L. Davis (1991)
#			 Predictive Modeling of Secondary Pulmonary Hypertension in Left Ventricular Diastolic Dysfunction, Harrod et.al (2021)
#			 CVSim: An Open-Source Cardiovascular Simulator for Teaching and Research, Heldt et.al (2010)	


#                         Left Heart
#                              |
#                           Artery
#                              |
#                            Veins
#                              |
#                         Right Heart
#                              |
#                       Pulmonary artery
#                              |
#                       Pulmonary veins

import os
import torch
import numpy as np
from cvsim6_simulator import *
from mpi4py import MPI

# MPI spec.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# create path to save the data, if not exist
os.makedirs('Data/',exist_ok = True)


# ****************** Capacitances (compliances) (ml/mmHg)********************** #
# 1: left ventricular diastolic capacitance 
# 2: left ventricular systolic capacitance 
# 3: arterial capacitance 
# 4: venous capacitance
# 5: right ventricular diastolic capacitance
# 6: right ventricular systolic capacitance
# 7: pulmonary arterial capacitance
# 8: pulmonary venous capacitance

C_ref  = np.array(	  [ 10.,    # C_{l,dia}
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

V_ref = np.array([ 	15,   # V_{0,lv}
					715,  # V_{0,a}
					2500, # V_{0,v}
					15,   # V_{0,rv}
					90,   # V_{0,pa}
					490,  # V){0,pv}
						])

# **************************************************************************** #


# --------------------- Data_preparation -----------------------------#
Sample_size = 6000

X = np.zeros((Sample_size,20))
Y = np.zeros((Sample_size,15))

# define lower and upper bounds
lower_bounds = np.concatenate( (C_ref * 0.7, R_ref * 0.7, V_ref * 0.7) ) 
upper_bounds = np.concatenate( (C_ref * 1.3, R_ref * 1.3, V_ref * 1.3) )

# random (uniform) sampling by parallel
np.random.seed(2113144 + rank)
Input_samples = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(Sample_size,20))

# construct the class
cvsim6 = simulator_cvsim6()

for j in range(Sample_size):

	if rank == 0 and j%100 ==0:
		print(j)

	X[j,:] = Input_samples[j,:]
	Y[j,:] = cvsim6.solve(torch.tensor(X[j,:]))

np.savetxt('Data/Rank-'+str(rank)+'-cvsim6_X.csv', X, delimiter = ',')
np.savetxt('Data/Rank-'+str(rank)+'-cvsim6_Y.csv', Y, delimiter = ',')