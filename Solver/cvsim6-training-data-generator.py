# The CVSIM-6 model data generator (parallel)
# Reference: Teaching physiology through interactive simulation of hemodynamics, T.L. Davis (1991)
#			 Predictive Modeling of Secondary Pulmonary Hypertension in Left Ventricular Diastolic Dysfunction, Harrod et.al (2021)
#			 CVSim: An Open-Source Cardiovascular Simulator for Teaching and Research, Heldt et.al (2010)	
#            PhysioNet: https://physionet.org/content/cvsim/1.0.0/

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
import numpy as np
from cvsim6_simulator import *
from mpi4py import MPI

# MPI spec.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ------------------------------------------------------------------------------ #
# define reference parameter (Davis 1991)
input_refs = np.array([72. , -4., 1./3, \
							10., 0.4, 1.6, 100., 20., 1.2, 4.3, 8.4, \
									0.01, 0.006, 1., 0.05, 0.003, 0.08, \
											15., 715., 2500., 15., 90., 490. ])
# ------------------------------------------------------------------------------ #

# -------------------------------------------------------------------------------------------- #
# ============ Tested: 04/30/2024 ===========
# get training data from the cvsim6 simulator by perturbing the reference input-parameter
# Inputs:
#       pth: where to save
#       N  : sample size
#       lower_bounds: lower bound factors for the parameter perturbation
#       upper_bounds: upper bound factors for the parameter perturbation
def get_training_data_cvsim6(pth, N, lower_f, upper_f):
	
	# each processor takes a different seed
	np.random.seed(rank) 
	os.makedirs(pth,exist_ok = True)

	# initialize data pair 23 --> 16
	X = np.random.uniform(low=lower_f*input_refs, high=upper_f*input_refs, size=(N, 23))
	Y = np.zeros((N,16))  

	# construct the class
	cvsim6 = simulator_cvsim6()

	# loop over each sample and solve the cvsim-6 system
	for j in range(N):

		# monitoring
		if rank == 0 and (j+1)%100 ==0:
			print('data generation: ' + str(j+1) + '/' + str(N) )

		# record the data
		Y[j,0]  = X[j,0] # heart rate copying
		Y[j,1:] = cvsim6.solve(X[j,:])
	
	# save the data
	np.savetxt(pth + 'Rank-'+str(rank)+'-cvsim6_X.csv', X, delimiter = ',')
	np.savetxt(pth + 'Rank-'+str(rank)+'-cvsim6_Y.csv', Y, delimiter = ',')
	return None
# ------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------- #
# ============ Tested: 04/30/2024 ===========
# Combine embrassingly parallel generated dataset for the cvsim6 data
# Input:
#       pth:  case path
#       numproc: how many processor used in data preparation
def parallel_data_loader(pth, numproc):
	for i in range(numproc):
		if i == 0:
			X = np.genfromtxt(pth+'Rank-'+str(i)+'-cvsim6_X.csv', delimiter = ',')
			Y = np.genfromtxt(pth+'Rank-'+str(i)+'-cvsim6_Y.csv', delimiter = ',')
		else:
			X_new = np.genfromtxt(pth+'Rank-'+str(i)+'-cvsim6_X.csv', delimiter = ',')
			X     = np.concatenate((X, X_new))
			Y_new = np.genfromtxt(pth+'Rank-'+str(i)+'-cvsim6_Y.csv', delimiter = ',')
			Y     = np.concatenate((Y, Y_new))
	np.savetxt(pth + 'cvsim6-input.csv',  X, delimiter=',')
	np.savetxt(pth + 'cvsim6-output.csv', Y, delimiter=',')
	return None
# ----------------------------------------------------------------------------------------- #


# =========================================================== #
# ============ Tested: 06/21/2024 ===========
# Dataset 1: dataset for the EHR
pth     = 'CVSIM6-Training-Data/EHR/'
N       = 5400        # how many data generated per processor
numproc = 10          # how many processors needed
lower_bounds = np.array([	0.8, 1.3, 0.7, \
							0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,\
							0.2, 0.2, 0.2, 0.2, 0.2, 0.2, \
							0.7, 0.7, 0.7, 0.7, 0.7, 0.7  ])

upper_bounds = np.array([	1.6, 0.7, 1.3, \
							1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6,\
							1.6, 1.6, 1.6, 1.6, 1.6, 1.6, \
							1.3, 1.3, 1.3, 1.3, 1.3, 1.3  ])
get_training_data_cvsim6(pth, N, lower_bounds, upper_bounds)
comm.Barrier()
if rank == 0:
	parallel_data_loader(pth, numproc)
# =========================================================== #



# # =========================================================== #
# # ============ Tested: 05/07/2024 ===========
# # Dataset 2: dataset for studying the structural identifiability
# pth     = 'CVSIM6-Training-Data/Structural_Study/'
# N       = 5400        # how many data generated per processor
# numproc = 10          # how many processors needed
# lower_bounds = np.array([	0.7, 1.3, 0.7, \
# 							0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,\
# 							0.7, 0.7, 0.7, 0.7, 0.7, 0.7, \
# 							0.7, 0.7, 0.7, 0.7, 0.7, 0.7  ])

# upper_bounds = np.array([	1.3, 0.7, 1.3, \
# 							1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,\
# 							1.3, 1.3, 1.3, 1.3, 1.3, 1.3, \
# 							1.3, 1.3, 1.3, 1.3, 1.3, 1.3  ])
# get_training_data_cvsim6(pth, N, lower_bounds, upper_bounds)
# comm.Barrier()
# if rank == 0:
# 	parallel_data_loader(pth, numproc)
# # =========================================================== #