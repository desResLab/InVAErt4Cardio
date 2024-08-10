# Stiffness analysis of the cvsim-6 system
from cvsim6_simulator import *
import numpy as np
import math
from cvsim6_plotting_functions import *
from matplotlib import pyplot as plt

# plotting args
plt.rc('font',  family='serif')
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['text.usetex'] = True
fs  = 16
DPI = 600

# ---------------------- Solve the system by implicit method ------------------------------ #
# Use the reference parameters as per Davis 1991 (unit not changed)
input_refs = np.array([72. , -4., 1./3, 10., 0.4, 1.6, 100., 20.,
							1.2, 4.3, 8.4, 0.01, 0.006, 1., 0.05, 0.003, 0.08, 
									15., 715., 2500., 15., 90., 490. ])
# init cvsim6 class
cvsim6     = simulator_cvsim6()

# solve the system 
# Note: save more points and use lower tolerance to reduce interpolation error
_, t_sol, P_sol, _, V_sol = cvsim6.solve( input_refs, save_traj = True, \
												num_t = 60000, rtol = 1e-8, atol = 1e-10) 

# where to save the figure
pth    = 'CVSim6_stiffness_pictures/'
# selected plotting duration
cutoff = int(len(t_sol)*5/6) # the last two cycles
# ---------------------------------------------------------------------------------------- #

# ------------------------Write the ODE system as: dPdt = AP + b -------------------------- #
# # ============== Tested: 04/18/2024 ============== #
# Input:
#        j: time index
# Output:
#       A: the coefficient matrix A 
#       [mitral, aortic, tricuspid, pulmonary]: 1/0, 1 means open, 0 means close
def get_A(j):

	# get the exact time
	time = t_sol[j]

	# ---- get sol ----- #
	Pl  = P_sol[0,j]
	Pa  = P_sol[1,j]
	Pv  = P_sol[2,j]
	Pr  = P_sol[3,j]
	Ppa = P_sol[4,j]
	Ppv = P_sol[5,j]
	# ------------------ #

	# ------ get time-varying capacitance and its derivative ------ #
	# consider periodicity
	t_per     =   math.fmod(time,  cvsim6.T_tot) 
	Cl, dCldt = cvsim6.C_lr(t_per, cvsim6.C_l_dia, cvsim6.C_l_sys)
	Cr, dCrdt = cvsim6.C_lr(t_per, cvsim6.C_r_dia, cvsim6.C_r_sys)
	# ------------------------------------------------------------- #

	# ------------ define indicator function ---------------- #
	# inputs:
	#        Pin: inflow pressure
	#        Pout: outflow pressure
	# if Pin > Pout, the valve is open, otherwise it is closed
	def indicator(Pin, Pout):
		if Pin > Pout:
			return 1.
		else:
			return 0.
	# ------------------------------------------------------- #

	# --------- record valve opening/closing time ------ #
	# mitral valve is open/close
	mitral = 1    if indicator(Ppv, Pl) == 1. else 0
	# aortic valve is open/close
	aortic = 1    if indicator(Pl, Pa) == 1.  else 0
	# tricuspid valve is open/close
	tricuspid = 1 if indicator(Pv, Pr) == 1.  else 0
	# pulmonary valve is open/close
	pulmonary = 1 if indicator(Pr, Ppa) == 1. else 0
	# --------------------------------------------------- #

	# ------------ init mat A --------------- #
	A = np.zeros((6,6))	
	# --------------------------------------- #

	# --------------------------- For Pl ---------------------------------------- #
	# Pl-Pl
	A[0,0] =  (-1./Cl * dCldt)  + ( -1./cvsim6.R_li * indicator(Ppv, Pl)  / Cl ) \
								+ ( -1./cvsim6.R_lo * indicator(Pl,  Pa)  / Cl )   
	# Pl-Pa							
	A[0,1] =  ( 1./cvsim6.R_lo * indicator(Pl,  Pa)  * 1./Cl )
	# Pl-Ppv
	A[0,5] =  ( 1./cvsim6.R_li * indicator(Ppv, Pl)  * 1./Cl )
	# ---------------------------------------------------------------------------- #

	# -------------------------------------- For Pa ---------------------------------------------- #
	# Pa-Pl
	A[1,0] = ( 1./ cvsim6.C_a * indicator(Pl, Pa) /  cvsim6.R_lo )  
	# Pa-Pa
	A[1,1] = (-1./ cvsim6.C_a * indicator(Pl, Pa) /  cvsim6.R_lo ) + (1./cvsim6.R_a * -1./cvsim6.C_a)
	# Pa-Pv
	A[1,2] =   1./ cvsim6.C_a / cvsim6.R_a
	# -------------------------------------------------------------------------------------------- #

	# --------------------------------------- For Pv ------------------------------------------- #
	# Pv-Pa
	A[2,1] = 1./ cvsim6.R_a  / cvsim6.C_v 
	# Pv-Pv
	A[2,2] = -1./ cvsim6.R_a  / cvsim6.C_v + 1./cvsim6.R_ri * indicator(Pv, Pr) * -1./cvsim6.C_v 
	# Pv-Pr
	A[2,3] = 1./ cvsim6.R_ri * indicator(Pv, Pr) / cvsim6.C_v
	# ------------------------------------------------------------------------------------------ #

	# --------------------------------------- For Pr --------------------------------------------- #
	# Pr-Pv
	A[3,2] =  1./cvsim6.R_ri * indicator(Pv, Pr) / Cr
	# Pr-Pr
	A[3,3] = -1./cvsim6.R_ri * indicator(Pv, Pr) / Cr + 1./cvsim6.R_ro * \
														indicator(Pr, Ppa) * -1./Cr +  -1 * dCrdt / Cr
	# Pr-Ppa
	A[3,4] =  1./cvsim6.R_ro * indicator(Pr, Ppa) / Cr
	# ----------------------------------------------------------------------------------------------#

	# -------------------------------------- For Ppa ----------------------------------------------- #
	# Ppa-Pr
	A[4,3] =  1./cvsim6.R_ro * indicator(Pr, Ppa) / cvsim6.C_pa
	# Ppa-Ppa
	A[4,4] = -1./cvsim6.R_ro * indicator(Pr, Ppa) / cvsim6.C_pa - 1./cvsim6.R_pv/cvsim6.C_pa
	# Ppa-Ppv
	A[4,5] =  1./cvsim6.R_pv / cvsim6.C_pa
	# ---------------------------------------------------------------------------------------------- #

	# ---------------------------------------- For Ppv --------------------------------------------- #
	# Ppv-Pl
	A[5,0] =  1. / cvsim6.R_li / cvsim6.C_pv * indicator(Ppv, Pl) 
	# Ppv-Ppa
	A[5,4] =  1. / cvsim6.R_pv / cvsim6.C_pv
	# Ppv-Ppv
	A[5,5] = -1. / cvsim6.R_pv / cvsim6.C_pv - 1. / cvsim6.C_pv / cvsim6.R_li * indicator(Ppv, Pl)
	# ---------------------------------------------------------------------------------------------- #

	return A, [mitral, aortic, tricuspid, pulmonary]
# ----------------------------------------------------------------------------------------------------- #


# ========================== Eigen-decomposition =============================== # 
# # ============== Tested: 05/22/2024 ============== #
Eigen_value_backup  = []       # placeholder for eigenvalues at each time step
Eigen_vector_backup = []       # placeholder for eigenvecs at each time step
valve_states_backup = []       # placeholder for the valve opening/closing backup

# get the A matrix at each time and apply eigen-decomposition
for j in range(len(t_sol)):

	# generate the A matrix and the valve opening states
	A, valve = get_A(j)

	# eigendecomposition
	# Note: eigenvecs are normalized and [:,k] is the k-th eig-vec w.r.t the k-th eig
	eigenvalues, eigenvectors = np.linalg.eig(A)

	# record valve opening/closing states
	valve_states_backup.append(valve)
	# record eigvalue, eigvector 
	Eigen_value_backup.append(eigenvalues)
	Eigen_vector_backup.append(eigenvectors)

# numpy the results
Eigen_value_backup     = np.array(Eigen_value_backup)
Eigen_vector_backup    = np.array(Eigen_vector_backup)
valve_states_backup    = np.array(valve_states_backup)

# re-order the eigenvalue and eigenvector by descending order
eig_indices = np.argsort( abs(Eigen_value_backup.real) )[:,::-1]

# sort the eigenvalues one at each time
for j in range(Eigen_value_backup.shape[0]):
	Eigen_value_backup[j]  = Eigen_value_backup[j, eig_indices[j]]
	for k in range(6): # sort the eigenvectors one at each time
		Eigen_vector_backup[j,k,:] = Eigen_vector_backup[j, k, eig_indices[j] ]

# plot eigenvalue dynamics in the last two heart cycles
eigen_plotter(pth, t_sol[cutoff:], Eigen_value_backup[cutoff:])
# =============================================================================== #


# -------------------------------------------------------------------------------------------------- #
# ============== Tested: 04/25/2024 ============== #
# plot eigenvectors when SR reaches its maximum or minimum value in the last two heart cycle
tol_choice          = 1e-14 # eigenvalue below that will be considered as zero

print('choice of tolerence is ' + str(tol_choice))
SR_choice_cut_off   = get_SR_with_tol(Eigen_value_backup, tol_choice)[cutoff:]
# warning: these are indices with respect to the cut-off-ed array
get_max_index       = np.argmax(SR_choice_cut_off)
get_min_index       = np.argmin(SR_choice_cut_off)
print('Stiffness ratio is at the maximum when t = ' + str(t_sol[cutoff:][get_max_index]) + \
						', ' + 'minimum when t = ' + str( t_sol[cutoff:][get_min_index]))

# plot eigenvector when SR reaches the maximum value
print('Eigenvalues when SR is maximized: ' + str(Eigen_value_backup[cutoff:][get_max_index]) )
radar_eigvec(pth + 'SR_max/', abs(Eigen_vector_backup[cutoff:][get_max_index]))
show_eigenvec_matrix(pth + 'SR_max/', 'SR_max_EVec', Eigen_vector_backup[cutoff:][get_max_index], \
														tol_choice)

# plot eigenvector when SR reaches the minimum value
print('Eigenvalues when SR is minimized: ' + str(Eigen_value_backup[cutoff:][get_min_index]) )
radar_eigvec(pth + 'SR_min/', abs(Eigen_vector_backup[cutoff:][get_min_index]))
show_eigenvec_matrix(pth + 'SR_min/', 'SR_min_EVec', Eigen_vector_backup[cutoff:][get_min_index], \
														tol_choice)
# -------------------------------------------------------------------------------------------------- #

# ---------------------------------- get valve opening/closing times ------------------------------------- #
# ============== Tested: 04/19/2024 ============== #
mitral_cut_off, aortic_cut_off       = valve_states_backup[cutoff:,0], valve_states_backup[cutoff:,1]
tricuspid_cut_off, pulmonary_cut_off = valve_states_backup[cutoff:,2], valve_states_backup[cutoff:,3]
# -------------------------------------------------------------------------------------------------------- #


# ------------------------------ plot systemic circulation, i.e. LV ---------------------------------------- #
# ============== Tested: 04/19/2024 ============== #
# plot lv volume
t_plot  = t_sol[cutoff:]
vcolor  = ['b', 'r'] # color for the shaded area under the valve opening times
y1_plot = V_sol[0,cutoff:] 
y1_pack = ['$V_l$', '$V_l$ (mL)']
curves_with_SR(pth, 'Vl', t_plot, y1_plot , SR_choice_cut_off, mitral_cut_off, aortic_cut_off, y1_pack, vcolor)

# plot Pl and Pa
y1_plot = P_sol[0,cutoff:] # Pl
y2_plot = P_sol[1,cutoff:] # Pa
y1_pack = ['$P_l$', 'Pressure (mmHg)']
y2_pack = ['$P_a$']
curves_with_SR(pth, 'pressureVL', t_plot, y1_plot , SR_choice_cut_off, mitral_cut_off, aortic_cut_off, \
														y1_pack, vcolor, y2 = y2_plot, y2pack = y2_pack)
# ---------------------------------------------------------------------------------------------------------- #


# ------------------------------ plot pulmonary circulation, i.e. RV ----------------------------------------------- #
# ============== Tested: 04/19/2024 ============== #
# plot rv volume
vcolor  = ['dimgray', 'lime']
y1_plot = V_sol[3,cutoff:] 
y1_pack = ['$V_r$', '$V_r$ (mL)']
curves_with_SR(pth, 'Vr', t_plot, y1_plot , SR_choice_cut_off, tricuspid_cut_off, pulmonary_cut_off, y1_pack, vcolor)

# plot Pr and Ppa
y1_plot = P_sol[3,cutoff:] # Pr
y2_plot = P_sol[4,cutoff:] # Ppa
y1_pack = ['$P_r$', 'Pressure (mmHg)']
y2_pack = ['$P_{pa}$']
curves_with_SR(pth, 'pressureVR', t_plot, y1_plot , SR_choice_cut_off, tricuspid_cut_off, pulmonary_cut_off, \
															y1_pack, vcolor, y2 = y2_plot, y2pack = y2_pack)
# ------------------------------------------------------------------------------------------------------------------ #



# ------------------------------ plot Cl, Cr, dCldt and dCrdt ----------------------------------------------- #
Cl_backup,     Cr_backup   = [], []
dCldt_backup, dCrdt_backup = [], []
# ============== Tested: 04/30/2024 ============== #
# -------- get time-varying capacitance and its derivative -------- #
for j in range(len(t_sol)):
	# get time
	time = t_sol[j]
	# consider periodicity
	t_per     =   math.fmod(time,  cvsim6.T_tot) 
	Cl, dCldt = cvsim6.C_lr(t_per, cvsim6.C_l_dia, cvsim6.C_l_sys)
	Cr, dCrdt = cvsim6.C_lr(t_per, cvsim6.C_r_dia, cvsim6.C_r_sys)

	# store values
	Cl_backup.append(Cl)
	Cr_backup.append(Cr)
	dCldt_backup.append(dCldt)
	dCrdt_backup.append(dCrdt)

# numpy them and normalize them
Cl_backup    = np.array(Cl_backup)
Cr_backup    = np.array(Cr_backup)
dCldt_backup = np.array(dCldt_backup)
dCrdt_backup = np.array(dCrdt_backup)
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# plot-Cl
vcolor  = ['b', 'r'] 
y1_plot = Cl_backup[cutoff:] 
y1_pack = ['$C_l$', '$C_l$ (mL/Barye)']
curves_with_SR(pth, 'Cl', t_plot, y1_plot , SR_choice_cut_off, mitral_cut_off, aortic_cut_off, y1_pack, \
																				vcolor, logy=True)
# plot-dCldt
y1_plot = dCldt_backup[cutoff:] 
y1_pack = [r'$\frac{d C_l}{dt}$', r'$\frac{dC_l}{dt}$ (mL/(Barye$\cdot$s))']
curves_with_SR(pth, 'dCldt', t_plot, y1_plot , SR_choice_cut_off, mitral_cut_off, aortic_cut_off, y1_pack, vcolor)

# plot-Cr
vcolor  = ['dimgray', 'lime']
y1_plot = Cr_backup[cutoff:] 
y1_pack = ['$C_r$', '$C_r$ (mL/Barye)']
curves_with_SR(pth, 'Cr', t_plot, y1_plot , SR_choice_cut_off, tricuspid_cut_off, pulmonary_cut_off, y1_pack, \
																				vcolor, logy=True)
# plot-dCrdt
y1_plot = dCrdt_backup[cutoff:] 
y1_pack = [r'$\frac{d C_r}{dt}$', r'$\frac{dC_r}{dt}$ (mL/(Barye$\cdot$s))']
curves_with_SR(pth, 'dCrdt', t_plot, y1_plot , SR_choice_cut_off, tricuspid_cut_off, pulmonary_cut_off, y1_pack, vcolor)
# --------------------------------------------------------------------------------------------------------------------- #













