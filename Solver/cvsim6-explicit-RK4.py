# using explicit methods to integrate the CVSim6 system for verification
# Note: the scipy RK45 is using adaptive step sizes, so a RK45 with fixed step size is hard-coded here
from cvsim6_simulator import *
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import math

fs = 16
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')  # Load amsmath for math features

DPI = 500
saving_pth    = 'CVSim6_stiffness_pictures/'
os.makedirs(saving_pth,exist_ok = True)

# ------------------------------- Set up ---------------------------------------- #
# ============ Tested: 05/09/2024 ===========
# Use the reference parameters as per Davis 1991 (unit not changed)
input_refs = np.array([72. , -4., 1./3, 10., 0.4, 1.6, 100., 20.,
							1.2, 4.3, 8.4, 0.01, 0.006, 1., 0.05, 0.003, 0.08, 
								15., 715., 2500., 15., 90., 490. ])

# init cvsim6 class
cvsim6     = simulator_cvsim6()

# name every parameter and other quantities
cvsim6.parameter_naming(input_refs)

# solve and get the initial condition. Note: the unit is in cgs not mmHg
P_ini = cvsim6.IC_solving()
# --------------------------------------------------------------------------------- #


# ----------------------- Solve the CVSIM-6 system with the implicit RK5 method -------------------------- #
# ============ Tested: 05/22/2024 ===========
# Note: save more points and use lower tolerance to reduce interpolation error
_, t_sol_ref, P_sol_ref, Q_sol_ref, V_sol_ref = cvsim6.solve( input_refs, save_traj = True, \
																num_t = 60000, rtol = 1e-8, atol = 1e-10) 
# --------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------- #
# ============ Tested: 05/09/2024 ===========
# define the time-dependent matrix and the forcing vector and compute the RHS of the ODE
# Input:
#        time: actual time
#        P0: previous solution
# Outputs:
#       The RHS of the ODE, defined by the previous solution
def Get_RHS(time, P0):

	# ---- get sol ----- #
	Pl  = P0[0]
	Pa  = P0[1]
	Pv  = P0[2]
	Pr  = P0[3]
	Ppa = P0[4]
	Ppv = P0[5]
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

	# ------------ init mat and b ----------- #
	A = np.zeros((6,6))
	b = np.zeros((6,1))	
	# --------------------------------------- #


	# --------------------------- For Pl ---------------------------------------- #
	# Pl-Pl
	A[0,0] =  (-1./Cl * dCldt)  + ( -1./cvsim6.R_li * indicator(Ppv, Pl)  / Cl ) \
								+ ( -1./cvsim6.R_lo * indicator(Pl,  Pa)  / Cl )   
	# Pl-Pa							
	A[0,1] =  ( 1./cvsim6.R_lo * indicator(Pl,  Pa)  * 1./Cl )
	# Pl-Ppv
	A[0,5] =  ( 1./cvsim6.R_li * indicator(Ppv, Pl)  * 1./Cl )

	# b_l
	b[0]   = cvsim6.Pth * dCldt/Cl
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

	# b_r
	b[3]   = cvsim6.Pth * dCrdt/Cr
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

	return A @ P0 + b
# -------------------------------------------------------------------------------------------------------------------- #


# --------------------------------------------------- #
# ============ Tested: 05/09/2024 ===========
# Define RK4 explicit integrator
# Inputs:
#       t: current time
#       dt: time step size
#       P0: previous solution
# Output:
#       Next solution
def RK4_integrator(t, dt, P0):

	# first stage
	k1 = Get_RHS(t, P0)	

	# second statge
	k2 = Get_RHS(t + dt/2, P0 + dt * k1 / 2 )

	# third statge
	k3 = Get_RHS(t + dt/2, P0 + dt * k2 / 2 )

	# fourth stage
	k4 = Get_RHS(t + dt, P0 + dt * k3 )

	return P0 + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
# ------------------------------------------------------ #



# ---------------------------------------- Start time integration ----------------------------------------- #
# ============ Tested: 06/06/2024 ===========
total_time  = 12 * 60/72.                     # total computational time (same as the implicit method)
color_pad   = ['b', 'r', 'lime', 'gold']      # list of colors
dt_list     = [4e-3, 8e-3, 2e-2]                          # list of time step size to be tested
counter     = 0                               # counter for color

# plots for pressure/volume and flows
fig1, ax1     = plt.subplots(figsize = (4,1.5))
fig2, ax2     = plt.subplots(figsize = (4,1.5))
fig3, ax3     = plt.subplots(figsize = (4,1.5))
# plot for volume conservation
fig4, ax4     = plt.subplots(figsize = (4,1.5))

# Loop tho different time step size for the RK4 integrator
for dt_ in dt_list:
	
	time        = 0.                        # starting time (t_n)
	time_backup = []                        # recording time instances
	sol_backup  = []                        # recording solutions
	P0          = P_ini.reshape(-1,1)       # initial condition, computed above
	
	# time loop
	for _ in range(int(total_time/dt_)):

		# forward RK4 time integrator
		P1 = RK4_integrator(time, dt_, P0)

		# update time, t_n+1
		time += dt_

		# record the solutions and time
		time_backup.append(time)
		sol_backup.append(P1)

		# sub
		P0 = P1

	# numpy the solutions
	P_sol    = np.array(sol_backup).squeeze()
	time_sol = np.array(time_backup)

	# compute flows based on the pressure solution
	Q_li, Q_lo, Q_a, Q_ri, Q_ro, Q_pv  = cvsim6.flow_update(P_sol.T)

	# compute volumes based on the pressure solution
	V_lv, V_a, V_v, V_rv, V_pa, V_pv  = cvsim6.volume_update(time_sol, P_sol.T)
	V_tot                             = V_lv + V_a + V_v + V_rv + V_pa + V_pv # compute total volume

	# prepare the solutions to be plotted
	Pl_plot  = P_sol[:,0]/1333.22 # cgs --> mmHg    # left-v pressure
	Vl_plot  = V_lv                # left-v volume
	Qlo_plot = Q_lo                # left-v outflow 
	Vtot_plot= V_tot               # total stressed volume

	# change legend notation 
	formatted_dt       = f"{dt_:.0e}"             # change it to floate-0exponent
	mantissa, exponent = formatted_dt.split('e')  # split the mantissa and exponent
	exponent           = int(exponent)            # int the expoenent

	# plot explicit RK4 Pl solution
	ax1.plot(time_sol, Pl_plot,  color = color_pad[counter],  alpha = 0.8, linewidth = 1.5)

	# for Pl, also show the Pa solution
	if dt_ == 2e-2:
		ax1.plot(time_sol, P_sol[:,1]/1333.22,  '-.', color = color_pad[counter],  alpha = 0.8, linewidth = 1.5)

	# plot explicit RK4 Vl solution
	ax2.plot(time_sol, Vl_plot,  color = color_pad[counter],  alpha = 0.8, linewidth = 1.5)

	# plot explicit RK4 Qlo solution
	ax3.plot(time_sol, Qlo_plot, color = color_pad[counter],  alpha = 0.8, linewidth = 1.5 , \
									label = r'$\Delta t= \ $' + f"${float(mantissa):.1f} \\cdot 10^{{ {exponent} }}$")

	# plot explicit RK4 Vtot solution
	ax4.plot(time_sol, Vtot_plot,  color = color_pad[counter], alpha = 0.8, linewidth = 1.5)

	
	counter+=1 # update counter

# ax1 spec for Pl
ax1.plot(t_sol_ref, P_sol_ref[0,:],'k--')
ax1.set_xlim([9,10])
ax1.set_xlabel('$t$ (s)',fontsize=fs-1)
ax1.set_ylabel('$P_l$ (mmHg)',fontsize=fs-1)
ax1.tick_params(axis='both', which='major', labelsize=fs-5)
fig1.savefig(saving_pth + 'Pl-RK4-comparison.png', bbox_inches='tight',pad_inches = 0, dpi = DPI)


# ax2 spec for Vl
ax2.plot(t_sol_ref, V_sol_ref[0,:],'k--')
ax2.set_xlim([9,10])
ax2.set_xlabel('$t$ (s)',fontsize=fs-1)
ax2.set_ylabel('$V_l$ (mL)',fontsize=fs-1)
ax2.tick_params(axis='both', which='major', labelsize=fs-5)
fig2.savefig(saving_pth + 'Vl-RK4-comparison.png', bbox_inches='tight',pad_inches = 0, dpi = DPI)


# ax3 spec for Qlo
ax3.plot(t_sol_ref, Q_sol_ref[1,:],'k--', label = 'Reference')
ax3.set_xlim([9,10])
ax3.set_xlabel('$t$ (s)',fontsize=fs-1)
ax3.set_ylabel('$Q_{l,out}$ (cc/s)',fontsize=fs-1)
ax3.tick_params(axis='both', which='major', labelsize=fs-5)
ax3.legend(fontsize=fs-6)
fig3.savefig(saving_pth + 'Qlo-RK4-comparison.png', bbox_inches='tight',pad_inches = 0, dpi = DPI)

# ax4 spec for Vtot
ax4.plot(t_sol_ref, V_sol_ref.sum(axis=0),'k--')
ax4.set_xlim([9,10])
ax4.set_xlabel('$t$ (s)',fontsize=fs-1)
ax4.set_ylabel(r'$\sum V \ (\textrm{mL})$', fontsize=fs-1)
ax4.tick_params(axis='both', which='major', labelsize=fs-5)
fig4.savefig(saving_pth + 'V-total-RK4-comparison.png', bbox_inches='tight',pad_inches = 0, dpi = DPI)
# #------------------------------------------------------------------------------------- #


















