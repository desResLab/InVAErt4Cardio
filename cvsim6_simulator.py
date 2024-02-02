# The CVSIM-6 simulator
# Reference: Teaching physiology through interactive simulation of hemodynamics, T.L. Davis (1991)
#			 Predictive Modeling of Secondary Pulmonary Hypertension in Left Ventricular Diastolic Dysfunction, Harrod et.al (2021)
#			 CVSim: An Open-Source Cardiovascular Simulator for Teaching and Research, Heldt et.al (2010)	
#            PhysioNet: https://physionet.org/content/cvsim/1.0.0/

import time
import torch
import math
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


#---------------------------------------------------------------#
# Get average quantity via integration 
# Input:
#      t: interested time-interval
#      y: interested quantity over the interested time interval
def get_average(t,y):
	return torch.trapz(y,t)/(t[-1] - t[0])
#---------------------------------------------------------------#


# ------------------------------------------------------------------------------------------ #
class simulator_cvsim6:

	def __init__(self):

		# constants
		# ------------------------------------------------- #
		self.mmHg2Barye = 1333.22 # unit conversion
		self.V_tot      = 5000.    # Total blood volume (ml)
		# ------------------------------------------------- #

	# ------------------------------------------------------------------------------------------ #
	# self the parameters of interest and apply unit conversion (input is of 23 dimension)
	# Inputs:
	#       input_para: Hr, Pth, rsys
	#                   8 capacitance, 6 resistance and 6 unstressed volumes
	def parameter_naming(self, input_para):

		# Physiological quantities
		self.Hr      = input_para[0]                          # heart rate 
		self.Pth     = input_para[1] * self.mmHg2Barye        # Transthoracic pressure 
		self.r_sys   = input_para[2]                          # systolic ratio per heart cycle
		
		# capacitances (ml/Barye)
		self.C_l_dia = input_para[3]  / self.mmHg2Barye       # left ventricular diastolic capacitance 
		self.C_l_sys = input_para[4]  / self.mmHg2Barye		  # left ventricular systolic capacitance 
		self.C_a     = input_para[5]  / self.mmHg2Barye		  # arterial capacitance
		self.C_v     = input_para[6]  / self.mmHg2Barye       # venous capacitance
		self.C_r_dia = input_para[7]  / self.mmHg2Barye       # right ventricular diastolic capacitance 
		self.C_r_sys = input_para[8]  / self.mmHg2Barye       # right ventricular systolic capacitance
		self.C_pa    = input_para[9]  / self.mmHg2Barye       # pulmonary arterial capacitance
		self.C_pv    = input_para[10] / self.mmHg2Barye       # pulmonary venous capacitance
		
		# resistances (Barye.s/ml)
		self.R_li    = input_para[11]  * self.mmHg2Barye      # left ventricular input resistance
		self.R_lo    = input_para[12]  * self.mmHg2Barye	  # left ventricular output resistance
		self.R_a     = input_para[13]  * self.mmHg2Barye	  # arterial resistance
		self.R_ri    = input_para[14]  * self.mmHg2Barye	  # right ventricular input resistance
		self.R_ro    = input_para[15]  * self.mmHg2Barye      # right ventricular output resistance
		self.R_pv    = input_para[16]  * self.mmHg2Barye      # pulmonary venous resistance

		# unstressed volumes (ml)
		self.V_lv    = input_para[17]                         # unstressed left ventricular volume
		self.V_a     = input_para[18]                         # unstressed arterial volume
		self.V_v     = input_para[19]                         # unstressed venous volume
		self.V_rv	 = input_para[20]						  # unstressed right ventricular volume
		self.V_pa    = input_para[21]                         # unstressed pulmonary arterial volume
		self.V_pv    = input_para[22]                         # unstressed pulmonary venous volume

		# other quantities
		self.V_0_tot = self.V_lv + self.V_a + self.V_v \
						+ self.V_rv + self.V_pa + self.V_pv   # total unstressed volume
		self.T_tot = 60./self.Hr                              # Cardiac cycle (seconds)
		self.T_sys = self.T_tot * self.r_sys                  # systolic ejection period (seconds)
		self.T_dia = self.T_tot - self.T_sys                  # diastolic filling period (seconds)

		return None
	# ---------------------------------------------------------------------------------------------- #


	# ---------------------------------------------------------------------------------------------- #
	# Solving the 8x8 linear system to obtain initial condition and return diastolic values
	def IC_solving(self):
		
		# initial condition, Sah et.al Davis et.al
		# P^0 = [ P_l,dia, P_l,sys, P_r,dia, P_r,sys, P_a, P_v, P_pa, P_pv ] 

		# initialization
		A = torch.zeros(8,8)
		b = torch.zeros(8)

		# equation 0:
		A[0,0] =  self.C_l_dia
		A[0,1] = -self.C_l_sys
		A[0,2] = -self.C_r_dia 
		A[0,3] =  self.C_r_sys
		b[0]   =  self.Pth*(self.C_l_dia - self.C_l_sys - self.C_r_dia + self.C_r_sys)

		# equation 1:
		A[1,0] =  self.C_l_dia
		A[1,1] = -self.C_l_sys - self.T_sys/self.R_lo
		A[1,4] =  self.T_sys/self.R_lo
		b[1]   =  self.Pth * (self.C_l_dia - self.C_l_sys)

		# equation 2:
		A[2,0] =   self.C_l_dia
		A[2,1] = - self.C_l_sys
		A[2,4] = - self.T_tot/self.R_a
		A[2,5] =   self.T_tot/self.R_a
		b[2]   =   self.Pth * (self.C_l_dia - self.C_l_sys)

		# equation 3: (Note: R_ri = R_v, R_ri is so small so merged to R_v)
		A[3,0] =   self.C_l_dia
		A[3,1] = - self.C_l_sys
		A[3,2] =   self.T_dia/self.R_ri
		A[3,5] = - self.T_dia/self.R_ri
		b[3]   =   self.Pth * (self.C_l_dia - self.C_l_sys)

		# equation 4:
		A[4,0] =   self.C_l_dia
		A[4,1] = - self.C_l_sys
		A[4,3] = - self.T_sys/self.R_ro
		A[4,6] =   self.T_sys/self.R_ro
		b[4]   =   self.Pth * (self.C_l_dia - self.C_l_sys)

		# equation 5:
		A[5,0] =   self.C_l_dia
		A[5,1] = - self.C_l_sys
		A[5,6] = - self.T_tot/self.R_pv
		A[5,7] =   self.T_tot/self.R_pv
		b[5]   =   self.Pth * (self.C_l_dia -self.C_l_sys)

		# equation 6:
		A[6,0] =   self.C_l_dia + self.T_dia/self.R_li
		A[6,1] = - self.C_l_sys
		A[6,7] = - self.T_dia /self.R_li
		b[6]   =   self.Pth * (self.C_l_dia - self.C_l_sys)

		# equation 7:
		A[7,0] = self.C_l_dia
		A[7,2] = self.C_r_dia 
		A[7,4] = self.C_a
		A[7,5] = self.C_v
		A[7,6] = self.C_pa
		A[7,7] = self.C_pv
		b[7]   = self.V_tot - self.V_0_tot + self.Pth * (self.C_l_dia + self.C_a/3. + self.C_r_dia + self.C_pa + self.C_pv)

		# solve the system
		P0 = torch.linalg.solve(A,b)

		# Take the diastolic quantities for initialization, ref: Davis 1991
		# [P_l, P_a, P_v, P_r, P_pa, P_pv] 
		return P0[[0,4,5,2,6,7]] 
	#--------------------------------------------------------------------------------------#


	#--------------------------------------------------------------------------------------#
	# Define flows of the compartment
	# Inputs:
	#       P_IN:  inflow pressure
	#       P_out: outflow pressure
	#       R_Comp: Resistance of the current compartment
	def flow(self, P_IN, P_OUT, R_COMP):
		if P_IN > P_OUT: # inflow pressure > outflow pressure, then you have a flow
			return (P_IN - P_OUT)/R_COMP
		else:
			return 0.     # cannot go backwards bc the valves are closed
	#--------------------------------------------------------------------------------------#


	#--------------------------------------------------------------------------------------------------------------------------#
	# Define driver functions and its derivatives at each ventricle 
	# Inputs:
	#        t    : current time (w.r.t one cardiac cycle)
	#        C_dia: diastolic capacitance
	#        C_sys: systolic capacitance 
	# Outputs:
	#        1/E: time-dependent capacitance
	#        -dEdt* 1/E^2: time derivative
	def C_lr(self, t, C_dia, C_sys):
		
		# ventricular diastole
		if (t <= 0.) or (t > 1.5 * self.T_sys) : 
			E    = 1./C_dia
			dEdt = 0.
		
		# systole (ventricle contraction)
		elif (t > 0.) and (t <= self.T_sys):
			E    = 0.5 * ( 1./C_sys - 1./C_dia ) * ( 1. - math.cos( math.pi * t/self.T_sys ) ) + 1./C_dia
			dEdt = 0.5 * ( 1./C_sys - 1./C_dia ) * ( math.pi/self.T_sys * math.sin( math.pi * t/self.T_sys )  )

		# end systole (early relaxtion)
		elif (t > self.T_sys) and (t <= 1.5 * self.T_sys):
			E    = 0.5 * ( 1./C_sys - 1./C_dia ) * ( 1. + math.cos( 2.*math.pi * (t-self.T_sys)/self.T_sys )  ) + 1./C_dia
			dEdt = 0.5 * ( 1./C_sys - 1./C_dia ) * ( -2.* math.pi/self.T_sys * math.sin(2.*math.pi * (t-self.T_sys)/self.T_sys) )

		return 1./E, - dEdt * 1./E/E
	#----------------------------------------------------------------------------------------------------------------------------#



	#--------------------------------------------------------------------------------------#
	# define ode system of the cvsim6 model
	# Inputs:
	#        t: actual time
	#        states: quantities of interest
	def f(self, t, states):

		# unpacking
		P_l_star, P_a_star, P_v_star, P_r_star, P_pa_star, P_pv_star  = states

		# compute C and dCdt
		t_per      = math.fmod(t, self.T_tot) # consider periodicity
		
		# call driver function and its derivatives
		C_l, dCldt = self.C_lr(t_per, self.C_l_dia, self.C_l_sys) 
		C_r, dCrdt = self.C_lr(t_per, self.C_r_dia, self.C_r_sys) 

		# compute flows tho each compartment
		Q_li = self.flow( P_pv_star,  P_l_star,   self.R_li)
		Q_lo = self.flow( P_l_star,   P_a_star,   self.R_lo )
		Q_a  = (          P_a_star -  P_v_star) / self.R_a
		Q_ri = self.flow( P_v_star,   P_r_star,   self.R_ri) # R_ri = R_v
		Q_ro = self.flow( P_r_star,   P_pa_star,  self.R_ro)
		Q_pv = (          P_pa_star - P_pv_star)/ self.R_pv
		
		# derivatives
		# 1: dPl/dt
		# 2: dPa/dt
		# 3: dPv/dt
		# 4: dPr/dt
		# 5: dPpa/dt
		# 6: dPpv/dt
		return ( Q_li - Q_lo - (P_l_star - self.Pth) * dCldt ) / C_l,\
				( Q_lo - Q_a ) / self.C_a,\
					( Q_a - Q_ri ) / self.C_v,\
						( Q_ri - Q_ro - (P_r_star - self.Pth) * dCrdt ) / C_r,\
							( Q_ro - Q_pv ) / self.C_pa,\
								( Q_pv - Q_li ) / self.C_pv
	#--------------------------------------------------------------------------------------#



	# ------------------------------------------------------------------------------------ #
	# Get flow trajectories
	# Inputs:
	#       P_array: pressure solution array: 
	#                [P_l(t), P_a(t), P_v(t), P_r(t), P_pa(t), P_pv(t)]
	#					0       1      2       3       4         5
	# Outputs:
	#       updated flow values
	def flow_update(self, P_array):
		
		# number of time steps
		sol_num = P_array.shape[1]

		# init
		Q_li_t = torch.zeros(sol_num)
		Q_lo_t = torch.zeros(sol_num)
		Q_a_t  = torch.zeros(sol_num)
		Q_ri_t = torch.zeros(sol_num)
		Q_ro_t = torch.zeros(sol_num)
		Q_pv_t = torch.zeros(sol_num)

		# update flows
		for j in range(sol_num):
			Q_li_t[j] = self.flow( P_array[5,j],  P_array[0,j],   self.R_li )  
			Q_lo_t[j] = self.flow( P_array[0,j],  P_array[1,j],   self.R_lo )
			Q_a_t[j]  = (          P_array[1,j] - P_array[2,j]) / self.R_a
			Q_ri_t[j] = self.flow( P_array[2,j],  P_array[3,j],   self.R_ri) 
			Q_ro_t[j] = self.flow( P_array[3,j],  P_array[4,j],   self.R_ro)
			Q_pv_t[j] = (          P_array[4,j] - P_array[5,j]) / self.R_pv

		return Q_li_t, Q_lo_t, Q_a_t, Q_ri_t, Q_ro_t, Q_pv_t
	# ------------------------------------------------------------------------------------ # 


	# ------------------------------------------------------------------------------------ #
	# volume update function via linear pressure/volume assumption
	# Inputs:
	#       t_array: time array
	#       P_array: pressure solution array
	# Outputs:
	#       stressed volumes
	def volume_update(self, t_array, P_array):

		# get driver function values at each time step
		C_l   =  torch.zeros(len(t_array))
		C_r   =  torch.zeros(len(t_array))
		
		for i in range(len(t_array)):
			t_per    = math.fmod(t_array[i], self.T_tot) # consider periodicity
			C_l[i],_ = self.C_lr(t_per, self.C_l_dia, self.C_l_sys)
			C_r[i],_ = self.C_lr(t_per, self.C_r_dia, self.C_r_sys)

		# compute stressed volumes
		V_lv_S    = (P_array[0,:] - self.Pth)     * C_l       + self.V_lv # left ventricle
		V_a_S     = (P_array[1,:] - self.Pth/3.)  * self.C_a  + self.V_a  # artery
		V_v_S     = (P_array[2,:])                * self.C_v  + self.V_v  # vein
		V_rv_S	  = (P_array[3,:] - self.Pth)     * C_r       + self.V_rv # right ventricle
		V_pa_S    = (P_array[4,:] - self.Pth)     * self.C_pa + self.V_pa # pulmonary artery
		V_pv_S    = (P_array[5,:] - self.Pth)     * self.C_pv + self.V_pv # pulmonary vein

		return V_lv_S, V_a_S, V_v_S, V_rv_S, V_pa_S, V_pv_S
	# ------------------------------------------------------------------------------------ #



	#--------------------------------------------------------------------------------------#
	# generate interested quantities from the Pressure/flow/volume solutions
	# Inputs:
	#       sol: pressure solution and the corresponding times
	#       save_traj: if true, output the pressure, volume and flow trajs
	# Outputs:
	#        interested output quantities
	def post_process(self, sol, save_traj=False):

		# time instances and pressure solutions
		t_sol = torch.tensor(sol.t)
		P_sol = torch.tensor(sol.y) # [P_l, P_a, P_v, P_r, P_pa, P_pv] 

		# Flow solutions
		Q_li_t, Q_lo_t, Q_a_t, Q_ri_t, Q_ro_t, Q_pv_t = self.flow_update(P_sol)

		# Stressed volume solutions 
		V_lv_S, V_a_S, V_v_S, V_rv_S, V_pa_S, V_pv_S  = self.volume_update(t_sol, P_sol)

		# Note: pressure solution will be taking from the last third to ensure stable cycles
		Tp    = int(3./4*self.num_t) # where to start in time for output calculation

		# ********************************************************************************** #
		# 1: systolic blood pressure (mmHg) 
		Pa_sys = P_sol[1, Tp:].max() / self.mmHg2Barye
		
		# 2: diastolic blood pressure (mmHg) 
		Pa_dia = P_sol[1, Tp:].min() / self.mmHg2Barye

		# 3: Right ventricular systolic pressure (mmHg)
		Pr_sys = P_sol[3, Tp:].max() / self.mmHg2Barye

		# 4: Right ventricular diastolic pressure (mmHg)
		Pr_dia = P_sol[3, Tp:].min() / self.mmHg2Barye

		# 5: Pulmonary arterial systolic pressure (mmHg)
		Ppa_sys = P_sol[4, Tp:].max() / self.mmHg2Barye

		# 6: Pulmonary arterial diastolic pressure (mmHg)
		Ppa_dia = P_sol[4, Tp:].min() / self.mmHg2Barye

		# 7: RVEDP (mmHg) (end of diastole, start of systole)
		Pr_edp  = P_sol[3, -1]        / self.mmHg2Barye

		# 8: pulmonary wedge pressure (mmHg)
		Pw      = get_average(t_sol[Tp:], P_sol[5,Tp:]) / self.mmHg2Barye 

		# 9: central venous pressure (mmHg)
		P_cvp   = get_average(t_sol[Tp:], P_sol[2,Tp:]) / self.mmHg2Barye

		# 10: systolic left ventricular volume (ml)
		V_l_sys = V_lv_S[Tp:].min()

		# 11: diastolic left ventricular volume (ml)
		V_l_dia = V_lv_S[Tp:].max()

		# 12: left ventricular ejection fraction
		#     Note: here we assume the End-systolic and End diastolic volume are just sys/dias volume
		Lvef    = ( V_l_dia - V_l_sys ) / V_l_dia

		# 13: Cariac Ouput (L/min)
		CO = get_average(t_sol[Tp:], Q_a_t[Tp:]) * 60/1000

		# 14: Systemic vascular resistance (mmHg.min/L) (HRU)
		R_svr = ( get_average(t_sol[Tp:], P_sol[1,Tp:]) - P_cvp ) / CO / self.mmHg2Barye

		# 15: Pulmonary vascular resistance (mmHg.min/L) (HRU)
		R_pvr = ( get_average(t_sol[Tp:], P_sol[4,Tp:]) - Pw ) / CO / self.mmHg2Barye
		# ********************************************************************************** #

		# group output set
		output_set = torch.tensor([Pa_sys, Pa_dia, Pr_sys, Pr_dia, Ppa_sys, Ppa_dia, Pr_edp, Pw, P_cvp,\
						V_l_sys, V_l_dia, Lvef, CO, R_svr, R_pvr])

		# to collect cvsim6 output data, do not need trajectories
		if save_traj == False:
			return output_set, None, None, None, None
		else: # to visualize pressure/volume/flow curves, return all
			return output_set, t_sol, P_sol, \
					torch.stack([Q_li_t, Q_lo_t, Q_a_t, Q_ri_t, Q_ro_t, Q_pv_t]), \
					torch.stack([V_lv_S, V_a_S, V_v_S, V_rv_S, V_pa_S, V_pv_S])
	# --------------------------------------------------------------------------------------#


	#--------------------------------------------------------------------------------------#
	# solve the cvsim6 system via build-in adpative solver and gather output data
	# Inputs:
	#        input_para: input parameters: C-R-V of each compartment plus Hr, Pth and r-sys
	#        save_traj: if True, return interested Trajectories as well
	# Outputs:
	#        interested output quantities if save_traj == False
	#        interested output quantities plus solution trajectories if save_traj == True
	def solve(self, input_para, save_traj = False):

		# ------- variable naming -------- #
		self.parameter_naming(input_para)
		# -------------------------------- #

		# --------------- solve for initial condition ---------------  #
		P0  = self.IC_solving()
		# ------------------------------------------------------------ #

		# --------------------------------------- Time Integration ---------------------------------------- #
		# total computation time
		self.T_total_cycle = 12. * self.T_tot
		
		# number of steps saved
		self.num_t         = 2000 

		# fix evaluation time stamps
		t_eval = torch.linspace(0., self.T_total_cycle, self.num_t) 

		# solve the system
		sol    = solve_ivp(self.f, [0., self.T_total_cycle], P0, method= 'Radau', t_eval = t_eval, rtol=1e-4)
		# -------------------------------------------------------------------------------------------------- #


		# ---------------------------- Post Process -------------------------------- #
		output,t_sol,P_sol,Q_sol,V_sol = self.post_process(sol, save_traj=save_traj)
		# -------------------------------------------------------------------------- #

		# make the output format work for both inVAErt and SBI
		if save_traj == False:
			return output
		else: # just for check and plotting, so it's more convenient to convert to numpy arrays
			return output.detach().numpy(),\
					t_sol.detach().numpy(),\
					P_sol.detach().numpy()/ self.mmHg2Barye,\
					Q_sol.detach().numpy(),\
					V_sol.detach().numpy()   
		#--------------------------------------------------------------------------------------------------#