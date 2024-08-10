
from cvsim6_simulator import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['text.usetex'] = True


# resistance unit conversion
HRU2cgs    = 80.

# ---------------------------------------------------------------------------------------------------- #
# Note: remove Hr here
output_id = ['$P_{a,sys}$ (mmHg)','$P_{a,dia}$ (mmHg)', '$P_{r,sys}$ (mmHg)', '$P_{r,dia}$ (mmHg)',
			'$P_{pa,sys}$ (mmHg)','$P_{pa,dia}$ (mmHg)','$P_{r,edp}$  (mmHg)', '$P_{w}$ (mmHg)',
			'$P_{cvp}$ (mmHg)','$V_{l,sys}$ (mL)','$V_{l,dia}$ (mL)','LVEF (-)','CO (L/min)',\
			'SVR (dyn$\cdot$s/cm$^5$)','PVR (dyn$\cdot$s/cm$^5$)']
# ---------------------------------------------------------------------------------------------------- #


# ------------------------- Test the CVSIM-6 system output ------------------------------ #
# reference parameter as per Davis 1991
input_refs = np.array([72. , -4., 1./3, \
							10., 0.4, 1.6, 100., 20., 1.2, 4.3, 8.4, \
									0.01, 0.006, 1., 0.05, 0.003, 0.08, \
											15., 715., 2500., 15., 90., 490. ] )
# init cvsim6 class
cvsim6     = simulator_cvsim6()

# solve the system and save the volume trajs
output,_,_,_,_ = cvsim6.solve( input_refs, save_traj = True) 

# print out the output quantities and compare with Table 4.4 of Davis 1991
for j in range(len(output_id)):
	if j == 13 or j == 14: # shifted one above since Hr is removed
		print('For ' + output_id[j] + ', the result is: ' + str(output[j]*HRU2cgs))
	else:
		print('For ' + output_id[j] + ', the result is: ' + str(output[j]))
# -------------------------------------------------------------------------------------- #
