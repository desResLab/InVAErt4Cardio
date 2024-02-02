# Test the cvsim-6 system

import torch
import numpy as np
from cvsim6_simulator import *
from matplotlib import pyplot as plt



# construct the class
cvsim6 = simulator_cvsim6()

# define test input
Input  = torch.tensor([72., -4., 1./3, 
						10., 0.4, 1.6, 100., 20., 1.2, 4.3, 8.4,
							0.01, 0.006, 1., 0.05, 0.003, 0.08,
								15., 715., 2500., 15., 90., 490.])

# solve the system
output,t_sol,P_sol,Q_sol,V_sol = cvsim6.solve( Input, save_traj = True ) 

print(output)


# num_t  = 1000 # total number of steps
# Tp     = np.arange(int(3./4*num_t), num_t)

# plt.figure()
# #plt.plot(t_sol[Tp], P_sol[1, Tp])
# plt.plot(t_sol, P_sol[3])
# plt.show()