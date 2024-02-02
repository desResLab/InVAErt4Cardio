# Using amortized NPE to solve for posterior distribution of the cvsim-6 system given an output observation
# Ref: https://sbi-dev.github.io/sbi/	
# Ref: https://github.com/sbi-dev/sbi/blob/main/tutorials/01_gaussian_amortized.ipynb
# REf: https://sbi-dev.github.io/sbi/tutorial/02_flexible_interface/


import pickle
import torch
from matplotlib import pyplot as plt
import numpy as np



# ------------- load the trained model --------------#
with open('SBI_models/sbi-cvsim6.pkl', 'rb') as f:
    posterior = pickle.load(f)
# ---------------------------------------------------#


# ---------------------------------------------------- #
# Provide an observation here
observation = torch.tensor([1.371471952701661792e+02,
							6.148587664079543913e+01,
							3.594182665112259656e+01,
							-1.163478298042583514e+00,
							3.539999992734676226e+01,
							1.765462247754142311e+01,
							5.143159210659513647e+00,
							1.676189549093835041e+01,
							1.149773791260637701e+01,
							7.025459810749148915e+01,
							1.965841206233300795e+02,
							6.426232297668408311e-01,
							6.377884387969970703e+00,
							1.539347443080172617e+01,
							3.945928476408148811e+00])
# ---------------------------------------------------- #


# ----------------------------------------------------- #
# Posterior sampling

# number of posterior samples needed
N_post    = 501 
# sampling
samples   = posterior.sample((N_post,), x=observation)



plt.figure()
plt.plot(samples[:,0], samples[:,3],'x')
plt.show()
# ----------------------------------------------------- #