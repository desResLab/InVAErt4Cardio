# making sure sbi likes my solver

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from cvsim6_simulator import *

# reference values, parametric system takes +/- 30/%
C_rv = torch.tensor([ 10, 20, 0.4, 1.2, 1.6, 100, 4.3, 8.4 ])
R_rv = torch.tensor([ 0.006, 1, 0.05, 0.003, 0.08, 0.01])
V_rv = torch.tensor([ 15, 715, 2500, 15, 90, 490])


# define lower and upper bounds
lower_bounds = torch.cat( (C_rv * 0.7, R_rv * 0.7, V_rv * 0.7) ) 
upper_bounds = torch.cat( (C_rv * 1.3, R_rv * 1.3, V_rv * 1.3) )

# define prior
prior     = utils.BoxUniform(low=lower_bounds, high=upper_bounds)

# define simulator
cvsim6 = simulator_cvsim6()

posterior = infer(cvsim6.solve, prior, method="SNPE", num_simulations=  100)
