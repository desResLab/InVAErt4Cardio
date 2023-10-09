# common helper functions

import torch

#---------------------------------------------------------------#
# Get average quantity via integration 
# Input:
#      t: interested time-interval
#      y: interested quantity over the interested time interval
def get_average(t,y):
	return torch.trapz(y,t)/(t[-1] - t[0])
#---------------------------------------------------------------#