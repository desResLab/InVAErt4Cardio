# plotting functions for the cvsim6-system, mostly for the stiffness analysis
from matplotlib import pyplot as plt
import matplotlib
import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# plotting args
plt.rc('font',  family='serif')
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['text.usetex'] = True
fs  = 16
DPI = 500

# define labels for plotting
labels    = ['$P_l$','$P_a$','$P_v$','$P_r$','$P_{pa}$','$P_{pv}$']	
Qlabels   = [   r'$|\boldsymbol{Q}_1|$', r'$|\boldsymbol{Q}_2|$', r'$|\boldsymbol{Q}_3|$',\
				r'$|\boldsymbol{Q}_4|$', r'$|\boldsymbol{Q}_5|$', r'$|\boldsymbol{Q}_6|$']
Qlabels_normal   = [   r'$\boldsymbol{Q}_1$', r'$\boldsymbol{Q}_2$', r'$\boldsymbol{Q}_3$',\
						r'$\boldsymbol{Q}_4$', r'$\boldsymbol{Q}_5$', r'$\boldsymbol{Q}_6$']

# ------------------------------------------------------------------------------------------ #
# # ============== Tested: 05/22/2024 ============== #
# Plot eigenvalue dynamics (eigenvalues has been re-ordered)
# Input: 
#       pth: where to save
#        t  : time
#       eigs: eigenvalues
def eigen_plotter(pth, t, eigs):
	os.makedirs(pth,exist_ok = True)
	dim = eigs.shape[1] # number of eigenvalues
	fig, ax  = plt.subplots(figsize=(6,2.3))
	for j in range(dim):
		ax.plot(t, eigs[:,j], '--', label = r'$\lambda_{' + str(j+1) + '}$', linewidth=1)
	ax.tick_params(axis='both', which='major', labelsize=fs-4)
	ax.set_xlabel(r'$t \ (\textrm{s})$',  fontsize=fs-2)
	ax.set_ylabel(r'$\lambda$', fontsize=fs-2)
	ax.legend(fontsize=fs-4)
	ax.set_xlim([t.min(), t.max()])
	fig.savefig(pth + 'Eigenvalue_last_two_cycles.png', bbox_inches='tight', pad_inches = 0, dpi = DPI)
	return None
# ------------------------------------------------------------------------------------------ #

# ---------------------------------------------------------------------------------- #
# ============== Tested: 04/18/2024 ============== #
# Compute stiffness ratio by a certain threshold tol
# Inputs:
#         eigs: the matrix of eigenvalues: number_step x dim
#         tol: certain threshold
# Outputs:
#         SR: stiffness ratio
def get_SR_with_tol(eigs, tol):
	# take real part and take abs value and sort in the feature dimension
	eigenvalues_real_abs = np.sort( abs(eigs.real), axis = 1 )
	SR = []
	# at each time, compute the stiffness ratio
	for j in range(eigs.shape[0]):
		eig_t = eigenvalues_real_abs[j] # eig values at time j
		# compute the stiffness ratio
		eig_max = eig_t[-1]
		# report if all eigs are zero
		assert eig_max > tol, "maximum eigenvalue == 0!"
		# find the smallest, non-zero eigenvalue 
		eig_min = eig_t[np.where(eig_t > tol)[0][0]]
		SR.append(eig_max/eig_min)
	return np.array(SR)
# ----------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------- #
# ============== Tested: 04/30/2024 ============== #
# prepare eigenvector radar plot
# Inputs:
#       added_path: mkdir
#       data: eigenvector matrix: 6x6, column is the vector
#       rename: if true, rename the label and file names
def radar_eigvec(added_path, data, rename = False):
	os.makedirs(added_path, exist_ok = True)
	# divide the pie
	splits    = np.linspace(0, 2 * np.pi, data.shape[0], endpoint=False).tolist()
	splits    += splits[:1]

	# start plotting, one eigenvector at a time
	for j in range(data.shape[1]):
		vec     = np.concatenate( (data[:,j], data[[0],j]), axis = 0)
		fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
		ax.fill(splits, vec,  color = 'b', alpha = 0.2) # shade area
		ax.plot(splits, vec, '--', color = 'r', linewidth = 2) # connect
		
		# rename if needed
		if type(rename) == list:
			plt.xticks(splits[:-1], rename, size = fs+22)
		else:	
			plt.xticks(splits[:-1], labels, size = fs+22)
		
		plt.yticks([0.2, 0.5, 0.8], ["0.2", "0.5", "0.8"], color="k", size=fs+12)
		plt.ylim(0, 1.0)
		ax.set_rlabel_position(-30) 
		plt.tick_params(axis='x', which='major', pad=15)
		fig.savefig(added_path + 'Q-' + str(j+1) + '.png', bbox_inches='tight', pad_inches = 0, dpi = DPI-200)
	return 0
# -------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------- #
# ============== Tested: 04/25/2024 ============== #
# show eigenvector matrix as an image
# inputs:
#       added_path: where to save the picture
#       name: name of the picture
#         B : matrix to be plotted
#        tol: tolerence of the eigenvalue, if less than that, regard as zero
def show_eigenvec_matrix(added_path, name, B, tol):
	fig, ax = plt.subplots(figsize=(6,6))
	im = ax.imshow(B, alpha = 0.75, cmap='autumn')
	ax.set_xticks(np.arange(6), labels= Qlabels_normal)
	ax.set_yticks(np.arange(6), labels= labels)
	ax.tick_params(axis='both', which='major', labelsize=fs+5)
	for j in range(6):
		for k in range(6):
			value = B[j,k]
			if abs(value) < tol: # if less than the selected tolerance, set as zero
				value = 0.0
			ax.text(k,j, f'{value:.3f}', ha='center', va='center', color='k', fontsize = fs)
	fig.savefig(added_path + name + '.png', bbox_inches='tight', pad_inches = 0, dpi = DPI)		
	return None
# -------------------------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------------- #
# ============== Tested: 04/30/2024 ============== #
# plot pressure/volume curves and superimpose SR, valve opening time
# Inputs:
#       pth: where to save 
#       fig_name: figure name
#       t: time duration
#       y1: first var to be plotted w.r.t. the left axis
#       SR: stiffness ratio
#       v1: first valve to be plotted
#       v2: second valve to be plotted
#       y1pack: legend and label for y1
#       vcolor: color for the valves
#       y2: if not None, plot the second var w.r.t the left axis
#       y2_pack: if not None, legend and label for y2
#       logy: if true, do semilogy to the left axis
def curves_with_SR(pth, fig_name, t, y1, SR, v1, v2, y1pack, vcolor, y2=None, y2pack = None, logy=False):
	fig, ax1 = plt.subplots(figsize=(5,1.8))
	
	# left axis: pressure or volume
	ax1.plot(t, y1, 'k', linewidth = 1, label = y1pack[0], alpha = 0.6)
	ax1.set_ylabel(y1pack[1], fontsize=fs)
	if type(y2) is np.ndarray:
		ax1.plot(t, y2, 'r', linewidth = 1, label = y2pack[0], alpha = 0.6)
	
	# right axis: stiffness ratio
	ax2      = ax1.twinx()
	ax2.semilogy(t, SR, 'k-.', linewidth=0.85, alpha = 0.5, label = 'SR')
	ax2.set_ylabel('SR', fontsize=fs)
	
	# superimposing valve opening/closing time
	ax3      = ax1.twinx()
	ax3.plot(t, v1, color=vcolor[0], alpha=0.0)
	ax3.plot(t, v2, color=vcolor[1], alpha=0.0)
	ax3.fill_between(t, v1, color=vcolor[0],  alpha=0.2, edgecolor='none')
	ax3.fill_between(t, v2, color=vcolor[1],  alpha=0.2, edgecolor='none')

	# hide valve's axis
	ax3.spines['right'].set_position(('outward', 60))
	ax3.yaxis.set_visible(False)
	ax3.spines['right'].set_visible(False)

	# merge legends
	lines1, labels1   = ax1.get_legend_handles_labels()
	lines2, labels2   = ax2.get_legend_handles_labels()
	ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=fs-4)
	ax1.set_xlabel('$t$ (s)', fontsize=fs)
	ax1.tick_params(axis='both', which='major', labelsize=fs-2)
	ax2.tick_params(axis='both', which='major', labelsize=fs-2)
	ax1.set_xlim([t.min(), t.max()])

	# if using semilogy
	if logy == True:
		ax1.set_yscale('log')
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.e'))

	fig.savefig(pth + fig_name + '.png', bbox_inches='tight', pad_inches = 0, dpi = DPI)
	return None
# ----------------------------------------------------------------------------------- #
