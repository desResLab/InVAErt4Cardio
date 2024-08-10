import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler

# plotting specs
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True   # show edgecolor
DPI = 200

# --------------------------------------------------------------------------------------- #
# # ******** Tested, 04/06/2024 ********
# 2D scatter plotter
# Inputs:
#       path: where to save the image
#       x: to be plotted, two columns
#       pic_name : picture name
#       x1label/x2label: labels
#       Cr: color, can also be a numpy array
#       Mkr : type of marker used
#       alpha: transparency control
#       Ms: marker size
#       fs: font size
#       superIP: if not 0, super impose a lambda function
#       label_free: if true, cancel the label
#       fmt: format control
#       x2/x3: another sets to be plotted
#       xyrange: if not false, control the range of axces, x1, x2, y1, y2
#       markone: if not None, mark one point (or an array of points) on the plot
#       logy:if true, apply semilogy
#       legend_list: if not None, show legend
#       cbarange: if Cr is a numpy array, than color every scatter point and then show the colorbar
# ------------------------------------------------------------------------------------------------------- #
def scatter2D_plot(path, x, pic_name, x1label, x2label, Cr = 'r', Mkr='*', alpha=0.4, Ms = 4, fs = 8, \
		superIP = 0, label_free = False, fmt=['%.1f','%.1f'], x2 = None, x3 = None, xyrange=False, markone=None, \
		logy = False, cbarrange=[0,1], legend_list = None):

	# create path if not exist
	os.makedirs(path, exist_ok = True)

	# ----------------- Basic functionality ------------------- #
	fig = plt.figure(figsize=(9, 8))
	ax  = fig.add_subplot()
	if type(Cr) != np.ndarray: # sometimes we need a color array
		plt.plot(x[:,0],x[:,1], Mkr, alpha= alpha ,markersize = Ms, color = Cr )
	plt.xlabel(x1label,fontsize=fs+8)
	plt.ylabel(x2label,fontsize=fs+8)
	plt.tick_params(labelsize=fs+4)
	ax.xaxis.set_major_formatter(FormatStrFormatter(fmt[0]))
	ax.yaxis.set_major_formatter(FormatStrFormatter(fmt[1]))
	# ---------------------------------------------------------- #

	# ----------------------------- Advanced functionality ---------------------------------- #
	# super-impose function if needed
	if superIP != 0: 
		plt.plot(x[:,0], superIP(x[:,0]), 'k-', alpha = 0.4, linewidth=1)

	# if excluding labels
	if label_free == True:
		ax.set(xlabel=None)
		ax.set(ylabel=None)

	# if plotting an another set
	if type(x2) is np.ndarray:
		assert Cr != 'r','same color!'
		plt.plot(x2[:,0],x2[:,1], Mkr, alpha= alpha ,markersize = Ms, color = 'red' )

	# if plotting an another set
	if type(x3) is np.ndarray:
		assert Cr != 'r','same color!'
		plt.plot(x3[:,0],x3[:,1], Mkr, alpha= alpha ,markersize = Ms, color = 'gold' )

	# if control the xy axces ranges
	if type(xyrange) is list:
		plt.xlim([xyrange[0],xyrange[1]])
		plt.ylim([xyrange[2],xyrange[3]])

	# if mark one of the point
	if markone is not None:
		plt.plot(x[markone,0],x[markone,1],'bx', markeredgewidth=3, markersize=10)

	# if apply semilogy 
	if logy == True:
		plt.yscale('log')

	# if show legend
	if type(legend_list) is list:
		assert superIP == 0 and markone is None, "need additional legends for superimposed function and marked point!"
		plt.legend(legend_list, fontsize=fs+6)

	# if color based on another array
	if type(Cr) is np.ndarray:
		assert x2 is None, "only one set of data is getting colored!"
		pl   = plt.scatter(x[:,0], x[:,1], c=Cr, cmap='viridis', alpha=alpha+0.1, s=100, vmin=cbarrange[0], vmax=cbarrange[1])  
		cbar = plt.colorbar(pl)
		cbar.ax.tick_params(labelsize=fs) 
		cbar.set_label('Relative error (\%)', fontsize = fs+6, labelpad=10)  # to be generalized
	# ------------------------------------------------------------------------------------------ #

	# save the figure
	fig_name = path + '/' + pic_name + '.pdf'
	plt.grid('on',color='0.9')
	plt.savefig(fig_name,bbox_inches='tight',pad_inches = 0)
	return None
# ------------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------- #
# # ******** Tested, 05/11/2024 ********
# General histogram plotter
# Inputs:
#       path: where to save the image
#       pic_name : picture name
#       x: dist to be plotted
#       xlabel: x label to be shown
#       y: dist to be compared against
#       ylabel: y label to be shown
#       bins: number of bins
#       fs: font size
#       Cr: color
#       alpha: alpha
#       xlegend/ylegend: if not none, show legend
#       resize: if true, change the default size (a given list)
#       legend_off: if true, turn off the legends
#       y_off: if true, turn off the y ticks and y label
#		x_off: if true, turn off the x ticks and x label
def hist_plot(path, pic_name, x, xlabel, y=None, ylabel=None, bins = 50, fs = 8, Cr='r', \
		alpha = 0.5, xlegend = 'NN', ylegend = None, resize=None, legend_off = False, y_off = False, x_off = False):

	# create path if not exist
	os.makedirs(path, exist_ok = True)

	# ------------------------- Basic functionality ------------------------------ #
	fig = plt.figure(figsize=(6, 6))
	ax  = fig.add_subplot()
	plt.hist(x, bins=bins, density=True, color=Cr, alpha = alpha, label = xlegend, edgecolor='none', linewidth=0)
	plt.xlabel(xlabel,fontsize=fs+8)
	plt.ylabel('Density',fontsize=fs+8)
	plt.tick_params(labelsize=fs+4)
	plt.grid(False)
	# ---------------------------------------------------------------------------- #

	# ----------------------------- Advanced functionality ---------------------------------- #
	# plot another distribution to compare
	if type(y) is np.ndarray:
		plt.hist(y, bins=bins, density=True, color='b', alpha = alpha, label = ylegend, edgecolor='none', linewidth=0)
		plt.legend(fontsize=fs+4)
	
	# if resize the figure
	if type(resize) is list:
		plt.gcf().set_size_inches(resize[0], resize[1])

	# if turn off the legend
	if legend_off == True:
		plt.legend('', frameon=False)

	# if cancel y label and y ticks
	if y_off == True:
		plt.yticks([])
		plt.ylabel('')

	# if cancel x label and x ticks
	if x_off == True:
		plt.xlabel('')
	# ---------------------------------------------------------------------------------------- #

	# save the figure 
	fig_name = path + '/' + pic_name + '.png'
	plt.savefig(fig_name, bbox_inches='tight', pad_inches = 0.05, dpi = DPI)

	return 0
# --------------------------------------------------------------------------------------- #








