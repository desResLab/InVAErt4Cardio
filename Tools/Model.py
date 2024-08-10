# Define InVAErt components

# reference for variational autoencoder
	# code:  ref: https://atcold.github.io/pytorch-Deep-Learning/en/week08/08-3/
	# paper: ref: Auto-Encoding Variational Bayes, Kingma et.al 2013 https://arxiv.org/abs/1312.6114

import torch
import torch.nn as nn
from Tools.DNN_tools import *
from Tools.NF_tools import *

torch.set_default_dtype(torch.float64)

#---------------------------------------------------------------------------------------------------------------#
# Define forward encoder, the emulator N_e
class Emulator(nn.Module):

	def __init__(self, emulator_in, emulator_out, emulator_paras):
		super().__init__()
		self.Ein  = emulator_in          # dimension of emulator input
		self.Eout = emulator_out	     # dimension of emulator output

		self.EHid = emulator_paras[0]    # hidden unit size
		self.EhL  = emulator_paras[1]    # hidden layer size
		self.Eact = emulator_paras[2]    # activation function type

		# ----------------------------------------------------------------------------------------- #
		# define emulator as a general MLP
		self.model_emulator = MLP_nonlinear(self.Ein, self.Eout, self.EHid, self.EhL, act=self.Eact)
		# ----------------------------------------------------------------------------------------- #

	# define forward function 
	def forward(self, v):
		return self.model_emulator(v)
#------------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------------#
# Define Real-NVP NF sampler
class NF_sampler(nn.Module):
	def __init__(self, nf_in, nf_para, device):
		super().__init__()

		self.NFin    = nf_in # input size of the NF, output is of the same size, due to bijection
		
		# network paras
		self.NFHid   = nf_para[0]    # num of hidden unit per layer
		self.NFhL    = nf_para[1]    # num of hidden layer per block
		self.NFblock = nf_para[2]    # num of alternating blocks
		self.NF_BN   = nf_para[3]    # Boolean, if use batch norm

		self.device  = device

		# # define activation function for the scaling (s) and translation (t) function, c.f. Dinh 2016
		self.act_s  = 'tanh'    # activation for s and t
		self.act_t  = 'relu'    # activation for s and t

		#self.act_s    = 'gelu'
		#self.act_t    = 'gelu'

		# ----------------------------------------------------------------------------- #
		# build NF via real nvp
		self.NF      = NF(self.device, self.NFin, 
								self.NFHid, self.NFhL,\
									self.NFblock, self.act_s, self.act_t, self.NF_BN)
		# ----------------------------------------------------------------------------- #

	# -------------------------------------------------- #
	# forward tranformation from y to z
	def forward(self, y):

		z, MiniBatchLogLikelihood = self.NF.LogProb(y)

		return z, MiniBatchLogLikelihood
	# -------------------------------------------------- #

	# -------------------------------------------------- #
	# inverse transformation from z to y
	def inverse(self, z):

		# neglect the likelihood here
		y, _ = self.NF.inverse(z)
		return y
	# -------------------------------------------------- #

	# ------------------------------------------------------------------------------- #
	# sampling from the model, i.e. sample z from standard normal and transform to y
	# Inputs:
		# num: sample size
		# model: trained real-nvp model
		# seed_control: rand seed
	# Output:
		# Samples of y, numpy it
	@torch.no_grad()
	def sampling(self, num, model, seed_control=0):
		
		model.eval()
		torch.manual_seed(seed_control)
		
		# sample z from N(0,1)
		Z_samples   = model.NF.base_dist.sample((num,))
		
		# invert the model to get y's
		Y_samples   = model.inverse(Z_samples)
		
		return Y_samples.detach().numpy()
	# --------------------------------------------------------------------------------- #



#------------------------------------------------------------------------------------------------------------------#
# Define variational auto-encoder and decoder
class VAEDecoder(nn.Module):
	def __init__(self, vae_in, vae_out, decoder_in, decoder_out, vae_para, decoder_para, device):
		super().__init__()

		self.device = device

		# define parameters for VAE encoder MLP
		self.VAEin      = vae_in     # input  for variational encoder, i.e. dim(V)- dim(aux)
		self.VAEout     = vae_out 	 # output for variational encoder, i.e. 2 x latent dimension 2xdim(W)
		self.latent_dim = int(self.VAEout/2) # number of latent dimension, i.e. dim(W)

		self.VAEHid     = vae_para[0] # number of hidden unit in VAE encoder
		self.VAEhL      = vae_para[1] # number of hidden layer in VAE encoder
		self.VAEact     = vae_para[2] # type of activation function in VAE encoder

		# define parameters for decoder MLP
		self.Din        = decoder_in
		self.Dout       = decoder_out
		self.DHid       = decoder_para[0] # number of hidden unit in decoder
		self.DhL        = decoder_para[1] # number of hidden layer in decoder
		self.Dact       = decoder_para[2] # Type of activation function in decoder

		# --------------------------------------------------------------------------------------- #
		# build VAE encoder as MLP
		self.VAE_encoder = MLP_nonlinear(self.VAEin, self.VAEout,\
											self.VAEHid, self.VAEhL, act=self.VAEact)

		# build decoder as MLP
		self.Decoder     = MLP_nonlinear(self.Din, self.Dout, self.DHid, self.DhL, act=self.Dact)
		# --------------------------------------------------------------------------------------- #

	#----------------------------------------------------------------------------------------#
	# the reparameterization trick for back prop
	# Inputs:
	#       mu : learned mean of the latent var
	#       logvar: learned logvar of the latent var
	#       Note: use logvar instead of var for numerical purposes
	def reparameterization(self, mu, logvar):
		
		# during training, apply the reparametrization trick
		if self.training:
			
			# standard deviation of the learned var
			std = torch.exp(0.5*logvar)

			# take std normal samples. Warning: do not apply random seed here!
			eps = torch.randn(std.size(), device=self.device)   

			return mu + eps * std
		# in testing, just use the learned mean
		else: 
			return mu
	#-----------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------#
	# Eval KL divergence via learned mu and logvar, take batch mean 
	# Input: 
	#      mu: batched mean vector
	#      logvar: batched logvar vector
	# Output:
	#      batch-averaged KL divergence loss
	def KL(self, mu, logvar):
		KL_loss = 0
		# loop tho each component of the latent variable w
		for i in range(mu.shape[1]):
			KL_loss += 0.5 * ( logvar[:,i].exp() - logvar[:,i] - 1 + mu[:,i].pow(2) )
		# take mini-batch mean
		return  torch.mean( KL_loss, dim = 0)
	#-----------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------#
	# forward of VAE+decoder during training and testing
	# Inputs:
		# x: sample of model input, i.e. v
		# y: sampler of model output, i.e. y
	# Outputs:
	#    KL_loss: KL divergence under Gaussian assumption
	#    x_hat = self.Decoder(y_tilde); inversion prediction
	def forward(self, x, y):

		# output of encoder is mu and logvar of the latent var
		mu_logvar = self.VAE_encoder(x).view(-1,2,self.latent_dim)

		# split mu and log sig^2
		mu      = mu_logvar[:,0,:]
		logvar  = mu_logvar[:,1,:]

		# sampling from the learned distribution via reparameterization
		w = self.reparameterization(mu, logvar) 

		# cat the latent variable with model output for inversion, i.e. build decoder input, i.e. \tilde{y}
		y_tilde = torch.cat((y, w), dim = 1) # dim 1 is the feature dimension

		# evaluate KL loss and forward the decoder
		return self.KL(mu, logvar), self.Decoder(y_tilde)
	#-----------------------------------------------------------------------------------------#


	#-------------------------------------------------------------------------------------------#
	# sampling from N(0,1), the prior, this is the classical way to draw samples from the VAE 
	# Inputs:
	# 		num: how many samples needed
	#       seed_control: repro 
	def VAE_sampling(self, num, seed_control=0):
		# control seed of torch    
		torch.manual_seed(seed_control)
		return torch.normal(0, 1, size = (num, self.latent_dim) ) # under uncorrelated assumption
	#--------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------#
	# Inversion prediction and sampling using N(0,1) prior and apply PC-sampling if needed
	# Inputs:
	#       model: trained varational auto-encoder and decoder
	#       sample_size: number of samples drawn
	#       seed_control: repro control for w only
	#       Task: which task to do
	#       y_given: always give y samples instead of sample inside this function
	#       w_given: default is N(0,1) sampling, change if use other sampling methods, e.g. see InVAErt Section 3.3
	#       denoise: if not None, apply $R$ rounds iteration for denoise-pc sampling, e.g. see InVAErt Section 3.3
	# Outputs:
	#        generated samples of v, i.e. inverse predictions
	@torch.no_grad()
	def inversion_sampling(self, model, sample_size, seed_control = 0, \
											Task = None, y_given = None, w_given = None, denoise = None):

		model.eval()
		# fix y and sample w to learn the non-identifiability
		if Task == 'FixY':

			# constant y_samples tensor, fixed
			y_samples = torch.zeros(sample_size, self.Din - self.latent_dim ) + y_given

			# Default: sample w from N(0,1) 
			if w_given is None:
				w_samples     = self.VAE_sampling(sample_size, seed_control=seed_control)
			
			# use given w samples from other sampling methods
			else:
				w_samples     = w_given

		# fix w and sample y, learn the most sensitive direction
		elif Task == 'FixW':
			
			# y samples are given, sampled from the trained NF model outside of this script
			y_samples  = y_given

			# w samples are fixed 
			w_samples  = torch.zeros(sample_size, self.latent_dim ) + self.VAE_sampling(1, seed_control=seed_control)

		# if task is None, nothing is fixed, ideally, we should recover the prior distribution of v
		else: 
			# y samples are given, sampled from the trained NF model outside of this script
			y_samples  = y_given

			# sample w from N(0,1)
			w_samples     = self.VAE_sampling(sample_size, seed_control = seed_control)


		# cat and decoding

		# build decoder input, i.e. \tilde{y} , cat in the feature dimension
		y_tilde_samples  = torch.cat((y_samples, w_samples), dim = 1)

		# decoding, i.e. inverse problem
		x_samples        = model.Decoder(y_tilde_samples)

		# if apply PC-sampling to denoise, see Section 3.3 of the InVAErt paper
		if denoise is not None:
			for r in range(denoise):
				# correction steps

				# pass predictor (x_samples) back to the trained VAE encoder
				mu_logvar = model.VAE_encoder(x_samples).view(-1,2, self.latent_dim)

				# extract mu only as the new latent variable
				w_corr      = mu_logvar[:,0,:]

				# cat 
				y_tilde_samples  = torch.cat((y_samples, w_corr), dim = 1)

				# decode again
				x_samples        = model.Decoder(y_tilde_samples)

		# return inversion samples of v
		return x_samples.detach().numpy()
		#-----------------------------------------------------------------------------------------#