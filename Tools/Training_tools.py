# Training algorithm of inVAErt

import os
from Tools.Model import *
from Tools.DNN_tools import *
import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)

# every xx epochs, save a copy of the model and print statistics
printing_period = 200


# -------------------------------------------------------------------------------------------------- #
class inVAErt:
	def __init__(self, dimV, dimY, para_dim, device, latent_plus):
		
		super().__init__()
		
		self.dimV        = dimV        # dimension of the input
		self.dimY        = dimY        # dimension of the output
		self.para_dim    = para_dim    # dimension of the input excluding the aux data
		self.latent_plus = latent_plus # number of additional dimensions added to the latent space
		self.device      = device      # device used


		# Emulator input/output
		# ---------------------------------------------------- #
		# define model input output dimensions for the emulator
		self.input_emulator  = self.dimV 
		self.output_emulator = self.dimY
		# ----------------------------------------------------- #

		# NF input
		# ------------------------------------------------- #
		# define input dimension for the density estimator
		self.input_NF        = self.dimY
		# ------------------------------------------------- #

		# VAE-encoder input/output
		# ------------------------------------------------------------------------------------- #
		self.input_VAE_encoder = self.para_dim # just the parameter, no aux data
		
		# determine the latent space dimension
		if self.para_dim - self.dimY <= 0: # if input dimension <= output dimension
			self.latent_VAE = self.latent_plus
			assert self.latent_VAE != 0, "Need at least one dimension for the latent var"

			print('dim V = ' + str(self.para_dim) + ', dim Y = ' + str(self.dimY) + \
					', Latent space dim = ' + str(self.latent_plus) )
		else:
			self.latent_VAE  = self.para_dim - self.dimY + self.latent_plus
		
			print('dim V = ' + str(self.para_dim) + ', dim Y = ' + str(self.dimY) + \
					', Latent space dim = '  + str(self.para_dim - self.dimY)+ \
						' + ' + str(self.latent_plus) )

		# mean and logvar of the latent var w, combined, i.e. [mu1, mu2, ..., logvar1 ...]
		self.output_VAE_encoder     = 2*self.latent_VAE	
		# ------------------------------------------------------------------------------------- #

		# Decoder input-output
		# ----------------------------------------------------------------- #
		self.input_decoder  = self.dimY + self.latent_VAE
		self.output_decoder = self.para_dim
		# ----------------------------------------------------------------- #



	# ---------------------------------------------------------------------------------------------------------- #
	# Define inVAErt components
	# Inputs:
	#       em_para: emulator parameters: hidden unit, hidden layer, activation function
	#       nf_para: normalizing flow parameters: hidden unit, hidden layer, alt-blocks, if BN
	#       vae_para: variational encoder parameters:hidden unit, hidden layer, activation function
	#       decoder_paras: decoder parameters: hidden unit, hidden layer, activation function
	def Define_Models(self, em_para, nf_para, vae_para, decoder_para):

		#------------------------------Forward emulator model --------------------------------------------#
		inVAErt_emulator      = Emulator(self.input_emulator, self.output_emulator, em_para)
		self.inVAErt_emulator = inVAErt_emulator.to(self.device) 

		Emulator_params       = sum(p.numel() for p in self.inVAErt_emulator.parameters() if p.requires_grad)
		print( 'Number of trainable para for emulator is:'  + str(Emulator_params) )
		#--------------------------------------------------------------------------------------------------#

		# ------------------------------Density estimator ------------------------------------------------- #
		inVAErt_NF       = NF_sampler(self.input_NF, nf_para, self.device)
		self.inVAErt_NF  = inVAErt_NF.to(self.device)

		NF_params        = sum(p.numel() for p in self.inVAErt_NF.parameters() if p.requires_grad)
		print( 'Number of trainable para for NF sampler is:'  + str(NF_params) )    
		# ------------------------------------------------------------------------------------------------- #

		# -------------------------------------------- VAE+decoder ------------------------------------------- #
		inVAErt_inverse = VAEDecoder(self.input_VAE_encoder, self.output_VAE_encoder, \
									self.input_decoder, self.output_decoder, vae_para, decoder_para, self.device)
		self.inVAErt_inverse = inVAErt_inverse.to(self.device)

		
		VAE_paras      = sum(p.numel() for p in self.inVAErt_inverse.VAE_encoder.parameters() if p.requires_grad)
		print( 'Number of trainable para for VAE encoder is:'  + str(VAE_paras) ) 

		Decoder_paras  = sum(p.numel() for p in self.inVAErt_inverse.Decoder.parameters() if p.requires_grad)
		print( 'Number of trainable para for decoder is:'  + str(Decoder_paras) ) 
		# ----------------------------------------------------------------------------------------------------- #

		return self.inVAErt_emulator.double(), self.inVAErt_NF.double(), self.inVAErt_inverse.double()
	# ---------------------------------------------------------------------------------------------------------- #



	# -------------------------------------------------------------------------------------------- #
	# training-testing data split
	# Inputs:
	#       X: input dataset
	#       Y: output dataset
	# Outputs:
	#       splited dataset for training and testing
	def prepare4training(self,X,Y):

		# Training - Testing split 
		train_tensor, test_tensor, train_truth_tensor, test_truth_tensor \
					= train_test_split(torch.from_numpy(X).double().to(self.device), \
						torch.from_numpy(Y).double().to(self.device), test_size=0.25, random_state=42)

		return train_tensor, test_tensor, train_truth_tensor, test_truth_tensor
	# --------------------------------------------------------------------------------------------- #



	# ------------------------------------------------------------------------------------------------------------ #
	# Training and testing steps for the emulator
	# Inputs:
	#		PATH: where to save the model
	#       X,Y: training/testing dataset
	#       model: the emulator model
	#       num_epochs: total number of epoches
	#       nB : minibatch size
	#       lr: initial learning rate
	#       steps: step in stepLR
	#       decay: lr decay rate
	#       l2_decay: l2 regularization penalty. Default: none
	def Emulator_train_test(self, PATH, X, Y, model, num_epochs, nB, lr, steps, decay, l2_decay = 0):

		# training/testing split
		train_tensor, test_tensor, train_truth_tensor, test_truth_tensor  = self.prepare4training(X, Y)

		print('\n')
		print('---------------Start to train the emulator-----------------')

		print('Total number of epoches for training emulator is:' + str(num_epochs))

		# define optimizer, always use Adam 
		opt        = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
		
		# define step LR scheduler
		scheduler  = torch.optim.lr_scheduler.StepLR(opt, step_size=steps, gamma=decay)

		# init loss/acc for the emulator training
		Emeasure = nn.MSELoss() 
		train_save, train_acc_save = [],[]
		test_save,  test_acc_save  = [],[]

		#-----------------------Start-training-testing-------------------------#
		for epoch in range(num_epochs):

			# at each epoch, save the loss for plotting 
			train_loss_per, train_acc_per = 0,0
			test_loss_per,  test_acc_per  = 0,0

			#----------------------use dataloader to do auto-batching--------------------------------#
			# Note: always shuffle the training batches after each epoch
			# 		no need to shuffle the testing batches
			traindata    =  MyDatasetXY(train_tensor, train_truth_tensor)
			trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True) 
		
			testdata     =  MyDatasetXY(test_tensor, test_truth_tensor)
			testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False) 
			
			num_batches_train = len(trainloader) # total number of training mini batches
			num_batches_test  = len(testloader)  # total number of testing mini batches
			
			if epoch == 0:
				print('Encoder: Total num of training batches:' + str(num_batches_train)+ \
														', testing batches:' + str(num_batches_test))
			#-----------------------------------------------------------------------------------------#

			# ------------------------Training: loop tho minibatches----------------------------#
			model.train()
			for X,Y in trainloader:

				# model forward 
				Y_hat = model(X)

				# loss function via mse functional
				Loss     = Emeasure(Y_hat, Y)
				
				# MSE accuracy
				Accuracy = (  1.0 - Loss  / Emeasure( Y, torch.zeros_like(Y) )    )*100

				# keep track of the loss
				train_loss_per += Loss
				train_acc_per  += Accuracy

				# Zero-out the gradient
				opt.zero_grad()

				# back-prop
				Loss.backward()

				# gradient update
				opt.step()
			# ---------------------------------------------------------------------------------- #
			
			# ---------------------------save epoch-wise quantities----------------------------- #
			train_save.append(train_loss_per.item()/num_batches_train)    # per batch loss
			train_acc_save.append(train_acc_per.item()/num_batches_train) # per batch accuracy
			
			# for every xx epoches, print statistics for monitering
			if epoch%printing_period == 0:
				print("Training: Epoch: %d, forward loss: %1.3e" % (epoch, train_save[epoch]) ,\
					", forward acc : %.6f%%"  % (train_acc_save[epoch]), \
					', current lr: %1.3e' % (opt.param_groups[0]['lr']))

			# update learning rate via scheduler
			scheduler.step()    
			#------------------------------------------------------------------------------------#

			
			# -------------------------Testing: loop tho minibatches-----------------------------#
			model.eval()
			with torch.no_grad():
				for X,Y in testloader:

					# loss function via mse functional
					Loss_test     = Emeasure(model(X), Y)
				
					# MSE accuracy
					Accuracy_test = (  1.0 - Loss_test / Emeasure( Y, torch.zeros_like(Y) )   )*100

					test_loss_per += Loss_test
					test_acc_per  += Accuracy_test
			# -----------------------------------------------------------------------------------#

			# ---------------------------save epoch-wise quantities----------------------------- #
			test_save.append(test_loss_per.item()/num_batches_test)
			test_acc_save.append(test_acc_per.item()/num_batches_test)
			
			# for every xx epoches, print statistics for monitering
			if epoch%printing_period == 0:
				print("Testing: forward loss: %1.3e" % (test_save[epoch]) ,\
					", forward acc : %.6f%%"  % (test_acc_save[epoch]))
			# ---------------------------------------------------------------------------------- #

			#------------------------------save the model--------------------------------------#
			if epoch % printing_period == 0 or epoch == num_epochs-1: 
				
				# save trained weights
				model_save_name   = PATH + '/Emulator_model.pth'
				torch.save(model.state_dict(), model_save_name)
				
				# plot training/testing loss curves
				TT_plot(PATH, train_save, test_save, 'EmulatorLoss', yscale = 'semilogy' )
				TT_plot(PATH, train_acc_save, test_acc_save, 'EmulatorAccuracy')
			#-----------------------------------------------------------------------------------#

		return None
	# -------------------------------------------------------------------------------------------------------------#

	
	# ------------------------------------------------------------------------------------------------------------ #
	# Training and testing steps for the Real-NVP sampler
	# Inputs:
	#		PATH: where to save the model
	#       X,Y: training/testing dataset 
	#       model: the NF model
	#       num_epochs: total number of epoches
	#       nB : minibatch size
	#       lr: initial learning rate
	#       steps: step in stepLR
	#       decay: lr decay rate
	#       l2_decay: l2 regularization penalty. Default: none
	#       noise: if true, add noise during training
	#       yscale: if not None, the scaling constants of the output, used to normalize noise
	#       noise_std: if not None, use as the standard deviation of the additive noise
	#       clip: if true, apply gradient clip to avoid gradient explosion
	def NF_train_test(self, PATH, X, Y, model, num_epochs, nB, lr, steps, decay, l2_decay = 0, \
													noise = False, yscale = None, noise_std = None, clip = False):

		# training/testing split (just need the Y's here)
		_, _, train_truth_tensor, test_truth_tensor  = self.prepare4training(X, Y)

		print('\n')
		print('---------------Start to train the Real NVP-----------------')
		
		print('Total number of epoches for training NF is:' + str(num_epochs))

		# define optimizer, always use Adam first
		opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
		
		# define step LR scheduler
		scheduler  = torch.optim.lr_scheduler.StepLR(opt, step_size=steps, gamma=decay)

		# init likelihood loss for nf
		train_NF_save = []
		test_NF_save  = []
		#----------------------------------------------------------------------#

		#-----------------------Start-training-testing-------------------------#
		for epoch in range(num_epochs):

			# for each epoch, save the loss for plotting 
			train_NF_per, test_NF_per = 0,0

			#----------------------use dataloader to do auto-batching--------------------------------#
			# Note: here we feed dataloader with the truth, i.e. y's
			# always shuffle the training batches after each epoch
			traindata    =  MyDatasetX(train_truth_tensor)
			trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True) 
		
			# no need to shuffle the testing batches
			testdata     =  MyDatasetX(test_truth_tensor)
			testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False) 
			
			num_batches_train = len(trainloader) # total number of training mini batches
			num_batches_test  = len(testloader)  # total number of testing mini batches
			
			if epoch == 0:
				print('NF: Total num of training batches:' + str(num_batches_train)+ ', testing batches:'\
									 + str(num_batches_test))
			#-----------------------------------------------------------------------------------------#

			# --------------------------------Training: loop tho minibatches----------------------------------------#
			model.train()
			for Y in trainloader:
				# ----------------------------------- Add noise during training -------------------------------------- #
				# define closed form noise, assuming zero-mean Gaussian
				if noise == True: # if add noise during training
					eta       = torch.randn(Y.shape[0], Y.shape[1], device = Y.device) * noise_std  # N(0,1) * sigma
					Y_noise   = Y + eta/yscale            # add the noise and consider scaling
					# forward transformation and get the log likelihood
					_, MiniBatchLogLikelihood = model(Y_noise) 
				else: # if no noise, just use Y
					# forward transformation and get the log likelihood
					_, MiniBatchLogLikelihood = model(Y)
				# ---------------------------------------------------------------------------------------------------- #

				# compute MLE loss
				Loss           = -1.0 * torch.mean(MiniBatchLogLikelihood, dim = 0) # take mean w.r.t mini-batches
				train_NF_per  += Loss

				# Zero-out the gradient
				opt.zero_grad()

				# back-prop
				Loss.backward()

				# if apply gradient clip 
				if clip == True:
					torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

				# gradient update
				opt.step()
			#-------------------------------------------------------------------------------------------------------#

			# ---------------------------save epoch-wise quantities----------------------------- #
			train_NF_save.append(train_NF_per.item()/num_batches_train)
			
			# for every xx epoches, print statistics for monitering
			if epoch%printing_period == 0:
				print("Training: Epoch: %d, likelihood loss: %1.3e" % (epoch, train_NF_save[epoch]) ,\
					', current lr: %1.3e' % (opt.param_groups[0]['lr']))

			# update learning rate via scheduler
			scheduler.step()    
			#--------------------------------------------------------------------------------------#

			# -------------------------Testing: loop tho minibatches-------------------------------#
			model.eval()
			with torch.no_grad():
				for Y in testloader:
					# ----------------------------------- Add noise during training -------------------------------------- #
					# define closed-form noise, assuming zero-mean Gaussian
					if noise == True: # if add noise during training
						eta       = torch.randn(Y.shape[0], Y.shape[1], device = Y.device) * noise_std  # N(0,1) * sigma
						Y_noise   = Y + eta/yscale # add the noise and consider scaling
						# forward transformation and get the log likelihood
						Z, MiniBatchLogLikelihood = model(Y_noise) 
					else: # if no noise, just use Y
						# forward transformation and get the log likelihood
						Z, MiniBatchLogLikelihood = model(Y)
					# ---------------------------------------------------------------------------------------------------- #

					# #--------------check if NF is correct-------------#
					# Y_inv = model.inverse(Z)
					# if noise == False:
					# 	print(torch.norm(Y-Y_inv))
					# else:
					# 	print(torch.norm(Y_noise-Y_inv))
					# # Single precision ~1e-6, double precision ~1e-12
					# #-------------------------------------------------#

					Loss          = -1.0 * torch.mean(MiniBatchLogLikelihood, dim = 0) # take mean w.r.t mini-batches
					test_NF_per  += Loss
			#--------------------------------------------------------------------------------------#
			
			# ---------------------------save epoch-wise quantities----------------------------- #
			test_NF_save.append(test_NF_per.item()/num_batches_test)

			# for every xx epoches, print statistics for monitering
			if epoch%printing_period == 0:
				print("Testing: likelihood loss: %1.3e" % (test_NF_save[epoch]))
			#--------------------------------------------------------------------------------------#


			#------------------------------------save the model-----------------------------------------#
			if epoch % printing_period == 0 or epoch == num_epochs-1:
				
				# save trained weights
				model_save_name   = PATH + '/NF_model.pth'
				
				torch.save(model.state_dict(), model_save_name)
				
				# plot likelihood loss curves
				TT_plot(PATH, train_NF_save, test_NF_save, 'LogNFLikelihood')
			#---------------------------------------------------------------------------------------------#

		return None
	# -------------------------------------------------------------------------------------------------------------#





	# ------------------------------------------------------------------------------------------------------------ #
	# Training and testing steps for the inverse model, i.e. VAE + decoder
	# Inputs:
	#		PATH: where to save the model
	#       X,Y: training/testing dataset
	#       model: the inverse model
	#       num_epochs: total number of epoches
	#       nB : minibatch size
	#       lr: initial learning rate
	#       steps: step in stepLR
	#       decay: lr decay rate
	#       penalty: importance assigned to kl, decoder mse loss and encoder reconstraint loss 
	#       l2_decay: l2 regularization penalty. Default: none
	#       EN: if not None, apply the loss Lr
	#       noise: if true, train the model with measurement noise (closed form only)
	#       yscale: the fixed std for scaling
	#       noise_std: standard deviation of each component 
	def Inverse_train_test(self, PATH, X, Y, model, num_epochs, nB, lr, steps, decay, penalty, l2_decay = 0, \
							EN = None, noise = False, yscale = None, noise_std = None):

		# training/testing split
		train_tensor, test_tensor, train_truth_tensor, test_truth_tensor  = self.prepare4training(X, Y)

		print('\n')
		print('---------------Start to train the inverse model-----------------')
		print('Total number of epoches for training the inverse model is:' + str(num_epochs))

		# define optimizer, always try Adam first
		opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)

		# define step LR scheduler
		scheduler  = torch.optim.lr_scheduler.StepLR(opt, step_size=steps, gamma=decay)

		# init loss/acc for VAE and decoder
		Emeasure = nn.MSELoss() # mse for error measure
		
		train_save, train_acc_save = [],[]     # decoder training loss and accuracy
		test_save,  test_acc_save  = [],[]     # decoder testing loss and accuracy
		train_KL_save, test_KL_save = [], []   # KL divergence loss: training and testing
		train_enc_save, test_enc_save = [], [] # emulator reconstraint loss and accuracy, if needed


		#-----------------------Start-training-testing-------------------------#
		for epoch in range(num_epochs):

			# for each epoch, save the loss for plotting 
			train_loss_per, train_acc_per = 0,0
			test_loss_per,  test_acc_per  = 0,0
			train_KL_per,   test_KL_per   = 0,0

			# if using the Lr loss, assign reconstraint loss placeholder as well
			if EN is not None:
				train_enc_per, test_enc_per = 0, 0

			#----------------------use dataloader to do auto-batching--------------------------------#
			traindata    =  MyDatasetXY(train_tensor, train_truth_tensor)
			trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=nB, shuffle=True)
	
			testdata     =  MyDatasetXY(test_tensor, test_truth_tensor)
			testloader   =  torch.utils.data.DataLoader(testdata, batch_size=nB, shuffle=False)
			
			num_batches_train = len(trainloader) # total number of training mini batches
			num_batches_test  = len(testloader)  # total number of testing mini batches
			
			if epoch == 0:
				print('VAEDecoder: Total num of training batches:' + str(num_batches_train) \
											+ ', testing batches:' + str(num_batches_test))
			#-----------------------------------------------------------------------------------------#
			

			# ---------------------------Training: loop tho minibatches-----------------------------------#
			model.train()           # training mode for VAE and decoder
			for X,Y in trainloader:
			
				if EN is not None: # if use Lr, the reconstraint loss will be computed by the trained emulator
					EN.eval()  # never fine-tune the trained emulator, put trained emulator in eval mode
				
				#-------do not use aux data in VAE encoding------------#
				X_aux = X[:,self.para_dim:] # this is the aux data
				X     = X[:,:self.para_dim] # this is the the parameter
				#------------------------------------------------------#

				# model forward, note that kl_loss is already the mean w.r.t the current mini-batch
				# --------------------- add closed-form measurement noise if needed --------------------------- #
				if noise == True:
					# define closed form noise, assuming zero-mean Gaussian
					eta        = torch.randn(Y.shape[0], Y.shape[1], device = X.device) * noise_std
					# add normalized noise
					Y_noise    = Y + eta/yscale
					
					# calculate kl div and decode
					kl_loss, X_hat = model(X, Y_noise)
				else:
					kl_loss, X_hat = model(X, Y)
				# --------------------------------------------------------------------------------------------- #

				# loss function via mse functional, i.e. re-construction loss
				Loss     = Emeasure(X_hat, X) 

				# MSE inversion accuracy
				Accuracy = (  1.0 - Loss  / Emeasure( X, torch.zeros_like(X) )    )*100

				# keep track of the losses
				train_loss_per += Loss      # decoder re-construction loss
				train_acc_per  += Accuracy  # decoder re-construction accuracy
				train_KL_per   += kl_loss   # KL loss

				# -------------------------------------------------------------------------------------- #
				# if imposing the re-constraint loss Lr, this is so called the knowledge distill
				if EN is not None:
					# cat prediction and true aux data
					X_hat_encoding          = torch.cat(( X_hat, X_aux  ),dim=1)

					# forward the trained emulator (residual tbd)
					Y_hat                   = EN(X_hat_encoding)

					Loss_trained_emulator   = Emeasure(Y_hat, Y) 

					# track the loss
					train_enc_per           += Loss_trained_emulator
				# -------------------------------------------------------------------------------------- #

				# Zero-out the gradient
				opt.zero_grad()

				# back-prop
				if EN is None: # if no re-constraint loss Lr
					(penalty[0]*kl_loss + penalty[1]*Loss).backward()
				else:          # if use re-constraint loss Lr
					(penalty[0]*kl_loss + penalty[1]*Loss + penalty[2]*Loss_trained_emulator).backward()
				
				# gradient update
				opt.step()
			# ---------------------------------------------------------------------------------- #
			

			# ---------------------------save and plot for training----------------------------- #
			train_save.append(train_loss_per.item()/num_batches_train)
			train_KL_save.append(train_KL_per.item()/num_batches_train)
			train_acc_save.append(train_acc_per.item()/num_batches_train)

			if EN is not None:
				train_enc_save.append(train_enc_per.item()/num_batches_train)
			
			# for every xx epoches, print statistics for monitering
			if epoch%printing_period == 0:
				
				if EN is not None:
					print("Training: Epoch: %d, inversion loss: %1.3e" % (epoch, train_save[epoch]) ,\
						", inversion acc : %.6f%%"  % (train_acc_save[epoch]), \
						", KL : %1.3e"  % (train_KL_save[epoch]), \
						", Emulator_reconstraint: %1.3e" % (train_enc_save[epoch]), \
						', current lr: %1.3e' % (opt.param_groups[0]['lr']))
				else:
					print("Training: Epoch: %d, inversion loss: %1.3e" % (epoch, train_save[epoch]) ,\
						", inversion acc : %.6f%%"  % (train_acc_save[epoch]), \
						", KL : %1.3e"  % (train_KL_save[epoch]), \
						', current lr: %1.3e' % (opt.param_groups[0]['lr']))

			# update learning rate via scheduler
			scheduler.step()    
			#------------------------------------------------------------------------------------#

			# -------------------------Testing: loop tho minibatches-----------------------------#
			model.eval()
			with torch.no_grad():
				for X,Y in testloader:

					if EN is not None: # if use Lr, the reconstraint loss will be computed by the trained emulator
						EN.eval()  # never fine-tune the trained emulator, put trained emulator in eval mode
				
					#-------do not use aux data in VAE encoding------------#
					X_aux = X[:,self.para_dim:] # this is the aux data
					X     = X[:,:self.para_dim] # this is the the parameter
					#------------------------------------------------------#

					# model forward, note that kl_loss is already the mean w.r.t the current mini-batch
					# --------------------- add closed-form measurement noise if needed --------------------------- #
					if noise == True:
						# define closed form noise, assuming zero-mean Gaussian
						eta = torch.randn(Y.shape[0], Y.shape[1], device = X.device) * noise_std
						# add normalized noise
						Y_noise = Y + eta/yscale
						
						# calculate kl div and decode
						kl_loss, X_hat = model(X, Y_noise)
					else:
						kl_loss, X_hat = model(X, Y)
					# --------------------------------------------------------------------------------------------- #

					# loss function via mse functional, i.e. re-construction loss
					Loss     = Emeasure(X_hat, X) 

					# MSE inversion accuracy
					Accuracy = (  1.0 - Loss  / Emeasure( X, torch.zeros_like(X) )    )*100


					# keep track of the losses
					test_loss_per += Loss
					test_acc_per  += Accuracy
					test_KL_per   += kl_loss

					# --------------------------------------------------------------------------------- #
					# if imposing the re-constraint loss Lr, this is so called the knowledge distill
					if EN is not None:

						# cat prediction and true aux data
						X_hat_encoding          = torch.cat(( X_hat, X_aux  ),dim=1)

						# forward the trained emulator (residual tbd)
						Y_hat                   = EN(X_hat_encoding)

						Loss_trained_emulator   = Emeasure(Y_hat, Y) 

						# track the loss
						test_enc_per           += Loss_trained_emulator
					# --------------------------------------------------------------------------------- #

			#---------------------------save and plot for testing----------------------------- #
			test_save.append(test_loss_per.item()/num_batches_test)
			test_KL_save.append(test_KL_per.item()/num_batches_test)
			test_acc_save.append(test_acc_per.item()/num_batches_test)

			if EN is not None:
				test_enc_save.append(test_enc_per.item()/num_batches_test)
			# ---------------------------------------------------------------------------------- #

			# -----------------------------------------------------------------------------------  #
			# for every xx epoches, print statistics for monitering
			if epoch % printing_period == 0:
				if EN is not None:
					print("Testing: inversion loss: %1.3e" % (test_save[epoch]) ,\
						", inversion acc : %.6f%%"  % (test_acc_save[epoch]), \
						", KL : %1.3e"  % (test_KL_save[epoch]),\
						", Emulator_reconstraint: %1.3e" % (test_enc_save[epoch]))
				else:
					print("Testing: inversion loss: %1.3e" % (test_save[epoch]) ,\
						", inversion acc : %.6f%%"  % (test_acc_save[epoch]), \
						", KL : %1.3e"  % (test_KL_save[epoch]))
			# -----------------------------------------------------------------------------------  #

			#----------------------------------------save the model-----------------------------------------------#
			if epoch % printing_period == 0 or epoch == num_epochs-1:
				
				# save the trained weights
				model_save_name   = PATH + '/Inverse_model.pth'
				torch.save(model.state_dict(), model_save_name)
				
				# plot loss curves
				TT_plot(PATH, train_save, test_save, 'DecoderLoss', yscale = 'semilogy' )
				TT_plot(PATH, train_acc_save, test_acc_save, 'DecoderAccuracy')
				TT_plot(PATH, train_KL_save, test_KL_save, 'KL divergence')
				
				if EN is not None:
					TT_plot(PATH, train_enc_save, test_enc_save, 'Emulator_reconstraint', yscale = 'semilogy')
			#-----------------------------------------------------------------------------------------------------#


		return None