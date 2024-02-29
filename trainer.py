import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Torch_Trainer(nn.Module):
  def __init__(self,torch_model,data_sampler,torch_optimizer,epochs=100, **kwargs):
    super().__init__()
    self.model = torch_model
    self.data_sampler = data_sampler
    self.optimizer = torch_optimizer
    self.epochs = epochs
    self.save_model = kwargs['save_model']
    if(self.save_model):
      self.model_path = kwargs['model_path']
    self.loss_step = kwargs['loss_step']
    self.verbose = kwargs['verbose']
    self.is_visualize = kwargs['is_visualize']
    self.loss = torch.zeros((self.epochs,))
    self.valid_loss = torch.zeros((self.epochs,))

  def visualize_training(self):
    fig,axs = plt.subplots(1,1)
    axs.plot(range(self.epochs),self.loss.detach().cpu().numpy(), label='Training Loss')
    if self.valid_loss is None:
      1==1
    else:
      axs.plot(range(self.epochs),self.valid_loss.detach().cpu().numpy(), label='Validation Loss')
    plt.show()

  def forward(self):

    for epoch in tqdm(range(self.epochs)):
      X_train,y_train,X_validation,y_validation,hidden = self.data_sampler
      #Training
      self.model.train()
      if(hidden is None):
        logits, curr_train_loss = self.model(X_train,y_train)
      else:
        logits,hidden, curr_train_loss = self.model(X_train,hidden,y_train)
      self.loss[epoch] = curr_train_loss.item()
      self.optimizer.zero_grad()
      curr_train_loss.backward()
      self.optimizer.step()

      #Eval
      self.model.eval()
      curr_valid_loss = torch.empty((self.epochs,))
      if(hidden is None):
        _, curr_valid_loss = self.model(X_validation,y_validation)
      else:
        _, _,curr_valid_loss = self.model(X_validation,hidden,y_validation)
      self.valid_loss[epoch] = curr_valid_loss.item()
      if(self.verbose):
        if(epoch % self.loss_step == 0 ):
          print(f"\nStep:{epoch}| Training Loss:{curr_train_loss}| Validation_loss:{curr_valid_loss}")
    if(self.verbose):
      print('\nTraining Finished...')
      print(f"Final Loss| Training Loss:{curr_train_loss}| Validation_loss:{curr_valid_loss}")

    if (self.is_visualize):
      self.visualize_training()
    if(self.save_model):
      torch.save(self.model.state_dict(), self.model_path)