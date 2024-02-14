import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Torch_Trainer(nn.Module):
  def __init__(self,torch_model,train_data:list,validation_data:list,torch_optimizer,epochs=100, **kwargs):
    super().__init__()
    self.model = torch_model
    self.X_train = train_data[0]
    self.y_train = train_data[1]
    self.X_valid = None
    self.y_valid = None
    if(validation_data):
      self.X_valid = validation_data[0]
      self.y_valid = validation_data[1]
    self.optimizer = torch_optimizer
    self.epochs = epochs
    self.save_model = kwargs['save_model']
    if(self.save_model):
      self.model_path = kwargs['model_path']
    self.loss_step = kwargs['loss_step']
    self.verbose = kwargs['verbose']
    self.hidden = kwargs['hidden_state']
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
    print('Starting Training...\n')
    for epoch in tqdm(range(self.epochs)):
      
      #Training
      self.model.train()
      if(self.hidden is None):
        logits, curr_train_loss = self.model(self.X_train,self.y_train)
      else:
        logits, curr_train_loss = self.model(self.X_train,self.hidden,self.y_train)
      self.loss[epoch] = curr_train_loss.item()
      self.optimizer.zero_grad()
      curr_train_loss.backward()
      self.optimizer.step()
      
      #Eval
      self.model.eval()
      curr_valid_loss = torch.empty((self.epochs,))
      if(self.hidden is None):
        _, curr_valid_loss = self.model(self.X_valid,self.y_valid)
      else:
        _, curr_valid_loss = self.model(self.X_train,self.hidden,self.y_train)
      self.valid_loss[epoch] = curr_valid_loss.item()
      if(self.verbose):
        if(epoch % self.loss_step == 0 ):
          print(f"\nStep:{epoch}| Training Loss:{curr_train_loss}| Validation_loss:{curr_valid_loss}")
    print('\nTraining Finished...')
    print(f"Final Loss| Training Loss:{curr_train_loss}| Validation_loss:{curr_valid_loss}")
    self.visualize_training()
    if(self.save_model):
      torch.save(self.model.state_dict(), self.model_path)