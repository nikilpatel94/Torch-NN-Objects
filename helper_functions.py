import torch

def simple_batch_maker(batch_size:int,seq_len:int,index:int,data:list,torch_tansforms=None):
  assert len(data ) >= seq_len *batch_size
  batch = torch.empty((batch_size,seq_len),dtype = torch.float)
  for i in range(batch_size):
    batch[i] = torch.tensor( data[index:index+seq_len],dtype = torch.float ).unsqueeze(dim=0)
    index = index+seq_len
  return batch