import torch

def simple_batch_maker(batch_size:int,seq_len:int,start_index:int,data:list,torch_tansforms=None):
  assert len(data ) >= seq_len *batch_size
  batch = torch.empty((batch_size,seq_len),dtype = torch.float)
  for i in range(batch_size):
    batch[i] = torch.tensor( data[start_index:start_index+seq_len],dtype = torch.float ).unsqueeze(dim=0)
    start_index = start_index+1
  return batch

def train_validation_test_splitter(data:list,train_test_split:float,valid_split=None,verbose=False):
  train_len = int(len(data)*train_test_split)
  data_test = data[train_len:]
  #Check if validation split is requested 
  if(valid_split):
    valid_len = train_len-int(train_len*valid_split)
    data_train = data[:train_len][:valid_len]
    data_validation = data[:train_len][valid_len:]
    assert len(data) == (len(data_train)+len(data_validation)+len(data_test))
    if verbose:
      print('Length of Training,Validation,Testing\n',len(data_train),len(data_validation),len(data_test))
    return data_train,data_validation,data_test
  
  else:
    data_train = data[:train_len]
    assert len(data) == (len(data_train)+len(data_test))
    if verbose:
      print('Length of Training,Testing:\n',len(data_train),len(data_test))
    return data_train,data_test
  