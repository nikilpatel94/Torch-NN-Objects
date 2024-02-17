
# PyTorch Objects



## Documentation

[Documentation](https://github.com/nikilpatel94/Torch-NN-Objects?tab=readme-ov-file#documentation)

This repository contains objects that abstracts various tasks while working with your Pytorch based Neural Networks. The goal is to avoid writing commonly used functions each time. 

This is Work in progress repository and will keep on updating frequently. 

The repository contains following helper function objects:

1. Torch_Trainer.py
  This python script can be used for training the Neural Networks. It accepts a fixed numbers of parameters to support the end to end training with basic visualization.
2. helper_functions.py
  This python script contains several helper functions that helps in common Pytorch tensor related tasks such as simple batch making, basic visulizations etc
  
    2.1 simple_batch_maker
    This function takes several args such as batch size, sequence lenght,start index( all mendatory) and data as python lists and returns batch of these data as torch tensors. This is useful in 
    care of batching sequetial data. 

 

## Usage

To use this any files from this repository, you need to import a raw file via python requests and write it into your current folder. 

For example to use helper_functions.py use follwoing python code to use and import this.
Credit: https://www.learnpytorch.io/02_pytorch_classification/#4-make-predictions-and-evaluate-the-model

```python
import requests
from pathlib import Path

if Path('helper_functions.py').is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print('Downloading helper_functions.py file...')
  response = requests.get('https://raw.githubusercontent.com/nikilpatel94/Torch-NN-Objects/main/helper_functions.py')
  with open('helper_functions.py','wb') as f:
    f.write(response.content)
    
from helper_functions import simple_batch_maker

```

