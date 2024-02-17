
# PyTorch Helper Functions




## Documentation

[Documentation](https://linktodocumentation)

This repository contains objects that abstracts various tasks while working with your Pytorch based Neural Networks. The goal is to avoid writing commonly used functions each time. 

This is Work in progress repository and will keep on updating frequently. 

The repository contains following helper function objects:

    1. Torch_Trainer.py
    This python script can be used for training the Neural Networks. It accepts a fixed numbers of parameters to support the end to end training with basic visualization.
    2. helper_functions.py
    This python script contains several helper functions that helps in common Pytorch tensor related tasks such as simple batch making, basic visulizations etc

 

## Usage

To use this function you need to take care of following things:
1 helper_functions.py

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

