# Decentralization of Artificial Intelligence with federated learning on Blockchain

This repo contains the code and data for doing federated learning on MNIST dataset on Blockchain.

## IEEE Paper - Record and reward federated learning contributions with blockchain

https://ieeexplore.ieee.org/document/8945913

## Installation

Before you do anything else you will first need to install the required Python packages. These are specified in the `src/combinedrequirements.txt` file.

This project was built using Python3 but may work with Python2 given a few minor tweaks.

## Preprocessing

The next step is to build the federated dataset to do federated learning on. You can prepare it by running this script:

```
cd src/data
python federated_data_extractor.py 10
```

This will create a dataset split among 10 clients. 

## Training

Once you've generated chunks of `federated_data_x.d`, you can begin training. For this, run the following commands in parallel:

```
cd src
python blockchain.py
```

In a separate terminal:

```
cd src
python federatedlearning.py
```

Assuming you've installed all dependencies and everything else successfully, this should start federated learning on the generated federated datasets on blockchain.

## Retrieving the models

Once you've finished training, you can get the aggregated globally updated model `federated_modelx.block` per round from the `src/blocks` folder.
