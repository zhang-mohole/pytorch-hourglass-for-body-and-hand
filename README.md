# Stacked Hourglass Networks in Pytorch and fitted for hand (FreiHand dataset)

most of this repo copy from **Stacked Hourglass Networks in Pytorch.** [princeton-vl/pytorch_stacked_hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass.git)

## Getting Started

This repository provides everything necessary to train and evaluate a single-person pose estimation model on MPII. If you plan on training your own model from scratch, we highly recommend using multiple GPUs.

Requirements:

- Python 3 (code has been tested on Python 3.6)
- PyTorch (code tested with 1.0)
- CUDA and cuDNN
- Python packages (not exhaustive): opencv-python, tqdm, cffi, h5py, scipy (tested with 1.1.0)

Structure:
- ```data/```: data loading and data augmentation code
- ```models/```: network architecture definitions
- ```task/```: task-specific functions and training configuration
- ```utils/```: image processing code and miscellaneous helper functions
- ```train.py```: code for model training
- ```test.py```: code for model evaluation

- ```train_freihand.py```: code for hand pose(i.e. for now, FreiHand only)

#### Dataset

> for body pose

Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/), and place the images directory in data/MPII/

> for hand pose (i.e. for now, FreiHand only)

[Freihand dataset](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip) and place the dataset to where you want, also you need to modify the *data_root* value of *get_Frihand_loader* in the script file: ```data/Frihand/Freihand_loader.py```
#### Training and Testing

To train a network, call:

> for body pose

```python train.py -e test_run_001``` (```-e,--exp``` allows you to specify an experiment name)

To continue an experiment where it left off, you can call:

```python train.py -c test_run_001```

All training hyperparameters are defined in ```task/pose.py```, and you can modify ```__config__``` to test different options. It is likely you will have to change the batchsize to accommodate the number of GPUs you have available.

Once a model has been trained, you can evaluate it with:

```python test.py -c test_run_001```

The option "-m n" will automatically stop training after n total iterations (if continuing, would look at total iterations)

> for hand pose

just the same as body pose. only modify the ```train.py``` to ```train_freihand.py```.

BTW. the test script for FreiHand dataset is not available now, but it's quite simple to implement by modifing the ```test.py```

#### Pretrained Models

An 8HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_8hg/checkpoint.pt). It should yield validation accuracy of 0.901.

A 2HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_2hg/checkpoint.pt). It should yield validation accuracy of 0.885.

Models should be formatted as exp/<exp_name>/checkpoint.pt

Note models were trained using batch size of 16 along with Adam optimizer with LR of 1e-3 (instead of RMSProp at 2.5e-4), as they outperformed in validation. Code can easily be modified to use original paper settings. The original paper reported validation accuracy of 0.881, which this code approximately replicated. Above results also were trained for approximately 200k iters, while the original paper trained for less.

#### Training/Validation split

The train/val split is same as that found in authors' [implementation](https://github.com/princeton-vl/pose-hg-train)

#### Note

During training, occasionaly "ConnectionResetError" warning was occasionally displayed between epochs, but did not affect training.  
