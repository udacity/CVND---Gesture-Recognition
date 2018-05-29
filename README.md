# Hand Gesture Recognition Tutorial

These scripts are modified from TwentyBN's [GulpIO-benchmarks](https://github.com/TwentyBN/GulpIO-benchmarks) repository, written by [Raghav Goyal](https://github.com/raghavgoyal14) and the TwentyBN team.

# Requirements

- Python 3.x
- PyTorch 0.4.0

# Instructions

## 1. Download The *Jester* Dataset

In order to train the gesture recognition system, we will use the [Jester Dataset](https://www.twentybn.com/datasets/jester). This dataset consists of 148,092 labeled videos, depicting 25 different classes of human hand gestures. This dataset is made available under the Creative Commons Attribution 4.0 International license CC BY-NC-ND 4.0. It can be used for academic research free of charge. In order to get access to the dataset you will need to register.

The Jester dataset is provided as one large TGZ archive and has a total download size of 22.8 GB, split into 23 parts of about 1 GB each. After downloading all the parts, you can extract the videos using:

`cat 20bn-jester-v1-?? | tar zx`

More information, including alternative ways to download the dataset, is available in the [Jester Dataset](https://www.twentybn.com/datasets/jester) website. 

## 2. Modify The Config File

In the **configs** folder you will find two config files: `config.json` and `config_5classes.json`. The `config.json` should be used for training the network and the `config_5classes.json` file should be used for quickly testing models. These config files contain the parameters to be used during training and testing, respectively. These files need to be modified to indicate the folder location of both the CSV files and the videos from the Jester dataset, a long with the parameters you want to use for training, such as the number of epochs. Please note that the default number of epochs in the `config.json` file used for training is set to `-1` which corresponds to `999999` epochs. 

## 3. Modify the CSV Files (Optional)

Please make sure to download all the **CSV files** as well. These CSV files contain the labels for the 25 different classes of hand gestures, and the video lables for both the training and validation sets. 


# CPU/GPU Option

The code allows you to choose whether you want to train the network using only the CPU or GPUs. Due to the very large size of the Jester dataset it is strongly recommended that you only perform the training using a GPU. The CPU mode is favorable when you just want to quickly test models. 

To specify whether you want to use GPUs or CPUs for your computation, use the `--use_gpu` flag as described below.

# Procedure

## Testing

When quickly testing models we recommend you use the `config_5classes.json` file and the CPU. To do this use the following commad:

`python train.py --config configs/config_5classes.json --use_gpu=False`

## Training

When training a model we recommend you use the `config.json` file and the GPUs. To do this use the following commad:

`python train.py --config configs/config.json -g 0`




