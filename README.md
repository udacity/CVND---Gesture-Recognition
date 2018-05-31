# Hand Gesture Recognition Tutorial

These scripts are modified from TwentyBN's [GulpIO-benchmarks](https://github.com/TwentyBN/GulpIO-benchmarks) repository, written by [Raghav Goyal](https://github.com/raghavgoyal14) and the TwentyBN team. These scripts serve as a starting point to create your own gesture recognition system. 

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

In the **configs** folder you will find two config files: `config.json` and `config_quick_testing.json`. The `config.json` should be used for training the network and the `config_quick_testing.json` file should be used for quickly testing models. These config files contain the parameters to be used during training and testing, respectively. These files need to be modified to indicate the folder location of both the CSV files and the videos from the Jester dataset, a long with the parameters you want to use for training, such as the number of epochs. Please note that the default number of epochs in the `config.json` file used for training is set to `-1` which corresponds to `999999` epochs.

## 3. Create Your Own Model

The `model.py` module already has a simple 3D CNN model that you can use to train your gesture recognition system. You are encouraged to modify `model.py` to create your own models. You can use a very small subset of the Jester dataset to quickly test your models before you train them on the full Jester dataset. 

## 4. Modify the CSV Files (Optional)

In the **20bn-jester-v1/annotations** folder you will find the CSV files containing the labels for the 25 different classes of hand gestures,`jester-v1-labels.csv`, the lables for the videos in the training set, `jester-v1-train.csv`, the lables for the videos in the validation set, `jester-v1-validation.csv`. These CSV files should **not** be modified and should be used for training the network.

In this folder you will also find the following files:
* `jester-v1-labels-quick-testing.csv`
* `jester-v1-train-quick-testing.csv`
* `jester-v1-validation-quick-testing.csv`

These files **can** be modified and we recommend you use these files when quickly testing models. These files contain labels for only 4 classes of hand gestures and contain the labels of 8 videos for training and 4 videos for validation. Feel free to modify these files as you see fit to add more classes or more videos to the training and validation sets. This is useful when doing quick tests or if you don't have a GPU and want to do training on the CPU but you don't want to use the entire Jester dataset. 


# CPU/GPU Option

The code allows you to choose whether you want to train the network using only a CPU or a GPU. Due to the very large size of the Jester dataset it is strongly recommended that you only perform the training using a GPU. The CPU mode is favorable when you just want to quickly test models. 

To specify whether you want to use the GPU or the CPU for your computation, use the `--use_gpu` flag as described below.

# Procedure

## Testing

When quickly testing models we recommend you use the `config_quick_testing.json` file and the CPU. To do this use the following commad:

`python train.py --config configs/config_quick_testing.json --use_gpu=False`

## Training

When training a model we recommend you use the `config.json` file and a GPU. To do this use the following commad:

`python train.py --config configs/config.json -g 0`




