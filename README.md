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

Please make sure to download all the **CSV files** as well. These CSV files contain the labels for the 25 different classes of hand gestures, and the video lables for both the training and validation sets. 

More information, including alternative ways to download the dataset, is available in the [Jester Dataset](https://www.twentybn.com/datasets/jester) website. 

## 2. Change The Config File

After you download the Jester dataset

# CPU/GPU option
Use `--use_gpu` flag to specify whether you want to use GPUs or CPUs for your computation

# Procedure

## Train
`python train.py --config configs/config.json -g 0`

## Test
`python train.py --config configs/config_5classes.json --use_gpu=False`
