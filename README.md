# Hand Gesture Recognition Tutorial

These scripts are modified from TwentyBN's [GulpIO-benchmarks](https://github.com/TwentyBN/GulpIO-benchmarks) repository, written by [Raghav Goyal](https://github.com/raghavgoyal14) and the [TwentyBN](https://20bn.com/) team. These scripts serve as a starting point to create your own gesture recognition system using a 3D CNN. 

# Requirements

- Python 3.x
- PyTorch 0.4.0

# Instructions

## 1. Download The *Jester* Dataset

In order to train the gesture recognition system, we will use TwentyBN's [Jester Dataset](https://www.twentybn.com/datasets/jester). This dataset consists of 148,092 labeled videos, depicting 25 different classes of human hand gestures. This dataset is made available under the Creative Commons Attribution 4.0 International license CC BY-NC-ND 4.0. It can be used for academic research free of charge. In order to get access to the dataset you will need to register.

The Jester dataset is provided as one large TGZ archive and has a total download size of 22.8 GB, split into 23 parts of about 1 GB each. After downloading all the parts, you can extract the videos using:

`cat 20bn-jester-v1-?? | tar zx`

The CSV files containing the labels for the videos in the Jester dataset have already been downloaded for you and can be found in the **20bn-jester-v1/annotations** folder.

More information, including alternative ways to download the dataset, is available in the [Jester Dataset](https://www.twentybn.com/datasets/jester) website. 

## 2. Modify The Config File

In the **configs** folder you will find two config files:

* `config.json`
* `config_quick_testing.json`

The `config.json` file should be used for training the network and the `config_quick_testing.json` file should be used for quickly testing models. These files need to be modified to indicate the location of both the CSV files and the videos from the Jester dataset. The default location is `./20bn-jester-v1/annotations/` for the CSV files and `./20bn-jester-v1/videos/` for the videos. 

These config files also contain the parameters to be used during training and quick testing, such as the number of epochs, batch size, learning rate, etc... Feel free to modify these parameters as you see fit.

Please note that the default number of epochs used for training is set to `-1` in the `config.json` file, which corresponds to `999999` epochs. 

## 3. Create Your Own Model

The `model.py` module already has a simple 3D CNN model that you can use to train your gesture recognition system. You are encouraged to modify `model.py` to create your own 3D CNN architecture.

## 4. Modify the CSV Files For Quick Testing (Optional)

In the **20bn-jester-v1/annotations** folder you will find the following CSV files:

* `jester-v1-labels-quick-testing.csv`
* `jester-v1-train-quick-testing.csv`
* `jester-v1-validation-quick-testing.csv`

These files are used when quickly testing models and can be modified as you see fit. By default, the `jester-v1-labels-quick-testing.csv` file contains labels for only 4 classes of hand gestures and 1 label for "Doing other things"; the `jester-v1-train-quick-testing.csv` file contains the video ID and the corresponding labels of only 8 videos for training; and the `jester-v1-validation-quick-testing.csv` file contains the video ID and the corresponding labels for only 4 videos for validation.

Feel free to add more classes of hand gestures or more videos to the training and validation sets. To add more classes of hand gestures, simply copy and paste from the `jester-v1-labels.csv` file that contains all the 25 different classes of hand gestures. Similarly, to add more videos to the training and validation sets, simply copy and paste from the `jester-v1-train.csv` and `jester-v1-validation.csv` files that contain all the video IDs and corresponding labels from the Jester dataset.

**NOTE**: In this folder you will also find the CSV files used for training: `jester-v1-labels.csv`, `jester-v1-train.csv`, and `jester-v1-validation.csv`. These CSV files should **NOT** be modified.


# CPU/GPU Option

You can choose whether you want to train the network using only a CPU or a GPU. Due to the very large size of the Jester dataset it is **strongly recommended** that you only perform the training using a GPU. The CPU mode is favorable when you just want to quickly test models.

To specify that you want to use the CPU for your computation, use the `--use_gpu=False` flag as described below.

# Procedure

## Testing

It is recommended that you quickly test your models before you train them on the full Jester dataset. When quickly testing models we suggest you use the `config_quick_testing.json` file and the CPU. To do this, use the following command:
 
`python train.py --config configs/config_quick_testing.json --use_gpu=False`

## Training

When training a model you should use the `config.json` file and a GPU (**strongly recommended**). To train your model using a GPU use the following command:

`python train.py --config configs/config.json -g 0`
