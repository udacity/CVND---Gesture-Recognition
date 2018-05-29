# Hand Gesture Recognition Tutorial

These scripts are modified from TwentyBN's [GulpIO-benchmarks](https://github.com/TwentyBN/GulpIO-benchmarks) repository, written by [Raghav Goyal](https://github.com/raghavgoyal14) and the TwentyBN team.

# Requirements

- Python 3.x
- PyTorch 0.4.0

# Instructions


## 1. Download *Jester* Dataset

For building a hand gesture recognizer, we will use the [Jester Dataset](https://www.twentybn.com/datasets/jester), a collection of almost 150k short videos of people performing hand
gestures. The dataset is available at: https://www.twentybn.com/datasets/jester

Besides downloading the data, please make sure to download all the **CSV files** as well.
(incomplete)

# CPU/GPU option
Use `--use_gpu` flag to specify whether you want to use GPUs or CPUs for your computation

# Procedure

## Train
`python train.py --config configs/config.json -g 0`

## Test
`python train.py --config configs/config_5classes.json --use_gpu=False`
